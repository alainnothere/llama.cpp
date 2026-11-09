#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>

#ifdef _WIN32
#    include <io.h>
#else
#    include <unistd.h>
#endif

#if defined(__linux__)
#    include <fcntl.h>
#endif

//
// disk-cache helpers (file-local)
//

namespace {

constexpr uint32_t DISK_CACHE_MAGIC   = 0x53504344; // 'SPCD' - Server Prompt Cache Disk
// v5: split files per conversation, one write mode each:
//   {conv}.hdr  - overwrite - magic/version/hashes, n_tok, sizes, segment table, checksum
//   {conv}.tok  - overwrite - raw llama_token array
//   {conv}.kv   - append    - pos-range attention-state blobs, applied in order on load
//   {conv}.drft - append    - same segmentation for the draft context (absent without a draft)
//   {conv}.rec  - overwrite - hybrid models only: [u32 n_tok][PARTIAL_ONLY recurrent blob];
//                             the recurrent state is a running state valid only at the latest
//                             position, so overwrite is its only correct mode (and it is small)
// checkpoints live in their own write-once cp_{conv}_{pos_min}.bin files (see below)
constexpr uint32_t DISK_CACHE_VERSION = 6;

// number of tokens covered by one .kv segment when writing a large state from scratch
// (bounds RAM on both save and load - the peak is one segment, ~500 MiB at ~60 KB/token)
constexpr uint32_t DISK_SEGMENT_TOKENS = 8192;

// backpressure for the async writer: enqueueing blocks while this many payload bytes
// are already in flight, so a huge flush cannot pile up unbounded in host RAM.
// sized to 4 segments (~512 MiB each at ~64 KB/token on the 27B): at 1 GiB the request thread
// stalled after two queued segments and waited for the disk (~200 ms per segment write). a 286k
// conversation flushed as a 35-segment rewrite then blocked the prompt for ~9 s. the normal
// 1-2 segment append must never wait here.
constexpr size_t DISK_QUEUE_MAX_BYTES = 2ull << 30; // 2 GiB

// [TAG_DISK_ENOSPC] free space every disk-cache write leaves behind on its filesystem. a write that would go below it is
// skipped with one log line (no retry loop, the next flush tries again). 2026-11-01: the hot dir's filesystem
// (/home/xlayn/theOtherDocs, shared with everything else on the box) hit 0 bytes, appends failed, and the old policy
// turned every failure into a rewrite that wrote 17 x 512 MiB and died again (~350 GiB of writes for one conversation).
// 2 GiB = the next 512 MiB segment (~200 ms on the NVMe) plus a dozen 150 MiB cp_ files of the other slot still in
// the queue, with room left for the .tok/.hdr rewrites and for whatever else writes to that disk
constexpr uint64_t DISK_FREE_MARGIN_BYTES = 2ull << 30;

// [TAG_COLD_TIER] the demote scan runs at startup, then at most once per this many seconds (or earlier when the hot dir
// runs short of space). a shorter --cache-disk-demote-after shortens it too (the tests use 1 s)
constexpr int64_t DISK_DEMOTE_SCAN_INTERVAL_S = 600;

// [TAG_COLD_TIER] copy granularity of a demotion. between chunks the writer checks for queued work and a load of the
// same conversation and abandons the copy if there is any: the cold disk is a spinner at ~200 MB/s, a 15 GB entry is
// ~75 s, and a flush that waited behind it would block the request thread at DISK_QUEUE_MAX_BYTES
constexpr size_t DISK_DEMOTE_CHUNK_BYTES = 32ull << 20;

// checkpoint files - written by spill_checkpoint() the moment create_checkpoint() captures a
// checkpoint. format: magic, version, arch_hash, vocab_hash, state_hash,
// pos_min (i32), pos_max (i32), n_tokens (i64), token_prefix_hash (u64),
// data_tgt_size (u64), data_tgt, data_dft_size (u64), data_dft.
// checkpoint files are named cp_{conversation_id}_{pos_min}.bin and live in disk_path
// v2: added token_prefix_hash. v1 files fail the version check and are ignored until evicted.
constexpr uint32_t CHECKPOINT_SPILL_MAGIC   = 0x43504B44; // 'CPKD'
constexpr uint32_t CHECKPOINT_SPILL_VERSION = 3;

// byte offsets inside a cp_ file (see format above) - used to register disk-backed checkpoints
// without reading the blobs: off_tgt = header, off_dft = header + tgt blob + its size field.
// the cp_ index ([TAG_CP_INDEX]: write_checkpoint_file's entry, read_checkpoint_file_entry) derives the
// offsets and the draft size (file size - off_dft) from it too
constexpr uint64_t CHECKPOINT_FILE_HEADER_BYTES = 5*sizeof(uint32_t) + 2*sizeof(int32_t) + sizeof(int64_t) + 2*sizeof(uint64_t);


static bool read_disk_u32(std::ifstream & f, uint32_t & v)
{
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return static_cast<bool>(f);
}

static bool read_disk_u64(std::ifstream & f, uint64_t & v)
{
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return static_cast<bool>(f);
}

// split-file extensions (v5)
constexpr std::string_view SPLIT_EXT_HDR  = ".hdr";
constexpr std::string_view SPLIT_EXT_TOK  = ".tok";
constexpr std::string_view SPLIT_EXT_KV   = ".kv";
constexpr std::string_view SPLIT_EXT_DRFT = ".drft";
constexpr std::string_view SPLIT_EXT_REC  = ".rec";

inline std::string split_path(const std::string & base, std::string_view ext) {
    return base + std::string(ext);
}

static bool disk_fsync(FILE * f) {
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

// write (truncate) + fsync, no rename. the optional second part is appended to the first
// (the .rec file is [u32 n_tok][shared blob], see disk_job::rec). a failed write removes the file
static bool disk_write_file(const std::string & path, const void * data, size_t n,
                            const void * data2 = nullptr, size_t n2 = 0) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    bool ok = n == 0 || fwrite(data, 1, n, f) == n;
    ok = ok && (n2 == 0 || fwrite(data2, 1, n2, f) == n2);
    ok = ok && fflush(f) == 0;
    ok = ok && disk_fsync(f);
    ok = fclose(f) == 0 && ok;
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    return ok;
}

// overwrite atomically: write tmp, fsync, rename over the target
static bool disk_write_file_atomic(const std::string & path, const void * data, size_t n,
                                   const void * data2 = nullptr, size_t n2 = 0) {
    const std::string tmppath = path + ".tmp";
    if (!disk_write_file(tmppath, data, n, data2, n2)) {
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmppath, path, ec);
    if (ec) {
        std::filesystem::remove(tmppath, ec);
        return false;
    }
    return true;
}

// [TAG_DISK_ENOSPC] test hook: LLAMA_DISK_CACHE_FAIL_AFTER_BYTES=N makes the .kv/.drft appends of this process fail once
// N bytes were appended in total (staged rewrites and promotions included). the failing append still writes the bytes
// up to N, like a disk that fills mid-write, so the rollback has an orphan tail to cut. unset or negative = off.
// returns true when this append must fail; n_ok = the bytes it may write first
static bool disk_fault_inject(size_t n, size_t & n_ok) {
    static const int64_t budget = []() {
        const char * e = getenv("LLAMA_DISK_CACHE_FAIL_AFTER_BYTES");
        return e ? (int64_t) std::strtoll(e, nullptr, 10) : (int64_t) -1;
    }();
    static std::atomic<int64_t> used{0};

    n_ok = n;
    if (budget < 0) {
        return false;
    }
    const int64_t before = used.fetch_add((int64_t) n);
    if (before + (int64_t) n <= budget) {
        return false;
    }
    n_ok = before >= budget ? 0 : (size_t) (budget - before);
    return true;
}

// append and fsync - the caller is responsible for truncating orphan tails first
static bool disk_append_file(const std::string & path, const void * data, size_t n) {
    size_t n_ok = n;
    const bool inject = disk_fault_inject(n, n_ok);

    FILE * f = fopen(path.c_str(), "ab");
    if (!f) {
        return false;
    }
    bool ok = n_ok == 0 || fwrite(data, 1, n_ok, f) == n_ok;
    ok = ok && fflush(f) == 0;
    ok = ok && disk_fsync(f);
    ok = fclose(f) == 0 && ok;
    return ok && !inject;
}

// cut a data file back to its committed size (an append that failed half way leaves a tail no header describes)
static void disk_truncate_to(const std::string & path, uint64_t size) {
    std::error_code ec;
    const uint64_t actual = (uint64_t) std::filesystem::file_size(path, ec);
    if (!ec && actual > size) {
        std::filesystem::resize_file(path, size, ec);
    }
}

// [TAG_COLD_TIER] copy src to dst (truncating dst) in DISK_DEMOTE_CHUNK_BYTES chunks, fsync. abort() is asked before
// every chunk. 1 = copied, 0 = I/O error, -1 = aborted. dst is removed unless the copy completed
template <typename F>
static int disk_copy_file(const std::string & src, const std::string & dst, F && abort) {
    FILE * in = fopen(src.c_str(), "rb");
    if (!in) {
        return 0;
    }
    FILE * out = fopen(dst.c_str(), "wb");
    if (!out) {
        fclose(in);
        return 0;
    }
    std::vector<char> buf(DISK_DEMOTE_CHUNK_BYTES);
    int res = 1;
    while (true) {
        if (abort()) {
            res = -1;
            break;
        }
        const size_t n = fread(buf.data(), 1, buf.size(), in);
        if (n > 0 && fwrite(buf.data(), 1, n, out) != n) {
            res = 0;
            break;
        }
        if (n < buf.size()) {
            if (ferror(in)) {
                res = 0;
            }
            break;
        }
    }
    fclose(in);
    if (res == 1 && (fflush(out) != 0 || !disk_fsync(out))) {
        res = 0;
    }
    if (fclose(out) != 0 && res == 1) {
        res = 0;
    }
    if (res != 1) {
        std::error_code ec;
        std::filesystem::remove(dst, ec);
    }
    return res;
}

// [TAG_COLD_TIER] seconds since the file was last written (-1 = no such file)
static int64_t disk_file_age_s(const std::string & path) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return -1;
    }
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(std::filesystem::file_time_type::clock::now() - t).count();
}

static int64_t disk_now_s() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// cp_{conv}_{pos_min}.bin -> pos_min. pos_min is the last field, terminated by the '.' of ".bin" and not by
// another underscore; anything else (the writer's .bin.tmp files, other conversations) is rejected
static bool parse_cp_file_name(const std::string & fn, const std::string & conv_prefix, int32_t & pmin) {
    if (fn.size() <= conv_prefix.size() + 4 || fn.compare(0, conv_prefix.size(), conv_prefix) != 0) {
        return false;
    }
    if (fn.compare(fn.size() - 4, 4, ".bin") != 0) {
        return false;
    }
    const size_t pmin_start = conv_prefix.size();
    const size_t pmin_end   = fn.size() - 4;
    int64_t v = 0;
    for (size_t i = pmin_start; i < pmin_end; ++i) {
        if (fn[i] < '0' || fn[i] > '9') {
            return false;
        }
        v = v * 10 + (fn[i] - '0');
        if (v > INT32_MAX) {
            return false;
        }
    }
    pmin = (int32_t) v;
    return true;
}

// xor of 8-byte words (tail zero-padded)
static uint64_t disk_hdr_checksum(const uint8_t * data, size_t n) {
    uint64_t sum = 0;
    for (size_t i = 0; i < n; i += 8) {
        uint64_t word = 0;
        memcpy(&word, data + i, std::min<size_t>(8, n - i));
        sum ^= word;
    }
    return sum;
}

// the .hdr bytes for a committed state (layout: see read_disk_state). state_hash is the identity the data files carry
static std::vector<uint8_t> disk_make_hdr(uint32_t arch_hash, uint32_t vocab_hash, uint32_t state_hash, uint32_t limit_tokens,
                                          const server_prompt_cache::disk_conv_state & st, uint64_t rec_size) {
    std::vector<uint8_t> hdr;
    auto put = [&hdr](const auto & v) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
        hdr.insert(hdr.end(), p, p + sizeof(v));
    };
    put(DISK_CACHE_MAGIC);
    put(DISK_CACHE_VERSION);
    put(arch_hash);
    put(vocab_hash);
    put(state_hash);
    put(limit_tokens);
    put(st.n_tok);
    put((uint32_t) st.segments.size());
    put(st.kv_size);
    put(st.drft_size);
    put(rec_size);
    for (const auto & s : st.segments) {
        put(s.kv_off);
        put(s.kv_size);
        put(s.drft_off);
        put(s.drft_size);
        put(s.p0);
        put(s.p1);
    }
    put(disk_hdr_checksum(hdr.data(), hdr.size()));
    return hdr;
}
} // anonymous namespace

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt.text},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt.text},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse(generated_input, true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_input.append(added);
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_input.text.c_str());
    auto new_msg = common_chat_parse(
        generated_input,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // --token-ledger: lifetime counters, they survive restarts
    if (metrics.ledger_enabled) {
        const auto & l = metrics.ledger;
        add_items("counter", {
            { "ledger_requests_total",                "Token ledger: requests since the ledger started",           (double) l.n_requests  },
            { "ledger_prompt_tokens_total",           "Token ledger: prompt tokens received",                      (double) l.n_prompt    },
            { "ledger_prompt_tokens_cached_total",    "Token ledger: prompt tokens served from the cache",         (double) l.n_cached    },
            { "ledger_prompt_tokens_processed_total", "Token ledger: prompt tokens processed",                     (double) l.n_processed },
            { "ledger_tokens_predicted_total",        "Token ledger: generated tokens",                            (double) l.n_gen       },
            { "ledger_reasoning_tokens_total",        "Token ledger: generated tokens inside the reasoning block", (double) l.n_reasoning },
            { "ledger_tokens_total",                  "Token ledger: prompt received plus generated tokens",       (double) l.n_total()   },
        });
        add_items("gauge", {
            { "ledger_since_seconds", "Token ledger: unix time of the first ledger entry", (double) metrics.ledger_since_unix },
        });
    }

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens          =*/ prompt.tokens.clone(),
            /*.checkpoints     =*/ prompt.checkpoints,
            /*.conversation_id =*/ prompt.conversation_id,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::find_state(const server_prompt & prompt, const server_tokens & tokens_new) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        if (f_keep_cur < 0.25f) {
            continue;
        }

        // sim-primary (see [TAG_CACHE_SELECT_SIM]). the f_keep tie-break only
        // applies between scanned candidates (it_best already set), NOT against
        // the slot's baseline f_sim_best — restoring a cache entry that merely ties
        // the slot's existing prefix buys zero extra reuse and is pure overhead.
        if (f_sim_cur > f_sim_best || (it_best != states.end() && f_sim_cur == f_sim_best && f_keep_cur > f_keep_best)) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    return it_best;
}

int32_t server_prompt_cache::restore_n_tokens(const server_prompt & prompt, const server_tokens & tokens_new, const std::string & conversation_id, bool has_dft, int * src) {
    int src_dummy = 0;
    int & src_out = src ? *src : src_dummy;
    src_out = 0;

    // same order as load(): Phase 1 RAM entry, then Phase 3 disk
    const auto it = find_state(prompt, tokens_new);
    if (it != states.end()) {
        src_out = 1;
        return it->prompt.n_tokens();
    }

    if (disk_path.empty() || conversation_id.empty()) {
        return -1;
    }

    // what the files will hold once the writer drained this conversation - load_from_disk() waits for exactly that.
    // the release flush ([TAG_RELEASE_FLUSH]) may still be in the queue, so the mirror beats the .hdr on disk
    {
        std::lock_guard<std::mutex> lock(mtx);
        const auto d = disk_convs.find(conversation_id);
        if (d != disk_convs.end() && !d->second.broken && !d->second.needs_reseed && d->second.scheduled_n_tok > 0) {
            const uint32_t n_tok = d->second.scheduled_n_tok;
            if (limit_tokens > 0 && n_tok > limit_tokens) {
                return -1;
            }
            src_out = 2;
            return (int32_t) n_tok;
        }
    }

    // the .hdr and the file sizes only: the .tok/.kv files are not read
    disk_conv_state st;
    if (!read_disk_state(conversation_id, st, /*with_tokens =*/ false)) {
        return -1;
    }
    // the same refusals as load_from_disk()
    if ((limit_tokens > 0 && st.n_tok > limit_tokens) || (has_dft && st.drft_size == 0)) {
        return -1;
    }

    src_out = 2;
    return (int32_t) st.n_tok;
}

// [TAG_POOL_TURNS_READAHEAD] a task deferred for pool room ([TAG_POOL_TURNS]) restores its conversation from disk as soon
// as the busy slot releases. measured restore rate from disk 368-483 MiB/s (2026-11-02 production, 13.8-16.4 GB per
// conversation = 15-25 s on the request path); the wait is usually minutes of the other slot's generation, so the files
// can be in the page cache by then and the restore runs at memory speed. nothing is measured here, the kernel decides.
// - not on the disk writer thread: load_from_disk() waits for every queued job of the conversation, a read-ahead job in
//   that queue would delay the very restore it is for. a detached thread holds only copies of the paths, so a shutdown
//   or the external sweep deleting the files under it is harmless (fadvise on an unlinked or truncated file is a no-op)
// - only when MemAvailable covers the files plus a margin: WILLNEED on 16 GB that does not fit evicts the model's own
//   mmap pages (or what the restore just read) and swaps; skipped and logged instead
// - Linux only (/proc/meminfo, posix_fadvise); elsewhere it logs and does nothing
bool server_prompt_cache::disk_readahead(const std::string & conversation_id) {
    if (disk_path.empty() || conversation_id.empty()) {
        return false;
    }
#if defined(__linux__)
    // [TAG_COLD_TIER] the files of a cold entry: the read-ahead warms exactly what load_from_disk will stream (and copy)
    std::string base;
    if (resolve_tier(conversation_id, base) == 0) {
        base = (std::filesystem::path(disk_path) / conversation_id).string();
    }

    std::vector<std::string> paths;
    uint64_t total = 0;
    for (const auto ext : { SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC }) {
        std::string p = split_path(base, ext);
        std::error_code ec;
        const uint64_t sz = (uint64_t) std::filesystem::file_size(p, ec);
        if (!ec && sz > 0) {
            total += sz;
            paths.push_back(std::move(p));
        }
    }
    if (paths.empty()) {
        SRV_WRN("disk read-ahead: '%s' has no .kv/.drft/.rec files - skipped\n", conversation_id.c_str());
        return false;
    }

    uint64_t mem_avail = 0;
    {
        std::ifstream mi("/proc/meminfo");
        std::string key;
        uint64_t val = 0;
        std::string unit;
        while (mi >> key >> val) {
            std::getline(mi, unit);
            if (key == "MemAvailable:") {
                mem_avail = val * 1024; // kB
                break;
            }
        }
    }

    // margin: 4 GiB or a quarter of the files, whichever is larger - room for the restore's own segment buffers
    // (one 512 MiB segment at a time) and whatever else the box runs
    const uint64_t margin = std::max<uint64_t>(4ull << 30, total / 4);
    if (mem_avail < total + margin) {
        SRV_WRN("disk read-ahead: '%s' skipped, %.1f GiB of files + %.1f GiB margin > %.1f GiB MemAvailable\n",
                conversation_id.c_str(), total / 1073741824.0, margin / 1073741824.0, mem_avail / 1073741824.0);
        return false;
    }

    SRV_WRN("disk read-ahead: '%s' %.1f GiB in %zu file(s) (MemAvailable %.1f GiB)\n",
            conversation_id.c_str(), total / 1073741824.0, paths.size(), mem_avail / 1073741824.0);

    // the thread does not log: it can outlive the logger when the server exits while a 16 GB submission is still
    // blocked in the kernel, and a log call after static destruction would crash the shutdown
    std::thread([paths]() {
        for (const auto & p : paths) {
            const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                continue;
            }
            // WILLNEED queues the reads and returns once they are submitted (the submission itself can block on a busy
            // device, hence the thread)
            posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
            close(fd);
        }
    }).detach();

    return true;
#else
    SRV_INF("disk read-ahead: '%s' skipped, not supported on this platform\n", conversation_id.c_str());
    return false;
#endif
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::string & conversation_id, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, int32_t n_room) {
    // [LLAMA_SERVER_UBATCH_STATS] 2026-10-22 night: which tier served the restore, read by the per-request line
    last_load_src   = 0;
    last_load_bytes = 0;
    last_load_lcp_us = 0;
    last_load_refused_room = false;
    static const bool lcp_timed = getenv("LLAMA_SERVER_UBATCH_STATS") != nullptr; // no clock reads when off
    const int64_t t_lcp0 = lcp_timed ? ggml_time_us() : 0;

    // Phase 1: find best match in states
    auto it_best = find_state(prompt, tokens_new);

    last_load_lcp_us = lcp_timed ? ggml_time_us() - t_lcp0 : 0;

    // [TAG_EVICT_ON_NEED] the caller evicted idle slots for this entry already; when it could not (the other slots are
    // busy), set_data would fail in state_read_meta after a partial write and wipe the slot. keep the slot's own state
    if (it_best != states.end() && n_room >= 0 && it_best->prompt.n_tokens() > n_room) {
        SRV_WRN("load: RAM entry with %d tokens does not fit in %d free cells of the unified KV pool - not restored\n",
                it_best->prompt.n_tokens(), n_room);
        last_load_refused_room = true; // [TAG_POOL_TURNS]
        return true;
    }

    if (it_best != states.end()) {
        last_load_src   = 1;
        last_load_bytes = (int64_t) (it_best->data.main.size() + it_best->data.drft.size());

        // NON-CONSUMING RESTORE — copy bytes into the live context, leave
        // the cached entry intact in `states` so future requests can match
        // it too.  See [TAG_CACHE_DONT_DELETE].  The earlier consume-on-load
        // pattern (move + erase + clear data) caused alternating-workload
        // thrash: on a poor cross-conversation match, the loaded bytes were
        // immediately discarded via do_reset in update_slots(), but the
        // cache entry was already gone — so when the original conversation
        // returned, its state had to be re-prefilled from scratch.  See
        // ContextFiles/context-disk-cache-eviction.md (2026-05-28 entry).
        //
        // Memory impact: the cached `data.main` / `data.drft` vectors stay
        // resident until natural eviction by update().  Slot KV (the live
        // copy inside ctx_tgt / ctx_dft) is in backend memory (GPU on Vulkan)
        // and not a duplicate of the host-side cached bytes.
        {
            const auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }
        }

        {
            const auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }
            }
        }

        // Copy the prompt (tokens + checkpoints + conversation_id) into the
        // slot. The state bytes were just loaded into ctx_tgt/ctx_dft via
        // set_data_ext, so the live KV is the source of truth - the blobs
        // themselves stay on the cached entry in `states` for the next
        // match.  alloc() will dedup on next save if this slot's
        // post-generation state strictly supersedes the cached entry
        // (prefix-of check in alloc()).
        prompt = it_best->prompt.clone();

        if (!disk_path.empty()) {
            merge_checkpoint_spills(prompt);
        }
        return true;
    }

    // Phase 3: disk fallback - direct lookup by conversation_id
    if (!disk_path.empty() && !conversation_id.empty()) {
        if (load_from_disk(prompt, conversation_id, ctx_tgt, ctx_dft, id_slot, n_room)) {
            last_load_src = 2;
            SRV_INF("load: restored '%s' from disk\n", conversation_id.c_str());
            merge_checkpoint_spills(prompt);
            return true;
        }
    }

    return true;
}

void server_prompt_cache::update() {
    auto evict_oldest = [this]() {
        // with the disk tier enabled, non-mtmd states are flushed write-through at
        // prompt_save time and never enter `states` - anything here is either mtmd
        // (not serializable) or RAM-only mode, so eviction is a plain destroy
        if (states.empty()) return;

        SRV_WRN(" - removing oldest entry (size = %.3f MiB)%s\n",
                states.front().size() / (1024.0 * 1024.0),
                states.front().prompt.tokens.has_mtmd ? " [mtmd: not spillable]" : "");
        states.pop_front();
    };

    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            evict_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            evict_oldest();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

bool server_prompt_cache::over_budget() {
    return limit_size > 0 && size() > limit_size;
}

void server_prompt_cache::spill_all_to_disk() {
    if (disk_path.empty()) {
        return;
    }

    // non-mtmd states are flushed write-through at prompt_save time and never enter
    // `states` when the disk tier is on - anything resident here is mtmd (not
    // serializable), so freeing RAM before a big load means dropping it
    size_t n_dropped = 0;

    while (!states.empty()) {
        ++n_dropped;
        states.pop_front();
    }

    if (n_dropped > 0) {
        SRV_WRN(" - valley: dropped %zu non-serializable state(s) - freed host RAM before load\n", n_dropped);
    }
}

//
// server_prompt_cache — disk tier
//

server_prompt_cache::server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens_in,
                                          std::string disk_path_in, int32_t queue_depth_in,
                                          int32_t checkpoint_spill_max_in,
                                          uint32_t arch_hash_in, uint32_t vocab_hash_in, uint32_t state_hash_in,
                                          std::string disk_path_cold_in, int32_t demote_after_s_in) {
    this->limit_size            = 1024ull * 1024ull * (limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens          = limit_tokens_in;
    this->disk_path             = std::move(disk_path_in);
    this->queue_depth           = queue_depth_in > 0 ? queue_depth_in : 16;
    this->checkpoint_spill_max  = checkpoint_spill_max_in;
    this->arch_hash             = arch_hash_in;
    this->vocab_hash            = vocab_hash_in;
    this->state_hash            = state_hash_in;
    this->disk_path_cold        = std::move(disk_path_cold_in);
    this->demote_after_s        = demote_after_s_in > 0 ? demote_after_s_in : 1;

    // [TAG_COLD_TIER] the cold tier only exists behind a hot one, and never as the same directory
    if (!this->disk_path_cold.empty()) {
        std::error_code ec;
        if (this->disk_path.empty()) {
            SRV_WRN("disk cache: --cache-disk-path-cold '%s' ignored without --cache-disk-path\n", this->disk_path_cold.c_str());
            this->disk_path_cold.clear();
        } else {
            std::filesystem::create_directories(this->disk_path_cold, ec);
            if (ec) {
                SRV_WRN("disk cache: cannot create cold directory '%s': %s - cold tier disabled\n",
                        this->disk_path_cold.c_str(), ec.message().c_str());
                this->disk_path_cold.clear();
            } else {
                std::filesystem::create_directories(this->disk_path, ec);
                if (std::filesystem::equivalent(this->disk_path, this->disk_path_cold, ec)) {
                    SRV_WRN("disk cache: cold directory '%s' is the hot directory - cold tier disabled\n", this->disk_path_cold.c_str());
                    this->disk_path_cold.clear();
                }
            }
        }
    }

    if (!this->disk_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(this->disk_path, ec);
        if (ec) {
            SRV_WRN("disk cache: cannot create directory '%s': %s — disk tier disabled\n",
                    this->disk_path.c_str(), ec.message().c_str());
            this->disk_path.clear();
        } else {
            SRV_WRN("disk cache: enabled at '%s' (queue depth %d, checkpoint spill max %d, arch=0x%08x, vocab=0x%08x, state=0x%08x)\n",
                    this->disk_path.c_str(), this->queue_depth, this->checkpoint_spill_max,
                    this->arch_hash, this->vocab_hash, this->state_hash);
            if (!this->disk_path_cold.empty()) {
                SRV_WRN("disk cache: cold tier at '%s', entries idle for %d s are demoted to it (scan at startup, then every %lld s)\n",
                        this->disk_path_cold.c_str(), this->demote_after_s,
                        (long long) std::min<int64_t>(DISK_DEMOTE_SCAN_INTERVAL_S, this->demote_after_s));
            }
            start_writer_if_needed();
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    // Destructor runs on the main thread when ctx_server is destroyed (post start_loop, post clean_up).
    // shutdown_and_spill stops the writer, joins, then synchronously spills remaining states + pending entries.
    // The signal handler never reaches this path (it only sets stop_flag indirectly via terminate()).
    shutdown_and_spill();
}

void server_prompt_cache::start_writer_if_needed() {
    if (worker.joinable() || disk_path.empty()) {
        return;
    }
    worker = std::thread([this]() { this->writer_loop(); });
}

void server_prompt_cache::writer_loop() {
    // [TAG_COLD_TIER] the demote scan is a writer job too (serialized with the appends), run only while the queue is
    // empty: at startup, then every min(DISK_DEMOTE_SCAN_INTERVAL_S, demote_after_s) seconds, or early on request
    const int64_t scan_interval_us = std::min<int64_t>(DISK_DEMOTE_SCAN_INTERVAL_S, demote_after_s) * 1000000;

    while (true) {
        disk_job job;
        {
            std::unique_lock<std::mutex> lock(mtx);
            if (disk_path_cold.empty()) {
                cv.wait(lock, [this]() { return stop_flag.load() || !queue.empty(); });
            } else {
                while (!stop_flag.load() && queue.empty()) {
                    const int64_t now = ggml_time_us();
                    if (demote_scan_requested || now >= t_next_demote_scan_us) {
                        job.demote_scan = true;
                        break;
                    }
                    cv.wait_for(lock, std::chrono::microseconds(t_next_demote_scan_us - now));
                }
            }
            if (stop_flag.load() && queue.empty()) {
                return;
            }
            if (job.demote_scan) {
                demote_scan_requested = false;
                t_next_demote_scan_us = ggml_time_us() + scan_interval_us;
            } else {
                job = std::move(queue.front());
                queue.pop_front();
                writer_active_conv = job.conv_id;
            }
        }

        // I/O outside the lock - writes can take seconds.
        // [TAG_DISK_ENOSPC] a failed write is rolled back to the committed header (files truncated, mirror re-seeded)
        // instead of retried: real failures here are persistent (ENOSPC, EIO, unmounted) and a retry loop would peg
        // the disk without making progress. nothing is ever deleted to make room
        process_disk_job(job);

        {
            std::lock_guard<std::mutex> lock(mtx);
            queue_bytes -= std::min(queue_bytes, job.bytes());
            writer_active_conv.clear();
            cv.notify_all();
        }
    }
}

bool server_prompt_cache::flush_from_context(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                                             common_state_pool * pool, server_rec_snapshot * rec_out) {
    if (disk_path.empty()) {
        return false;
    }

    const std::string & conv_id = prompt.conversation_id;
    if (conv_id.empty()) {
        SRV_WRN("%s", "disk cache: conversation_id is empty, refusing to flush (no fallback)\n");
        return false;
    }

    GGML_ASSERT(!prompt.tokens.has_mtmd);

    const auto & toks = prompt.tokens.get_tokens();
    const uint32_t n_tok = (uint32_t) toks.size();
    if (n_tok == 0) {
        return false;
    }

    // pos-range delta segments require the attention KV to cover the whole history
    // (no SWA sliding window) and a non-recurrent-only model. hybrid models qualify:
    // their attention child is range-sliceable and the recurrent child (a running state,
    // valid only at the latest position) is snapshotted separately into the .rec file.
    // pure-recurrent and SWA models get a single full-state rewrite instead (their
    // serialized state is small by construction).
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    bool delta_capable = llama_model_n_swa(model_tgt) == 0 &&
                         !llama_model_is_recurrent(model_tgt);
    if (ctx_dft) {
        const llama_model * model_dft = llama_get_model(ctx_dft);
        delta_capable = delta_capable &&
                        llama_model_n_swa(model_dft) == 0 &&
                        !llama_model_is_recurrent(model_dft);
    }

    // note: a hybrid draft with hybrid memory (external -md model) would lose its recurrent
    // state on restore - acceptance re-warms, correctness is unaffected. the MTP draft on
    // hybrid Qwen3.5/3.6 uses a plain attention KV, which the segments cover fully.
    const bool rec_needed = delta_capable && llama_model_is_hybrid(model_tgt);

    // first touch of this conversation: seed the in-RAM mirror from an existing .hdr.
    // [TAG_DISK_ENOSPC] same after a failed write (needs_reseed): the writer rolled the files back to the committed header
    // and bumped the epoch, so the queued jobs of the failed timeline are dropped; plan against what is on disk
    {
        bool have_state = false;
        bool reseed     = false;
        {
            std::lock_guard<std::mutex> lock(mtx);
            const auto d = disk_convs.find(conv_id);
            have_state = d != disk_convs.end();
            reseed     = have_state && d->second.needs_reseed;
        }
        if (!have_state || reseed) {
            disk_conv_state seed;
            read_disk_state(conv_id, seed); // missing/invalid files leave the default (rewrite)

            std::lock_guard<std::mutex> lock(mtx);
            auto d = disk_convs.find(conv_id);
            if (d == disk_convs.end()) {
                disk_convs.emplace(conv_id, std::move(seed));
            } else if (d->second.needs_reseed) {
                SRV_WRN("disk cache: '%s' re-planned from the committed header (%u tokens) after a failed write\n",
                        conv_id.c_str(), seed.n_tok);
                seed.epoch      = d->second.epoch;
                seed.t_last_use = d->second.t_last_use;
                d->second       = std::move(seed);
            }
        }
    }

    // decide append-from-high-water-mark vs truncate-then-append vs rewrite-from-scratch.
    // the saved list includes the generated reply, so a divergence below the high-water mark is
    // common: a UI rewind/edit, or a client that resends the previous reply slightly differently
    // (seen at f_sim 0.998). a full rewrite there cost a 286k conversation 35 x 512 MiB segment
    // writes (~200 ms each, ~9 s) on the request path. instead keep every segment that ends at or
    // below the divergence point and rewrite only from that segment end.
    uint32_t base        = 0;
    bool     rewrite     = true;
    int32_t  truncate_to = -1;
    uint32_t old_n_tok   = 0;
    uint32_t epoch0      = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & st = disk_convs[conv_id];
        epoch0 = st.epoch;
        st.t_last_use = disk_now_s();

        // longest common prefix of the scheduled tokens and the live tokens
        const size_t n_cmp = std::min<size_t>(std::min<size_t>(st.scheduled_n_tok, st.scheduled_tokens.size()), n_tok);
        const uint32_t lcp = (uint32_t) (std::mismatch(st.scheduled_tokens.begin(), st.scheduled_tokens.begin() + n_cmp, toks.begin()).first - st.scheduled_tokens.begin());

        // [fork, 2026-10-30] a file in another K/V type (restored through --cache-convert) is never appended to: the
        // next flush rewrites it in this server's type. one-time cost per conversation (2026-10 numbers: ~200 ms per
        // 8192-token segment on the 27B, ~2.5 s for a 98k conversation) against ~10 min of prefill it saved
        const bool same_type = st.file_state_hash == state_hash;
        if (!st.broken && st.scheduled_n_tok > 0 && !same_type) {
            SRV_WRN("disk cache: '%s' was saved with another K/V type (state=0x%08x, this server 0x%08x) - rewriting it in this server's type\n",
                    conv_id.c_str(), st.file_state_hash, state_hash);
        }
        const bool usable = !st.broken && !st.needs_reseed && st.scheduled_n_tok > 0 && same_type;

        // [TAG_DISK_ENOSPC] a hybrid file whose .rec is missing or stale (an append that failed after an intermediate
        // header commit, or a restore through the checkpoint fallback) is not "already flushed" even with no new token:
        // the last segment is rewritten with a fresh .rec (truncate branch below), else every restore of it reprocesses
        // the tail after the newest cp_ file again
        const bool rec_ok = !rec_needed || st.scheduled_rec;

        if (usable && lcp == st.scheduled_n_tok && n_tok == lcp && rec_ok) {
            SRV_INF("disk cache: '%s' already flushed (%u tokens) - nothing to write\n", conv_id.c_str(), n_tok);
            return true;
        }
        // [TAG_FLUSH_PREFIX_GUARD] the live tokens are a strict prefix of what the files hold: a released partial prefill
        // (cancel or error: release() keeps prompt.tokens, and the release flush only runs on a normal stop). the files
        // already cover every live token with the same tokens, so there is nothing to write. before this guard the
        // truncate branch below ran (keep = last segment end <= lcp): 2026-11-02 production, a cancelled 250801-token
        // prefill at ~115k would have chopped the good 250744-token file of d6b9 to ~114688 and deleted its .rec on the
        // next prompt_save of that slot (in-place reuse, LRU update_cache, kv_make_room eviction, the cache-idle-slots
        // pass, shutdown flush). the mirror keeps the longer list, so afterwards:
        // - the slot continues in place (its next turn shares the prefix): it prefills on top of its live cells and the
        //   release flush appends at the old high-water mark once the live tokens pass it (lcp == scheduled_n_tok), or
        //   truncates at the real divergence if the new turn differs below it - both against the files' own tokens
        // - the slot is evicted or cleared: the conversation comes back as the full disk restore
        // the K/V type does not matter here (nothing is written; a foreign-type file stays readable through
        // --cache-convert and gets rewritten in this server's type by the first flush that really writes). a broken
        // mirror is excluded: its scheduled list does not describe the files
        if (!st.broken && st.scheduled_n_tok > 0 && lcp == n_tok && n_tok < st.scheduled_n_tok) {
            SRV_WRN("disk cache: '%s' live tokens (%u) are a prefix of the %u on disk - nothing to write, the file is kept\n",
                    conv_id.c_str(), n_tok, st.scheduled_n_tok);
            return true;
        }
        if (usable && delta_capable) {
            if (lcp == st.scheduled_n_tok && n_tok > lcp) {
                base    = st.scheduled_n_tok;
                rewrite = false;
            } else if (!st.scheduled_seg_ends.empty() && st.scheduled_seg_ends.back() == st.scheduled_n_tok) {
                // segment ends are not multiples of DISK_SEGMENT_TOKENS (appends start at the old
                // high-water mark), so take them from the scheduled list. st.segments is not used:
                // it is the writer's committed view and lags the queue. keep < n_tok so at least
                // one segment is written (the hybrid .rec snapshot rides on the last one).
                uint32_t keep = 0;
                for (const uint32_t p1 : st.scheduled_seg_ends) {
                    if (p1 <= lcp && p1 < n_tok) {
                        keep = p1;
                    }
                }
                if (keep > 0) {
                    base        = keep;
                    rewrite     = false;
                    truncate_to = (int32_t) keep;
                    old_n_tok   = st.scheduled_n_tok;
                }
            }
        }
    }

    const uint32_t chunk = delta_capable ? DISK_SEGMENT_TOKENS : n_tok - base;
    const uint32_t n_seg = (n_tok - base + chunk - 1) / chunk;

    if (truncate_to >= 0) {
        SRV_WRN("disk cache: flushing '%s' tokens [%u, %u) as %u segment(s) (truncate from %u to %u, append)\n",
                conv_id.c_str(), base, n_tok, n_seg, old_n_tok, base);
    } else {
        SRV_WRN("disk cache: flushing '%s' tokens [%u, %u) as %u segment(s) (%s)\n",
                conv_id.c_str(), base, n_tok, n_seg, rewrite ? "rewrite" : "append");
    }

    bool first = true;
    for (uint32_t p0 = base; p0 < n_tok; p0 += chunk) {
        const uint32_t p1 = std::min(p0 + chunk, n_tok);

        disk_job job;
        job.conv_id     = conv_id;
        job.rewrite     = first && rewrite;
        job.truncate_to = first ? truncate_to : -1;
        job.epoch       = epoch0;
        job.staged      = rewrite; // [TAG_DISK_ENOSPC] every job of a rewrite goes to the .tmp files
        job.last        = p1 == n_tok;
        first = false;
        job.p0 = (llama_pos) p0;
        job.p1 = (llama_pos) p1;

        auto capture_failed = [&](const char * which) {
            SRV_WRN("disk cache: state capture (%s) failed for '%s' pos [%u, %u) - marking for rewrite\n",
                    which, conv_id.c_str(), p0, p1);
            std::lock_guard<std::mutex> lock(mtx);
            disk_convs[conv_id].broken = true;
            // [TAG_DISK_ENOSPC] the jobs of this flush already queued are dropped (a staged rewrite's .tmp files with them)
            disk_convs[conv_id].epoch++;
        };

        {
            const size_t sz = delta_capable
                ? llama_state_seq_get_size_range(ctx_tgt, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (sz == 0) {
                capture_failed("main");
                return false;
            }
            uint8_t * dst = job.kv.reserve_for_write(sz);
            const size_t got = delta_capable
                ? llama_state_seq_get_data_range(ctx_tgt, dst, sz, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_data_ext(ctx_tgt, dst, sz, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (got != sz) {
                capture_failed("main");
                return false;
            }
        }

        if (ctx_dft) {
            const size_t sz = delta_capable
                ? llama_state_seq_get_size_range(ctx_dft, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (sz == 0) {
                capture_failed("drft");
                return false;
            }
            uint8_t * dst = job.drft.reserve_for_write(sz);
            const size_t got = delta_capable
                ? llama_state_seq_get_data_range(ctx_dft, dst, sz, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_data_ext(ctx_dft, dst, sz, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (got != sz) {
                capture_failed("drft");
                return false;
            }
        }

        job.tokens.assign(toks.begin(), toks.begin() + p1);

        // [TAG_DISK_ENOSPC] the rewrite's space check covers the whole staged flush (the old entry stays on disk until
        // the swap, so both copies coexist): segments are equal-sized except the tail, the .rec (~157 MB on the 27B)
        // fits in DISK_FREE_MARGIN_BYTES
        if (job.rewrite) {
            job.est_bytes = (uint64_t) (job.kv.size() + job.drft.size()) * n_seg + (uint64_t) n_tok * sizeof(llama_token);
        }

        // the last chunk of a hybrid flush carries the recurrent-state snapshot - it is
        // captured from the live context and only valid at the current position (n_tok)
        if (rec_needed && p1 == n_tok) {
            const size_t sz = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (sz == 0) {
                capture_failed("rec");
                return false;
            }
            // the n_tok prefix of the .rec file is written separately (process_disk_job), so the buffer
            // holds exactly the PARTIAL_ONLY blob a checkpoint of this position would hold
            uint8_t * dst = job.rec.reserve_for_write(sz, pool);
            const size_t got = llama_state_seq_get_data_ext(ctx_tgt, dst, sz, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (got != sz) {
                capture_failed("rec");
                return false;
            }
            job.rec_n_tok = n_tok;

            // [TAG_REC_REUSE] hand the same bytes to the slot: checkpoint A of the next request sits at
            // this position (create_checkpoint checks position, token prefix and state_gen)
            if (rec_out) {
                rec_out->blob              = job.rec;
                rec_out->n_tokens          = n_tok;
                rec_out->token_prefix_hash = hash_token_prefix(prompt.tokens, n_tok);
                rec_out->pos_min           = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id_slot);
                rec_out->pos_max           = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id_slot);
            }
        }

        // enqueue with byte-budget backpressure so a big flush cannot pile up in host RAM
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return queue_bytes < DISK_QUEUE_MAX_BYTES; });

            auto & st = disk_convs[conv_id];
            // [TAG_DISK_ENOSPC] a write of this conversation failed while this flush was enqueueing: the jobs already
            // queued carry the old epoch and are dropped, the rest is not worth capturing. the next flush re-plans
            if (st.epoch != epoch0) {
                SRV_WRN("disk cache: '%s' flush stopped at pos %u - an earlier write failed, the next flush re-plans from the committed header\n",
                        conv_id.c_str(), p0);
                return false;
            }
            st.scheduled_rec    = !job.rec.empty();
            st.scheduled_n_tok  = p1;
            st.scheduled_tokens = job.tokens;
            if (job.rewrite) {
                st.scheduled_seg_ends.clear();
                st.file_state_hash = state_hash; // the rewrite writes this server's header (process_disk_job)
            }
            while (job.truncate_to >= 0 && !st.scheduled_seg_ends.empty() && st.scheduled_seg_ends.back() > (uint32_t) job.truncate_to) {
                st.scheduled_seg_ends.pop_back();
            }
            st.scheduled_seg_ends.push_back(p1);

            queue_bytes += job.bytes();
            queue.push_back(std::move(job));
            cv.notify_all();
        }
    }

    return true;
}

void server_prompt_cache::process_disk_job(const disk_job & job) {
    if (job.demote_scan) {
        demote_scan();
        return;
    }
    if (job.ckpt) {
        write_checkpoint_file(*job.ckpt, job.conv_id);
        return;
    }

    // snapshot the committed state
    disk_conv_state st;
    {
        std::lock_guard<std::mutex> lock(mtx);
        st = disk_convs[job.conv_id];
    }

    // [TAG_DISK_ENOSPC] planned against a timeline that a failed write rolled back: its segment would leave a hole
    if (job.epoch != st.epoch) {
        SRV_INF("disk cache: dropping write of '%s' pos [%d, %d) - planned before a failed write\n", job.conv_id.c_str(), job.p0, job.p1);
        if (job.staged) {
            abort_staging(job.conv_id);
        }
        return;
    }
    if (st.broken && !job.rewrite) {
        SRV_WRN("disk cache: dropping append for broken conversation '%s' (a rewrite will follow)\n", job.conv_id.c_str());
        return;
    }
    if (job.staged) {
        process_staged_job(job, st);
        return;
    }

    // appends go to the tier that holds the committed files ([TAG_COLD_TIER]: cold only when a promotion was skipped)
    const std::string & dir     = st.tier == 2 && !disk_path_cold.empty() ? disk_path_cold : disk_path;
    const std::string base      = (std::filesystem::path(dir) / job.conv_id).string();
    const std::string kv_path   = split_path(base, SPLIT_EXT_KV);
    const std::string drft_path = split_path(base, SPLIT_EXT_DRFT);
    const std::string rec_path  = split_path(base, SPLIT_EXT_REC);
    const std::string tok_path  = split_path(base, SPLIT_EXT_TOK);
    const std::string hdr_path  = split_path(base, SPLIT_EXT_HDR);

    const int64_t t_start = ggml_time_us();

    // the files are inconsistent with the header (smaller than it says): nothing to roll back to, rewrite. the staged
    // rewrite leaves this entry in place until the new one is complete
    auto mark_broken = [&](const char * what) {
        SRV_WRN("disk cache: %s failed for '%s' - marking conversation for rewrite\n", what, job.conv_id.c_str());
        std::lock_guard<std::mutex> lock(mtx);
        disk_convs[job.conv_id].broken = true;
    };

    std::error_code ec;

    if (job.truncate_to >= 0) {
        // divergence below the high-water mark: drop the segments above truncate_to. lower segments are not touched.
        // [TAG_DISK_ENOSPC] the shorter header is committed BEFORE the tails are cut: until then the old header describes
        // the old files, after it the new header describes their prefix, so a failure anywhere in this job (or in the
        // append that follows) leaves an entry that loads. not a staged copy: the kept prefix is up to 15 GB (286k
        // tokens on the 27B), copying it to make a tail edit atomic would cost more than the ~9 s rewrite this path
        // exists to avoid
        while (!st.segments.empty() && st.segments.back().p1 > job.truncate_to) {
            st.segments.pop_back();
        }
        if (st.segments.empty() || st.segments.back().p1 != job.truncate_to) {
            SRV_WRN("disk cache: truncate boundary %d of '%s' is not a committed segment end - flush dropped\n", job.truncate_to, job.conv_id.c_str());
            rollback_mirror(job.conv_id, "truncate boundary");
            return;
        }
        const auto & last = st.segments.back();
        st.kv_size   = last.kv_off + last.kv_size;
        st.drft_size = last.drft_off + last.drft_size;
        st.n_tok     = (uint32_t) job.truncate_to;
        // .rec is valid at the old n_tok only: the new header has none (the loader falls back to a cp_ file if the
        // append below fails), the last job of this flush writes the new one
        st.rec_size  = 0;

        const std::vector<uint8_t> hdr = disk_make_hdr(arch_hash, vocab_hash, state_hash, (uint32_t) limit_tokens, st, 0);
        if (!disk_write_file_atomic(hdr_path, hdr.data(), hdr.size())) {
            SRV_WRN("disk cache: truncate header write failed for '%s' - flush dropped, the entry is unchanged\n", job.conv_id.c_str());
            rollback_mirror(job.conv_id, "truncate hdr");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto & live = disk_convs[job.conv_id];
            live.n_tok     = st.n_tok;
            live.kv_size   = st.kv_size;
            live.drft_size = st.drft_size;
            live.rec_size  = 0;
            live.segments  = st.segments;
        }
        std::filesystem::remove(rec_path, ec);
    } else if (job.p0 != (llama_pos) st.n_tok) {
        // [TAG_DISK_ENOSPC] an append must start at the committed high-water mark. it cannot after a dropped segment of
        // the same flush (that one already rolled back and bumped the epoch, so this is a defensive check)
        SRV_WRN("disk cache: append of '%s' pos [%d, %d) does not start at the committed %u tokens - flush dropped\n",
                job.conv_id.c_str(), job.p0, job.p1, st.n_tok);
        rollback_mirror(job.conv_id, "gap");
        return;
    }

    // crash-orphan cleanup: the data files must be at least as large as the committed
    // sizes; extra tail bytes come from an append whose header never committed
    {
        const uint64_t kv_actual = (uint64_t) std::filesystem::file_size(kv_path, ec);
        if (ec || kv_actual < st.kv_size) {
            mark_broken("kv size check");
            return;
        }
        if (st.drft_size > 0) {
            const uint64_t drft_actual = (uint64_t) std::filesystem::file_size(drft_path, ec);
            if (ec || drft_actual < st.drft_size) {
                mark_broken("drft size check");
                return;
            }
        }
        disk_truncate_to(kv_path, st.kv_size);
        if (st.drft_size > 0 || !job.drft.empty()) {
            disk_truncate_to(drft_path, st.drft_size);
        }
    }

    // [TAG_DISK_ENOSPC] skip (one line, no retry) instead of filling the disk
    if (!disk_space_ok(dir, job.kv.size() + job.drft.size() + job.rec.size() + job.tokens.size() * sizeof(llama_token),
                       job.conv_id, "append")) {
        rollback_mirror(job.conv_id, "space");
        return;
    }

    // [TAG_DISK_ENOSPC] a failed write is rolled back to the committed header: the data files are cut back to the sizes
    // it names, the conversation is NOT marked broken (the old policy deleted the good entry and rewrote it from scratch,
    // and under ENOSPC that rewrite failed on the same segment again, every turn), and the mirror re-plans from disk
    const std::string rec_new = rec_path + ".new";
    auto fail = [&](const char * what) {
        disk_truncate_to(kv_path, st.kv_size);
        disk_truncate_to(drft_path, st.drft_size);
        std::filesystem::remove(rec_new, ec);
        SRV_WRN("disk cache: %s failed for '%s' pos [%d, %d) - rolled back to the committed %u tokens, the entry stays loadable\n",
                what, job.conv_id.c_str(), job.p0, job.p1, st.n_tok);
        rollback_mirror(job.conv_id, what);
    };

    disk_conv_state::seg_entry seg;
    seg.kv_off    = st.kv_size;
    seg.kv_size   = job.kv.size();
    seg.drft_off  = st.drft_size;
    seg.drft_size = job.drft.size();
    seg.p0        = job.p0;
    seg.p1        = job.p1;

    if (!disk_append_file(kv_path, job.kv.data(), job.kv.size())) {
        fail("kv append");
        return;
    }
    if (!job.drft.empty() && !disk_append_file(drft_path, job.drft.data(), job.drft.size())) {
        fail("drft append");
        return;
    }

    // recurrent-state snapshot (hybrid models, last chunk of a flush) - overwrite mode. written next to the old one and
    // renamed over it only after the header commit: until then the old .rec stays valid for the old header
    uint64_t rec_size_new = st.rec_size;
    if (!job.rec.empty()) {
        if (!disk_write_file(rec_new, &job.rec_n_tok, sizeof(uint32_t), job.rec.data(), job.rec.size())) {
            fail("rec write");
            return;
        }
        rec_size_new = sizeof(uint32_t) + job.rec.size();
    }

    if (!disk_write_file_atomic(tok_path, job.tokens.data(), job.tokens.size() * sizeof(llama_token))) {
        fail("tok write");
        return;
    }

    disk_conv_state nst = st;
    nst.segments.push_back(seg);
    nst.kv_size   += job.kv.size();
    nst.drft_size += job.drft.size();
    nst.n_tok      = (uint32_t) job.tokens.size();

    // header last - a crash before this point leaves only an orphan tail in the data
    // files, which the next append (or load) truncates away using the old header
    const std::vector<uint8_t> hdr = disk_make_hdr(arch_hash, vocab_hash, state_hash, (uint32_t) limit_tokens, nst, rec_size_new);
    if (!disk_write_file_atomic(hdr_path, hdr.data(), hdr.size())) {
        fail("hdr write");
        return;
    }

    if (!job.rec.empty()) {
        std::filesystem::rename(rec_new, rec_path, ec);
        if (ec) {
            // the header names a .rec this file is not: read_disk_state flags it, the loader takes a cp_ file instead
            SRV_WRN("disk cache: '%s' .rec rename failed (%s) - the next restore uses a checkpoint\n", job.conv_id.c_str(), ec.message().c_str());
        }
    }

    const size_t n_segs = nst.segments.size();

    // commit
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & live = disk_convs[job.conv_id];
        live.n_tok          = nst.n_tok;
        live.kv_size        = nst.kv_size;
        live.drft_size      = nst.drft_size;
        live.rec_size       = rec_size_new;
        live.segments       = std::move(nst.segments);
        live.broken         = false;
        live.t_last_use     = disk_now_s();
    }

    const double t_ms = (ggml_time_us() - t_start) / 1e3;
    SRV_WRN("disk cache: wrote '%s' segment pos [%d, %d) %.1f MiB (drft %.1f MiB, rec %.1f MiB) in %.0f ms - now %u tokens in %zu segment(s)\n",
            job.conv_id.c_str(), job.p0, job.p1,
            job.kv.size() / (1024.0 * 1024.0), job.drft.size() / (1024.0 * 1024.0),
            job.rec.size() / (1024.0 * 1024.0),
            t_ms, nst.n_tok, n_segs);
}

// [TAG_DISK_ENOSPC] one job of a from-scratch flush. the segments go to <id>.kv.tmp/.drft.tmp in the hot dir and the
// last job renames everything over the old entry once its own header is on disk. 2026-11-01: the old rewrite removed
// the valid .kv/.drft/.rec first, committed a header per segment, and died on segment 18 of a 141k conversation under
// ENOSPC, leaving a hybrid header with rec_size 0 that the loader then read as a full-state file (151k-token reprocess,
// 152 s, again 2026-11-04). now a failure at any point leaves the previous entry exactly as it was
void server_prompt_cache::process_staged_job(const disk_job & job, disk_conv_state & st) {
    const std::string base     = (std::filesystem::path(disk_path) / job.conv_id).string();
    const std::string kv_tmp   = split_path(base, SPLIT_EXT_KV)   + ".tmp";
    const std::string drft_tmp = split_path(base, SPLIT_EXT_DRFT) + ".tmp";
    const std::string rec_tmp  = split_path(base, SPLIT_EXT_REC)  + ".tmp";
    const std::string tok_tmp  = split_path(base, SPLIT_EXT_TOK)  + ".tmp";
    const std::string hdr_tmp  = split_path(base, SPLIT_EXT_HDR)  + ".tmp";

    const int64_t t_start = ggml_time_us();

    auto fail = [&](const char * what) {
        SRV_WRN("disk cache: %s failed for '%s' (staged rewrite, pos [%d, %d)) - abandoned, the previous entry (%u tokens) is untouched\n",
                what, job.conv_id.c_str(), job.p0, job.p1, st.n_tok);
        abort_staging(job.conv_id);
        rollback_mirror(job.conv_id, what);
    };

    if (job.rewrite) {
        abort_staging(job.conv_id); // leftovers of an earlier flush that never reached its last job
        {
            std::lock_guard<std::mutex> lock(mtx);
            disk_convs[job.conv_id].broken = false; // the jobs after this one belong to this rewrite
        }
        if (!disk_space_ok(disk_path, job.est_bytes, job.conv_id, "rewrite")) {
            rollback_mirror(job.conv_id, "space");
            return;
        }
        disk_conv_state sg;
        sg.tier = 1;
        staging[job.conv_id] = std::move(sg);
    }

    auto it = staging.find(job.conv_id);
    if (it == staging.end()) {
        SRV_INF("disk cache: dropping staged write of '%s' pos [%d, %d) - no rewrite in progress\n", job.conv_id.c_str(), job.p0, job.p1);
        return;
    }
    disk_conv_state & sg = it->second;

    if (job.p0 != (llama_pos) sg.n_tok) {
        fail("staged segment order");
        return;
    }
    if (!job.rewrite && !disk_space_ok(disk_path, job.kv.size() + job.drft.size() + job.rec.size(), job.conv_id, "staged segment")) {
        abort_staging(job.conv_id);
        rollback_mirror(job.conv_id, "space");
        return;
    }

    disk_conv_state::seg_entry seg;
    seg.kv_off    = sg.kv_size;
    seg.kv_size   = job.kv.size();
    seg.drft_off  = sg.drft_size;
    seg.drft_size = job.drft.size();
    seg.p0        = job.p0;
    seg.p1        = job.p1;

    if (!disk_append_file(kv_tmp, job.kv.data(), job.kv.size())) {
        fail("kv append");
        return;
    }
    if (!job.drft.empty() && !disk_append_file(drft_tmp, job.drft.data(), job.drft.size())) {
        fail("drft append");
        return;
    }
    sg.segments.push_back(seg);
    sg.kv_size   += job.kv.size();
    sg.drft_size += job.drft.size();
    sg.n_tok      = (uint32_t) job.tokens.size();

    if (!job.last) {
        SRV_WRN("disk cache: wrote '%s' segment pos [%d, %d) %.1f MiB (drft %.1f MiB) in %.0f ms - staged, %u tokens so far\n",
                job.conv_id.c_str(), job.p0, job.p1, job.kv.size() / (1024.0 * 1024.0), job.drft.size() / (1024.0 * 1024.0),
                (ggml_time_us() - t_start) / 1e3, sg.n_tok);
        return;
    }

    if (!job.rec.empty()) {
        if (!disk_write_file(rec_tmp, &job.rec_n_tok, sizeof(uint32_t), job.rec.data(), job.rec.size())) {
            fail("rec write");
            return;
        }
        sg.rec_size = sizeof(uint32_t) + job.rec.size();
    }
    if (!disk_write_file(tok_tmp, job.tokens.data(), job.tokens.size() * sizeof(llama_token))) {
        fail("tok write");
        return;
    }
    const std::vector<uint8_t> hdr = disk_make_hdr(arch_hash, vocab_hash, state_hash, (uint32_t) limit_tokens, sg, sg.rec_size);
    if (!disk_write_file(hdr_tmp, hdr.data(), hdr.size())) {
        fail("hdr write");
        return;
    }

    // swap. the old header goes first: from here until the new header lands the entry is invisible (a cache miss after a
    // crash), never a new data file under an old header. renames within one directory do not need space
    std::string old_base;
    const int old_tier = resolve_tier(job.conv_id, old_base);
    std::error_code ec;
    if (old_tier != 0) {
        std::filesystem::remove(split_path(old_base, SPLIT_EXT_HDR), ec);
    }
    bool ok = true;
    auto put = [&](const std::string & tmp, std::string_view ext, bool present) {
        const std::string dst = split_path(base, ext);
        if (present) {
            std::filesystem::rename(tmp, dst, ec);
            ok = ok && !ec;
        } else {
            std::filesystem::remove(dst, ec);
        }
    };
    put(kv_tmp,   SPLIT_EXT_KV,   true);
    put(drft_tmp, SPLIT_EXT_DRFT, sg.drft_size > 0);
    put(rec_tmp,  SPLIT_EXT_REC,  sg.rec_size > 0);
    put(tok_tmp,  SPLIT_EXT_TOK,  true);
    put(hdr_tmp,  SPLIT_EXT_HDR,  true);
    if (!ok) {
        SRV_ERR("disk cache: '%s' staged rewrite: rename failed (%s) - the entry is gone, the next flush rewrites it\n",
                job.conv_id.c_str(), ec.message().c_str());
        abort_staging(job.conv_id);
        rollback_mirror(job.conv_id, "rename");
        return;
    }
    // [TAG_COLD_TIER] the entry now lives in hot: the old cold copy goes (its header is already gone)
    if (old_tier == 2) {
        for (const auto ext : { SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC, SPLIT_EXT_TOK }) {
            std::filesystem::remove(split_path(old_base, ext), ec);
        }
    }

    const size_t   n_segs = sg.segments.size();
    const uint32_t n_tok  = sg.n_tok;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & live = disk_convs[job.conv_id];
        live.n_tok           = sg.n_tok;
        live.kv_size         = sg.kv_size;
        live.drft_size       = sg.drft_size;
        live.rec_size        = sg.rec_size;
        live.segments        = std::move(sg.segments);
        live.broken          = false;
        live.tier            = 1;
        live.t_last_use      = disk_now_s();
    }
    staging.erase(it);

    SRV_WRN("disk cache: wrote '%s' segment pos [%d, %d) %.1f MiB (drft %.1f MiB, rec %.1f MiB) in %.0f ms - now %u tokens in %zu segment(s) (staged rewrite committed)\n",
            job.conv_id.c_str(), job.p0, job.p1,
            job.kv.size() / (1024.0 * 1024.0), job.drft.size() / (1024.0 * 1024.0), job.rec.size() / (1024.0 * 1024.0),
            (ggml_time_us() - t_start) / 1e3, n_tok, n_segs);
}

void server_prompt_cache::abort_staging(const std::string & conv_id) {
    const std::string base = (std::filesystem::path(disk_path) / conv_id).string();
    std::error_code ec;
    for (const auto ext : { SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC, SPLIT_EXT_TOK, SPLIT_EXT_HDR }) {
        std::filesystem::remove(split_path(base, ext) + ".tmp", ec);
    }
    staging.erase(conv_id);
}

void server_prompt_cache::rollback_mirror(const std::string & conv_id, const char * why) {
    GGML_UNUSED(why);
    abort_staging(conv_id);
    std::lock_guard<std::mutex> lock(mtx);
    auto & live = disk_convs[conv_id];
    live.epoch++;
    live.needs_reseed = true;
}

bool server_prompt_cache::disk_space_ok(const std::string & dir, uint64_t need, const std::string & conv_id, const char * what) {
    std::error_code ec;
    const auto sp = std::filesystem::space(dir, ec);
    if (ec || sp.available >= need + DISK_FREE_MARGIN_BYTES) {
        return true; // unknown free space: the write itself reports a failure
    }
    SRV_WRN("disk cache: skipping %s of '%s' - %.1f MiB free in '%s', need %.1f MiB + %.0f MiB margin\n",
            what, conv_id.c_str(), sp.available / (1024.0 * 1024.0), dir.c_str(), need / (1024.0 * 1024.0),
            DISK_FREE_MARGIN_BYTES / (1024.0 * 1024.0));
    // [TAG_COLD_TIER] a short hot dir brings the demote scan forward. it still only moves entries idle for demote_after_s:
    // nothing is evicted to make room
    if (dir == disk_path && !disk_path_cold.empty()) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!demote_scan_requested) {
            demote_scan_requested = true;
            SRV_WRN("%s", "disk cache: hot dir short of space - the demote scan runs early\n");
        }
    }
    return false;
}

bool server_prompt_cache::read_disk_state(const std::string & conversation_id, disk_conv_state & st, bool with_tokens) {
    // [TAG_COLD_TIER] hot first, then cold
    std::string base;
    const int tier = resolve_tier(conversation_id, base);
    if (tier == 0) {
        return false;
    }

    std::vector<uint8_t> hdr;
    {
        std::ifstream f(split_path(base, SPLIT_EXT_HDR), std::ios::binary);
        if (!f) {
            return false;
        }
        hdr.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // fixed part: 7 x u32 + 3 x u64, then the segment table, then a u64 checksum
    constexpr size_t fixed_bytes = 8*sizeof(uint32_t) + 3*sizeof(uint64_t);
    constexpr size_t seg_bytes   = 4*sizeof(uint64_t) + 2*sizeof(int32_t);
    if (hdr.size() < fixed_bytes + sizeof(uint64_t)) {
        return false;
    }

    uint64_t stored_checksum = 0;
    memcpy(&stored_checksum, hdr.data() + hdr.size() - sizeof(uint64_t), sizeof(uint64_t));
    if (disk_hdr_checksum(hdr.data(), hdr.size() - sizeof(uint64_t)) != stored_checksum) {
        SRV_WRN("disk cache: '%s.hdr' checksum mismatch - ignoring\n", conversation_id.c_str());
        return false;
    }

    size_t off = 0;
    auto get = [&hdr, &off](auto & v) {
        memcpy(&v, hdr.data() + off, sizeof(v));
        off += sizeof(v);
    };

    uint32_t magic = 0, version = 0, fa = 0, fv = 0, fs = 0, lt = 0, n_tok = 0, n_segs = 0;
    uint64_t kv_size = 0, drft_size = 0, rec_size = 0;
    get(magic);
    get(version);
    get(fa);
    get(fv);
    get(fs);
    get(lt);
    get(n_tok);
    get(n_segs);
    get(kv_size);
    get(drft_size);
    get(rec_size);
    GGML_UNUSED(lt);

    if (magic != DISK_CACHE_MAGIC || version != DISK_CACHE_VERSION) {
        SRV_INF("disk cache: '%s.hdr' wrong magic/version (0x%08x, %u)\n", conversation_id.c_str(), magic, version);
        return false;
    }
    if (fa != arch_hash || fv != vocab_hash) {
        SRV_INF("disk cache: '%s.hdr' arch/vocab mismatch\n", conversation_id.c_str());
        return false;
    }
    // same arch and vocab does not mean readable state: KV cache types and model
    // geometry must match too, or state_read aborts mid-restore on the first
    // typed tensor (observed: two qwen35-family models sharing one cache dir)
    // [fork, 2026-10-30] state_hash_ok: with --cache-convert the other f16/q8_0 type pairs of this geometry are
    // readable too (server-context.cpp, make_state_hash); st.file_state_hash records which one this file has
    if (!state_hash_ok(fs)) {
        SRV_INF("disk cache: '%s.hdr' kv-state identity mismatch (0x%08x != 0x%08x) - ignoring\n",
                conversation_id.c_str(), fs, state_hash);
        return false;
    }
    if (n_tok == 0 || hdr.size() != fixed_bytes + (size_t) n_segs * seg_bytes + sizeof(uint64_t)) {
        return false;
    }

    st.segments.resize(n_segs);
    for (auto & s : st.segments) {
        get(s.kv_off);
        get(s.kv_size);
        get(s.drft_off);
        get(s.drft_size);
        get(s.p0);
        get(s.p1);
    }

    // the data files must hold at least the committed bytes (orphan tails are fine)
    std::error_code ec;
    const uint64_t kv_actual = (uint64_t) std::filesystem::file_size(split_path(base, SPLIT_EXT_KV), ec);
    if (ec || kv_actual < kv_size) {
        SRV_WRN("disk cache: '%s.kv' smaller than header says - ignoring\n", conversation_id.c_str());
        return false;
    }
    if (drft_size > 0) {
        const uint64_t drft_actual = (uint64_t) std::filesystem::file_size(split_path(base, SPLIT_EXT_DRFT), ec);
        if (ec || drft_actual < drft_size) {
            SRV_WRN("disk cache: '%s.drft' smaller than header says - ignoring\n", conversation_id.c_str());
            return false;
        }
    }
    // [TAG_DISK_ENOSPC] a missing, short or stale .rec no longer hides the entry: the attention segments are intact and
    // load_from_disk restores the recurrent state from the newest matching cp_ file instead (the tail after it is
    // reprocessed). before, the entry was ignored here, or worse, a rec_size 0 header was read as a full-state file
    bool rec_valid = true;
    if (rec_size > 0) {
        const std::string rec_path = split_path(base, SPLIT_EXT_REC);
        const uint64_t rec_actual = (uint64_t) std::filesystem::file_size(rec_path, ec);
        uint32_t rec_n_tok = 0;
        if (!ec && rec_actual == rec_size) {
            std::ifstream rf(rec_path, std::ios::binary);
            rec_valid = read_disk_u32(rf, rec_n_tok) && rec_n_tok == n_tok;
        } else {
            rec_valid = false;
        }
        if (!rec_valid) {
            SRV_WRN("disk cache: '%s.rec' does not match the header (size %llu vs %llu, n_tok %u vs %u) - recurrent state comes from a checkpoint\n",
                    conversation_id.c_str(), ec ? 0ull : (unsigned long long) rec_actual, (unsigned long long) rec_size, rec_n_tok, n_tok);
        }
    }

    std::vector<llama_token> toks(with_tokens ? n_tok : 0);
    if (with_tokens) {
        std::ifstream f(split_path(base, SPLIT_EXT_TOK), std::ios::binary);
        if (!f) {
            return false;
        }
        f.read(reinterpret_cast<char *>(toks.data()), (std::streamsize)(n_tok * sizeof(llama_token)));
        if (!f) {
            return false;
        }
    }

    st.n_tok            = n_tok;
    st.kv_size          = kv_size;
    st.drft_size        = drft_size;
    st.rec_size         = rec_size;
    st.file_state_hash  = fs;
    st.scheduled_n_tok  = n_tok;
    st.scheduled_tokens = std::move(toks);
    st.broken           = false;
    st.needs_reseed     = false;
    st.rec_valid        = rec_valid;
    st.scheduled_rec    = rec_size > 0 && rec_valid;
    st.tier             = tier;

    st.scheduled_seg_ends.clear();
    for (const auto & s : st.segments) {
        st.scheduled_seg_ends.push_back((uint32_t) s.p1);
    }

    return true;
}

void server_prompt_cache::spill_checkpoint(const common_prompt_checkpoint & cp, const std::string & conversation_id) {
    if (disk_path.empty()) {
        return;
    }

    if (conversation_id.empty()) {
        SRV_WRN("%s", "checkpoint spill: conversation_id is empty, refusing to write (no fallback)\n");
        return;
    }

    if (cp.data_tgt.empty() && cp.data_dft.empty()) {
        // disk-backed (lazy) or empty checkpoint - its file already exists
        return;
    }

    // write-at-creation, async: the job holds its own reference to the blobs, so the resident checkpoint
    // can be converted to disk-backed (or evicted) while the write is still in flight
    disk_job job;
    job.conv_id = conversation_id;
    job.ckpt = std::make_shared<common_prompt_checkpoint>();
    job.ckpt->pos_min           = cp.pos_min;
    job.ckpt->pos_max           = cp.pos_max;
    job.ckpt->n_tokens          = cp.n_tokens;
    job.ckpt->token_prefix_hash = cp.token_prefix_hash;
    // [TAG_STATE_BLOB] shared references, not copies: 150 MB per checkpoint used to be memcpy'd here
    // on the request thread. the blobs are immutable, so the writer reads the same bytes the slot holds
    job.ckpt->data_tgt          = cp.data_tgt;
    job.ckpt->data_dft          = cp.data_dft;

    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return queue_bytes < DISK_QUEUE_MAX_BYTES; });
        queue_bytes += job.bytes();
        queue.push_back(std::move(job));
        cv.notify_all();
    }
}

void server_prompt_cache::lazify_checkpoints(server_prompt & prompt) {
    if (disk_path.empty() || prompt.conversation_id.empty() || prompt.checkpoints.size() < 2) {
        return;
    }

    // keep the newest checkpoint resident (including its speculative state); convert older
    // ones whose cp_ file has committed to disk-backed form so at most one checkpoint's
    // KV blobs occupy host RAM
    auto last = std::prev(prompt.checkpoints.end());
    for (auto it = prompt.checkpoints.begin(); it != last; ++it) {
        auto & cp = *it;

        if (!cp.src_path.empty()) {
            continue; // already disk-backed
        }
        if (cp.data_tgt.empty() && cp.data_dft.empty()) {
            continue;
        }

        const std::string filename = "cp_" + prompt.conversation_id + "_" + std::to_string(cp.pos_min) + ".bin";
        const std::string path     = (std::filesystem::path(disk_path) / filename).string();

        const uint64_t sz_tgt   = cp.data_tgt.size();
        const uint64_t sz_dft   = cp.data_dft.size();
        const uint64_t expected = CHECKPOINT_FILE_HEADER_BYTES + sz_tgt + sizeof(uint64_t) + sz_dft;

        std::error_code ec;
        const uint64_t actual = (uint64_t) std::filesystem::file_size(path, ec);
        if (ec || actual != expected) {
            continue; // write still in flight (or failed) - try again on the next checkpoint
        }

        cp.src_path = path;
        cp.off_tgt  = CHECKPOINT_FILE_HEADER_BYTES;
        cp.sz_tgt   = sz_tgt;
        cp.off_dft  = CHECKPOINT_FILE_HEADER_BYTES + sz_tgt + sizeof(uint64_t);
        cp.sz_dft   = sz_dft;
        // [TAG_STATE_BLOB] drops the reference. it used to be vector::clear(), which kept the capacity: the
        // "freed" below was false and the 27B server held 9 dead 150 MiB blocks (624 MiB RAM + 727 MiB swap,
        // 2026-10-22). now the last reference returns the buffer to the slot pool (max_free) or frees it
        cp.clear_tgt();
        cp.clear_dft();

        SRV_INF("checkpoint: converted pos_min=%d to disk-backed ('%s', %.1f MiB released)\n",
                cp.pos_min, filename.c_str(), (sz_tgt + sz_dft) / (1024.0 * 1024.0));
    }
}

bool server_prompt_cache::write_checkpoint_file(const common_prompt_checkpoint & cp, const std::string & conversation_id) {
    // filename encodes conversation_id so the limit and merge logic can filter per conversation
    const std::string filename = "cp_" + conversation_id + "_" + std::to_string(cp.pos_min) + ".bin";
    const std::filesystem::path filepath = std::filesystem::path(disk_path) / filename;
    const std::filesystem::path tmppath  = std::filesystem::path(disk_path) / (filename + ".tmp");

    // [TAG_DISK_ENOSPC] a checkpoint that does not fit is skipped like a segment (no retry)
    if (!disk_space_ok(disk_path, CHECKPOINT_FILE_HEADER_BYTES + cp.data_tgt.size() + cp.data_dft.size() + sizeof(uint64_t),
                       conversation_id, "checkpoint")) {
        return false;
    }

    {
        std::ofstream f(tmppath, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_WRN("checkpoint spill: failed to open '%s' for write\n", tmppath.string().c_str());
            return false;
        }

        auto put_u32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        auto put_u64 = [&f](uint64_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };

        put_u32(CHECKPOINT_SPILL_MAGIC);
        put_u32(CHECKPOINT_SPILL_VERSION);
        put_u32(arch_hash);
        put_u32(vocab_hash);
        put_u32(state_hash);

        const int32_t pos_min_v  = cp.pos_min;
        const int32_t pos_max_v  = cp.pos_max;
        const int64_t n_tokens_v = cp.n_tokens;
        f.write(reinterpret_cast<const char *>(&pos_min_v),  sizeof(pos_min_v));
        f.write(reinterpret_cast<const char *>(&pos_max_v),  sizeof(pos_max_v));
        f.write(reinterpret_cast<const char *>(&n_tokens_v), sizeof(n_tokens_v));

        put_u64(cp.token_prefix_hash);

        put_u64(static_cast<uint64_t>(cp.data_tgt.size()));
        if (!cp.data_tgt.empty()) {
            f.write(reinterpret_cast<const char *>(cp.data_tgt.data()),
                    static_cast<std::streamsize>(cp.data_tgt.size()));
        }

        put_u64(static_cast<uint64_t>(cp.data_dft.size()));
        if (!cp.data_dft.empty()) {
            f.write(reinterpret_cast<const char *>(cp.data_dft.data()),
                    static_cast<std::streamsize>(cp.data_dft.size()));
        }

        if (!f) {
            std::error_code ec;
            std::filesystem::remove(tmppath, ec);
            SRV_WRN("checkpoint spill: write error for '%s'\n", tmppath.string().c_str());
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmppath, filepath, ec);
    if (ec) {
        std::filesystem::remove(tmppath, ec);
        SRV_WRN("checkpoint spill: rename failed '%s': %s\n", tmppath.string().c_str(), ec.message().c_str());
        return false;
    }

    // [TAG_CP_INDEX] the file is committed: publish it to the index merge_checkpoint_spills() reads.
    // seeding first (one scan, writer thread, first file of the conversation in this process) makes the
    // spill limit below count the files earlier runs left behind
    cp_index_seed(conversation_id);
    {
        cp_file_entry e;
        e.path              = filepath.string();
        e.pos_min           = cp.pos_min;
        e.pos_max           = cp.pos_max;
        e.n_tokens          = cp.n_tokens;
        e.token_prefix_hash = cp.token_prefix_hash;
        e.off_tgt           = CHECKPOINT_FILE_HEADER_BYTES;
        e.sz_tgt            = cp.data_tgt.size();
        e.off_dft           = CHECKPOINT_FILE_HEADER_BYTES + cp.data_tgt.size() + sizeof(uint64_t);
        e.sz_dft            = cp.data_dft.size();
        e.status            = cp_file_entry::CP_OK;

        std::lock_guard<std::mutex> lock(mtx);
        cp_index[conversation_id].files[cp.pos_min] = std::move(e);
    }

    // enforce the per-conversation checkpoint spill limit, thinning the retained set towards
    // uniform spacing over the conversation (see the victim choice below)
    if (checkpoint_spill_max > 0) {
        struct cp_entry {
            std::filesystem::path path;
            int32_t pos_min;
        };
        std::vector<cp_entry> cp_files;

        // [TAG_CP_INDEX] from the index (ascending pos_min), not a directory scan
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto & kv : cp_index[conversation_id].files) {
                cp_files.push_back({ kv.second.path, kv.first });
            }
        }

        if ((int32_t) cp_files.size() > checkpoint_spill_max) {
            // only now pay a stat per file: entries whose file the external sweep deleted must not
            // count against the limit, or a live file would be evicted in their place
            for (auto it = cp_files.begin(); it != cp_files.end(); ) {
                std::error_code ec_ex;
                if (!std::filesystem::exists(it->path, ec_ex)) {
                    std::lock_guard<std::mutex> lock(mtx);
                    cp_index[conversation_id].files.erase(it->pos_min);
                    it = cp_files.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if ((int32_t) cp_files.size() > checkpoint_spill_max) {
            // the newest files are never thinned - a rewind almost always lands near the tip
            int n_protect = 8;
            if (checkpoint_spill_max <= n_protect) {
                n_protect = checkpoint_spill_max / 2;
                if (n_protect < 1) {
                    n_protect = 1;
                }
            }

            // below the protected tail, drop the file closest to its predecessor. that keeps
            // the survivors spread over the whole conversation depth instead of collapsing
            // them onto its tail, which is what makes deep rewinds cheap.
            while ((int32_t) cp_files.size() > checkpoint_spill_max) {
                const int n_deletable = (int) cp_files.size() - n_protect;
                if (n_deletable <= 0) {
                    break;
                }

                int     victim  = 0;
                int32_t gap_min = cp_files[0].pos_min; // gap of the first file: distance from position 0
                for (int i = 1; i < n_deletable; ++i) {
                    const int32_t gap = cp_files[i].pos_min - cp_files[i - 1].pos_min;
                    if (gap < gap_min) {
                        gap_min = gap;
                        victim  = i;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    cp_index[conversation_id].files.erase(cp_files[victim].pos_min);
                }
                std::error_code ec3;
                std::filesystem::remove(cp_files[victim].path, ec3);
                SRV_WRN("checkpoint spill: evicted '%s' (pos_min=%d, gap=%d, limit=%d)\n",
                        cp_files[victim].path.filename().string().c_str(), cp_files[victim].pos_min,
                        gap_min, checkpoint_spill_max);

                cp_files.erase(cp_files.begin() + victim);
            }
        }
    }

    return true;
}

void server_prompt_cache::merge_checkpoint_spills(server_prompt & prompt) {
    if (disk_path.empty()) {
        return;
    }

    if (prompt.conversation_id.empty()) {
            SRV_WRN("%s", "checkpoint merge: conversation_id is empty, refusing to scan disk (no fallback)\n");
        return;
    }

    int n_scanned    = 0;
    int n_loaded     = 0;
    int n_skip_magic = 0;
    int n_skip_hash  = 0;
    int n_skip_tok   = 0;
    int n_skip_dup   = 0;

    std::list<common_prompt_checkpoint> new_ckpts;

    // [TAG_CP_INDEX] the writer-maintained index replaces the directory scan, the per-file open and the
    // seek past the target blob (157 MB on the 27B) to read the draft size. the first call for a
    // conversation in this process seeds the index with one scan
    cp_index_seed(prompt.conversation_id);

    std::vector<cp_file_entry> entries;
    {
        std::lock_guard<std::mutex> lock(mtx);
        const auto & files = cp_index[prompt.conversation_id].files;
        entries.reserve(files.size());
        for (const auto & kv : files) {
            entries.push_back(kv.second);
        }
    }

    for (const auto & e : entries) {
        ++n_scanned;

        if (e.status == cp_file_entry::CP_BAD_MAGIC) {
            ++n_skip_magic;
            continue;
        }
        if (e.status == cp_file_entry::CP_BAD_HASH) {
            ++n_skip_hash;
            continue;
        }
        if (e.status != cp_file_entry::CP_OK) {
            continue; // truncated
        }

        const int32_t  pos_min_v  = e.pos_min;
        const int64_t  n_tokens_v = e.n_tokens;

        // the file must belong to this exact token timeline. a cp_ file left by an abandoned
        // branch shares the conversation_id but not the prefix, and restoring it would put
        // KV from the other branch into this prompt.
        if (n_tokens_v < 0 || (size_t) n_tokens_v > prompt.tokens.size() ||
            hash_token_prefix(prompt.tokens, (size_t) n_tokens_v) != e.token_prefix_hash) {
            ++n_skip_tok;
            continue;
        }

        // skip if an equivalent checkpoint is already known. pos_min identifies the file
        // (cp_<conv>_<pos_min>.bin), n_tokens identifies the capture point. create_checkpoint()
        // supersedes an in-list checkpoint at an equal n_tokens, and when the superseding
        // capture landed on a different pos_min (SWA: the window start moves after a restore)
        // the superseded file stays on disk with a still-matching token_prefix_hash. Without
        // the n_tokens test it would be resurrected here on the next rewind, undoing that.
        bool dup = false;
        for (const auto & cp : prompt.checkpoints) {
            if (cp.pos_min == pos_min_v || cp.n_tokens == n_tokens_v) {
                dup = true;
                break;
            }
        }
        // two files of this conversation can collide on n_tokens with different pos_min too
        for (auto it = new_ckpts.begin(); !dup && it != new_ckpts.end(); ++it) {
            if (it->pos_min == pos_min_v || it->n_tokens == n_tokens_v) {
                dup = true;
            }
        }
        if (dup) {
            ++n_skip_dup;
            continue;
        }

        // LAZY (disk-backed) registration: record the file + the byte offsets of
        // the tgt/dft blobs, but DO NOT read them into RAM. The KV bytes stay on
        // disk; load_tgt/load_dft fault them in only if a rewind actually selects
        // this checkpoint, then discard them again. This is what keeps reviewed-
        // from-disk checkpoints off the anonymous heap. The file is not consumed
        // here - see [TAG_CACHE_DONT_DELETE]. a file that vanished since it was indexed
        // fails in load_tgt and the caller falls back to an older checkpoint.
        common_prompt_checkpoint cp;
        cp.pos_min           = pos_min_v;
        cp.pos_max           = e.pos_max;
        cp.n_tokens          = n_tokens_v;
        cp.token_prefix_hash = e.token_prefix_hash;
        cp.src_path          = e.path;
        cp.off_tgt           = e.off_tgt;
        cp.sz_tgt            = e.sz_tgt;
        cp.off_dft           = e.off_dft;
        cp.sz_dft            = e.sz_dft;

        new_ckpts.push_back(std::move(cp));
        ++n_loaded;
    }

    SRV_TRC("checkpoint merge: scanned %d, loaded %d, skipped: magic %d, hash %d, tokens %d, dup %d\n",
            n_scanned, n_loaded, n_skip_magic, n_skip_hash, n_skip_tok, n_skip_dup);

    if (new_ckpts.empty()) {
        return;
    }

    // insert into prompt.checkpoints keeping sorted order by pos_min
    for (auto & cp : new_ckpts) {
        auto it = prompt.checkpoints.begin();
        while (it != prompt.checkpoints.end() && it->pos_min < cp.pos_min) {
            ++it;
        }
        prompt.checkpoints.insert(it, std::move(cp));
    }
}


void server_prompt_cache::drop_stale_checkpoint_spills(const server_prompt & prompt, int64_t n_valid) {
    if (disk_path.empty() || prompt.conversation_id.empty()) {
        return;
    }

    // [TAG_CP_INDEX] candidates come from the index (headers already parsed), no scan, no opens
    cp_index_seed(prompt.conversation_id);

    std::vector<cp_file_entry> victims;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & files = cp_index[prompt.conversation_id].files;
        for (auto it = files.begin(); it != files.end(); ) {
            const auto & e = it->second;
            // unreadable for this build: dead weight in the spill budget. otherwise bound to more
            // tokens than the shared prefix: no future request of this conversation can bind it again,
            // whatever the client sends next
            const bool drop = e.status == cp_file_entry::CP_BAD_MAGIC || e.n_tokens > n_valid;
            if (drop) {
                victims.push_back(e);
                it = files.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (const auto & e : victims) {
        std::error_code ec2;
        std::filesystem::remove(e.path, ec2);
        SRV_WRN("checkpoint spill: dropping checkpoint from abandoned branch '%s' (pos_min = %d, n_tokens = %" PRId64 ", n_valid = %" PRId64 ")\n",
                std::filesystem::path(e.path).filename().string().c_str(), e.pos_min, e.n_tokens, n_valid);
    }
}

void server_prompt_cache::forget_checkpoint_spill(const std::string & conversation_id, int32_t pos_min) {
    if (disk_path.empty() || conversation_id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cp_index.find(conversation_id);
    if (it != cp_index.end()) {
        it->second.files.erase(pos_min);
    }
}

bool server_prompt_cache::read_checkpoint_file_entry(const std::string & path, cp_file_entry & e) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false; // vanished between the listing and the open
    }

    e.path   = path;
    e.status = cp_file_entry::CP_BAD_MAGIC;

    uint32_t magic = 0, version = 0, fa = 0, fv = 0, fs = 0;
    f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (!f || magic != CHECKPOINT_SPILL_MAGIC || version != CHECKPOINT_SPILL_VERSION) {
        return true;
    }

    f.read(reinterpret_cast<char *>(&fa), sizeof(fa));
    f.read(reinterpret_cast<char *>(&fv), sizeof(fv));
    f.read(reinterpret_cast<char *>(&fs), sizeof(fs));
    f.read(reinterpret_cast<char *>(&e.pos_min),           sizeof(e.pos_min));
    f.read(reinterpret_cast<char *>(&e.pos_max),           sizeof(e.pos_max));
    f.read(reinterpret_cast<char *>(&e.n_tokens),          sizeof(e.n_tokens));
    f.read(reinterpret_cast<char *>(&e.token_prefix_hash), sizeof(e.token_prefix_hash));
    if (!f) {
        return true; // short header
    }

    // [fork, 2026-10-30] a checkpoint of another convertible K/V type pair is readable with --cache-convert: the
    // recurrent part is f32 and type-independent, the KV part (SWA models) goes through the converting reader
    if (fa != arch_hash || fv != vocab_hash || !state_hash_ok(fs)) {
        e.status = cp_file_entry::CP_BAD_HASH;
        return true;
    }

    e.status = cp_file_entry::CP_TRUNCATED;

    uint64_t tgt_sz = 0;
    f.read(reinterpret_cast<char *>(&tgt_sz), sizeof(tgt_sz));
    if (!f) {
        return true;
    }

    // the draft size comes from the file size instead of the size field behind the target blob: that
    // read was a seek 157 MB into every file on the 27B (the writer writes exactly
    // header + tgt + u64 + dft, so both agree for every file this server wrote)
    std::error_code ec;
    const uint64_t fsize = (uint64_t) std::filesystem::file_size(path, ec);
    const uint64_t off_dft = CHECKPOINT_FILE_HEADER_BYTES + tgt_sz + sizeof(uint64_t);
    if (ec || fsize < off_dft) {
        return true;
    }

    e.off_tgt = CHECKPOINT_FILE_HEADER_BYTES;
    e.sz_tgt  = tgt_sz;
    e.off_dft = off_dft;
    e.sz_dft  = fsize - off_dft;
    e.status  = cp_file_entry::CP_OK;
    return true;
}

void server_prompt_cache::cp_index_seed(const std::string & conversation_id) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cp_index[conversation_id].seeded) {
            return;
        }
    }

    // outside the mutex: the request thread and the writer keep moving while the directory is listed.
    // a file the writer adds meanwhile is inserted by the writer itself; one it evicts meanwhile can be
    // re-inserted here as a stale entry, which every reader tolerates (see cp_index)
    const std::string conv_prefix = "cp_" + conversation_id + "_";
    std::vector<cp_file_entry> found;

    // [TAG_COLD_TIER] hot first: a pos_min present in both (a promotion or demotion cut short) resolves to the hot file
    for (const std::string * dir : { &disk_path, &disk_path_cold }) {
        if (dir->empty()) {
            continue;
        }
        std::error_code ec;
        for (const auto & de : std::filesystem::directory_iterator(*dir, ec)) {
            if (ec) {
                break;
            }
            int32_t pmin = 0;
            if (!parse_cp_file_name(de.path().filename().string(), conv_prefix, pmin)) {
                continue;
            }
            cp_file_entry e;
            if (!read_checkpoint_file_entry(de.path().string(), e)) {
                continue;
            }
            e.pos_min = pmin; // the name is the key (the header can be unreadable)
            found.push_back(std::move(e));
        }
    }

    std::lock_guard<std::mutex> lock(mtx);
    auto & ci = cp_index[conversation_id];
    for (auto & e : found) {
        const int32_t key = e.pos_min;
        ci.files.emplace(key, std::move(e)); // entries the writer added meanwhile win (and hot wins over cold)
    }
    ci.seeded = true;

    SRV_INF("checkpoint index: seeded '%s' with %zu file(s)\n", conversation_id.c_str(), found.size());
}


bool server_prompt_cache::load_from_disk(server_prompt & prompt, const std::string & conversation_id,
                                          llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, int32_t n_room) {
    if (disk_path.empty() || conversation_id.empty()) {
        return false;
    }

    // wait until no write for this conversation is queued or in flight.
    // [TAG_COLD_TIER] registered as loading before the wait: a demotion of this conversation that is copying right now
    // sees it between two chunks and gives up (its partial cold copy is removed), and the scan skips it from here on
    {
        std::unique_lock<std::mutex> lock(mtx);
        loading_convs.insert(conversation_id);
        cv.wait(lock, [&]() {
            if (writer_active_conv == conversation_id) {
                return false;
            }
            for (const auto & j : queue) {
                if (j.conv_id == conversation_id) {
                    return false;
                }
            }
            return true;
        });
    }
    struct loading_guard {
        server_prompt_cache * c;
        const std::string   & id;
        ~loading_guard() {
            std::lock_guard<std::mutex> lock(c->mtx);
            c->loading_convs.erase(id);
        }
    } loading { this, conversation_id };

    disk_conv_state st;
    if (!read_disk_state(conversation_id, st)) {
        SRV_INF("load_from_disk: no valid cache files for '%s'\n", conversation_id.c_str());
        return false;
    }

    if (limit_tokens > 0 && st.n_tok > limit_tokens) {
        SRV_INF("load_from_disk: '%s' too many tokens (%u > %zu)\n", conversation_id.c_str(), st.n_tok, limit_tokens);
        return false;
    }

    const bool runtime_has_dft = ctx_dft != nullptr;
    if (runtime_has_dft && st.drft_size == 0) {
        SRV_INF("load_from_disk: '%s' asymmetric (runtime has draft, file does not)\n", conversation_id.c_str());
        return false;
    }

    // [TAG_EVICT_ON_NEED] refuse a state that cannot fit before reading a byte of it. the segments are applied one by
    // one with APPEND and only the one that runs out of cells fails, in state_read_meta. 2026-10-28 production (27B
    // hybrid, -np 2 --kv-unified, 420096-cell pool): slot 1 held 172070 cells, this conversation needed 248289, 263
    // short. all 118 segments (15.4 GB, 14 s) streamed in, the last one ("failed to find 1767 available cells")
    // failed, the wipe below threw it all away, and the request re-prefilled 248k tokens (~5 min). a partial restore
    // is no use on a hybrid model anyway: the attention cells need the .rec state valid at n_tok, which goes in last.
    // the caller (get_available_slot) evicts idle slots first, so this triggers only when every other slot is busy
    if (n_room >= 0 && st.n_tok > (uint32_t) n_room) {
        SRV_WRN("load_from_disk: '%s' needs %u cells, only %d free in the unified KV pool - not restored\n",
                conversation_id.c_str(), st.n_tok, n_room);
        last_load_refused_room = true; // [TAG_POOL_TURNS]
        return false;
    }

    const bool        from_cold = st.tier == 2;
    const std::string base      = (std::filesystem::path(from_cold ? disk_path_cold : disk_path) / conversation_id).string();

    // [TAG_DISK_ENOSPC] the file layout comes from the model, never from rec_size: a delta-capable hybrid model writes
    // attention-only range blobs (APPEND) plus a .rec, everything else writes full-state blobs. 2026-11-01 and again
    // 2026-11-04 (production log 20261104_082417 ~L18285): a rewrite that died under ENOSPC left a hybrid header with
    // rec_size 0, segment 0 was applied with flags NONE, state_read wanted a recurrent section the range blob does not
    // have ("unexpectedly reached end of buffer", set_data returned 0 of 537067936), and 151k tokens were reprocessed
    // from zero (152 s). same test as flush_from_context's delta_capable / rec_needed
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    bool delta_capable = llama_model_n_swa(model_tgt) == 0 && !llama_model_is_recurrent(model_tgt);
    if (ctx_dft) {
        const llama_model * model_dft = llama_get_model(ctx_dft);
        delta_capable = delta_capable && llama_model_n_swa(model_dft) == 0 && !llama_model_is_recurrent(model_dft);
    }
    const bool hybrid = delta_capable && llama_model_is_hybrid(model_tgt);

    std::ifstream kf(split_path(base, SPLIT_EXT_KV), std::ios::binary);
    if (!kf) {
        return false;
    }

    // [TAG_COLD_TIER] promotion: a cold entry is copied to hot in the same pass that restores it - every segment read
    // from the spinner (~200 MB/s) is written to <hot>/<id>.kv.tmp (NVMe, ~512 MiB in ~200 ms) before it is applied, so
    // the copy hides behind the read. the cold files go only after the hot ones are complete and renamed
    bool promote = from_cold;
    const std::string hot_base = (std::filesystem::path(disk_path) / conversation_id).string();
    const std::string hot_kv_tmp   = split_path(hot_base, SPLIT_EXT_KV)   + ".tmp";
    const std::string hot_drft_tmp = split_path(hot_base, SPLIT_EXT_DRFT) + ".tmp";
    const std::string hot_rec_tmp  = split_path(hot_base, SPLIT_EXT_REC)  + ".tmp";
    const std::string hot_tok_tmp  = split_path(hot_base, SPLIT_EXT_TOK)  + ".tmp";
    const std::string hot_hdr_tmp  = split_path(hot_base, SPLIT_EXT_HDR)  + ".tmp";
    FILE * pk = nullptr;
    FILE * pd = nullptr;
    auto promo_close = [&]() {
        if (pk) { fclose(pk); pk = nullptr; }
        if (pd) { fclose(pd); pd = nullptr; }
    };
    auto promo_remove = [&]() {
        std::error_code ec;
        for (const auto & p : { hot_kv_tmp, hot_drft_tmp, hot_rec_tmp, hot_tok_tmp, hot_hdr_tmp }) {
            std::filesystem::remove(p, ec);
        }
    };
    auto promo_abort = [&](const char * why) {
        if (!promote) {
            return;
        }
        promote = false;
        promo_close();
        promo_remove();
        SRV_WRN("disk cache: promotion of '%s' abandoned (%s) - the cold entry is kept\n", conversation_id.c_str(), why);
    };
    if (promote) {
        // the segments must tile the files from offset 0 so the copy reproduces the committed bytes exactly
        bool tiles = true;
        uint64_t kv_off = 0, drft_off = 0;
        for (const auto & seg : st.segments) {
            tiles = tiles && seg.kv_off == kv_off && seg.drft_off == drft_off;
            kv_off   += seg.kv_size;
            drft_off += seg.drft_size;
        }
        if (!tiles || kv_off != st.kv_size || drft_off != st.drft_size) {
            promo_abort("segments do not tile the files");
        } else {
            std::error_code ec;
            const auto sp = std::filesystem::space(disk_path, ec);
            const uint64_t need = st.kv_size + st.drft_size + st.rec_size + (uint64_t) st.n_tok * sizeof(llama_token);
            if (!ec && sp.available < need + DISK_FREE_MARGIN_BYTES) {
                promote = false;
                SRV_WRN("disk cache: '%s' restored from cold without promotion - %.1f MiB free in the hot dir, need %.1f MiB + margin\n",
                        conversation_id.c_str(), sp.available / (1024.0 * 1024.0), need / (1024.0 * 1024.0));
            } else {
                pk = fopen(hot_kv_tmp.c_str(), "wb");
                pd = st.drft_size > 0 ? fopen(hot_drft_tmp.c_str(), "wb") : nullptr;
                if (!pk || (st.drft_size > 0 && !pd)) {
                    promo_abort("cannot create the hot files");
                }
            }
        }
    }

    std::ifstream df;
    if (runtime_has_dft || (promote && st.drft_size > 0)) {
        df.open(split_path(base, SPLIT_EXT_DRFT), std::ios::binary);
        if (!df) {
            promo_abort("no .drft");
            if (runtime_has_dft) {
                return false;
            }
        }
    }

    const int64_t t_start = ggml_time_us();

    // [fork, 2026-10-30] --cache-convert: the counters are cumulative per context, the difference is this restore
    const llama_state_convert_stats conv0_tgt = llama_state_get_convert_stats(ctx_tgt);
    const llama_state_convert_stats conv0_dft = ctx_dft ? llama_state_get_convert_stats(ctx_dft) : llama_state_convert_stats { 0, 0, 0, -1, -1 };

    SRV_INF("load_from_disk: '%s' n_tok=%u kv=%.1f MiB drft=%.1f MiB segments=%zu%s\n",
            conversation_id.c_str(), st.n_tok,
            st.kv_size / (1024.0 * 1024.0), st.drft_size / (1024.0 * 1024.0), st.segments.size(), from_cold ? " (cold)" : "");

    // stream the segments into the context one buffer at a time; peak host RAM = the
    // largest single segment.
    //
    // non-hybrid: full-state blobs - the first segment replaces the sequence (flags=0
    // parses the complete state layout), later segments merge in with APPEND.
    // hybrid: every segment is attention-only, which only the APPEND path parses - so wipe the sequence explicitly and
    // apply all segments with APPEND, then restore the recurrent state from .rec (or a cp_ file) with PARTIAL_ONLY.
    // the prompt is cleared with the memory: load() returns true after a failed disk load and the caller keeps the
    // slot, so stale tokens left here would claim cells that are gone (the next turn would reuse a prefix of them)
    auto wipe_slot = [&]() {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
        prompt.clear();
    };
    auto fail = [&]() {
        promo_abort("restore failed");
        wipe_slot();
        return false;
    };

    if (hybrid) {
        wipe_slot();
    }

    std::vector<uint8_t> buf;
    for (size_t i = 0; i < st.segments.size(); ++i) {
        const auto & seg = st.segments[i];
        const llama_state_seq_flags flags = (hybrid || i > 0) ? LLAMA_STATE_SEQ_FLAGS_APPEND : LLAMA_STATE_SEQ_FLAGS_NONE;

        buf.resize(seg.kv_size);
        kf.seekg((std::streamoff) seg.kv_off);
        kf.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) seg.kv_size);
        if (!kf) {
            SRV_WRN("load_from_disk: '%s.kv' read failed at segment %zu\n", conversation_id.c_str(), i);
            return fail();
        }
        if (promote) {
            size_t n_ok = buf.size();
            const bool inject = disk_fault_inject(buf.size(), n_ok); // [TAG_DISK_ENOSPC] the same test hook as appends
            if (fwrite(buf.data(), 1, n_ok, pk) != n_ok || inject) {
                promo_abort("hot .kv write failed");
            }
        }

        size_t n = llama_state_seq_set_data_ext(ctx_tgt, buf.data(), buf.size(), id_slot, flags);
        if (n != buf.size()) {
            SRV_WRN("load_from_disk: set_data (main, segment %zu) returned %zu of %zu\n", i, n, buf.size());
            return fail();
        }

        if (seg.drft_size > 0 && (runtime_has_dft || promote)) {
            buf.resize(seg.drft_size);
            df.seekg((std::streamoff) seg.drft_off);
            df.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) seg.drft_size);
            if (!df) {
                SRV_WRN("load_from_disk: '%s.drft' read failed at segment %zu\n", conversation_id.c_str(), i);
                if (runtime_has_dft) {
                    return fail();
                }
                promo_abort("cold .drft read failed");
                continue;
            }
            if (promote && fwrite(buf.data(), 1, buf.size(), pd) != buf.size()) {
                promo_abort("hot .drft write failed");
            }

            if (runtime_has_dft) {
                n = llama_state_seq_set_data_ext(ctx_dft, buf.data(), buf.size(), id_slot, flags);
                if (n != buf.size()) {
                    SRV_WRN("load_from_disk: set_data (drft, segment %zu) returned %zu of %zu\n", i, n, buf.size());
                    return fail();
                }
            }
        }
    }

    // hybrid models: restore the recurrent state (valid at n_tok) on top of the attention cells
    uint32_t n_tok_restored = st.n_tok;
    common_state_data rec; // [TAG_STATE_BLOB] no zero fill of 157 MB before the read
    bool rec_ok = false;
    if (hybrid) {
        if (st.rec_size > sizeof(uint32_t) && st.rec_valid) {
            uint8_t * rec_dst = rec.reserve_for_write(st.rec_size);
            std::ifstream rf(split_path(base, SPLIT_EXT_REC), std::ios::binary);
            rf.read(reinterpret_cast<char *>(rec_dst), (std::streamsize) rec.size());
            uint32_t rec_n_tok = 0;
            if (rf) {
                memcpy(&rec_n_tok, rec.data(), sizeof(uint32_t));
            }
            // the snapshot must match the header's token count - a crash between the segment
            // append and the .rec overwrite leaves them inconsistent
            if (!rf || rec_n_tok != st.n_tok) {
                SRV_WRN("load_from_disk: '%s.rec' unreadable or for %u tokens (header %u) - trying a checkpoint\n",
                        conversation_id.c_str(), rec_n_tok, st.n_tok);
            } else {
                const size_t n = llama_state_seq_set_data_ext(ctx_tgt, rec.data() + sizeof(uint32_t),
                                                              rec.size() - sizeof(uint32_t), id_slot,
                                                              LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                rec_ok = n == rec.size() - sizeof(uint32_t);
                if (!rec_ok) {
                    SRV_WRN("load_from_disk: set_data (rec) returned %zu of %zu - trying a checkpoint\n", n, rec.size() - sizeof(uint32_t));
                }
            }
        }

        if (!rec_ok) {
            // [TAG_DISK_ENOSPC] no usable .rec: the attention cells are all in, the recurrent state comes from the newest
            // cp_ file of this exact token timeline at or below n_tok. the attention cells past it are dropped (the
            // recurrent state only knows the tokens up to it) and the server reprocesses the tail from there.
            // 2026-11-01: cp_9b88741580953422_139224.bin would have saved 139k of the 186k tokens reprocessed from zero
            const int64_t n_cp = restore_rec_from_checkpoint(conversation_id, st, ctx_tgt, ctx_dft, id_slot);
            if (n_cp <= 0) {
                SRV_WRN("load_from_disk: '%s' has no valid .rec (rec_size %llu) and no matching checkpoint - the %u tokens are reprocessed from zero\n",
                        conversation_id.c_str(), (unsigned long long) st.rec_size, st.n_tok);
                return fail();
            }
            n_tok_restored = (uint32_t) n_cp;
        }
    }

    // populate the prompt. checkpoints are not stored with the entry - the caller runs
    // merge_checkpoint_spills() which registers this conversation's cp_ files lazily.
    prompt.tokens.clear();
    {
        llama_tokens toks(st.scheduled_tokens.begin(), st.scheduled_tokens.begin() + n_tok_restored);
        prompt.tokens.insert(toks);
    }
    prompt.checkpoints.clear();
    prompt.conversation_id = conversation_id;

    // [LLAMA_SERVER_UBATCH_STATS] 2026-10-22 night: per-request restore size (load() sets last_load_src)
    last_load_bytes = (int64_t) (st.kv_size + st.drft_size + st.rec_size);

    const double t_ms = (ggml_time_us() - t_start) / 1e3;

    // [TAG_COLD_TIER] finish the promotion: .rec/.tok/.hdr next to the copied segments, renames (header last), then the
    // cold files go. any failure removes the hot files and keeps the cold entry, the restore itself already succeeded
    if (promote) {
        bool ok = true;
        ok = ok && fflush(pk) == 0 && disk_fsync(pk);
        ok = ok && (!pd || (fflush(pd) == 0 && disk_fsync(pd)));
        ok = (fclose(pk) == 0) && ok;
        pk = nullptr;
        if (pd) {
            ok = (fclose(pd) == 0) && ok;
            pd = nullptr;
        }
        // the .rec only travels when it was valid; otherwise the hot header says rec_size 0 (checkpoint fallback)
        const uint64_t rec_size_hot = rec_ok ? rec.size() : 0;
        ok = ok && (!rec_ok || disk_write_file(hot_rec_tmp, rec.data(), rec.size()));
        ok = ok && disk_write_file(hot_tok_tmp, st.scheduled_tokens.data(), st.scheduled_tokens.size() * sizeof(llama_token));
        if (ok) {
            const std::vector<uint8_t> hdr = disk_make_hdr(arch_hash, vocab_hash, st.file_state_hash, (uint32_t) limit_tokens, st, rec_size_hot);
            ok = disk_write_file(hot_hdr_tmp, hdr.data(), hdr.size());
        }
        std::error_code ec;
        auto put = [&](const std::string & tmp, std::string_view ext, bool present) {
            const std::string dst = split_path(hot_base, ext);
            if (present) {
                std::filesystem::rename(tmp, dst, ec);
                ok = ok && !ec;
            } else {
                std::filesystem::remove(dst, ec);
            }
        };
        if (ok) {
            put(hot_kv_tmp,   SPLIT_EXT_KV,   true);
            put(hot_drft_tmp, SPLIT_EXT_DRFT, st.drft_size > 0);
            put(hot_rec_tmp,  SPLIT_EXT_REC,  rec_ok);
            put(hot_tok_tmp,  SPLIT_EXT_TOK,  true);
        }
        if (ok) {
            put(hot_hdr_tmp,  SPLIT_EXT_HDR,  true);
        }
        if (!ok) {
            promo_remove();
            for (const auto ext : { SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC, SPLIT_EXT_TOK }) {
                std::filesystem::remove(split_path(hot_base, ext), ec); // renamed already, no hot header names them
            }
            SRV_WRN("disk cache: promotion of '%s' failed at the commit - the cold entry is kept\n", conversation_id.c_str());
        } else {
            // the hot header exists: the entry resolves to hot from here. the cold header goes first so the entry is never
            // listed in both tiers longer than these unlinks take
            for (const auto ext : { SPLIT_EXT_HDR, SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC, SPLIT_EXT_TOK }) {
                std::filesystem::remove(split_path(base, ext), ec);
            }
            move_cp_files(conversation_id, disk_path_cold, disk_path);
            st.tier          = 1;
            st.rec_size      = rec_size_hot;
            st.rec_valid     = rec_ok;
            st.scheduled_rec = rec_ok;

            const double t_promo_ms = (ggml_time_us() - t_start) / 1e3;
            SRV_WRN("disk cache: promoted '%s' %u tokens %.1f MiB from cold in %.0f ms\n",
                    conversation_id.c_str(), st.n_tok,
                    (st.kv_size + st.drft_size + rec_size_hot) / (1024.0 * 1024.0), t_promo_ms);
        }
    }

    // seed the in-RAM mirror so the next flush appends instead of re-reading the header
    {
        std::lock_guard<std::mutex> lock(mtx);
        st.t_last_use = disk_now_s();
        auto d = disk_convs.find(conversation_id);
        if (d == disk_convs.end()) {
            disk_convs.emplace(conversation_id, std::move(st));
        } else {
            // [TAG_COLD_TIER] an existing mirror keeps its scheduled view; only where the files live and the idle clock change
            d->second.tier       = st.tier;
            d->second.t_last_use = st.t_last_use;
            if (from_cold && st.tier == 1) {
                d->second.rec_size      = st.rec_size;
                d->second.rec_valid     = st.rec_valid;
                d->second.scheduled_rec = d->second.scheduled_rec && st.rec_valid;
            }
        }
    }

    SRV_WRN("load_from_disk: restored '%s' tokens=%zu in %.0f ms (%.1f MiB/s)\n",
            conversation_id.c_str(), prompt.tokens.size(), t_ms,
            t_ms > 0.0 ? ((last_load_bytes) / (1024.0 * 1024.0)) / (t_ms / 1e3) : 0.0);

    // [fork, 2026-10-30] one line per converted restore (the library adds one per segment it converted); the next
    // flush of this conversation rewrites it in this server's type, see flush_from_context
    {
        const llama_state_convert_stats c1_tgt = llama_state_get_convert_stats(ctx_tgt);
        const llama_state_convert_stats c1_dft = ctx_dft ? llama_state_get_convert_stats(ctx_dft) : conv0_dft;

        const uint64_t src_tgt = c1_tgt.n_bytes_src - conv0_tgt.n_bytes_src;
        const uint64_t src_dft = c1_dft.n_bytes_src - conv0_dft.n_bytes_src;
        const uint64_t dst_all = (c1_tgt.n_bytes_dst - conv0_tgt.n_bytes_dst) + (c1_dft.n_bytes_dst - conv0_dft.n_bytes_dst);

        if (src_tgt + src_dft > 0) {
            const llama_state_convert_stats & c = src_tgt > 0 ? c1_tgt : c1_dft;
            SRV_WRN("disk cache: restored '%s' with K/V conversion %s->%s (%.1f MiB) - main %.1f MiB, drft %.1f MiB as saved, %.1f MiB in the cache\n",
                    conversation_id.c_str(),
                    c.type_src >= 0 ? ggml_type_name((ggml_type) c.type_src) : "?",
                    c.type_dst >= 0 ? ggml_type_name((ggml_type) c.type_dst) : "?",
                    (src_tgt + src_dft) / (1024.0 * 1024.0),
                    src_tgt / (1024.0 * 1024.0), src_dft / (1024.0 * 1024.0), dst_all / (1024.0 * 1024.0));
        }
    }

    return true;
}

// [TAG_DISK_ENOSPC] the recurrent half of a hybrid restore from a cp_ file, for a .hdr whose .rec is missing or stale. the
// attention segments of [0, st.n_tok) are already applied. newest first: the first cp_ file of this token timeline
// (token_prefix_hash over st's tokens) with n_tokens <= st.n_tok that loads wins. returns its n_tokens, 0 = none
int64_t server_prompt_cache::restore_rec_from_checkpoint(const std::string & conversation_id, const disk_conv_state & st,
                                                         llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    cp_index_seed(conversation_id);

    std::vector<cp_file_entry> entries;
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto & kv : cp_index[conversation_id].files) {
            entries.push_back(kv.second);
        }
    }
    std::sort(entries.begin(), entries.end(), [](const cp_file_entry & a, const cp_file_entry & b) { return a.n_tokens > b.n_tokens; });

    const server_tokens toks(llama_tokens(st.scheduled_tokens.begin(), st.scheduled_tokens.end()), false);

    for (const auto & e : entries) {
        // pos_max + 1 == n_tokens: text-only timelines (the disk tier never holds mtmd), the cut below is by position
        if (e.status != cp_file_entry::CP_OK || e.n_tokens <= 0 || e.n_tokens > (int64_t) st.n_tok || e.pos_max + 1 != e.n_tokens) {
            continue;
        }
        if (hash_token_prefix(toks, (size_t) e.n_tokens) != e.token_prefix_hash) {
            continue;
        }

        // the recurrent state of this checkpoint has seen exactly the first n_tokens tokens: drop the attention cells
        // past it (the recurrent child holds nothing for this sequence yet, so only the attention child is cut)
        const llama_pos p0 = (llama_pos) e.n_tokens;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, p0, -1) ||
            (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, p0, -1))) {
            SRV_WRN("load_from_disk: '%s' cannot cut the attention cells at %d - no checkpoint restore\n", conversation_id.c_str(), p0);
            return 0;
        }

        common_prompt_checkpoint cp;
        cp.pos_min           = e.pos_min;
        cp.pos_max           = e.pos_max;
        cp.n_tokens          = e.n_tokens;
        cp.token_prefix_hash = e.token_prefix_hash;
        cp.src_path          = e.path;
        cp.off_tgt           = e.off_tgt;
        cp.sz_tgt            = e.sz_tgt;
        cp.off_dft           = e.off_dft;
        cp.sz_dft            = e.sz_dft;

        if (!cp.load_tgt(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
            !cp.load_dft(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
            SRV_WRN("load_from_disk: checkpoint '%s' failed to load - trying an older one\n", std::filesystem::path(e.path).filename().string().c_str());
            forget_checkpoint_spill(conversation_id, e.pos_min);
            continue;
        }

        SRV_WRN("load_from_disk: '%s' recurrent state from checkpoint '%s' (%" PRId64 " of %u tokens, the last %" PRId64 " are reprocessed)\n",
                conversation_id.c_str(), std::filesystem::path(e.path).filename().string().c_str(), e.n_tokens, st.n_tok,
                (int64_t) st.n_tok - e.n_tokens);
        return e.n_tokens;
    }

    return 0;
}

void server_prompt_cache::wait_idle() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]() { return queue.empty() && writer_active_conv.empty(); });
}

void server_prompt_cache::shutdown_and_spill() {
    // idempotent: everything durable was flushed write-through at prompt_save time, so
    // shutdown only needs to drain the async writer queue
    size_t queued_jobs  = 0;
    size_t queued_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        queued_jobs  = queue.size();
        queued_bytes = queue_bytes;
    }

    if (queued_jobs > 0) {
        SRV_WRN("disk cache: shutdown - draining %zu queued write(s) (%.1f MiB)\n",
                queued_jobs, queued_bytes / (1024.0 * 1024.0));
    }

    stop_flag.store(true);
    cv.notify_all();
    if (worker.joinable()) {
        worker.join(); // the writer drains the whole queue before exiting
        SRV_WRN("%s", "disk cache: async writer drained - shutdown complete\n");
    }

    // mtmd states cannot be persisted
    states.clear();
}

//
// server_prompt_cache - cold tier ([TAG_COLD_TIER])
//

void server_prompt_cache::set_live_conversations(const std::vector<std::string> & ids) {
    if (disk_path_cold.empty()) {
        return;
    }
    std::unordered_set<std::string> next(ids.begin(), ids.end());
    std::lock_guard<std::mutex> lock(mtx);
    if (next != live_convs) {
        live_convs = std::move(next);
    }
}

int server_prompt_cache::resolve_tier(const std::string & conversation_id, std::string & base) const {
    std::error_code ec;
    if (!disk_path.empty()) {
        const std::string b = (std::filesystem::path(disk_path) / conversation_id).string();
        if (std::filesystem::exists(split_path(b, SPLIT_EXT_HDR), ec)) {
            base = b;
            return 1;
        }
    }
    if (!disk_path_cold.empty()) {
        const std::string b = (std::filesystem::path(disk_path_cold) / conversation_id).string();
        if (std::filesystem::exists(split_path(b, SPLIT_EXT_HDR), ec)) {
            base = b;
            return 2;
        }
    }
    return 0;
}

void server_prompt_cache::move_cp_files(const std::string & conv_id, const std::string & from_dir, const std::string & to_dir) {
    const std::string conv_prefix = "cp_" + conv_id + "_";
    std::vector<std::pair<int32_t, std::string>> files;
    {
        std::error_code ec;
        for (const auto & de : std::filesystem::directory_iterator(from_dir, ec)) {
            if (ec) {
                break;
            }
            int32_t pmin = 0;
            const std::string fn = de.path().filename().string();
            if (parse_cp_file_name(fn, conv_prefix, pmin)) {
                files.emplace_back(pmin, fn);
            }
        }
    }

    size_t n_moved = 0;
    for (const auto & [pmin, fn] : files) {
        const std::string src = (std::filesystem::path(from_dir) / fn).string();
        const std::string dst = (std::filesystem::path(to_dir)   / fn).string();
        std::error_code ec;
        // rename first (same filesystem); hot and cold are normally different disks, then copy + unlink
        std::filesystem::rename(src, dst, ec);
        if (ec) {
            const std::string tmp = dst + ".tmp";
            if (disk_copy_file(src, tmp, []() { return false; }) != 1) {
                SRV_WRN("disk cache: cannot copy '%s' to '%s' - left in place\n", src.c_str(), to_dir.c_str());
                continue;
            }
            const auto t = std::filesystem::last_write_time(src, ec);
            if (!ec) {
                std::filesystem::last_write_time(tmp, t, ec);
            }
            std::filesystem::rename(tmp, dst, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                continue;
            }
            std::filesystem::remove(src, ec);
        }
        ++n_moved;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cp_index.find(conv_id);
        if (it != cp_index.end()) {
            auto f = it->second.files.find(pmin);
            if (f != it->second.files.end()) {
                f->second.path = dst;
            }
        }
    }
    if (n_moved > 0) {
        SRV_INF("disk cache: moved %zu checkpoint file(s) of '%s' to '%s'\n", n_moved, conv_id.c_str(), to_dir.c_str());
    }
}

// writer thread. lists the hot .hdr files and demotes every entry that is idle (last write or restore older than
// demote_after_s), held by no slot, not being loaded and has nothing queued. yields to queued work between entries and
// inside an entry's copy
void server_prompt_cache::demote_scan() {
    if (disk_path_cold.empty()) {
        return;
    }

    std::vector<std::string> ids;
    {
        std::error_code ec;
        for (const auto & de : std::filesystem::directory_iterator(disk_path, ec)) {
            if (ec) {
                break;
            }
            const std::string fn = de.path().filename().string();
            if (fn.size() > SPLIT_EXT_HDR.size() && fn.compare(fn.size() - SPLIT_EXT_HDR.size(), SPLIT_EXT_HDR.size(), SPLIT_EXT_HDR) == 0) {
                ids.push_back(fn.substr(0, fn.size() - SPLIT_EXT_HDR.size()));
            }
        }
    }

    const int64_t now = disk_now_s();
    size_t n_demoted = 0;
    for (const auto & id : ids) {
        if (stop_flag.load()) {
            break;
        }
        const int64_t age_file = disk_file_age_s(split_path((std::filesystem::path(disk_path) / id).string(), SPLIT_EXT_HDR));
        if (age_file < 0) {
            continue; // gone since the listing
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!queue.empty()) {
                break; // appends first, the next idle period resumes
            }
            if (live_convs.count(id) || loading_convs.count(id) || staging.count(id)) {
                continue;
            }
            int64_t age = age_file;
            const auto d = disk_convs.find(id);
            if (d != disk_convs.end() && d->second.t_last_use > 0) {
                age = std::min(age, now - d->second.t_last_use);
            }
            if (age < demote_after_s) {
                continue;
            }
            // load_from_disk() of this conversation now waits until the move is done (or abandoned)
            writer_active_conv = id;
        }

        const int r = demote_one(id);

        {
            std::lock_guard<std::mutex> lock(mtx);
            writer_active_conv.clear();
            cv.notify_all();
        }
        if (r < 0) {
            break;
        }
        n_demoted += r;
    }
    if (n_demoted > 0) {
        SRV_INF("disk cache: demote scan moved %zu of %zu hot entr%s to cold\n", n_demoted, ids.size(), ids.size() == 1 ? "y" : "ies");
    }
}

int server_prompt_cache::demote_one(const std::string & conv_id) {
    const std::string hot_base  = (std::filesystem::path(disk_path)      / conv_id).string();
    const std::string cold_base = (std::filesystem::path(disk_path_cold) / conv_id).string();

    // the files of the entry: data first, then the cp_ files, the header last (it makes the cold copy visible)
    std::vector<std::pair<std::string, std::string>> moves; // hot path, cold path
    uint64_t total = 0;
    std::error_code ec;
    auto add = [&](const std::string & hot, const std::string & cold) {
        const uint64_t sz = (uint64_t) std::filesystem::file_size(hot, ec);
        if (!ec) {
            moves.emplace_back(hot, cold);
            total += sz;
        }
    };
    for (const auto ext : { SPLIT_EXT_KV, SPLIT_EXT_DRFT, SPLIT_EXT_REC, SPLIT_EXT_TOK }) {
        add(split_path(hot_base, ext), split_path(cold_base, ext));
    }
    {
        const std::string conv_prefix = "cp_" + conv_id + "_";
        for (const auto & de : std::filesystem::directory_iterator(disk_path, ec)) {
            if (ec) {
                break;
            }
            int32_t pmin = 0;
            const std::string fn = de.path().filename().string();
            if (parse_cp_file_name(fn, conv_prefix, pmin)) {
                add(de.path().string(), (std::filesystem::path(disk_path_cold) / fn).string());
            }
        }
    }
    add(split_path(hot_base, SPLIT_EXT_HDR), split_path(cold_base, SPLIT_EXT_HDR));
    if (moves.empty() || moves.back().first != split_path(hot_base, SPLIT_EXT_HDR)) {
        return 0; // the header vanished (the external sweep)
    }

    const auto sp = std::filesystem::space(disk_path_cold, ec);
    if (!ec && sp.available < total + DISK_FREE_MARGIN_BYTES) {
        SRV_WRN("disk cache: not demoting '%s' (%.1f MiB) - %.1f MiB free in the cold dir\n",
                conv_id.c_str(), total / (1024.0 * 1024.0), sp.available / (1024.0 * 1024.0));
        return 0;
    }

    const int64_t t_start = ggml_time_us();

    // a load of this conversation or any queued write wins over the move: the partial cold copy is dropped
    auto yield = [this, &conv_id]() {
        if (stop_flag.load()) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mtx);
        return !queue.empty() || loading_convs.count(conv_id) > 0;
    };

    std::vector<std::string> copied; // cold .tmp paths
    auto drop_copies = [&]() {
        for (const auto & t : copied) {
            std::filesystem::remove(t, ec);
        }
    };
    for (const auto & [hot, cold] : moves) {
        const std::string tmp = cold + ".tmp";
        const int r = disk_copy_file(hot, tmp, yield);
        if (r != 1) {
            drop_copies();
            if (r < 0) {
                SRV_INF("disk cache: demotion of '%s' yielded to queued work - retried on the next idle scan\n", conv_id.c_str());
                return -1;
            }
            SRV_WRN("disk cache: demotion of '%s' failed copying '%s' - the hot entry is kept\n", conv_id.c_str(), hot.c_str());
            return 0;
        }
        // keep the last-write time: the external sweep (3 days on the .kv mtime, both dirs) measures idle time from the
        // last real use, not from the day the entry moved
        const auto t = std::filesystem::last_write_time(hot, ec);
        if (!ec) {
            std::filesystem::last_write_time(tmp, t, ec);
        }
        copied.push_back(tmp);
    }

    // commit the cold copy (header last), then remove the hot files (header first: from here the entry resolves to cold)
    for (size_t i = 0; i < moves.size(); ++i) {
        std::filesystem::rename(copied[i], moves[i].second, ec);
        if (ec) {
            SRV_WRN("disk cache: demotion of '%s' failed renaming '%s' (%s) - the hot entry is kept\n",
                    conv_id.c_str(), copied[i].c_str(), ec.message().c_str());
            for (size_t j = 0; j < i; ++j) {
                std::filesystem::remove(moves[j].second, ec); // no cold header was renamed yet (it is the last move)
            }
            drop_copies();
            return 0;
        }
    }
    std::filesystem::remove(moves.back().first, ec);
    for (size_t i = 0; i + 1 < moves.size(); ++i) {
        std::filesystem::remove(moves[i].first, ec);
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        auto d = disk_convs.find(conv_id);
        if (d != disk_convs.end()) {
            d->second.tier = 2;
        }
        auto c = cp_index.find(conv_id);
        if (c != cp_index.end()) {
            for (auto & kv : c->second.files) {
                const std::string cold = (std::filesystem::path(disk_path_cold) / std::filesystem::path(kv.second.path).filename()).string();
                for (const auto & m : moves) {
                    if (m.second == cold) {
                        kv.second.path = cold;
                        break;
                    }
                }
            }
        }
    }

    SRV_WRN("disk cache: demoted '%s' %.1f MiB (%zu files) to cold in %.0f ms\n",
            conv_id.c_str(), total / (1024.0 * 1024.0), moves.size(), (ggml_time_us() - t_start) / 1e3);
    return 1;
}
