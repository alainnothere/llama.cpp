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

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <system_error>

#ifdef _WIN32
#    include <io.h>
#else
#    include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// disk-cache helpers (file-local)
//

namespace {

constexpr uint32_t DISK_CACHE_MAGIC   = 0x53504344; // 'SPCD' - Server Prompt Cache Disk
// v4: split files per conversation, one write mode each:
//   {conv}.hdr  - overwrite - magic/version/hashes, n_tok, sizes, segment table, checksum
//   {conv}.tok  - overwrite - raw llama_token array
//   {conv}.kv   - append    - self-contained pos-range state blobs, applied in order on load
//   {conv}.drft - append    - same segmentation for the draft context (absent without a draft)
// checkpoints live in their own write-once cp_{conv}_{pos_min}.bin files (see below)
constexpr uint32_t DISK_CACHE_VERSION = 4;

// number of tokens covered by one .kv segment when writing a large state from scratch
// (bounds RAM on both save and load - the peak is one segment, ~500 MiB at ~60 KB/token)
constexpr uint32_t DISK_SEGMENT_TOKENS = 8192;

// backpressure for the async writer: enqueueing blocks while this many payload bytes
// are already in flight, so a huge flush cannot pile up unbounded in host RAM
constexpr size_t DISK_QUEUE_MAX_BYTES = 1ull << 30; // 1 GiB

// checkpoint files - written by spill_checkpoint() the moment create_checkpoint() captures a
// checkpoint. format: magic, version, arch_hash, vocab_hash,
// pos_min (i32), pos_max (i32), n_tokens (i64), data_tgt_size (u64), data_tgt, data_dft_size (u64), data_dft.
// checkpoint files are named cp_{conversation_id}_{pos_min}.bin and live in disk_path
constexpr uint32_t CHECKPOINT_SPILL_MAGIC   = 0x43504B44; // 'CPKD'
constexpr uint32_t CHECKPOINT_SPILL_VERSION = 1;

// byte offsets inside a cp_ file (see format above) - used to register disk-backed checkpoints
// without reading the blobs: off_tgt = header, off_dft = header + tgt blob + its size field
constexpr uint64_t CHECKPOINT_FILE_HEADER_BYTES = 4*sizeof(uint32_t) + 2*sizeof(int32_t) + sizeof(int64_t) + sizeof(uint64_t);


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

// split-file extensions (v4)
constexpr std::string_view SPLIT_EXT_HDR  = ".hdr";
constexpr std::string_view SPLIT_EXT_TOK  = ".tok";
constexpr std::string_view SPLIT_EXT_KV   = ".kv";
constexpr std::string_view SPLIT_EXT_DRFT = ".drft";

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

// overwrite atomically: write tmp, fsync, rename over the target
static bool disk_write_file_atomic(const std::string & path, const void * data, size_t n) {
    const std::string tmppath = path + ".tmp";

    FILE * f = fopen(tmppath.c_str(), "wb");
    if (!f) {
        return false;
    }
    bool ok = n == 0 || fwrite(data, 1, n, f) == n;
    ok = ok && fflush(f) == 0;
    ok = ok && disk_fsync(f);
    ok = fclose(f) == 0 && ok;

    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(tmppath, ec);
        return false;
    }
    std::filesystem::rename(tmppath, path, ec);
    if (ec) {
        std::filesystem::remove(tmppath, ec);
        return false;
    }
    return true;
}

// append and fsync - the caller is responsible for truncating orphan tails first
static bool disk_append_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "ab");
    if (!f) {
        return false;
    }
    bool ok = n == 0 || fwrite(data, 1, n, f) == n;
    ok = ok && fflush(f) == 0;
    ok = ok && disk_fsync(f);
    ok = fclose(f) == 0 && ok;
    return ok;
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
            {"generation_prompt",         chat_parser_params.generation_prompt},
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
        {"generation_prompt",         chat_parser_params.generation_prompt},
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
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
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

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
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
    // nlohmann::json converts -inf to null, so we need to prevent that
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
        {"content",             content},
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
        {"timings",             timings.to_json()},
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
                {"text",          content},
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
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
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
        msg.content = content;
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
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
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

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
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
        msg.content = content;
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
        msg.content = content;
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

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

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
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
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
                {"text",          content},
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
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
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

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
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
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
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
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
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
        res += state.n_tokens();
    }

    return res;
}

server_prompt * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // next, remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
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
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.checkpoints =*/ prompt.checkpoints,
        /*.conversation_id =*/ prompt.conversation_id,
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::string & conversation_id, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best = states.end();

    // Phase 1: find best match in states
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float sim_cur    = float(lcp_cur) / tokens_new.size();

        if (f_keep_cur < 0.25f) {
            continue;
        }

        // sim-primary (see [TAG_CACHE_SELECT_SIM]). the f_keep tie-break only
        // applies between scanned candidates (it_best already set), NOT against
        // the slot's baseline sim_best — restoring a cache entry that merely ties
        // the slot's existing prefix buys zero extra reuse and is pure overhead.
        if (sim_cur > sim_best || (it_best != states.end() && sim_cur == sim_best && f_keep_cur > f_keep_best)) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

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

        // Copy tokens + checkpoints into the slot's prompt. NOT data — the
        // bytes were just loaded into ctx_tgt/ctx_dft via set_data_ext, so
        // the live KV is the source of truth. server_slot::prompt_save()
        // asserts prompt.data.size() == 0 (server-context.cpp:112) before
        // it captures a fresh save from the live context, so leaving
        // data empty here is required.  Cached entry's own copies stay
        // in `states` for the next match.  alloc() will dedup on next
        // save if this slot's post-generation state strictly supersedes
        // the cached entry (prefix-of check at server-task.cpp:2078).
        prompt.tokens      = it_best->tokens.clone();
        prompt.data        = {};
        prompt.checkpoints = it_best->checkpoints;
        prompt.conversation_id = it_best->conversation_id;

        if (!disk_path.empty()) {
            merge_checkpoint_spills(prompt);
        }
        return true;
    }

    // Phase 3: disk fallback - direct lookup by conversation_id
    if (!disk_path.empty() && !conversation_id.empty()) {
        if (load_from_disk(prompt, conversation_id, ctx_tgt, ctx_dft, id_slot)) {
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
                states.front().tokens.has_mtmd ? " [mtmd: not spillable]" : "");
        states.pop_front();
    };

    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            evict_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            evict_oldest();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
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
                                          uint32_t arch_hash_in, uint32_t vocab_hash_in) {
    this->limit_size            = 1024ull * 1024ull * (limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens          = limit_tokens_in;
    this->disk_path             = std::move(disk_path_in);
    this->queue_depth           = queue_depth_in > 0 ? queue_depth_in : 16;
    this->checkpoint_spill_max  = checkpoint_spill_max_in;
    this->arch_hash             = arch_hash_in;
    this->vocab_hash            = vocab_hash_in;

    if (!this->disk_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(this->disk_path, ec);
        if (ec) {
            SRV_WRN("disk cache: cannot create directory '%s': %s — disk tier disabled\n",
                    this->disk_path.c_str(), ec.message().c_str());
            this->disk_path.clear();
        } else {
            SRV_WRN("disk cache: enabled at '%s' (queue depth %d, checkpoint spill max %d, arch=0x%08x, vocab=0x%08x)\n",
                    this->disk_path.c_str(), this->queue_depth, this->checkpoint_spill_max,
                    this->arch_hash, this->vocab_hash);
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
    while (true) {
        disk_job job;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return stop_flag.load() || !queue.empty(); });
            if (stop_flag.load() && queue.empty()) {
                return;
            }
            job = std::move(queue.front());
            queue.pop_front();
            writer_active_conv = job.conv_id;
        }

        // I/O outside the lock - writes can take seconds.
        // write failures mark the conversation broken (next flush rewrites from scratch)
        // rather than retrying: real failures here are persistent (ENOSPC, EIO, unmounted)
        // and a retry loop would peg the disk without making progress.
        process_disk_job(job);

        {
            std::lock_guard<std::mutex> lock(mtx);
            queue_bytes -= std::min(queue_bytes, job.bytes());
            writer_active_conv.clear();
            cv.notify_all();
        }
    }
}

bool server_prompt_cache::flush_from_context(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
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

    // pos-range delta segments are only self-contained when the serialized state covers the
    // whole history (plain full-attention unified KV). windowed state (SWA / hybrid /
    // recurrent) is small by construction and gets a single full-state rewrite instead.
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    bool delta_capable = llama_model_n_swa(model_tgt) == 0 &&
                         !llama_model_is_recurrent(model_tgt) &&
                         !llama_model_is_hybrid(model_tgt);
    if (ctx_dft) {
        const llama_model * model_dft = llama_get_model(ctx_dft);
        delta_capable = delta_capable &&
                        llama_model_n_swa(model_dft) == 0 &&
                        !llama_model_is_recurrent(model_dft) &&
                        !llama_model_is_hybrid(model_dft);
    }

    // first touch of this conversation: seed the in-RAM mirror from an existing .hdr
    {
        bool have_state = false;
        {
            std::lock_guard<std::mutex> lock(mtx);
            have_state = disk_convs.count(conv_id) > 0;
        }
        if (!have_state) {
            disk_conv_state seed;
            read_disk_state(conv_id, seed); // missing/invalid files leave the default (rewrite)

            std::lock_guard<std::mutex> lock(mtx);
            disk_convs.emplace(conv_id, std::move(seed));
        }
    }

    // decide append-from-high-water-mark vs rewrite-from-scratch
    uint32_t base    = 0;
    bool     rewrite = true;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & st = disk_convs[conv_id];

        const bool prefix_ok = !st.broken &&
                               st.scheduled_n_tok > 0 &&
                               st.scheduled_n_tok <= n_tok &&
                               std::equal(st.scheduled_tokens.begin(), st.scheduled_tokens.end(), toks.begin());

        if (prefix_ok && st.scheduled_n_tok == n_tok) {
            SRV_INF("disk cache: '%s' already flushed (%u tokens) - nothing to write\n", conv_id.c_str(), n_tok);
            return true;
        }
        if (prefix_ok && delta_capable) {
            base    = st.scheduled_n_tok;
            rewrite = false;
        }
    }

    const uint32_t chunk = delta_capable ? DISK_SEGMENT_TOKENS : n_tok - base;

    SRV_WRN("disk cache: flushing '%s' tokens [%u, %u) as %u segment(s) (%s)\n",
            conv_id.c_str(), base, n_tok,
            (n_tok - base + chunk - 1) / chunk,
            rewrite ? "rewrite" : "append");

    bool first = rewrite;
    for (uint32_t p0 = base; p0 < n_tok; p0 += chunk) {
        const uint32_t p1 = std::min(p0 + chunk, n_tok);

        disk_job job;
        job.conv_id = conv_id;
        job.rewrite = first;
        first = false;
        job.p0 = (llama_pos) p0;
        job.p1 = (llama_pos) p1;

        auto capture_failed = [&](const char * which) {
            SRV_WRN("disk cache: state capture (%s) failed for '%s' pos [%u, %u) - marking for rewrite\n",
                    which, conv_id.c_str(), p0, p1);
            std::lock_guard<std::mutex> lock(mtx);
            disk_convs[conv_id].broken = true;
        };

        {
            const size_t sz = delta_capable
                ? llama_state_seq_get_size_range(ctx_tgt, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (sz == 0) {
                capture_failed("main");
                return false;
            }
            job.kv.resize(sz);
            const size_t got = delta_capable
                ? llama_state_seq_get_data_range(ctx_tgt, job.kv.data(), sz, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_data_ext(ctx_tgt, job.kv.data(), sz, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
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
            job.drft.resize(sz);
            const size_t got = delta_capable
                ? llama_state_seq_get_data_range(ctx_dft, job.drft.data(), sz, id_slot, (llama_pos) p0, (llama_pos) p1, LLAMA_STATE_SEQ_FLAGS_NONE)
                : llama_state_seq_get_data_ext(ctx_dft, job.drft.data(), sz, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (got != sz) {
                capture_failed("drft");
                return false;
            }
        }

        job.tokens.assign(toks.begin(), toks.begin() + p1);

        // enqueue with byte-budget backpressure so a big flush cannot pile up in host RAM
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return queue_bytes < DISK_QUEUE_MAX_BYTES; });

            auto & st = disk_convs[conv_id];
            st.scheduled_n_tok  = p1;
            st.scheduled_tokens = job.tokens;

            queue_bytes += job.bytes();
            queue.push_back(std::move(job));
            cv.notify_all();
        }
    }

    return true;
}

void server_prompt_cache::process_disk_job(const disk_job & job) {
    if (job.ckpt) {
        write_checkpoint_file(*job.ckpt, job.conv_id);
        return;
    }

    const std::string base      = (std::filesystem::path(disk_path) / job.conv_id).string();
    const std::string kv_path   = split_path(base, SPLIT_EXT_KV);
    const std::string drft_path = split_path(base, SPLIT_EXT_DRFT);

    const int64_t t_start = ggml_time_us();

    // snapshot the committed state
    disk_conv_state st;
    {
        std::lock_guard<std::mutex> lock(mtx);
        st = disk_convs[job.conv_id];
    }
    if (st.broken && !job.rewrite) {
        SRV_WRN("disk cache: dropping append for broken conversation '%s' (a rewrite will follow)\n", job.conv_id.c_str());
        return;
    }

    auto fail = [&](const char * what) {
        SRV_WRN("disk cache: %s failed for '%s' - marking conversation for rewrite\n", what, job.conv_id.c_str());
        std::lock_guard<std::mutex> lock(mtx);
        disk_convs[job.conv_id].broken = true;
    };

    std::error_code ec;

    if (job.rewrite) {
        st.segments.clear();
        st.kv_size   = 0;
        st.drft_size = 0;
        st.n_tok     = 0;

        std::filesystem::remove(kv_path, ec);
        std::filesystem::remove(drft_path, ec);
    } else {
        // crash-orphan cleanup: the data files must be at least as large as the committed
        // sizes; extra tail bytes come from an append whose header never committed
        const uint64_t kv_actual = (uint64_t) std::filesystem::file_size(kv_path, ec);
        if (ec || kv_actual < st.kv_size) {
            fail("kv size check");
            return;
        }
        if (kv_actual > st.kv_size) {
            std::filesystem::resize_file(kv_path, st.kv_size, ec);
            if (ec) {
                fail("kv truncate");
                return;
            }
        }
        if (st.drft_size > 0) {
            const uint64_t drft_actual = (uint64_t) std::filesystem::file_size(drft_path, ec);
            if (ec || drft_actual < st.drft_size) {
                fail("drft size check");
                return;
            }
            if (drft_actual > st.drft_size) {
                std::filesystem::resize_file(drft_path, st.drft_size, ec);
                if (ec) {
                    fail("drft truncate");
                    return;
                }
            }
        }
    }

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

    st.segments.push_back(seg);
    st.kv_size   += job.kv.size();
    st.drft_size += job.drft.size();
    st.n_tok      = (uint32_t) job.tokens.size();

    if (!disk_write_file_atomic(split_path(base, SPLIT_EXT_TOK), job.tokens.data(), job.tokens.size() * sizeof(llama_token))) {
        fail("tok write");
        return;
    }

    // header last - a crash before this point leaves only an orphan tail in the data
    // files, which the next append (or load) truncates away using the old header
    std::vector<uint8_t> hdr;
    auto put = [&hdr](const auto & v) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
        hdr.insert(hdr.end(), p, p + sizeof(v));
    };
    put(DISK_CACHE_MAGIC);
    put(DISK_CACHE_VERSION);
    put(arch_hash);
    put(vocab_hash);
    put((uint32_t) limit_tokens);
    put(st.n_tok);
    put((uint32_t) st.segments.size());
    put(st.kv_size);
    put(st.drft_size);
    for (const auto & s : st.segments) {
        put(s.kv_off);
        put(s.kv_size);
        put(s.drft_off);
        put(s.drft_size);
        put(s.p0);
        put(s.p1);
    }
    put(disk_hdr_checksum(hdr.data(), hdr.size()));

    if (!disk_write_file_atomic(split_path(base, SPLIT_EXT_HDR), hdr.data(), hdr.size())) {
        fail("hdr write");
        return;
    }

    const size_t n_segs = st.segments.size();

    // commit
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto & live = disk_convs[job.conv_id];
        live.n_tok     = st.n_tok;
        live.kv_size   = st.kv_size;
        live.drft_size = st.drft_size;
        live.segments  = std::move(st.segments);
        live.broken    = false;
    }

    const double t_ms = (ggml_time_us() - t_start) / 1e3;
    SRV_WRN("disk cache: wrote '%s' segment pos [%d, %d) %.1f MiB (drft %.1f MiB) in %.0f ms - now %u tokens in %zu segment(s)\n",
            job.conv_id.c_str(), job.p0, job.p1,
            job.kv.size() / (1024.0 * 1024.0), job.drft.size() / (1024.0 * 1024.0),
            t_ms, st.n_tok, n_segs);
}

bool server_prompt_cache::read_disk_state(const std::string & conversation_id, disk_conv_state & st) {
    const std::string base = (std::filesystem::path(disk_path) / conversation_id).string();

    std::vector<uint8_t> hdr;
    {
        std::ifstream f(split_path(base, SPLIT_EXT_HDR), std::ios::binary);
        if (!f) {
            return false;
        }
        hdr.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // fixed part: 7 x u32 + 2 x u64, then the segment table, then a u64 checksum
    constexpr size_t fixed_bytes = 7*sizeof(uint32_t) + 2*sizeof(uint64_t);
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

    uint32_t magic = 0, version = 0, fa = 0, fv = 0, lt = 0, n_tok = 0, n_segs = 0;
    uint64_t kv_size = 0, drft_size = 0;
    get(magic);
    get(version);
    get(fa);
    get(fv);
    get(lt);
    get(n_tok);
    get(n_segs);
    get(kv_size);
    get(drft_size);
    GGML_UNUSED(lt);

    if (magic != DISK_CACHE_MAGIC || version != DISK_CACHE_VERSION) {
        SRV_INF("disk cache: '%s.hdr' wrong magic/version (0x%08x, %u)\n", conversation_id.c_str(), magic, version);
        return false;
    }
    if (fa != arch_hash || fv != vocab_hash) {
        SRV_INF("disk cache: '%s.hdr' arch/vocab mismatch\n", conversation_id.c_str());
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

    std::vector<llama_token> toks(n_tok);
    {
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
    st.scheduled_n_tok  = n_tok;
    st.scheduled_tokens = std::move(toks);
    st.broken           = false;

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

    // write-at-creation, async: copy the blobs into the job so the resident checkpoint can be
    // converted to disk-backed (or evicted) while the write is still in flight
    disk_job job;
    job.conv_id = conversation_id;
    job.ckpt = std::make_shared<common_prompt_checkpoint>();
    job.ckpt->pos_min  = cp.pos_min;
    job.ckpt->pos_max  = cp.pos_max;
    job.ckpt->n_tokens = cp.n_tokens;
    job.ckpt->data_tgt = cp.data_tgt;
    job.ckpt->data_dft = cp.data_dft;

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
        cp.clear_tgt();
        cp.clear_dft();

        SRV_INF("checkpoint: converted pos_min=%d to disk-backed ('%s', %.1f MiB freed)\n",
                cp.pos_min, filename.c_str(), (sz_tgt + sz_dft) / (1024.0 * 1024.0));
    }
}

bool server_prompt_cache::write_checkpoint_file(const common_prompt_checkpoint & cp, const std::string & conversation_id) {
    // filename encodes conversation_id so the limit and merge logic can filter per conversation
    const std::string filename = "cp_" + conversation_id + "_" + std::to_string(cp.pos_min) + ".bin";
    const std::filesystem::path filepath = std::filesystem::path(disk_path) / filename;
    const std::filesystem::path tmppath  = std::filesystem::path(disk_path) / (filename + ".tmp");

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

        const int32_t pos_min_v  = cp.pos_min;
        const int32_t pos_max_v  = cp.pos_max;
        const int64_t n_tokens_v = cp.n_tokens;
        f.write(reinterpret_cast<const char *>(&pos_min_v),  sizeof(pos_min_v));
        f.write(reinterpret_cast<const char *>(&pos_max_v),  sizeof(pos_max_v));
        f.write(reinterpret_cast<const char *>(&n_tokens_v), sizeof(n_tokens_v));

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

    // enforce the per-conversation checkpoint spill limit by deleting the files that cover
    // the lowest positions (they are the least useful for future rewinds)
    if (checkpoint_spill_max > 0) {
        struct cp_entry {
            std::filesystem::path path;
            int32_t pos_min;
        };
        std::vector<cp_entry> cp_files;

        // only consider files belonging to this conversation: cp_{conversation_id}_{pos_min}.bin
        const std::string conv_prefix = "cp_" + conversation_id + "_";

        std::error_code ec2;
        for (const auto & e : std::filesystem::directory_iterator(disk_path, ec2)) {
            if (ec2) break;
            const std::string fn = e.path().filename().string();
            if (e.path().extension() != ".bin") {
                continue;
            }
            if (fn.size() <= conv_prefix.size() || fn.compare(0, conv_prefix.size(), conv_prefix) != 0) {
                continue;
            }
            // parse pos_min: the field immediately after the conversation_id prefix
            const size_t pmin_start = conv_prefix.size();
            const size_t pmin_end   = fn.find('_', pmin_start);
            if (pmin_end == std::string::npos) {
                continue;
            }
            int32_t pmin = 0;
            bool ok = true;
            for (size_t i = pmin_start; i < pmin_end; ++i) {
                if (fn[i] < '0' || fn[i] > '9') { ok = false; break; }
                pmin = pmin * 10 + (fn[i] - '0');
            }
            if (!ok) {
                continue;
            }
            cp_entry entry;
            entry.path    = e.path();
            entry.pos_min = pmin;
            cp_files.push_back(entry);
        }

        if ((int32_t) cp_files.size() > checkpoint_spill_max) {
            // sort ascending by pos_min so the front of the vector has the oldest entries
            for (size_t i = 0; i < cp_files.size(); ++i) {
                for (size_t j = i + 1; j < cp_files.size(); ++j) {
                    if (cp_files[j].pos_min < cp_files[i].pos_min) {
                        cp_entry tmp  = cp_files[i];
                        cp_files[i]   = cp_files[j];
                        cp_files[j]   = tmp;
                    }
                }
            }
            const int n_delete = (int) cp_files.size() - checkpoint_spill_max;
            for (int i = 0; i < n_delete; ++i) {
                std::error_code ec3;
                std::filesystem::remove(cp_files[i].path, ec3);
                SRV_WRN("checkpoint spill: evicted '%s' (pos_min=%d, limit=%d)\n",
                        cp_files[i].path.filename().string().c_str(), cp_files[i].pos_min, checkpoint_spill_max);
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

    std::error_code ec;
    if (!std::filesystem::exists(disk_path, ec) || ec) {
        return;
    }

    int n_scanned    = 0;
    int n_loaded     = 0;
    int n_skip_magic = 0;
    int n_skip_hash  = 0;
    int n_skip_dup   = 0;

    std::list<common_prompt_checkpoint> new_ckpts;

    // only load files that belong to this conversation: cp_{conversation_id}_{pos_min}.bin
    const std::string conv_prefix = "cp_" + prompt.conversation_id + "_";

    for (const auto & entry : std::filesystem::directory_iterator(disk_path, ec)) {
        if (ec) break;
        const std::string fname = entry.path().filename().string();
        if (entry.path().extension() != ".bin") {
            continue;
        }
        if (fname.size() <= conv_prefix.size() || fname.compare(0, conv_prefix.size(), conv_prefix) != 0) {
            continue;
        }
        ++n_scanned;

        std::ifstream f(entry.path(), std::ios::binary);
        if (!f) {
            continue;
        }

        uint32_t magic = 0, version = 0, fa = 0, fv = 0;
        f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (magic != CHECKPOINT_SPILL_MAGIC || version != CHECKPOINT_SPILL_VERSION) {
            ++n_skip_magic;
            continue;
        }

        f.read(reinterpret_cast<char *>(&fa), sizeof(fa));
        f.read(reinterpret_cast<char *>(&fv), sizeof(fv));

        if (fa != arch_hash || fv != vocab_hash) {
            ++n_skip_hash;
            continue;
        }

        int32_t pos_min_v  = 0;
        int32_t pos_max_v  = 0;
        int64_t n_tokens_v = 0;
        f.read(reinterpret_cast<char *>(&pos_min_v),  sizeof(pos_min_v));
        f.read(reinterpret_cast<char *>(&pos_max_v),  sizeof(pos_max_v));
        f.read(reinterpret_cast<char *>(&n_tokens_v), sizeof(n_tokens_v));
        if (!f) {
            continue;
        }

        // skip if this pos_min is already in the checkpoint list
        bool dup = false;
        for (const auto & cp : prompt.checkpoints) {
            if (cp.pos_min == pos_min_v) {
                dup = true;
                break;
            }
        }
        if (dup) {
            ++n_skip_dup;
            continue;
        }

        uint64_t tgt_sz = 0;
        f.read(reinterpret_cast<char *>(&tgt_sz), sizeof(tgt_sz));
        if (!f) {
            continue;
        }

        // LAZY (disk-backed) registration: record the file + the byte offsets of
        // the tgt/dft blobs, but DO NOT read them into RAM. The KV bytes stay on
        // disk; load_tgt/load_dft fault them in only if a rewind actually selects
        // this checkpoint, then discard them again. This is what keeps reviewed-
        // from-disk checkpoints off the anonymous heap. The file is not consumed
        // here — see [TAG_CACHE_DONT_DELETE].
        common_prompt_checkpoint cp;
        cp.pos_min  = pos_min_v;
        cp.pos_max  = pos_max_v;
        cp.n_tokens = n_tokens_v;
        cp.src_path = entry.path().string();

        // tgt blob starts at the current position (just past the tgt_sz field)
        const std::streamoff pos_tgt = f.tellg();
        cp.off_tgt = static_cast<uint64_t>(pos_tgt);
        cp.sz_tgt  = tgt_sz;

        // skip over the tgt blob to reach the dft_sz field
        f.seekg(static_cast<std::streamoff>(tgt_sz), std::ios::cur);

        uint64_t dft_sz = 0;
        f.read(reinterpret_cast<char *>(&dft_sz), sizeof(dft_sz));
        if (!f) {
            // truncated / malformed file — skip without registering
            continue;
        }

        // dft blob starts at the current position (just past the dft_sz field)
        const std::streamoff pos_dft = f.tellg();
        cp.off_dft = static_cast<uint64_t>(pos_dft);
        cp.sz_dft  = dft_sz;

        new_ckpts.push_back(std::move(cp));
        ++n_loaded;
    }

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


bool server_prompt_cache::load_from_disk(server_prompt & prompt, const std::string & conversation_id,
                                          llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (disk_path.empty() || conversation_id.empty()) {
        return false;
    }

    // wait until no write for this conversation is queued or in flight
    {
        std::unique_lock<std::mutex> lock(mtx);
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

    const std::string base = (std::filesystem::path(disk_path) / conversation_id).string();

    std::ifstream kf(split_path(base, SPLIT_EXT_KV), std::ios::binary);
    if (!kf) {
        return false;
    }
    std::ifstream df;
    if (runtime_has_dft) {
        df.open(split_path(base, SPLIT_EXT_DRFT), std::ios::binary);
        if (!df) {
            return false;
        }
    }

    const int64_t t_start = ggml_time_us();

    SRV_INF("load_from_disk: '%s' n_tok=%u kv=%.1f MiB drft=%.1f MiB segments=%zu\n",
            conversation_id.c_str(), st.n_tok,
            st.kv_size / (1024.0 * 1024.0), st.drft_size / (1024.0 * 1024.0), st.segments.size());

    auto wipe_slot = [&]() {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
    };

    // stream the segments into the context one buffer at a time: the first replaces the
    // sequence, the rest merge into it. peak host RAM = the largest single segment.
    std::vector<uint8_t> buf;
    for (size_t i = 0; i < st.segments.size(); ++i) {
        const auto & seg = st.segments[i];
        const llama_state_seq_flags flags = i == 0 ? LLAMA_STATE_SEQ_FLAGS_NONE : LLAMA_STATE_SEQ_FLAGS_APPEND;

        buf.resize(seg.kv_size);
        kf.seekg((std::streamoff) seg.kv_off);
        kf.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) seg.kv_size);
        if (!kf) {
            SRV_WRN("load_from_disk: '%s.kv' read failed at segment %zu\n", conversation_id.c_str(), i);
            wipe_slot();
            return false;
        }

        size_t n = llama_state_seq_set_data_ext(ctx_tgt, buf.data(), buf.size(), id_slot, flags);
        if (n != buf.size()) {
            SRV_WRN("load_from_disk: set_data (main, segment %zu) returned %zu of %zu\n", i, n, buf.size());
            wipe_slot();
            return false;
        }

        if (runtime_has_dft && seg.drft_size > 0) {
            buf.resize(seg.drft_size);
            df.seekg((std::streamoff) seg.drft_off);
            df.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) seg.drft_size);
            if (!df) {
                SRV_WRN("load_from_disk: '%s.drft' read failed at segment %zu\n", conversation_id.c_str(), i);
                wipe_slot();
                return false;
            }

            n = llama_state_seq_set_data_ext(ctx_dft, buf.data(), buf.size(), id_slot, flags);
            if (n != buf.size()) {
                SRV_WRN("load_from_disk: set_data (drft, segment %zu) returned %zu of %zu\n", i, n, buf.size());
                wipe_slot();
                return false;
            }
        }
    }

    // populate the prompt. checkpoints are not stored with the entry - the caller runs
    // merge_checkpoint_spills() which registers this conversation's cp_ files lazily.
    prompt.tokens.clear();
    {
        llama_tokens toks = st.scheduled_tokens;
        prompt.tokens.insert(toks);
    }
    prompt.data = {};
    prompt.checkpoints.clear();
    prompt.conversation_id = conversation_id;

    // seed the in-RAM mirror so the next flush appends instead of re-reading the header
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (disk_convs.count(conversation_id) == 0) {
            disk_convs.emplace(conversation_id, std::move(st));
        }
    }

    const double t_ms = (ggml_time_us() - t_start) / 1e3;
    SRV_WRN("load_from_disk: restored '%s' tokens=%zu in %.0f ms (%.1f MiB/s)\n",
            conversation_id.c_str(), prompt.tokens.size(), t_ms,
            t_ms > 0.0 ? ((st.kv_size + st.drft_size) / (1024.0 * 1024.0)) / (t_ms / 1e3) : 0.0);

    return true;
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
