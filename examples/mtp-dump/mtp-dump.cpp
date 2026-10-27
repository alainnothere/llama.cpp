// llama-mtp-dump: write the hidden states that the NextN/MTP draft head consumes, for training a new head.
//
// dump mode (default):
//   llama-mtp-dump -m model.gguf -c N --out-dir DIR conv1.tok conv2.tok ...
//   each .tok is a raw int32 llama_token stream of one conversation. per conversation it writes
//   <name>.h.f16 ([n_tokens, n_embd] f16, row i = h_i), <name>.tok (copy) and <name>.meta.json (last, marks done).
//   --truncate: a conversation longer than -c is dumped for its first n_ctx tokens instead of skipped (h is causal, so
//   the prefix rows equal the full stream's); <name>.tok then holds only those tokens and meta.json records
//   "truncated": true and "n_tokens_orig". Added 2026-10-25 for the Qwen3.8 redo: f16 KV at 287k does not fit next to
//   the trunk on two 24 GB XTX (pass 1 at 217088 left ~2 GB/card), and q8_0 KV would make the h differ from what
//   production (f16 KV) computes, so the 4 longest conversations (238k..288k) lose their tail instead.
// validate mode:
//   llama-mtp-dump -m model.gguf --out-dir DIR --validate conv.tok [--n-validate 256]
//   runs the shipped MTP head the way draft-mtp does and writes validate.json (full-vocab top-10 per position).
//
// the llama-context.cpp comment "hidden state before the final output norm" does not hold for qwen35: t_h_nextn is taken after output_norm
// h_i is the target graph's res->t_h_nextn: the last trunk layer output AFTER output_norm (qwen35.cpp build
// graph, cb "h_nextn"). draft-mtp pairs it with token i+1 at MTP position i+1 to predict token i+2
// (common/speculative.cpp process(): h_row = h_tgt + (k - 1) * n_embd). See HEAD_CONTRACT.md in the dump dir.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn(_ith)

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const char * H_DEFINITION =
    "h_i = target hidden state of position i after the last trunk block and output_norm (graph tensor h_nextn, "
    "llama_get_embeddings_nextn unmasked); draft-mtp feeds h_i with tok_embd[t_{i+1}] at MTP position i+1 to predict t_{i+2}";

static bool read_tokens(const std::string & path, std::vector<llama_token> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const std::streamsize sz = f.tellg();
    if (sz < 0 || sz % (std::streamsize) sizeof(llama_token) != 0) {
        return false;
    }
    out.resize((size_t) sz / sizeof(llama_token));
    f.seekg(0);
    return (bool) f.read((char *) out.data(), sz);
}

static std::string json_str(const std::string & s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); r += b; }
        else { r += c; }
    }
    return r + "\"";
}

// decode tokens[0, n) at positions 0..n-1 on ctx (memory must be empty) and hand each batch of h rows to sink.
// the unmasked nextn buffer holds one row per batch token (llama-context.cpp output_reserve), so a decode
// batch must not exceed n_batch; only the last token of a batch requests logits.
template <typename F>
static bool decode_h(llama_context * ctx, const llama_token * tokens, int64_t n, int32_t n_batch, F && sink) {
    common_batch batch(ctx);
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i0 = 0; i0 < n; i0 += n_batch) {
        const int32_t nb = (int32_t) std::min<int64_t>(n_batch, n - i0);
        batch.clear();
        for (int32_t j = 0; j < nb; ++j) {
            batch.add(tokens[i0 + j], (llama_pos) (i0 + j), 0, j == nb - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("decode failed at pos %" PRId64 "\n", i0);
            return false;
        }
        llama_synchronize(ctx);
        const float * h = llama_get_embeddings_nextn(ctx);
        if (h == nullptr) {
            LOG_ERR("no nextn embeddings (pooling must be none and the model must export h_nextn)\n");
            return false;
        }
        if (!sink(h, nb)) {
            return false;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        LOG_INF("  %8" PRId64 " / %8" PRId64 "  %7.1f tok/s\n", i0 + nb, n, (double) (i0 + nb) / dt);
    }
    return true;
}

// debug (validate mode only, env MTP_DUMP_CB=DIR): write every F32 contiguous "mtp_*" graph tensor of the MTP
// context's decode to DIR/<name>.f32 + DIR/<name>.shape (ne0 ne1 ne2 ne3). Added 2026-10-25 to localize where the
// PyTorch port departs from the CPU reference on Qwen3.8 (records 0/1 were 7.7e-7 on 3.6, 6.6e-3 on 3.8).
static bool mtp_dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const char * nm = ggml_get_name(t);
    const bool want = strncmp(nm, "mtp_", 4) == 0 && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    if (ask) {
        return want;
    }
    if (want) {
        const std::string dir = (const char *) ud;
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        std::string fn = nm;
        std::replace(fn.begin(), fn.end(), '/', '_');
        FILE * f = fopen((dir + "/" + fn + ".f32").c_str(), "wb");
        if (f) { fwrite(buf.data(), sizeof(float), buf.size(), f); fclose(f); }
        f = fopen((dir + "/" + fn + ".shape").c_str(), "w");
        if (f) { fprintf(f, "%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]); fclose(f); }
    }
    return true;
}

static int run_validate(common_params & params, llama_model * model, llama_context * ctx_tgt,
        const std::string & tok_path, int n_val, const std::string & out_dir) {
    std::vector<llama_token> toks;
    if (!read_tokens(tok_path, toks)) {
        LOG_ERR("cannot read %s\n", tok_path.c_str());
        return 1;
    }
    // record i needs h_i, t_{i+1} and the target t_{i+2}
    n_val = (int) std::min<int64_t>(n_val, (int64_t) toks.size() - 2);
    if (n_val <= 0) {
        LOG_ERR("%s is too short\n", tok_path.c_str());
        return 1;
    }
    const int     n_rows = n_val + 1; // MTP rows at positions 0..n_val, row p = (t_p, h_{p-1})
    const int32_t n_embd = llama_model_n_embd_out(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // target h for positions 0..n_val-1 (h is causal, so the prefix gives the same rows as the full stream)
    std::vector<float> h_tgt;
    h_tgt.reserve((size_t) n_val * n_embd);
    if (!decode_h(ctx_tgt, toks.data(), n_val, llama_n_batch(ctx_tgt), [&](const float * h, int32_t nb) {
            h_tgt.insert(h_tgt.end(), h, h + (size_t) nb * n_embd);
            return true;
        })) {
        return 1;
    }

    // MTP context exactly as common_speculative_init_result builds it for draft-mtp
    auto cparams = common_context_params_to_llama(params);
    cparams.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;
    cparams.n_ctx     = std::max<uint32_t>(512, (uint32_t) GGML_PAD(n_rows, 256));
    cparams.n_batch   = cparams.n_ctx;
    cparams.n_ubatch  = std::min<uint32_t>(cparams.n_ubatch, cparams.n_batch);
    cparams.n_outputs_max = 0;
    static std::string cb_dir;
    if (const char * e = getenv("MTP_DUMP_CB")) {
        cb_dir = e;
        fs::create_directories(cb_dir);
        cparams.cb_eval           = mtp_dump_cb;
        cparams.cb_eval_user_data = (void *) cb_dir.c_str();
    }
    llama_context * ctx_dft = llama_init_from_model(model, cparams);
    if (ctx_dft == nullptr) {
        LOG_ERR("failed to create the MTP context\n");
        return 1;
    }
    llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

    // one catch-up decode like draft-mtp process(): row p gets token t_p and the target h of position p-1, with
    // zeros for p = 0 (pending_h starts zeroed, speculative.cpp ctor). every row is an output, so row p+1 is the
    // first draft row that predicts t_{p+2} from (h_p, t_{p+1}) with the MTP KV of rows 0..p behind it, which is
    // what the seed row of a draft step sees. no logits subset is set, so the head is the full output.weight.
    std::vector<float> h_zero(n_embd, 0.0f);
    common_batch batch(ctx_dft);
    for (int p = 0; p < n_rows; ++p) {
        const int32_t idx = batch.add(toks[p], p, 0, true);
        const float * h_row = p == 0 ? h_zero.data() : h_tgt.data() + (size_t) (p - 1) * n_embd;
        batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
    }
    if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("MTP decode failed\n");
        llama_free(ctx_dft);
        return 1;
    }
    llama_synchronize(ctx_dft);

    const std::string jpath = out_dir + "/validate.json";
    FILE * fj = fopen(jpath.c_str(), "w");
    FILE * fh = fopen((out_dir + "/validate.h_in.f32").c_str(), "wb");
    FILE * fo = fopen((out_dir + "/validate.mtp_h_out.f32").c_str(), "wb");
    if (!fj || !fh || !fo) {
        LOG_ERR("cannot write validate files in %s\n", out_dir.c_str());
        return 1;
    }
    fwrite(h_tgt.data(), sizeof(float), (size_t) n_val * n_embd, fh);

    fprintf(fj, "{\"model\": %s, \"source\": %s, \"n_validate\": %d, \"n_vocab\": %d, \"n_embd\": %d,\n",
            json_str(params.model.path).c_str(), json_str(tok_path).c_str(), n_val, n_vocab, n_embd);
    fprintf(fj, " \"how\": %s,\n", json_str(
        "one llama_process on a LLAMA_CONTEXT_TYPE_MTP context (as draft-mtp builds it) with rows p=0..n_validate: "
        "token t_p, embd h_{p-1} (zeros at p=0), pos p, all outputs; record i = MTP row i+1; logits = "
        "llama_get_logits_ith (result_output of graph_mtp, full output.weight, no vocab subset); "
        "validate.h_in.f32 = [n_validate, n_embd] f32 h_0.. fed to the head; validate.mtp_h_out.f32 = "
        "[n_validate, n_embd] f32 MTP h_nextn (rms(x)*shared_head_norm, the h of a chained draft row) per record").c_str());
    fprintf(fj, " \"records\": [\n");
    std::vector<int32_t> ids(n_vocab);
    for (int i = 0; i < n_val; ++i) {
        const int32_t row = i + 1;
        // measured 2026-10-25 on Vulkan (2x 7900 XTX, F16 output.weight): every logit here is exactly f16-representable
        // (range 4.6..39.7, so 1/64..1/32 steps); a port must compare against f16-rounded logits, not raw f32
        // the --mtp-head round trip (same bytes) gave a byte-identical validate.json; shared_head_norm * 0.9 moved top-1 logits by up to 4.25
        const float * lg = llama_get_logits_ith(ctx_dft, row);
        const float * ho = llama_get_embeddings_nextn_ith(ctx_dft, row);
        fwrite(ho, sizeof(float), n_embd, fo);
        for (int32_t k = 0; k < n_vocab; ++k) { ids[k] = k; }
        std::partial_sort(ids.begin(), ids.begin() + 10, ids.end(), [&](int32_t a, int32_t b) {
            return lg[a] > lg[b] || (lg[a] == lg[b] && a < b);
        });
        fprintf(fj, "  {\"pos\": %d, \"token_next\": %d, \"target\": %d, \"top10_ids\": [", i, toks[i + 1], toks[i + 2]);
        for (int k = 0; k < 10; ++k) { fprintf(fj, "%s%d", k ? ", " : "", ids[k]); }
        fprintf(fj, "], \"top10_logits\": [");
        for (int k = 0; k < 10; ++k) { fprintf(fj, "%s%.9g", k ? ", " : "", lg[ids[k]]); }
        fprintf(fj, "]}%s\n", i + 1 < n_val ? "," : "");
    }
    fprintf(fj, " ]}\n");
    fclose(fj);
    fclose(fh);
    fclose(fo);
    llama_free(ctx_dft);

    LOG_INF("wrote %s (%d records)\n", jpath.c_str(), n_val);
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string out_dir;
    std::string validate_path;
    int n_validate = 256;
    bool truncate = false;
    std::vector<std::string> inputs;

    // own args first, the rest goes to the common parser
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--validate" && i + 1 < argc) {
            validate_path = argv[++i];
        } else if (a == "--truncate") {
            truncate = true;
        } else if (a == "--n-validate" && i + 1 < argc) {
            n_validate = std::stoi(argv[++i]);
        } else if (a.size() > 4 && a.compare(a.size() - 4, 4, ".tok") == 0) {
            inputs.push_back(a);
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (out_dir.empty() || (inputs.empty() && validate_path.empty())) {
        LOG_ERR("usage: %s -m model.gguf -c N --out-dir DIR [--validate conv.tok [--n-validate 256]] [--truncate] [conv.tok ...]\n", argv[0]);
        return 1;
    }
    fs::create_directories(out_dir);

    params.pooling_type   = LLAMA_POOLING_TYPE_NONE;
    params.n_outputs_max  = 1;   // one logits row per batch, the h rows come from the unmasked nextn buffer
    params.n_parallel     = 1;
    params.warmup         = false;
    params.load_mtp       = true; // blk.64 is skipped at load unless asked for (nextn_flags), validate and --mtp-head need it

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load the model / create the context\n");
        return 1;
    }
    if (llama_model_n_layer_nextn(model) <= 0) {
        LOG_ERR("model has no NextN/MTP block\n");
        return 1;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

    const int32_t n_embd  = llama_model_n_embd_out(model);
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int64_t n_ctx   = llama_n_ctx(ctx);

    if (!validate_path.empty()) {
        const int rc = run_validate(params, model, ctx, validate_path, n_validate, out_dir);
        if (rc != 0 || inputs.empty()) {
            return rc;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    const char * ctk = ggml_type_name(params.cache_type_k);
    const char * ctv = ggml_type_name(params.cache_type_v);

    int64_t n_done = 0, n_skip = 0, bytes = 0;
    const auto t_all = std::chrono::steady_clock::now();

    for (const auto & in : inputs) {
        const std::string name = fs::path(in).stem().string();
        const std::string base = out_dir + "/" + name;

        std::vector<llama_token> toks;
        if (!read_tokens(in, toks) || toks.empty()) {
            LOG_WRN("SKIP %s: unreadable or empty\n", in.c_str());
            n_skip++;
            continue;
        }
        const int64_t n_orig = (int64_t) toks.size();
        const bool    trunc  = truncate && n_orig > n_ctx;
        if (trunc) {
            toks.resize((size_t) n_ctx);
            LOG_WRN("TRUNCATE %s: %" PRId64 " tokens > n_ctx %" PRId64 ", dumping the first %" PRId64 "\n", in.c_str(), n_orig, n_ctx, n_ctx);
        }
        const int64_t n = (int64_t) toks.size();
        if (n > n_ctx) {
            LOG_WRN("SKIP %s: %" PRId64 " tokens > n_ctx %" PRId64 "\n", in.c_str(), n, n_ctx);
            n_skip++;
            continue;
        }
        if (fs::exists(base + ".meta.json") && fs::exists(base + ".h.f16") &&
            fs::file_size(base + ".h.f16") == (uintmax_t) n * n_embd * sizeof(ggml_fp16_t)) {
            LOG_INF("DONE already: %s (%" PRId64 " tokens)\n", name.c_str(), n);
            continue;
        }
        fs::remove(base + ".meta.json");

        LOG_INF("dump %s: %" PRId64 " tokens\n", name.c_str(), n);
        const auto t0 = std::chrono::steady_clock::now();

        FILE * fh = fopen((base + ".h.f16").c_str(), "wb");
        if (!fh) {
            LOG_ERR("cannot write %s.h.f16\n", base.c_str());
            return 1;
        }
        std::vector<ggml_fp16_t> h16((size_t) n_batch * n_embd);
        llama_memory_clear(llama_get_memory(ctx), true);
        const bool ok = decode_h(ctx, toks.data(), n, n_batch, [&](const float * h, int32_t nb) {
            ggml_fp32_to_fp16_row(h, h16.data(), (int64_t) nb * n_embd);
            return fwrite(h16.data(), sizeof(ggml_fp16_t), (size_t) nb * n_embd, fh) == (size_t) nb * n_embd;
        });
        fclose(fh);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!ok) {
            LOG_ERR("FAILED %s\n", name.c_str());
            return 1;
        }

        {
            std::ofstream ft(base + ".tok", std::ios::binary);
            ft.write((const char *) toks.data(), (std::streamsize) (toks.size() * sizeof(llama_token)));
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        {
            std::ofstream fm(base + ".meta.json");
            fm << "{\"n_tokens\": " << n << ", \"d\": " << n_embd << ", \"dtype\": \"f16\""
               << ", \"model\": " << json_str(params.model.path)
               << ", \"h_definition\": " << json_str(H_DEFINITION)
               << ", \"source\": " << json_str(fs::absolute(in).string())
               << ", \"n_ctx_used\": " << n_ctx
               << ", \"n_batch\": " << n_batch << ", \"n_ubatch\": " << llama_n_ubatch(ctx)
               << ", \"cache_type_k\": \"" << ctk << "\", \"cache_type_v\": \"" << ctv << "\""
               << ", \"mtp_head\": " << json_str(params.mtp_head)
               << ", \"truncated\": " << (trunc ? "true" : "false") << ", \"n_tokens_orig\": " << n_orig
               << ", \"seconds\": " << dt << "}\n";
        }
        n_done++;
        bytes += n * n_embd * (int64_t) sizeof(ggml_fp16_t);
        LOG_INF("DONE %s: %" PRId64 " tokens in %.1f s (%.1f tok/s)\n", name.c_str(), n, dt, n / dt);
    }

    const double dt_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    LOG_INF("summary: %" PRId64 " dumped, %" PRId64 " skipped, %.2f GB h written, %.1f s\n", n_done, n_skip, bytes / 1e9, dt_all);
    return 0;
}
