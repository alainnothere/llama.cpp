// llama-mtp-replay: offline, head-only replay of the NextN/MTP draft chain against recorded target hidden states.
//
//   llama-mtp-replay -m model.gguf [common args] --circuit M0 [--circuit M0*2 ...] [--depth 8] [--stride 8]
//       [--bucket-filter dim:label] [--vocab-subset FILE] [--out result.json] DUMP_DIR/NAME [DUMP_DIR/NAME ...]
//
// inputs are mtp-dump conversations (examples/mtp-dump): NAME.tok (int32 t_0..t_{N-1}), NAME.h.f16 ([N, n_embd]
// row i = h_i, the target's last-trunk-layer output after output_norm) and NAME.meta.json; plus the bucket sidecar
// NAME.buckets.json written by tag_buckets.py (looked up in --buckets-dir, default DUMP_DIR/../buckets, then next to
// the .tok). No target context is created and no trunk layer runs: the h rows come from the dump.
//
// what one conversation does, per circuit (one LLAMA_CONTEXT_TYPE_MTP context each, built like draft-mtp's ctx_dft):
//   1. catch-up prefill of MTP rows p = 0..n_rows-1 on seq 0, exactly like draft-mtp process(): token t_p, h_{p-1}
//      (zeros at p = 0), pos p. Rows that are chain starts are outputs: row s is the seed row of a draft step and
//      its logits ARE depth 1 (predicts t_{s+1}), its hm (MTP h_nextn) seeds depth 2. This is the validate mode of
//      mtp-dump and eval_long's step 1.
//   2. per start s: seq j gets seq 0's cells [0, s] (llama_memory_seq_cp, unified KV, no copy), then depth k = 2..D
//      decodes one row per chain at pos s+k-1 with token t_{s+k-1} and h = hm of the chain's previous row, like the
//      growing-KV branch of draft() (speculative.cpp: batch.add(id, pos0 + i + 1), set_embd(h_row)). Up to --n-par
//      chains share one decode. Then seq j is dropped (llama_memory_seq_rm).
//   The chain is TEACHER-FORCED on the recorded tokens, like eval_long.py: depth k is the head's accuracy GIVEN
//   that depths 1..k-1 were accepted (in production a rejected draft ends the step, so later depths never run).
//   'alive' (all of 1..k top-1 right) and 'cum' (product of acc_prob 1..k) are the unconditional views.
//
// metrics per (circuit, bucket, depth); the bucket is the one of the PREDICTED token t_{s+k} (eval_long's asst mask
// at r+1). q = the draft sampler's distribution: top-k 10 of the head's logits (full vocab, or the --vocab-subset
// rows), softmax at --temp. p = the target's distribution for the same position: softmax(output.weight @ h_r / temp)
// over the full vocab (optionally top-k --target-top-k renormalized), with h_r from the dump (output.weight run with
// ggml on the CPU from the mmapped gguf, not through libllama).
//   top1      argmax of the head == recorded token
//   top10     recorded token in the head's top 10
//   p_rec     q(recorded token), 0 when outside the top 10
//   acc_prob  sum over the head's top 10 of min(q(x), p(x)) = expected acceptance of the probabilistic verifier
//             (speculative sampling: accept x ~ q with prob min(1, p/q)). eval_long.py's acc_prob, same definition.
//   tgt_top1  argmax of p == recorded token (how predictable the position is for the target itself)
//   alive     top1 at every depth 1..k of this chain (greedy accepted length = sum over k)
//   cum       product of acc_prob over depths 1..k (probabilistic accepted length proxy = sum over k)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

// [CIRCUIT] the circuit engine (src/llama-mtp-circuit.*, src/llama-ext.h, other owner of this worktree, 2026-11-07):
//   int32_t llama_mtp_set_circuit(llama_context * ctx_dft, const char * circuit);   + get / last_error / max_passes
//   llama_context_params.mtp_circuit_max_passes = own-K/V passes the draft KV is allocated for (one K/V layer each)
// When the engine is in the tree (its header exists) the real API is used and the draft context is sized for the
// largest own-pass count among the --circuit list. Without it (a libllama from before 2026-11-07) the function is
// declared weak, resolves to null, and only the implicit default circuit "M0" (the shipped single-pass head) runs.
// MTP_REPLAY_CIRCUIT_ENGINE comes from CMakeLists.txt (src/llama-mtp-circuit.h exists in the configured tree): a
// __has_include here would resolve against the real path of this file, not the tree being built
#ifndef MTP_REPLAY_CIRCUIT_ENGINE
#define MTP_REPLAY_CIRCUIT_ENGINE 0
#endif
#if !MTP_REPLAY_CIRCUIT_ENGINE
int32_t llama_mtp_set_circuit(llama_context * ctx, const char * circuit) __attribute__((weak));
#endif

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------------------
// target head: logits = output.weight @ h, straight from the mmapped gguf with a CPU ggml graph. The dumped h is
// already after output_norm (HEAD_CONTRACT s1), so this is exactly the target's result_output for that position.
struct target_head {
    void *         map      = nullptr;
    size_t         map_size = 0;
    ggml_context * ctx_w    = nullptr;
    ggml_tensor *  W        = nullptr;
    int64_t        n_embd   = 0;
    int64_t        n_vocab  = 0;

    bool init(const std::string & path) {
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &ctx_w };
        gguf_context * g = gguf_init_from_file(path.c_str(), gp);
        if (!g) {
            LOG_ERR("target head: cannot read gguf %s\n", path.c_str());
            return false;
        }
        const int64_t id = gguf_find_tensor(g, "output.weight");
        if (id < 0) {
            LOG_ERR("target head: no output.weight in %s (tied embeddings are not handled)\n", path.c_str());
            gguf_free(g);
            return false;
        }
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        W = ggml_get_tensor(ctx_w, "output.weight");
        n_embd  = W->ne[0];
        n_vocab = W->ne[1];
        gguf_free(g);

        const int fd = open(path.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            LOG_ERR("target head: cannot open %s\n", path.c_str());
            return false;
        }
        map_size = (size_t) st.st_size;
        map = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (map == MAP_FAILED) {
            map = nullptr;
            LOG_ERR("target head: mmap failed\n");
            return false;
        }
        W->data = (char *) map + off;
        LOG_INF("target head: output.weight %s [%" PRId64 " x %" PRId64 "] mmapped from %s\n",
                ggml_type_name(W->type), n_embd, n_vocab, path.c_str());
        return true;
    }

    ~target_head() {
        if (map) {
            munmap(map, map_size);
        }
        if (ctx_w) {
            ggml_free(ctx_w);
        }
    }

    // out[i * n_vocab ..] = logits of h row i, rows in chunks of 64 (one ggml graph per chunk)
    void logits(const float * h, int64_t n_rows, float * out, int n_threads) const {
        const int64_t CH = 64;
        for (int64_t r0 = 0; r0 < n_rows; r0 += CH) {
            const int64_t nr = std::min(CH, n_rows - r0);
            const size_t mem = (size_t) nr * (n_embd + n_vocab) * sizeof(float) + (size_t) nr * n_embd * 2 +
                               ggml_graph_overhead() + 16 * ggml_tensor_overhead() + (64u << 20);
            ggml_init_params ip = { mem, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nr);
            memcpy(x->data, h + r0 * n_embd, (size_t) nr * n_embd * sizeof(float));
            ggml_tensor * y = ggml_mul_mat(ctx, W, x);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            memcpy(out + r0 * n_vocab, y->data, (size_t) nr * n_vocab * sizeof(float));
            ggml_free(ctx);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// conversation + buckets

struct conversation {
    std::string              name;
    std::string              base;
    std::vector<llama_token> tok;
    const ggml_fp16_t *      h = nullptr;  // [n_h, n_embd] mmapped
    void *                   h_map = nullptr;
    size_t                   h_size = 0;
    int64_t                  n_h = 0;
    int64_t                  n_samples = 0; // eval_long's c.n_samples: MTP rows 0..n_samples, targets t_1..t_{n_samples+1}
    std::vector<std::vector<int32_t>> bucket_of; // [dim][pos] global bucket id, -1 = none

    ~conversation() {
        if (h_map) {
            munmap(h_map, h_size);
        }
    }
};

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

struct bucket_registry {
    std::vector<std::string>         dims;    // requested dims, order = conversation.bucket_of index
    std::vector<std::string>         names;   // global bucket id -> "dim:label"; id 0 = "all"
    std::map<std::string, int32_t>   ids;

    int32_t id(const std::string & n) {
        auto it = ids.find(n);
        if (it != ids.end()) {
            return it->second;
        }
        ids[n] = (int32_t) names.size();
        names.push_back(n);
        return (int32_t) names.size() - 1;
    }
};

static bool load_buckets(conversation & c, const std::string & path, bucket_registry & reg) {
    c.bucket_of.assign(reg.dims.size(), std::vector<int32_t>(c.tok.size(), -1));
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded() || !j.contains("dims")) {
        LOG_WRN("%s: not a buckets sidecar\n", path.c_str());
        return false;
    }
    const int64_t n_side = j.value("n_tokens", (int64_t) 0);
    if (n_side < (int64_t) c.tok.size()) {
        // the dump .tok of a --truncate'd conversation is a prefix of the archived one, so a longer sidecar is fine
        LOG_WRN("%s: sidecar covers %" PRId64 " tokens < %zu, tags past it are 'none'\n", path.c_str(), n_side, c.tok.size());
    }
    for (size_t d = 0; d < reg.dims.size(); ++d) {
        const auto & dj = j["dims"];
        if (!dj.contains(reg.dims[d])) {
            LOG_WRN("%s: no dim '%s'\n", path.c_str(), reg.dims[d].c_str());
            continue;
        }
        const auto & labels = dj[reg.dims[d]]["labels"];
        std::vector<int32_t> gid(labels.size());
        for (size_t l = 0; l < labels.size(); ++l) {
            const std::string lab = labels[l].get<std::string>();
            // '-' = not applicable (e.g. asst dim on non-assistant positions): not a bucket
            gid[l] = lab == "-" ? -1 : reg.id(reg.dims[d] + ":" + lab);
        }
        for (const auto & r : dj[reg.dims[d]]["runs"]) {
            const int64_t a = r[0].get<int64_t>(), b = std::min<int64_t>(r[1].get<int64_t>(), (int64_t) c.tok.size());
            const int32_t g = gid[r[2].get<size_t>()];
            for (int64_t p = a; p < b; ++p) {
                c.bucket_of[d][p] = g;
            }
        }
    }
    return true;
}

static bool load_conversation(const std::string & arg, int64_t n_embd, int64_t max_rows, conversation & c) {
    std::string base = arg;
    for (const char * ext : { ".tok", ".h.f16", ".meta.json" }) {
        const size_t le = strlen(ext);
        if (base.size() > le && base.compare(base.size() - le, le, ext) == 0) {
            base = base.substr(0, base.size() - le);
        }
    }
    c.base = base;
    c.name = fs::path(base).filename().string();
    if (!read_tokens(base + ".tok", c.tok)) {
        LOG_ERR("%s: cannot read .tok\n", base.c_str());
        return false;
    }
    const int fd = open((base + ".h.f16").c_str(), O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        LOG_ERR("%s: cannot open .h.f16\n", base.c_str());
        return false;
    }
    c.h_size = (size_t) st.st_size;
    if (c.h_size % (n_embd * sizeof(ggml_fp16_t)) != 0) {
        LOG_ERR("%s.h.f16: size %zu is not a multiple of n_embd %" PRId64 " f16\n", base.c_str(), c.h_size, n_embd);
        close(fd);
        return false;
    }
    c.h_map = mmap(nullptr, c.h_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (c.h_map == MAP_FAILED) {
        c.h_map = nullptr;
        return false;
    }
    c.h   = (const ggml_fp16_t *) c.h_map;
    c.n_h = (int64_t) (c.h_size / (n_embd * sizeof(ggml_fp16_t)));
    // same as data.py Conv: n_samples = min(n_h, len(tok) - 2), eval_long --max-rows truncates it
    c.n_samples = std::max<int64_t>(0, std::min<int64_t>(c.n_h, (int64_t) c.tok.size() - 2));
    if (max_rows > 0) {
        c.n_samples = std::min(c.n_samples, max_rows);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// scoring

struct cfg_t {
    int   depth         = 8;
    int   stride        = 8;
    int   n_par         = 64;
    int   n_batch       = 512;
    int   top_k         = 10;
    float temp          = 1.0f;  // acc_prob: eval_long.py's definition (draft and target at temp 1, full-vocab target)
    int   target_top_k  = 0;
    // acc2: the production-like acceptance (2026-11-07 server: --temp 0.2 --top-k 20 --min-p 0, top_p default 0.95,
    // --spec-draft-sampling probabilistic): q2 = softmax(head top-10 / acc2_temp) (spec_retune: draft top-k 10 + the
    // target's temp), p2 = target top-k acc2_top_k, then top-p acc2_top_p on the temp-1 probabilities of those
    // (common sampler order: top_k, top_p, min_p, temp), then softmax at acc2_temp. acc2 = sum min(q2, p2).
    // An approximation of the verifier's p (penalties / dry / xtc are off in production, so nothing else applies).
    bool  acc2          = true;
    float acc2_temp     = 0.2f;
    int   acc2_top_k    = 20;
    float acc2_top_p    = 0.95f;
    bool  target        = true;
    int   n_threads     = 8;
    int64_t max_rows    = 0;
    int64_t max_starts  = 0;
    std::string filter_dim, filter_label;
};

// target distribution of the MTP rows the current chains score (row r = h_r, predicts t_{r+1}); only the rows in
// the union of [s, s + depth - 1] over the starts are computed (a bucket filter leaves gaps between starts)
struct target_rows {
    int64_t r0 = 0;
    std::vector<int32_t> slot; // [r - r0] -> row index, -1 = not computed
    int64_t n = 0;
    std::vector<float>   logits;  // [n, n_vocab] raw target logits (needed for the full-vocab acc_prob lookup)
    std::vector<float>   lse;     // log-normalizer at cfg.temp
    std::vector<int32_t> argmax;
    std::vector<std::vector<std::pair<int32_t, float>>> topk;  // cfg.target_top_k > 0: renormalized top-k at temp
    std::vector<std::vector<std::pair<int32_t, float>>> cand2; // acc2 target distribution
};

struct row_result {
    bool  top1, top10, tgt_top1;
    float p_rec, acc, acc2;
};

static void parallel_for(int64_t n, int n_threads, const std::function<void(int64_t)> & fn) {
    if (n <= 0) {
        return;
    }
    const int nt = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (int64_t i; (i = next.fetch_add(1)) < n; ) {
                fn(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

// draft row lg (n_vocab logits, -INF outside a vocab subset) vs recorded token rec at MTP row r
static row_result score_row(const float * lg, int64_t n_vocab, llama_token rec, int64_t r, const target_rows * tr, const cfg_t & cfg) {
    const int K = cfg.top_k;
    int32_t ids[64];
    float   val[64];
    int     nk = 0;
    for (int64_t v = 0; v < n_vocab; ++v) {
        const float x = lg[v];
        if (!(x > -INFINITY)) {
            continue;
        }
        if (nk == K && x <= val[K - 1]) {
            continue; // ties keep the lower id, like validate's partial_sort
        }
        int p = nk < K ? nk++ : K - 1;
        while (p > 0 && val[p - 1] < x) {
            val[p] = val[p - 1];
            ids[p] = ids[p - 1];
            --p;
        }
        val[p] = x;
        ids[p] = (int32_t) v;
    }
    row_result res{};
    if (nk == 0) {
        return res;
    }
    auto softmax_top = [&](float t, float * q) {
        float sum = 0.0f;
        for (int i = 0; i < nk; ++i) {
            q[i] = expf((val[i] - val[0]) / t);
            sum += q[i];
        }
        for (int i = 0; i < nk; ++i) {
            q[i] /= sum;
        }
    };
    // the draft sampler's q: top-k then softmax at temp (eval_long: top-10, temp 1)
    float q[64];
    softmax_top(cfg.temp, q);
    res.top1 = ids[0] == rec;
    for (int i = 0; i < nk; ++i) {
        if (ids[i] == rec) {
            res.top10 = true;
            res.p_rec = q[i];
        }
    }
    if (tr) {
        const int64_t i = tr->slot[r - tr->r0];
        GGML_ASSERT(i >= 0 && i < tr->n);
        res.tgt_top1 = tr->argmax[i] == rec;
        auto acc_list = [&](const float * qq, const std::vector<std::pair<int32_t, float>> & pl) {
            double acc = 0.0;
            for (int k = 0; k < nk; ++k) {
                for (const auto & pr : pl) {
                    if (pr.first == ids[k]) {
                        acc += std::min<double>(qq[k], pr.second);
                    }
                }
            }
            return (float) acc;
        };
        if (cfg.target_top_k > 0) {
            res.acc = acc_list(q, tr->topk[i]);
        } else {
            const float * tl = tr->logits.data() + i * n_vocab;
            double acc = 0.0;
            for (int k = 0; k < nk; ++k) {
                const double p = exp((double) tl[ids[k]] / cfg.temp - tr->lse[i]);
                acc += std::min<double>(q[k], p);
            }
            res.acc = (float) acc;
        }
        if (cfg.acc2) {
            float q2[64];
            softmax_top(cfg.acc2_temp, q2);
            res.acc2 = acc_list(q2, tr->cand2[i]);
        }
    }
    return res;
}

// top-k (by logit, ties to the lower id) of one row, renormalized at temp, optionally cut by top-p first
static std::vector<std::pair<int32_t, float>> target_cands(const float * l, int64_t nv, int K, float top_p, float temp) {
    std::vector<int32_t> ids(K);
    std::vector<float>   val(K, -INFINITY);
    int nk = 0;
    for (int64_t v = 0; v < nv; ++v) {
        const float x = l[v];
        if (nk == K && x <= val[K - 1]) {
            continue;
        }
        int p = nk < K ? nk++ : K - 1;
        while (p > 0 && val[p - 1] < x) {
            val[p] = val[p - 1];
            ids[p] = ids[p - 1];
            --p;
        }
        val[p] = x;
        ids[p] = (int32_t) v;
    }
    int keep = nk;
    if (top_p < 1.0f) {
        // llama_sampler_top_p: cumulative temp-1 probabilities over the kept candidates, keep the minimal prefix >= p
        double z = 0.0;
        for (int k = 0; k < nk; ++k) {
            z += exp((double) (val[k] - val[0]));
        }
        double cum = 0.0;
        for (int k = 0; k < nk; ++k) {
            cum += exp((double) (val[k] - val[0])) / z;
            if (cum >= top_p) {
                keep = k + 1;
                break;
            }
        }
    }
    double z = 0.0;
    for (int k = 0; k < keep; ++k) {
        z += exp((double) (val[k] - val[0]) / temp);
    }
    std::vector<std::pair<int32_t, float>> out;
    for (int k = 0; k < keep; ++k) {
        out.push_back({ ids[k], (float) (exp((double) (val[k] - val[0]) / temp) / z) });
    }
    return out;
}

static void build_target_rows(const target_head & th, const conversation & c, const std::vector<int64_t> & rows, const cfg_t & cfg, target_rows & tr) {
    tr.r0 = rows.front();
    tr.slot.assign(rows.back() - rows.front() + 1, -1);
    tr.n  = (int64_t) rows.size();
    for (int64_t i = 0; i < tr.n; ++i) {
        tr.slot[rows[i] - tr.r0] = (int32_t) i;
    }
    const int64_t nv = th.n_vocab, ne = th.n_embd;
    std::vector<float> h((size_t) tr.n * ne);
    for (int64_t i = 0; i < tr.n; ++i) {
        ggml_fp16_to_fp32_row(c.h + rows[i] * ne, h.data() + i * ne, ne);
    }
    tr.logits.resize((size_t) tr.n * nv);
    th.logits(h.data(), tr.n, tr.logits.data(), cfg.n_threads);
    tr.lse.assign(tr.n, 0.0f);
    tr.argmax.assign(tr.n, 0);
    tr.topk.assign(cfg.target_top_k > 0 ? tr.n : 0, {});
    tr.cand2.assign(cfg.acc2 ? tr.n : 0, {});
    parallel_for(tr.n, cfg.n_threads, [&](int64_t i) {
        const float * l = tr.logits.data() + i * nv;
        int32_t am = 0;
        for (int64_t v = 1; v < nv; ++v) {
            if (l[v] > l[am]) {
                am = (int32_t) v;
            }
        }
        double s = 0.0;
        for (int64_t v = 0; v < nv; ++v) {
            s += exp((double) (l[v] - l[am]) / cfg.temp);
        }
        tr.argmax[i] = am;
        tr.lse[i] = (float) ((double) l[am] / cfg.temp + log(s));
        if (cfg.target_top_k > 0) {
            tr.topk[i] = target_cands(l, nv, cfg.target_top_k, 1.0f, cfg.temp);
        }
        if (cfg.acc2) {
            tr.cand2[i] = target_cands(l, nv, cfg.acc2_top_k, cfg.acc2_top_p, cfg.acc2_temp);
        }
    });
}

struct stat_t {
    double n = 0, top1 = 0, top10 = 0, p_rec = 0, acc = 0, acc2 = 0, tgt_top1 = 0, alive = 0, cum = 0, cum2 = 0;
};

// [circuit][bucket][depth]
using stats_t = std::vector<std::vector<std::vector<stat_t>>>;

// ---------------------------------------------------------------------------------------------------------------

struct replay_ctx {
    std::string     circuit;
    llama_context * ctx = nullptr;
};

static llama_context * make_ctx_dft(const common_params & params, llama_model * model, const cfg_t & cfg, int64_t n_rows, uint32_t n_own_passes) {
    auto cparams = common_context_params_to_llama(params);
#if MTP_REPLAY_CIRCUIT_ENGINE
    // every own-K/V pass of the circuit is one more K/V layer of n_ctx cells (the NextN block's size)
    cparams.mtp_circuit_max_passes = std::max<uint32_t>(1, n_own_passes);
#else
    GGML_ASSERT(n_own_passes <= 1);
#endif
    cparams.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq      = 0;
    cparams.ctx_other     = nullptr; // qwen35 MTP needs no target context (only gemma4 / eagle3 / dflash do)
    // the whole conversation + one live chain per parallel seq (depth - 1 own rows each), like ctx_dft at n_ctx_tgt
    cparams.n_ctx         = (uint32_t) GGML_PAD(n_rows + (int64_t) cfg.n_par * cfg.depth + 256, 256);
    cparams.n_batch       = (uint32_t) std::max(cfg.n_batch, cfg.n_par);
    cparams.n_ubatch      = cparams.n_batch;
    cparams.n_seq_max     = (uint32_t) cfg.n_par + 1;
    cparams.kv_unified    = true;  // seq_cp shares the prefix cells instead of copying them
    cparams.n_outputs_max = 0;     // = n_batch: every prefill row can be a chain start (stride 1)
    cparams.embeddings    = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    }
    return ctx;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    cfg_t cfg;
    std::vector<std::string> circuits;
    std::vector<std::string> inputs;
    std::string out_path;
    std::string buckets_dir;
    std::string vocab_subset;
    bool vocab_subset_forced = false;
    std::string bucket_dims = "role,think,kind,lang,script,asst";
    bool repack = false;
    double min_frac_print = 0.0;
    int64_t min_n_print = 200;

    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                LOG_ERR("%s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--circuit")         { circuits.push_back(next()); }
        else if (a == "--depth")           { cfg.depth = std::stoi(next()); }
        else if (a == "--stride")          { cfg.stride = std::stoi(next()); }
        else if (a == "--n-par")           { cfg.n_par = std::stoi(next()); }
        else if (a == "--prefill-batch")   { cfg.n_batch = std::stoi(next()); }
        else if (a == "--temp")            { cfg.temp = std::stof(next()); }
        else if (a == "--draft-top-k")     { cfg.top_k = std::stoi(next()); }
        else if (a == "--target-top-k")    { cfg.target_top_k = std::stoi(next()); }
        else if (a == "--acc2")            {
            // "temp,top_k,top_p" of the production-like acceptance, "off" disables it
            const std::string v = next();
            if (v == "off") { cfg.acc2 = false; }
            else if (sscanf(v.c_str(), "%f,%d,%f", &cfg.acc2_temp, &cfg.acc2_top_k, &cfg.acc2_top_p) != 3) { LOG_ERR("--acc2 wants temp,top_k,top_p\n"); return 1; }
        }
        else if (a == "--no-target")       { cfg.target = false; }
        else if (a == "--max-rows")        { cfg.max_rows = std::stoll(next()); }
        else if (a == "--max-starts")      { cfg.max_starts = std::stoll(next()); }
        else if (a == "--bucket-filter")   {
            const std::string f = next();
            const size_t p = f.find(':');
            if (p == std::string::npos) { LOG_ERR("--bucket-filter wants dim:label\n"); return 1; }
            cfg.filter_dim = f.substr(0, p); cfg.filter_label = f.substr(p + 1);
        }
        else if (a == "--bucket-dims")     { bucket_dims = next(); }
        else if (a == "--buckets-dir")     { buckets_dir = next(); }
        else if (a == "--vocab-subset")    { vocab_subset = next(); }
        else if (a == "--vocab-subset-forced") { vocab_subset_forced = true; }
        else if (a == "--out")             { out_path = next(); }
        else if (a == "--min-n")           { min_n_print = std::stoll(next()); }
        else if (a == "--repack")          { repack = true; }
        else if (a.size() > 0 && a[0] != '-' && (fs::exists(a + ".tok") || (a.size() > 4 && a.compare(a.size() - 4, 4, ".tok") == 0))) {
            inputs.push_back(a);
        } else {
            rest.push_back(argv[i]);
        }
    }
    (void) min_frac_print;
    if (circuits.empty()) {
        circuits.push_back("M0");
    }
    cfg.top_k = std::min(cfg.top_k, 64);

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (inputs.empty()) {
        LOG_ERR("usage: %s -m model.gguf [--circuit C ...] [--depth 8] [--stride 8] [--n-par 64] [--bucket-filter dim:label]\n"
                "         [--vocab-subset FILE [--vocab-subset-forced]] [--temp 1] [--target-top-k 0] [--no-target]\n"
                "         [--max-rows N] [--max-starts N] [--buckets-dir DIR] [--out res.json] DUMP/NAME ...\n", argv[0]);
        return 1;
    }
    cfg.n_threads = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : (int) std::thread::hardware_concurrency();

    // the circuit API is optional: without it only the implicit default circuit can run
#if MTP_REPLAY_CIRCUIT_ENGINE
    const bool have_circuit_api = true;
#else
    const bool have_circuit_api = llama_mtp_set_circuit != nullptr; // weak: null without the engine
#endif
    // own-K/V passes per circuit (sizes the draft KV); without the engine every circuit must be the default M0
    std::vector<uint32_t> n_own(circuits.size(), 1);
    for (size_t i = 0; i < circuits.size(); ++i) {
#if MTP_REPLAY_CIRCUIT_ENGINE
        std::vector<llama_mtp_pass> passes;
        std::string canon, err;
        if (!llama_mtp_circuit_parse(circuits[i].c_str(), passes, canon, err)) {
            LOG_ERR("circuit '%s': %s\n", circuits[i].c_str(), err.c_str());
            return 1;
        }
        n_own[i] = 0;
        for (const auto & ps : passes) {
            n_own[i] += ps.borrow < 0 ? 1 : 0;
        }
        LOG_INF("circuit '%s' = %s: %zu passes, %u own K/V\n", circuits[i].c_str(), canon.c_str(), passes.size(), n_own[i]);
#else
        if (circuits[i] != "M0") {
            LOG_ERR("circuit '%s' needs the circuit engine (llama_mtp_set_circuit), this build has none: only M0\n", circuits[i].c_str());
            return 1;
        }
#endif
    }

    params.load_mtp       = true;  // blk.64 is skipped at load unless asked for
    params.fit_params     = false;
    params.warmup         = false;
    params.no_extra_bufts = !repack; // a CPU-repacked output.weight cannot take llama_set_logits_subset
    llama_backend_init();
    llama_numa_init(params.numa);

    // [MEM] the loader maps the whole gguf (all 64 trunk layers too): on CPU the trunk pages are mapped but never
    // touched, the resident set is output.weight + the used token_embd rows + blk.64 + the MTP KV
    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    if (!params.mtp_head.empty() && llama_model_load_mtp_head(model, params.mtp_head.c_str()) < 0) {
        LOG_ERR("failed to apply --mtp-head %s\n", params.mtp_head.c_str());
        return 1;
    }
    if (llama_model_n_layer_nextn(model) <= 0) {
        LOG_ERR("model has no NextN/MTP block\n");
        return 1;
    }
    const int64_t n_embd  = llama_model_n_embd_out(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);

    target_head th;
    if (cfg.target) {
        if (!th.init(params.model.path) || th.n_embd != n_embd || th.n_vocab != n_vocab) {
            LOG_ERR("target head unusable, run with --no-target to skip acc_prob\n");
            return 1;
        }
    }

    // vocab subset: the file's ids (eval_long's Ws), optionally + the forced set of draft-mtp vs_init (control /
    // user-defined / EOG / digit-only / newline-only pieces). production also unions each request's prompt ids,
    // which this tool does not (it would leak future ids from the conversation's later turns).
    std::vector<llama_token> vs_ids;
    if (!vocab_subset.empty()) {
        std::vector<uint8_t> in(n_vocab, 0);
        std::ifstream f(vocab_subset);
        if (!f) {
            LOG_ERR("cannot open --vocab-subset %s\n", vocab_subset.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            const size_t p = line.find_first_not_of(" \t");
            if (p == std::string::npos || line[p] == '#') {
                continue;
            }
            const long v = std::strtol(line.c_str() + p, nullptr, 10);
            if (v >= 0 && v < n_vocab && !in[v]) {
                in[v] = 1;
                vs_ids.push_back((llama_token) v);
            }
        }
        if (vocab_subset_forced) {
            for (llama_token t = 0; t < n_vocab; ++t) {
                if (in[t]) {
                    continue;
                }
                bool force = llama_vocab_is_eog(vocab, t) ||
                    (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0;
                if (!force) {
                    const std::string piece = common_token_to_piece(vocab, t, true);
                    const bool digits = !piece.empty() && std::all_of(piece.begin(), piece.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
                    const bool nl     = !piece.empty() && piece.find('\n') != std::string::npos &&
                        std::all_of(piece.begin(), piece.end(), [](char ch) { return ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t'; });
                    force = digits || nl;
                }
                if (force) {
                    in[t] = 1;
                    vs_ids.push_back(t);
                }
            }
        }
        LOG_INF("vocab subset: %zu ids from %s%s\n", vs_ids.size(), vocab_subset.c_str(), vocab_subset_forced ? " + forced" : "");
    }

    bucket_registry reg;
    reg.id("all");
    {
        size_t p0 = 0;
        while (p0 <= bucket_dims.size()) {
            const size_t p1 = bucket_dims.find(',', p0);
            const std::string d = bucket_dims.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
            if (!d.empty()) {
                reg.dims.push_back(d);
            }
            if (p1 == std::string::npos) {
                break;
            }
            p0 = p1 + 1;
        }
        if (!cfg.filter_dim.empty() && std::find(reg.dims.begin(), reg.dims.end(), cfg.filter_dim) == reg.dims.end()) {
            reg.dims.push_back(cfg.filter_dim);
        }
    }

    const int D = cfg.depth;
    stats_t stats(circuits.size());
    auto stat_at = [&](size_t ci, int32_t b, int k) -> stat_t & {
        auto & sc = stats[ci];
        if ((int32_t) sc.size() <= b) {
            sc.resize(b + 1, std::vector<stat_t>(D));
        }
        return sc[b][k];
    };

    double t_prefill = 0, t_chain = 0, t_target = 0, t_score = 0;
    int64_t n_starts_total = 0, n_rows_scored = 0, n_rows_prefill = 0;
    const double t_all0 = now_s();
    json conv_meta = json::array();

    for (const auto & in : inputs) {
        conversation c;
        if (!load_conversation(in, n_embd, cfg.max_rows, c)) {
            return 1;
        }
        if (params.n_ctx > 0) {
            // -c caps the replayed prefix (the KV also holds n_par chains of depth rows)
            c.n_samples = std::min<int64_t>(c.n_samples, std::max<int64_t>(1, params.n_ctx - (int64_t) cfg.n_par * D - 2));
        }
        {
            std::string bp;
            const std::vector<std::string> cands = {
                buckets_dir.empty() ? "" : buckets_dir + "/" + c.name + ".buckets.json",
                (fs::path(c.base).parent_path().parent_path() / "buckets" / (c.name + ".buckets.json")).string(),
                c.base + ".buckets.json",
            };
            for (const auto & p : cands) {
                if (!p.empty() && fs::exists(p)) {
                    bp = p;
                    break;
                }
            }
            if (bp.empty() || !load_buckets(c, bp, reg)) {
                LOG_WRN("%s: no bucket sidecar (run tag_buckets.py), reporting 'all' only\n", c.name.c_str());
                c.bucket_of.assign(reg.dims.size(), std::vector<int32_t>(c.tok.size(), -1));
            }
        }
        const int64_t n_rows = c.n_samples + 1; // MTP rows 0..n_samples
        if (c.n_samples < 1) {
            LOG_WRN("%s: too short, skipped\n", c.name.c_str());
            continue;
        }

        // chain starts: s in [1, n_samples], every stride-th, optionally only where t_{s+1} is in the filter bucket
        int32_t filter_gid = -2;
        size_t  filter_dim = 0;
        if (!cfg.filter_dim.empty()) {
            filter_dim = std::find(reg.dims.begin(), reg.dims.end(), cfg.filter_dim) - reg.dims.begin();
            filter_gid = reg.id(cfg.filter_dim + ":" + cfg.filter_label);
        }
        std::vector<uint8_t> is_start(n_rows, 0);
        int64_t n_starts = 0;
        for (int64_t s = 1; s <= c.n_samples; ++s) {
            if (filter_gid != -2 && c.bucket_of[filter_dim][s + 1] != filter_gid) {
                continue;
            }
            is_start[s] = 1;
        }
        // stride over the eligible starts (so a bucket filter keeps every stride-th position of that bucket)
        {
            int64_t k = 0;
            for (int64_t s = 1; s <= c.n_samples; ++s) {
                if (is_start[s]) {
                    is_start[s] = (k++ % cfg.stride) == 0;
                    if (is_start[s] && cfg.max_starts > 0 && n_starts_total + n_starts >= cfg.max_starts) {
                        is_start[s] = 0;
                    }
                    n_starts += is_start[s];
                }
            }
        }
        LOG_INF("conv %s: %" PRId64 " tokens, %" PRId64 " MTP rows, %" PRId64 " chain starts, depth %d, %zu circuit(s)\n",
                c.name.c_str(), (int64_t) c.tok.size(), n_rows, n_starts, D, circuits.size());
        if (n_starts == 0) {
            continue;
        }
        // rows past the last start never need to be decoded
        int64_t last_start = 0;
        for (int64_t s = n_rows - 1; s >= 1; --s) {
            if (is_start[s]) {
                last_start = s;
                break;
            }
        }
        const int64_t n_rows_run = last_start + 1;

        std::vector<replay_ctx> rcs;
        for (const auto & circ : circuits) {
            replay_ctx rc;
            rc.circuit = circ;
            rc.ctx = make_ctx_dft(params, model, cfg, n_rows_run, n_own[&circ - &circuits[0]]);
            if (!rc.ctx) {
                LOG_ERR("failed to create the MTP context\n");
                return 1;
            }
            if (have_circuit_api) {
                if (llama_mtp_set_circuit(rc.ctx, circ.c_str()) != 0) {
#if MTP_REPLAY_CIRCUIT_ENGINE
                    char eb[512];
                    llama_mtp_circuit_last_error(rc.ctx, eb, sizeof(eb));
                    LOG_ERR("llama_mtp_set_circuit('%s') failed: %s\n", circ.c_str(), eb);
#else
                    LOG_ERR("llama_mtp_set_circuit('%s') failed\n", circ.c_str());
#endif
                    return 1;
                }
            }
            if (!vs_ids.empty() && llama_set_logits_subset(rc.ctx, vs_ids.data(), (int32_t) vs_ids.size()) != 0) {
                LOG_ERR("llama_set_logits_subset failed (repacked CPU head? drop --repack)\n");
                return 1;
            }
            rcs.push_back(rc);
        }

        std::vector<float> h_in((size_t) cfg.n_batch * n_embd);
        std::vector<float> zero(n_embd, 0.0f);
        target_rows tr;

        for (int64_t a = 0; a < n_rows_run; a += cfg.n_batch) {
            const int64_t b = std::min<int64_t>(a + cfg.n_batch, n_rows_run);
            std::vector<int64_t> starts;
            for (int64_t s = a; s < b; ++s) {
                if (is_start[s]) {
                    starts.push_back(s);
                }
            }
            // h rows for MTP rows a..b-1: h_{p-1}, zeros at p = 0
            for (int64_t p = a; p < b; ++p) {
                float * dst = h_in.data() + (p - a) * n_embd;
                if (p == 0) {
                    std::fill(dst, dst + n_embd, 0.0f);
                } else {
                    ggml_fp16_to_fp32_row(c.h + (p - 1) * n_embd, dst, n_embd);
                }
            }
            // target logits for every row the chains of these starts score: [first start, last start + D - 1]
            if (cfg.target && !starts.empty()) {
                const double t0 = now_s();
                std::vector<int64_t> rows;
                for (const int64_t s0 : starts) {
                    for (int64_t r = std::max(s0, rows.empty() ? s0 : rows.back() + 1); r < std::min<int64_t>(s0 + D, c.n_samples + 1); ++r) {
                        rows.push_back(r);
                    }
                }
                build_target_rows(th, c, rows, cfg, tr);
                t_target += now_s() - t0;
            }

            for (size_t ci = 0; ci < rcs.size(); ++ci) {
                llama_context * ctx = rcs[ci].ctx;
                llama_memory_t mem = llama_get_memory(ctx);
                common_batch batch(ctx);

                // 1. catch-up prefill of rows a..b-1 on seq 0
                double t0 = now_s();
                std::vector<int32_t> idx_of(b - a, -1);
                for (int64_t p = a; p < b; ++p) {
                    const int32_t idx = batch.add(c.tok[p], (llama_pos) p, 0, is_start[p] != 0);
                    batch.set_embd(idx, { h_in.data() + (p - a) * n_embd, 1, (size_t) n_embd });
                    idx_of[p - a] = idx;
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    LOG_ERR("prefill decode failed at row %" PRId64 "\n", a);
                    return 1;
                }
                llama_synchronize(ctx);
                t_prefill += now_s() - t0;
                if (ci == 0) {
                    n_rows_prefill += b - a;
                }

                // per chain state; depth 1 = the seed rows of this prefill
                const size_t ns = starts.size();
                std::vector<float> hm(ns * n_embd);
                std::vector<uint8_t> alive(ns, 1);
                std::vector<double>  cum(ns, 1.0), cum2(ns, 1.0);
                auto score_depth = [&](int k, const std::vector<std::pair<size_t, int32_t>> & rows /* chain, out idx */) {
                    const double ts = now_s();
                    std::vector<row_result> rr(rows.size());
                    parallel_for((int64_t) rows.size(), cfg.n_threads, [&](int64_t i) {
                        const size_t j = rows[i].first;
                        const int64_t r = starts[j] + k - 1;
                        rr[i] = score_row(llama_get_logits_ith(ctx, rows[i].second), n_vocab, c.tok[r + 1], r, cfg.target ? &tr : nullptr, cfg);
                    });
                    for (size_t i = 0; i < rows.size(); ++i) {
                        const size_t j = rows[i].first;
                        const int64_t r = starts[j] + k - 1;
                        const float * ho = llama_get_embeddings_nextn_ith(ctx, rows[i].second);
                        memcpy(hm.data() + j * n_embd, ho, n_embd * sizeof(float));
                        alive[j] = alive[j] && rr[i].top1;
                        cum[j]  *= cfg.target ? rr[i].acc : 1.0;
                        cum2[j] *= cfg.target ? rr[i].acc2 : 1.0;
                        auto add = [&](int32_t bkt) {
                            stat_t & st = stat_at(ci, bkt, k - 1);
                            st.n += 1;
                            st.top1 += rr[i].top1;
                            st.top10 += rr[i].top10;
                            st.p_rec += rr[i].p_rec;
                            st.acc += rr[i].acc;
                            st.acc2 += rr[i].acc2;
                            st.cum2 += cum2[j];
                            st.tgt_top1 += rr[i].tgt_top1;
                            st.alive += alive[j];
                            st.cum += cum[j];
                        };
                        add(0);
                        for (size_t d = 0; d < reg.dims.size(); ++d) {
                            const int32_t bkt = c.bucket_of[d][r + 1];
                            if (bkt >= 0) {
                                add(bkt);
                            }
                        }
                    }
                    if (ci == 0) {
                        n_rows_scored += (int64_t) rows.size();
                    }
                    t_score += now_s() - ts;
                };
                {
                    std::vector<std::pair<size_t, int32_t>> rows;
                    for (size_t j = 0; j < ns; ++j) {
                        rows.push_back({ j, idx_of[starts[j] - a] });
                    }
                    score_depth(1, rows);
                }

                // 2. chains, n_par at a time, one decode per depth
                t0 = now_s();
                for (size_t g0 = 0; g0 < ns; g0 += cfg.n_par) {
                    const size_t g1 = std::min(ns, g0 + (size_t) cfg.n_par);
                    for (size_t j = g0; j < g1; ++j) {
                        llama_memory_seq_cp(mem, 0, (llama_seq_id) (1 + j - g0), 0, (llama_pos) starts[j] + 1);
                    }
                    for (int k = 2; k <= D; ++k) {
                        batch.clear();
                        std::vector<std::pair<size_t, int32_t>> rows;
                        for (size_t j = g0; j < g1; ++j) {
                            const int64_t r = starts[j] + k - 1; // MTP row of this depth, predicts t_{r+1}
                            if (r > c.n_samples) {
                                continue;
                            }
                            const int32_t idx = batch.add(c.tok[r], (llama_pos) r, (llama_seq_id) (1 + j - g0), true);
                            batch.set_embd(idx, { hm.data() + j * n_embd, 1, (size_t) n_embd });
                            rows.push_back({ j, idx });
                        }
                        if (rows.empty()) {
                            break;
                        }
                        const double td = now_s();
                        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                            LOG_ERR("chain decode failed (depth %d)\n", k);
                            return 1;
                        }
                        llama_synchronize(ctx);
                        t_chain += now_s() - td;
                        score_depth(k, rows); // overwrites hm[j] after the decode consumed it
                    }
                    for (size_t j = g0; j < g1; ++j) {
                        llama_memory_seq_rm(mem, (llama_seq_id) (1 + j - g0), -1, -1);
                    }
                }
                (void) t0;
            }
            if (!starts.empty()) {
                n_starts_total += (int64_t) starts.size();
            }
            LOG_INF("  rows %8" PRId64 " / %8" PRId64 "  starts %8" PRId64 "  %.1f s\n", b, n_rows_run, n_starts_total, now_s() - t_all0);
        }
        for (auto & rc : rcs) {
            llama_free(rc.ctx);
        }
        conv_meta.push_back({ { "name", c.name }, { "n_tokens", c.tok.size() }, { "n_samples", c.n_samples },
                              { "n_rows_run", n_rows_run }, { "n_starts", n_starts } });
    }
    const double t_all = now_s() - t_all0;

    // ---- report
    json out;
    out["model"] = params.model.path;
    out["mtp_head"] = params.mtp_head;
    out["inputs"] = conv_meta;
    out["config"] = { { "depth", D }, { "stride", cfg.stride }, { "n_par", cfg.n_par }, { "draft_top_k", cfg.top_k },
                      { "temp", cfg.temp }, { "target_top_k", cfg.target_top_k },
                      { "acc2", cfg.acc2 ? json{ { "temp", cfg.acc2_temp }, { "top_k", cfg.acc2_top_k }, { "top_p", cfg.acc2_top_p } } : json(nullptr) }, { "target", cfg.target },
                      { "max_rows", cfg.max_rows }, { "vocab_subset", vocab_subset }, { "vocab_subset_forced", vocab_subset_forced },
                      { "vocab_subset_n", vs_ids.size() }, { "bucket_filter", cfg.filter_dim.empty() ? "" : cfg.filter_dim + ":" + cfg.filter_label },
                      { "cache_type_k", ggml_type_name(params.cache_type_k) }, { "cache_type_v", ggml_type_name(params.cache_type_v) },
                      { "circuit_api", have_circuit_api }, { "circuit_engine_in_tree", (bool) MTP_REPLAY_CIRCUIT_ENGINE }, { "teacher_forced", true } };
    out["timing"] = { { "total_s", t_all }, { "prefill_s", t_prefill }, { "chain_decode_s", t_chain }, { "target_s", t_target },
                      { "score_s", t_score }, { "n_starts", n_starts_total }, { "n_rows_scored", n_rows_scored },
                      { "n_rows_prefill", n_rows_prefill }, { "threads", cfg.n_threads } };
    out["definitions"] = "per (circuit, bucket of the predicted token, depth k): chain teacher-forced on recorded tokens; "
        "top1 = head argmax == recorded; top10 = recorded in head top-k; q = softmax(top-k head logits / temp); "
        "p = softmax(output.weight @ h_target / temp) (full vocab or target_top_k); p_rec = q(recorded); "
        "acc_prob = sum_x in top-k min(q(x), p(x)); tgt_top1 = argmax p == recorded; alive = top1 at depths 1..k; "
        "cum = prod acc_prob 1..k; acc2/cum2 = the same with the production-like sampler settings of config.acc2 "
        "(q2 = softmax(head top-k / temp2), p2 = target top_k2 -> top_p2 (temp 1) -> softmax / temp2); all metrics are means over n";
    json res = json::object();
    for (size_t ci = 0; ci < circuits.size(); ++ci) {
        json jc = json::object();
        for (size_t bk = 0; bk < stats[ci].size(); ++bk) {
            json jd = json::array();
            for (int k = 0; k < D; ++k) {
                const stat_t & s = stats[ci][bk][k];
                const double n = std::max(1.0, s.n);
                jd.push_back({ { "k", k + 1 }, { "n", (int64_t) s.n }, { "top1", s.top1 / n }, { "top10", s.top10 / n },
                               { "p_rec", s.p_rec / n }, { "acc_prob", cfg.target ? s.acc / n : NAN }, { "acc2", cfg.target && cfg.acc2 ? s.acc2 / n : NAN },
                               { "tgt_top1", cfg.target ? s.tgt_top1 / n : NAN }, { "alive", s.alive / n },
                               { "cum", cfg.target ? s.cum / n : NAN }, { "cum2", cfg.target && cfg.acc2 ? s.cum2 / n : NAN } });
            }
            if (stats[ci][bk][0].n > 0) {
                jc[reg.names[bk]] = jd;
            }
        }
        res[circuits[ci]] = jc;
    }
    out["results"] = res;
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << out.dump(1) << "\n";
        LOG_INF("wrote %s\n", out_path.c_str());
    }

    // table: rows = circuit x bucket (n at depth 1 >= --min-n), cols = depth: top1 / acc_prob, then n and sums
    printf("\n== mtp-replay: %zu conv, %" PRId64 " starts, depth %d, stride %d, %s vocab%s, temp %.2f, %.1f s "
           "(prefill %.1f, chain decode %.1f, target %.1f, score %.1f)\n",
           conv_meta.size(), n_starts_total, D, cfg.stride, vs_ids.empty() ? "full" : "subset", vs_ids.empty() ? "" : (" " + std::to_string(vs_ids.size())).c_str(),
           cfg.temp, t_all, t_prefill, t_chain, t_target, t_score);
    printf("   cell = top1/acc_prob per depth; sumA = sum of acc_prob over depths; Eg = sum alive (greedy accepted length); "
           "Ep = sum cum (probabilistic proxy); sumA2/Ep2 = the same at the acc2 sampler settings (production-like)\n");
    printf("%-8s %-24s %8s", "circuit", "bucket", "n(k1)");
    for (int k = 1; k <= D; ++k) {
        printf("   k%-2d t1  acc ", k);
    }
    printf("   sumA    Eg    Ep  sumA2   Ep2\n");
    for (size_t ci = 0; ci < circuits.size(); ++ci) {
        for (size_t bk = 0; bk < stats[ci].size(); ++bk) {
            const auto & sd = stats[ci][bk];
            if (sd[0].n < min_n_print && bk != 0) {
                continue;
            }
            printf("%-8s %-24s %8" PRId64, circuits[ci].c_str(), reg.names[bk].c_str(), (int64_t) sd[0].n);
            double sa = 0, eg = 0, ep = 0, sa2 = 0, ep2 = 0;
            for (int k = 0; k < D; ++k) {
                const double n = std::max(1.0, sd[k].n);
                printf("  %.3f %.3f", sd[k].top1 / n, cfg.target ? sd[k].acc / n : NAN);
                sa += sd[k].acc / n;
                eg += sd[k].alive / n;
                ep += sd[k].cum / n;
                sa2 += sd[k].acc2 / n;
                ep2 += sd[k].cum2 / n;
            }
            const bool t2 = cfg.target && cfg.acc2;
            printf("  %5.3f %5.3f %5.3f  %5.3f %5.3f\n", cfg.target ? sa : NAN, eg, cfg.target ? ep : NAN, t2 ? sa2 : NAN, t2 ? ep2 : NAN);
        }
    }
    fflush(stdout);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
