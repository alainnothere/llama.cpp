#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

// TEMP DIAGNOSTIC, env: LLAMA_GRAPH_REUSE_LOG=1 logs every failing can_reuse sub-check
static bool graph_reuse_log_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_GRAPH_REUSE_LOG");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// [KQ_MASK_INC] static part of the gate, see llama_kv_cache::kq_mask_inc_state
static bool kq_mask_dev_decide(
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const ggml_tensor * kq_mask) {
    return !cparams.training && kq_mask && mctx->kq_mask_dev_usable(ubatch, kq_mask->ne[3], kq_mask->type, cparams.causal_attn);
}

// [KQ_MASK_DEV] per-graph part of the gate, see llama_kv_cache::kq_build_state. kq_mask is the graph input
// (created either way: its shape keeps the graph-reuse check exact and models that read get_kq_mask() still work)
static bool kq_build_decide(
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const ggml_tensor * kq_mask) {
    return !cparams.training && cparams.flash_attn && kq_mask && kq_mask->type == GGML_TYPE_F16 && kq_mask->ne[3] == 1 &&
           mctx->kq_build_enabled() && (!ubatch.is_pos_2d() || !mctx->kq_build_ext_2d());
}

// cells cap. Every n_kv-sized graph node is a re-plan + drain of all backends when it outgrows the gallocr plan (1b of
// the design: 22 re-plans / 26.3 s of drain for one 46.8k prompt with the n_kv-sized host mask).
// pp (n_tokens > 64): kv_size, so consecutive ubatches never re-plan (4-8 B x 288k = 1.2-2.3 MB per device per ubatch).
//   A first version used a power-of-two bucket of n_kv here: 5 re-plans / 1.38 s of drain left per 46.8k prompt.
// decode (<= 64 rows, the 6a row cap): exactly n_kv, a re-plan only when n_kv crosses a 256 boundary and the steps
//   arrive synchronized anyway; the bucket version copied 2.34 MB per step at 47k (65536 x 8 B x 3 devices) vs 718 KB
//   with 6a, t_copy p50 1.30 -> 2.01 ms, step p50 25.3 -> 26.1 ms.
static int64_t kq_build_cells_cap(int64_t n_kv, int64_t kv_size, int64_t n_tokens) {
    return n_tokens > 64 ? std::max<int64_t>(n_kv, kv_size) : n_kv;
}

void llm_graph_input_attn_kv::kq_build_verify(const llama_ubatch * ubatch) const {
    std::vector<ggml_tensor *> outs;
    for (ggml_tensor * t : kq_build_out) {
        if (t && t->buffer) {
            outs.push_back(t);
        }
    }
    if (!outs.empty() && kq_build_mctx) {
        kq_build_mctx->kq_build_verify(outs, ubatch, self_kq_mask->ne[0], cparams.causal_attn);
    }
}

// impl

ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_dev_view(ggml_context * ctx, ggml_tensor * dev) {
    for (const auto & [d, v] : kq_mask_dev_views) {
        if (d == dev) {
            return v;
        }
    }

    const int64_t n_kv  = self_kq_mask->ne[0];
    const int64_t n_tps = self_kq_mask->ne[1];
    const size_t  es    = ggml_element_size(dev);

    GGML_ASSERT(self_kq_mask->ne[2] == 1 && self_kq_mask->ne[3] == 1 && dev->type == self_kq_mask->type);

    ggml_tensor * v = ggml_view_4d(ctx, dev, n_kv, n_tps, 1, 1, n_kv*es, n_kv*n_tps*es, n_kv*n_tps*es, 0);
    ggml_format_name(v, "attn_inp_kq_mask_dev%zu", kq_mask_dev_views.size());

    kq_mask_dev_views.emplace_back(dev, v);

    return v;
}

ggml_tensor * llm_graph_input_attn_kv::get_kq_build_view(ggml_context * ctx, int idx) {
    GGML_ASSERT(kq_build_on && idx >= 0);
    if ((int) kq_build_out.size() <= idx) {
        kq_build_out.resize(idx + 1, nullptr);
        kq_build_view.resize(idx + 1, nullptr);
    }
    if (!kq_build_view[idx]) {
        const int64_t n_kv     = self_kq_mask->ne[0];
        const int64_t n_tokens = self_kq_mask->ne[1];
        // fixed size: kv_size x n_ubatch whatever this graph's n_kv / n_tokens (no gallocr re-plan between ubatches);
        // the reserve sizes it once per device (876 MiB at 224k/2048 replaces the sched's 4 input copies of that size)
        const int64_t n_out = std::max<int64_t>((int64_t) mctx->kq_build_kv_size()*cparams.n_ubatch, n_kv*n_tokens);
        ggml_tensor * out = ggml_kq_mask_build(ctx, kq_build_cells, kq_build_rows, mctx->kq_build_anchor(idx), n_out,
                (int32_t) n_kv, (int32_t) n_tokens, mctx->kq_build_causal(cparams.causal_attn));
        ggml_format_name(out, "kq_mask_build_d%d", idx);
        if (mctx->kq_build_check()) {
            ggml_set_output(out); // keep it alive past its last FA for the read-back
        }
        // dense view (row stride n_kv): the Vulkan FA indexes the mask with row stride KV and would copy a
        // strided mask into a scratch per FA call (ggml_vk_flash_attn, mask_dense)
        const size_t es = ggml_element_size(out);
        ggml_tensor * v = ggml_view_4d(ctx, out, n_kv, n_tokens, 1, 1, n_kv*es, n_kv*n_tokens*es, n_kv*n_tokens*es, 0);
        ggml_format_name(v, "kq_mask_build_d%d_view", idx);
        kq_build_out[idx]  = out;
        kq_build_view[idx] = v;
    }
    return kq_build_view[idx];
}

ggml_tensor * llm_graph_input_attn_kv::get_kq_mask_il(ggml_context * ctx, int il) {
    // [KQ_MASK_DEV] one op per attention device per graph, shared by all its layers: the Vulkan FA caches mask_opt /
    // sparse per mask tensor (ggml_vk_fa_mask_scratch), so one view per device also keeps that pre-pass at one per graph
    if (kq_build_on && self_kq_mask_cnv == self_kq_mask) {
        const int idx = mctx->kq_build_dev(il);
        if (idx >= 0) {
            return get_kq_build_view(ctx, idx);
        }
    }

    if (kq_mask_dev_ok && self_kq_mask_cnv == self_kq_mask) {
        ggml_tensor * dev = mctx->get_kq_mask_dev(il);
        if (dev) {
            return get_kq_mask_dev_view(ctx, dev);
        }
    }

    return get_kq_mask();
}

void llm_graph_input_attn_kv::set_input_kq_mask_any(const llama_kv_cache_context * mctx, const llama_ubatch * ubatch) {
    // [KQ_MASK_DEV] ordinary graph inputs: the sched's pipelined input copies move them, no host write into live
    // device memory and therefore no synchronize (the 6a path's sync is what kept it off pp ubatches)
    if (kq_build_on && kq_build_cells && kq_build_cells->buffer) {
        mctx->set_input_kq_build(kq_build_cells, kq_build_rows, ubatch, self_kq_mask->ne[0]);
        kq_build_mctx = mctx;
    }

    if (!kq_mask_dev_views.empty()) {
        mctx->set_input_kq_mask_dev(ubatch, self_kq_mask->ne[0], cparams.causal_attn);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass) or when every layer reads the persistent mask
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    ASSERT_EMBD_OR_TOKEN(*ubatch);

    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

// [CP] an idx input is unallocated when no node reads it: a range that is not live (h_r >= n_kv) has its input in the
// graph but no store reads it. Kept as an unread input on purpose: the input set stays fixed per cache (can_reuse
// compares it), only the nodes change at an n_kv step, and the setter skips what the sched did not allocate.
void llm_graph_input_attn_kv::set_input_cp_idxs(const llama_kv_cache_context * mctx, const llama_ubatch * ubatch) {
    for (size_t r = 0; r < self_cp_idxs.size(); ++r) {
        if (self_cp_idxs[r] && self_cp_idxs[r]->buffer) {
            mctx->set_input_cp_idxs(self_cp_idxs[r], ubatch, (int) r);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    // the idxs are left unallocated when the KV cache is bypassed during training
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }
    set_input_cp_idxs(mctx, ubatch);

    set_input_kq_mask_any(mctx, ubatch);

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    res &= kq_mask_dev_ok == kq_mask_dev_decide(mctx, params.ubatch, params.cparams, self_kq_mask);

    // [KQ_MASK_DEV] the op params and the cells bucket follow (n_kv, n_tokens), which can_reuse_kq_mask pins
    res &= kq_build_on == kq_build_decide(mctx, params.ubatch, params.cparams, self_kq_mask);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    // left unallocated when the indexer does not use the rotation
    if (self_k_rot_lid && self_k_rot_lid->buffer) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // [CP] same guard as llm_graph_input_attn_kv::set_input: an input no node reads gets no buffer. With every
    // attention layer context-parallel and no LLAMA_CP_CHECK, no node reads the global k/v idxs (each layer stores
    // through the per-range cp idxs), so the sched leaves them unallocated (2026-10-23: GGML_ASSERT(buffer) in
    // set_input_k_idxs at the warm-up decode of the 27B, LLAMA_CP_LAYERS=all)
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    }
    if (inp_attn->self_v_idxs && inp_attn->self_v_idxs->buffer) {
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }
    inp_attn->set_input_cp_idxs(mctx->get_attn(), ubatch);

    inp_attn->set_input_kq_mask_any(mctx->get_attn(), ubatch);

    if (inp_attn->self_k_rot && inp_attn->self_k_rot->buffer) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot && inp_attn->self_v_rot->buffer) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // TEMP DIAGNOSTIC: each sub-check logged on failure when LLAMA_GRAPH_REUSE_LOG=1
    const auto * recr = mctx->get_recr();
    const auto * attn = mctx->get_attn();
    const auto & ub   = params.ubatch;

    if (inp_attn->self_k_idxs->ne[0] != ub.n_tokens) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: k_idxs ne0 %" PRId64 " != n_tokens %u\n", __func__, inp_attn->self_k_idxs->ne[0], ub.n_tokens);
        res = false;
    }
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    if (inp_attn->kq_mask_dev_ok != kq_mask_dev_decide(attn, ub, params.cparams, inp_attn->self_kq_mask)) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: persistent kq mask decision changed\n", __func__);
        res = false;
    }

    if (inp_attn->kq_build_on != kq_build_decide(attn, ub, params.cparams, inp_attn->self_kq_mask)) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: device-built kq mask decision changed\n", __func__);
        res = false;
    }

    if (!can_reuse_kq_mask(inp_attn->self_kq_mask, attn, ub, params.cparams)) {
        const auto * m = inp_attn->self_kq_mask;
        const uint32_t n_stream = params.cparams.kv_unified ? 1 : ub.n_seqs_unq;
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: kq_mask [%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] vs want [%u,%u,1,%u]\n", __func__,
                m->ne[0], m->ne[1], m->ne[2], m->ne[3], attn->get_n_kv(), ub.n_tokens/n_stream, n_stream);
        res = false;
    }

    if (inp_rs->s_copy->ne[0] != recr->get_n_rs()) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: s_copy ne0 %" PRId64 " != n_rs %u\n", __func__, inp_rs->s_copy->ne[0], recr->get_n_rs());
        res = false;
    }

    if (inp_rs->s_copy_main->ne[0] != ub.n_seqs) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: s_copy_main ne0 %" PRId64 " != n_seqs %u\n", __func__, inp_rs->s_copy_main->ne[0], ub.n_seqs);
        res = false;
    }

    if (inp_rs->head != recr->get_head()) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: head %u != %u\n", __func__, inp_rs->head, recr->get_head());
        res = false;
    }
    if (inp_rs->rs_z != recr->get_rs_z()) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: rs_z %d != %d\n", __func__, inp_rs->rs_z, recr->get_rs_z());
        res = false;
    }

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    kq_mask_dev   = false;
    kq_build_check.clear();
    t_cp_merged.clear();
    t_cp_full.clear();

    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (graph_reuse_log_enabled()) LLAMA_LOG_WARN("%s: no reuse: graph params differ (n_tokens %u->%u n_seqs %u->%u n_outputs %d->%d gtype %d->%d mctx %p->%p)\n", __func__,
                this->params.ubatch.n_tokens, params.ubatch.n_tokens, this->params.ubatch.n_seqs, params.ubatch.n_seqs,
                (int) this->params.n_outputs, (int) params.n_outputs, (int) this->params.gtype, (int) params.gtype,
                (const void *) this->params.mctx, (const void *) params.mctx);

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    moe_cache        (params.moe_cache),
    prec_policy      (params.prec_policy),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s,
          ggml_tensor * slots) const {
    // the experts in the MoE cache are selected by their slots
    ggml_tensor * res = slots == nullptr ?
        ggml_mul_mat_id(ctx0, w, cur, ids) :
        ggml_mul_mat_id(ctx0, moe_cache->get_experts(w), cur, slots);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    // the experts of host-resident layers may be read from the MoE cache
    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s, slots); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                            cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                        } else {
                            up = ggml_clamp(ctx0, up, -limit, limit);
                            cb(up, "ffn_moe_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s, slots); // [n_embd, n_expert_used, n_tokens]
    if (arch == LLM_ARCH_MISTRAL4) {
        // src1 can exceed F16 range
        ggml_prec_set_src(experts, GGML_PREC_F32, 1);
    }
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

ggml_tensor * llm_graph_context::build_moe_cache_slots(
         ggml_tensor * selected_experts,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * gate_up_exps,
                 int   il) const {
    if (moe_cache == nullptr) {
        return nullptr;
    }

    ggml_tensor * slot_map = moe_cache->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0]);
    if (slot_map == nullptr) {
        return nullptr;
    }
    for (ggml_tensor * w : { up_exps, gate_exps, down_exps, gate_up_exps }) {
        if (w != nullptr && moe_cache->get_experts(w) == nullptr) {
            return nullptr;
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(moe_cache->backend(), slots)) {
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend());
    cb(slots, "ffn_moe_slots", il);

    return ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]); // [n_expert_used, n_tokens]
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // token embeddings with lora and padding
    auto build_tok = [&](ggml_tensor * ids) {
        ggml_tensor * cur = ggml_get_rows(ctx0, tok_embd, ids);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = inp->embd;

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;
    if (has_mixed) {
        const int64_t n_tok_rows = llm_graph_n_tok_rows(ubatch);

        inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
        cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
        ggml_set_input(inp->mixed_tokens);

        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), build_tok(inp->mixed_tokens), inp->mixed_slots);
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    const bool scale_tok_only = hparams.f_embedding_scale != 0.0f && hparams.n_deepstack_layers > 0;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f && !scale_tok_only) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale*(scale_tok_only ? hparams.f_embedding_scale : 1.0f);
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    cb(cur, "inp_attn_scale", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);
    ggml_set_name(cur, "cross_embd");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_enc");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_dec");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_prec_set_acc(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    cb(inp->self_kq_mask, "self_kq_mask", -1);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static ggml_backend_t cp_backend_of(ggml_backend_sched_t sched, ggml_backend_dev_t dev);

// [CP] node names of the context-parallel ranges: range 0 is "a" (the owner), range 1 "b" as before N ranges
static const char * cp_name(const char * fmt, int r) {
    static thread_local char buf[GGML_MAX_SRC][64];
    GGML_ASSERT(r >= 0 && r < GGML_MAX_SRC);
    snprintf(buf[r], sizeof(buf[r]), fmt, 'a' + r);
    return buf[r];
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        // [CP] one per range; same shape as self_k_idxs (pinned by can_reuse), present iff the cache has a CP layer
        for (int r = 0; r < mctx_cur->cp_n_ranges(); ++r) {
            inp->self_cp_idxs.push_back(mctx_cur->build_input_cp_idxs(ctx0, ubatch, r));
        }

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;

        inp->kq_mask_dev_ok = kq_mask_dev_decide(mctx_cur, ubatch, cparams, inp->self_kq_mask);

        // [KQ_MASK_DEV] the two small inputs; the ops themselves are built per device on first use (get_kq_mask_il)
        inp->kq_build_on = kq_build_decide(mctx_cur, ubatch, cparams, inp->self_kq_mask);
        if (inp->kq_build_on) {
            inp->kq_mask_dev_ok = false; // 6a is not allocated when the op gate is on; belt and braces

            const int64_t n_kv = inp->self_kq_mask->ne[0];
            inp->kq_build_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_cur->kq_build_cell_stride(),
                    kq_build_cells_cap(n_kv, mctx_cur->kq_build_kv_size(), ubatch.n_tokens));
            inp->kq_build_rows  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 2, std::max<int64_t>(cparams.n_ubatch, ubatch.n_tokens));
            ggml_set_input(inp->kq_build_cells);
            ggml_set_input(inp->kq_build_rows);
            ggml_set_name(inp->kq_build_cells, "kq_build_cells");
            ggml_set_name(inp->kq_build_rows,  "kq_build_rows");
        }
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_tensor * q = q_cur;
    ggml_tensor * k;
    ggml_tensor * v;

    // [CP] split cell of the context-parallel layer (0: plain attention); a function of n_kv only, so it is known before
    // the K/V store and the store can already feed the helper from the packed Q|K|V (cp_pack_qkv)
    const uint32_t cp_h = cparams.training || kq_b || sinks || !cparams.flash_attn ? 0 : mctx_cur->cp_split(il);
    // [CP] range split: above h there is no single K/V tensor, so a graph that cannot run build_attn_cp cannot attend
    GGML_ASSERT(!(cp_h == 0 && !cparams.training && mctx_cur->cp_split(il) > 0) &&
            "context-parallel layer above h needs the build_attn_cp path (flash attention, no kq_b, no sinks)");
    std::vector<ggml_tensor *> q_h; // [CP] Q on helper r when packed, else the sched hops q itself
    int cp_live = 1;                // [CP] ranges with cells < n_kv

    if (cparams.training) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);

        k = k_cur;
        v = v_cur;
    } else {
        {
            const auto & k_idxs = inp->get_k_idxs();
            const auto & v_idxs = inp->get_v_idxs();

            if (!mctx_cur->cp_active(il)) {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
            } else {
                // [CP] range-split store (llama_kv_cache::cp_state): range 0 on the owner always; range r >= 1 on its device
                // only when it is live (h_r < n_kv, else no cell of the ubatch is in it, see cp_split / cp_n_live). Each
                // set_rows takes all n_tokens rows, other ranges' tokens land in its trash row (fixed shape -> graph reuse
                // holds; a ubatch across a bound is split by the indices, not by the graph). The 2026-10-21 mirror wrote all
                // rows into a full copy on the helper (1056 MiB per layer at -c 270336).
                const int n_ranges = mctx_cur->cp_n_ranges();
                GGML_ASSERT((int) inp->self_cp_idxs.size() == n_ranges && n_ranges >= 2);

                ggml_build_forward_expand(gf, mctx_cur->cpy_k_range(ctx0, k_cur, inp->self_cp_idxs[0], il, 0));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v_range(ctx0, v_cur, inp->self_cp_idxs[0], il, 0));
                if (mctx_cur->cp_check(il)) {
                    // LLAMA_CP_CHECK reference: the whole layer on the owner, global indices
                    ggml_build_forward_expand(gf, mctx_cur->cpy_k_range(ctx0, k_cur, k_idxs, il, -1));
                    ggml_build_forward_expand(gf, mctx_cur->cpy_v_range(ctx0, v_cur, v_idxs, il, -1));
                }
                if (cp_h > 0) {
                    cp_live = mctx_cur->cp_n_live(il);
                    GGML_ASSERT(cp_live >= 2);

                    q_h.assign(cp_live, nullptr);
                    std::vector<ggml_tensor *> k_h(cp_live, k_cur);
                    std::vector<ggml_tensor *> v_h(cp_live, v_cur);
                    if (mctx_cur->cp_pack()) {
                        cp_pack_qkv(mctx_cur, q_cur, k_cur, v_cur, q_h, k_h, v_h, cp_live, il);
                    }
                    for (int r = 1; r < cp_live; ++r) {
                        ggml_tensor * sk = mctx_cur->cpy_k_range(ctx0, k_h[r], inp->self_cp_idxs[r], il, r);
                        ggml_tensor * sv = mctx_cur->cpy_v_range(ctx0, v_h[r], inp->self_cp_idxs[r], il, r);
                        // pinned: a set_rows follows its destination's buffer anyway, the pin keeps the sched from
                        // splitting the helper's two stores around an owner node
                        ggml_backend_t be_r = cp_backend_of(sched, mctx_cur->cp_dev(il, r));
                        GGML_ASSERT(be_r && "context-parallel attention: helper device has no backend in the scheduler");
                        ggml_backend_sched_set_tensor_backend(sched, sk, be_r);
                        ggml_backend_sched_set_tensor_backend(sched, sv, be_r);
                        ggml_build_forward_expand(gf, sk);
                        ggml_build_forward_expand(gf, sv);
                    }
                }
            }
        }

        // [CP] above h get_k/get_v of a CP layer would abort (no single tensor): build_attn_cp reads the ranges
        if (cp_h == 0) {
            k = mctx_cur->get_k(ctx0, il);
            v = mctx_cur->get_v(ctx0, il);
        } else {
            k = nullptr;
            v = nullptr;
        }
    }

    ggml_tensor * kq_mask = inp->get_kq_mask_il(ctx0, il);
    if (kq_mask != inp->get_kq_mask() && !inp->kq_build_on) {
        res->kq_mask_dev = true;
    }
    if (inp->kq_build_on && mctx_cur->kq_build_check() &&
            std::find(res->kq_build_check.begin(), res->kq_build_check.end(), inp) == res->kq_build_check.end()) {
        res->kq_build_check.push_back(inp);
    }

    ggml_tensor * cur = cp_h > 0
        ? build_attn_cp(inp, q, q_h, kq_mask, cp_live, kq_scale, il)
        : build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// [CP] 2026-10-21 context-parallel attention prototype for one layer (llama_kv_cache::cp_state has the memory side).
// 2026-10-22 night: the helper holds only cells [h, kv_size) of the layer (range split, llama_kv_cache::cp_state), not a
// full mirror; h is fixed at construction, so below h this function is not called (cp_split 0, plain graph on range 0).
//
// Node order is the schedule: the sched runs splits in graph order and copies a split's inputs before issuing it.
//   A1 [... Q/K/V, store into A's cache] -> B1 [range-1 store, FA_B over [h, n_kv)] -> A2 [FA_A over [0, h)]
//   -> B2 [B's partial out] -> A3 [merge, then the rest of the layer]
// B1's inputs wait for A1 only, so B starts FA_B before A has FA_A queued; A2 has no input copies and with
// ggml_backend_sched_set_split_sync_relax its issue does not wait for B either; B2 (a tiny node on B) is the split
// boundary that keeps the merge's wait for B's partial out of A2. Every node is pinned: FLASH_ATTN_EXT is excluded
// from the sched's weight rule and the merge's neighbours would pull it anywhere.
// Merge (the recipe tests/test-flash-attn-merge.cpp validates): w = softmax over the two lse, O = w_a*O_a + w_b*O_b.
//
// 2026-10-22 evening, LLAMA_CP_PACK (default 1, 0 = the graph above with one hop per tensor). Why, from the decode
// tapes of ~/ai/claude_scratch/cp_overlap (Qwen 27B, -c 150000, ~120k context, 1-12 token steps, owner Vulkan3,
// helper Vulkan1, layer 27; step = t_total + next t_sync_prev): A2 (CP off) 63.7 ms, C2 (CP on) 64.3, D2 (CP on,
// LLAMA_CP_SYNC_RELAX=0) 65.2, B2 (CHECK) identical.
//   - the host does NOT hold FA_A back: C2 t_split_sync_us p50 0 (D2: 30.3 ms), A2's t_issue_us 19 us, every CP-layer
//     input t_dst_wait_us <= 3 us, no CPU-sourced CP input takes the synchronous bounce (k/v idxs go path "in"; the
//     only bounces in the step are node_1 / model.input_embed into Vulkan2, the 2026-10-22 race fix). The relax is
//     real: D2 starts FA_A 1.45 ms after A1 (C2 0.53 ms) and D2's step is 0.9 ms longer than C2's, the same 0.9 ms.
//   - the "owner idles while the helper runs FA_B" picture of the --timeline bars is a cross-card clock offset:
//     in A2 (CP off, strictly serial V2 -> V3 -> V1) Vulkan1's split starts 1.17 ms (p50) BEFORE the Vulkan3 split
//     whose l_out it waits on, so the Vulkan3&Vulkan1 busy intersection reads 1.18 ms with zero real overlap;
//     C2's 1.64 ms is that 1.18 plus FA_B (0.44 ms) landing inside A1's interval on the skewed clock. Bounded by
//     causality (A3 waits for B2's hop, V1's next split waits for A3's) the V1 clock reads 1.21-1.61 ms early, so
//     FA_B really runs at A1_end + [0.51, 0.91] ms against FA_A at A1_end + [0.53, 1.00]: the halves overlap.
//   - what the owner does pay, on its own clock (same-card gaps, no offset): A1_end -> FA_A start 0.53 ms p50 =
//     the 4 hop-1 copies (Qcur 172 KB, Kcur/Vcur 28 KB, kq_mask_build view 1.5 MB at 7 rows) the sched records on
//     Vulkan3's queue for B1, each its own batch + submit + timeline signal (ggml_vk_d2d_copy_async); FA_A end ->
//     A3 start 0.39 ms = the 2 hop-2 copies back (cp_lse_b, cp_fa_b view). Two splits on one card with no hop
//     between run 0.027 ms apart (B1 -> B2), so ~0.13 ms per hop; 6 hops ~0.8 ms of owner bubble against ~0.45 ms
//     of FA saved = the +0.6 ms of C2 over A2.
// So the pack cuts the hops to one each way:
//   - forward: Q|K|V concatenated on A (two concat kernels, ~250 KB at 7 rows), one hop, one cont on B (the sched
//     copies the tensor a B node reads; views of an A tensor would be hopped one by one), range-1 store and FA_B read
//     views of it. Needs F32 Q/K/V with one head size (Qwen 3.x: 256/256/256); otherwise the per-tensor hops stay.
//   - mask: with the device-built mask (KQ_MASK_DEV, kq_build_on) B reads the op of its own device instead of a
//     1.5 MB view of A's (the op already exists when B holds attention layers, llama_kv_cache::cp_kq_build_dev; the
//     same node then serves B's own layers 43-63, so no extra GPU work and its 482 KB cells input moves from B's last
//     split into B1). Other mask modes are unchanged.
//   - backward: B2 = cont(p_b) on B, A3 starts with cont of that on A (one hop of O|lse, ~175 KB), lse_b / O_b are
//     views on A. The cont on B is the B2 split boundary that the old lse_b node was.
// Expected on the owner: 4 -> 1 hop-1 submits before FA_A (~0.4 ms), 2 -> 1 hop-2 before A3 (~0.13 ms). Not yet
// measured on GPU (the GPU gate is in contextFiles/context-perf-step-budget-whatifs.md, 2026-10-22 evening).
// Preconditions elsewhere: ggml_backend_sched_split_graph copies an input once per (tensor, backend) and points
// every later reader at the copy (tensor_id_copy), and a view is placed on its view_src's backend; the Vulkan
// concat takes strided F32 (ggml_vk_concat_supported), set_rows takes the strided K/V views (row stride = the packed
// token stride), FA takes a strided Q (q_stride = nbq1, ggml_vk_flash_attn).
static ggml_backend_t cp_backend_of(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(b) == dev) {
            return b;
        }
    }
    return nullptr;
}

// [CP] LLAMA_CP_PACK forward half (comment above build_attn_cp): Q|K|V as one tensor on the owner, one hop per helper on
// another device, views there. Leaves q_h / k_h / v_h untouched (per-tensor hops) when the shapes or types do not allow
// one concat, and for helpers on the owner's device (the CPU test mode: no hop to save, no concat built).
void llm_graph_context::cp_pack_qkv(
        const llama_kv_cache_context * mctx_cur,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        std::vector<ggml_tensor *> & q_h,
        std::vector<ggml_tensor *> & k_h,
        std::vector<ggml_tensor *> & v_h,
                int   n_live,
                int   il) const {
    ggml_backend_dev_t dev_a = mctx_cur->cp_dev(il, 0);
    bool any = false;
    for (int r = 1; r < n_live; ++r) {
        any = any || mctx_cur->cp_dev(il, r) != dev_a;
    }
    if (!any) {
        return;
    }

    const int64_t D = q_cur->ne[0];
    const bool ok =
        q_cur->type == GGML_TYPE_F32 && k_cur->type == GGML_TYPE_F32 && v_cur->type == GGML_TYPE_F32 &&
        k_cur->ne[0] == D && v_cur->ne[0] == D &&
        k_cur->ne[2] == q_cur->ne[2] && v_cur->ne[2] == q_cur->ne[2] &&
        q_cur->ne[3] == 1 && k_cur->ne[3] == 1 && v_cur->ne[3] == 1;
    if (!ok) {
        return;
    }

    ggml_backend_t be_a = cp_backend_of(sched, dev_a);
    GGML_ASSERT(be_a && "context-parallel attention: owner device has no backend in the scheduler");

    ggml_tensor * qk = ggml_concat(ctx0, q_cur, k_cur, 1);
    ggml_backend_sched_set_tensor_backend(sched, qk, be_a);
    ggml_tensor * qkv = ggml_concat(ctx0, qk, v_cur, 1);  // [D, n_head + 2*n_head_kv, n_tokens], contiguous
    ggml_backend_sched_set_tensor_backend(sched, qkv, be_a);
    cb(qkv, "cp_qkv", il);

    const int64_t nh    = q_cur->ne[1];
    const int64_t nh_kv = k_cur->ne[1];
    const int64_t nt    = q_cur->ne[2];

    for (int r = 1; r < n_live; ++r) {
        if (mctx_cur->cp_dev(il, r) == dev_a) {
            continue;
        }
        ggml_backend_t be_r = cp_backend_of(sched, mctx_cur->cp_dev(il, r));
        GGML_ASSERT(be_r && "context-parallel attention: helper device has no backend in the scheduler");

        ggml_tensor * qkv_r = ggml_cont(ctx0, qkv);        // the one hop: the sched copies qkv into helper r for this node
        ggml_backend_sched_set_tensor_backend(sched, qkv_r, be_r);
        cb(qkv_r, cp_name("cp_qkv_%c", r), il);

        const size_t nb1 = qkv_r->nb[1];
        const size_t nb2 = qkv_r->nb[2];

        q_h[r] = ggml_view_3d(ctx0, qkv_r, D, nh,    nt, nb1, nb2, 0);
        k_h[r] = ggml_view_3d(ctx0, qkv_r, D, nh_kv, nt, nb1, nb2, nh*nb1);
        v_h[r] = ggml_view_3d(ctx0, qkv_r, D, nh_kv, nt, nb1, nb2, (nh + nh_kv)*nb1);
    }
}

// [CP] 2026-10-23, N ranges: helper r (range [h_r, h_r+1) on its own device) is what "B" is below; every helper gets
// its own B1 (store + FA over [h_r, min(h_r+1, n_kv))) and B2 (its partial out), all B1 before A2 so every helper
// starts before the owner queues FA_A, all B2 after it. Hops per layer per step: one forward and one back per helper
// on another device (pack). N = 2 is node for node the graph below (same calls, same order).
ggml_tensor * llm_graph_context::build_attn_cp(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q,
        const std::vector<ggml_tensor *> & q_h,
        ggml_tensor * kq_mask,
                int   n_live,
              float   kq_scale,
                int   il) const {
    const auto * mctx_cur = inp->mctx;

    GGML_ASSERT(n_live >= 2 && (int) q_h.size() == n_live);

    std::vector<ggml_backend_t> be(n_live);
    for (int r = 0; r < n_live; ++r) {
        be[r] = cp_backend_of(sched, mctx_cur->cp_dev(il, r));
        GGML_ASSERT(be[r] && "context-parallel attention: owner or helper device has no backend in the scheduler");
    }
    ggml_backend_t be_a = be[0];

    const bool pack = mctx_cur->cp_pack();

    auto on = [&](ggml_tensor * t, ggml_backend_t b) {
        ggml_backend_sched_set_tensor_backend(sched, t, b);
        return t;
    };

    GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);

    const uint32_t n_kv = kq_mask->ne[0];
    const int64_t  D    = hparams.n_embd_head_v(il);
    const int64_t  nh   = q->ne[1];
    const int64_t  nt   = q->ne[2];

    GGML_ASSERT(mctx_cur->cp_bound(n_live - 1) < n_kv);

    // each helper reads its own persistent mask when the owner does, else a column view of the graph input
    // [KQ_MASK_DEV] with the device-built mask a helper reads the owner's op output (the sched copies it over),
    // unless LLAMA_CP_PACK and the helper's device has its own op (cp_kq_build_dev): then it reads that one, no 1.5 MB hop
    std::vector<ggml_tensor *> mask(n_live, kq_mask);
    for (int r = 1; r < n_live; ++r) {
        if (kq_mask != inp->get_kq_mask() && !inp->kq_build_on) {
            ggml_tensor * dev_r = mctx_cur->cp_kq_mask_dev(il, r);
            GGML_ASSERT(dev_r);
            mask[r] = inp->get_kq_mask_dev_view(ctx0, dev_r);
        } else if (pack && inp->kq_build_on && kq_mask != inp->get_kq_mask()) {
            const int idx_r = mctx_cur->cp_kq_build_dev(il, r);
            if (idx_r >= 0) {
                mask[r] = inp->get_kq_build_view(ctx0, idx_r);
                GGML_ASSERT(mask[r]->ne[0] == kq_mask->ne[0] && mask[r]->ne[1] == kq_mask->ne[1]);
                on(mask[r]->view_src, be[r]);
            }
        }
    }

    // a column view keeps the row stride n_kv; FA takes strided masks (ggml.h)
    auto cols = [&](ggml_tensor * m, uint32_t c0, uint32_t c1) {
        return ggml_view_4d(ctx0, m, c1 - c0, m->ne[1], m->ne[2], m->ne[3], m->nb[1], m->nb[2], m->nb[3], c0*ggml_element_size(m));
    };

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    // [CP] i0/i1 are global cells; get_k_range picks the range holding them (each range at its local row 0), ref = the
    // LLAMA_CP_CHECK reference on the owner
    auto fa = [&](ggml_tensor * qq, uint32_t i0, uint32_t i1, bool ref, ggml_tensor * m, bool lse) {
        ggml_tensor * k = ggml_permute(ctx0, mctx_cur->get_k_range(ctx0, il, i0, i1, ref), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(ctx0, mctx_cur->get_v_range(ctx0, il, i0, i1, ref), 0, 2, 1, 3);

        ggml_tensor * cur = ggml_flash_attn_ext(ctx0, qq, k, v, m, kq_scale, hparams.f_max_alibi_bias,
                                                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        if (lse) {
            ggml_flash_attn_ext_set_emit_lse(cur, true);
        }
        return cur; // [D (+4), nh, nt]
    };

    std::vector<ggml_tensor *> p(n_live, nullptr);
    for (int r = 1; r < n_live; ++r) {
        const uint32_t c0 = mctx_cur->cp_bound(r);
        const uint32_t c1 = std::min(mctx_cur->cp_bound(r + 1), n_kv);
        ggml_tensor * qq = q_h[r] ? ggml_permute(ctx0, q_h[r], 0, 2, 1, 3) : q;
        p[r] = on(fa(qq, c0, c1, false, cols(mask[r], c0, c1), true), be[r]);
        cb(p[r], cp_name("cp_fa_%c", r), il);
        ggml_build_forward_expand(gf, p[r]);
    }

    const uint32_t h = mctx_cur->cp_bound(1);
    p[0] = on(fa(q, 0, h, false, cols(kq_mask, 0, h), true), be_a);
    res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, p[0], il});
    cb(p[0], "cp_fa_a", il);
    ggml_build_forward_expand(gf, p[0]);

    auto lse_of = [&](ggml_tensor * t, ggml_backend_t b) {
        return on(ggml_cont(ctx0, ggml_view_3d(ctx0, t, 1, nh, nt, t->nb[1], t->nb[2], D*sizeof(float))), b);
    };
    auto o_of = [&](ggml_tensor * t) {
        return ggml_view_3d(ctx0, t, D, nh, nt, t->nb[1], t->nb[2], 0);
    };

    // B2 and the hop back, per helper. Pack: B2 = cont(p_r) on the helper, A3 opens with a cont of it on the owner (the
    // one hop: an owner node reads the whole helper tensor), lse_r / O_r are views on the owner. Else B2 = lse_r on the
    // helper and the owner hops lse_r and the O_r view (N >= 3: the merge op hops the whole p_r instead).
    // Partials in range order (owner first); lse/o are only read by the 2-way ops path, p (the whole [D + 4] tensor)
    // only by GGML_OP_FLASH_ATTN_MERGE, so the 2-way non-pack graph keeps its two separate hops.
    std::vector<llm_cp_partial> parts(n_live);
    for (int r = 1; r < n_live; ++r) {
        if (pack) {
            ggml_tensor * x = on(ggml_cont(ctx0, p[r]), be[r]);
            cb(x, cp_name("cp_fa_%c_out", r), il);
            ggml_build_forward_expand(gf, x);

            ggml_tensor * p_r_a = on(ggml_cont(ctx0, x), be_a);
            cb(p_r_a, cp_name("cp_fa_%c_on_a", r), il);

            parts[r] = { p_r_a, nullptr, nullptr };
            parts[r].lse = lse_of(p_r_a, be_a);
            parts[r].o   = o_of(p_r_a);
        } else {
            ggml_tensor * lse_r = lse_of(p[r], be[r]);
            cb(lse_r, cp_name("cp_lse_%c", r), il);
            ggml_build_forward_expand(gf, lse_r);
            parts[r] = { n_live > 2 ? p[r] : nullptr, o_of(p[r]), lse_r };
        }
    }
    parts[0] = { pack || n_live > 2 ? p[0] : nullptr, o_of(p[0]), lse_of(p[0], be_a) };

    ggml_tensor * cur = build_attn_cp_merge(parts, be_a, il);

    if (mctx_cur->cp_check(il)) {
        ggml_tensor * full = on(fa(q, 0, n_kv, true, kq_mask, false), be_a);
        cb(full, "cp_fa_full", il);
        ggml_set_output(full);
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, full);

        res->t_cp_full.push_back(full);
        res->t_cp_merged.push_back(cur);
    }

    return ggml_reshape_2d(ctx0, cur, D*nh, nt);
}

// [CP] 2026-10-23: merge of N context-parallel partials (design-context-parallel-at-scale.md step 3 needs N = 3: owner +
// two helpers). All merge nodes run on `be` (the owner).
//   N == 2 and LLAMA_CP_MERGE_OP unset (default): the 2-way ops recipe of 2026-10-21, node for node as before
//     (concat of the two lse -> soft_max -> 2 views/cont -> 2 mul -> add). Kept because the LLAMA_CP_CHECK gate numbers
//     (nmse 2.3e-7, argmax 1.0 on Vulkan, 2026-10-22) were taken on exactly this arithmetic.
//   N >= 3, or N == 2 with LLAMA_CP_MERGE_OP=1: one GGML_OP_FLASH_ATTN_MERGE over the packed [D + 4] partials (needs
//     every p, i.e. the LLAMA_CP_PACK path; N == 2 without it falls back to the ops, N >= 3 asserts). Softmax over N lse = pairwise fold up to f32 rounding:
//     test-flash-attn-merge 2026-10-23 on CPU, nway vs fold and vs the ops recipe <= 1e-6 row-relative. A partial with
//     lse -inf (its range holds no attended cell for this row, e.g. a helper range beyond n_kv) is skipped, not multiplied.
//     Vulkan: compiled, NOT gated (gate: test-backend-ops -o FLASH_ATTN_MERGE -b Vulkan<n>, then LLAMA_CP_CHECK with
//     LLAMA_CP_MERGE_OP=1). Vulkan takes N <= 4 (supports_op), more falls back to the CPU via the sched.
// Shapes: every partial is [D (+4), nh, nt] for this ubatch; nothing here depends on n_kv, so N fixed per layer keeps the
// graph shape fixed (no gallocr re-plan per ubatch).
ggml_tensor * llm_graph_context::build_attn_cp_merge(
        const std::vector<llm_cp_partial> & parts,
        ggml_backend_t be,
        int il) const {
    static const bool use_op = [] {
        const char * e = getenv("LLAMA_CP_MERGE_OP");
        return e && atoi(e) != 0;
    }();

    auto on = [&](ggml_tensor * t) {
        ggml_backend_sched_set_tensor_backend(sched, t, be);
        return t;
    };

    const int N = (int) parts.size();
    GGML_ASSERT(N >= 2 && N <= GGML_MAX_SRC);

    // LLAMA_CP_MERGE_OP=1 with LLAMA_CP_PACK=0: no whole partial on the owner, keep the 2-way ops
    const bool have_p = std::all_of(parts.begin(), parts.end(), [](const llm_cp_partial & p) { return p.p != nullptr; });

    ggml_tensor * cur;
    if (N == 2 && (!use_op || !have_p)) {
        ggml_tensor * o_a   = parts[0].o;
        ggml_tensor * lse_a = parts[0].lse;
        ggml_tensor * o_b   = parts[1].o;
        ggml_tensor * lse_b = parts[1].lse;
        const int64_t nh = o_a->ne[1];
        const int64_t nt = o_a->ne[2];

        ggml_tensor * w     = on(ggml_soft_max(ctx0, on(ggml_concat(ctx0, lse_a, lse_b, 0)))); // [2, nh, nt]
        ggml_tensor * w_a   = on(ggml_cont(ctx0, ggml_view_3d(ctx0, w, 1, nh, nt, w->nb[1], w->nb[2], 0)));
        ggml_tensor * w_b   = on(ggml_cont(ctx0, ggml_view_3d(ctx0, w, 1, nh, nt, w->nb[1], w->nb[2], sizeof(float))));

        cur = on(ggml_add(ctx0,
                on(ggml_mul(ctx0, o_a, w_a)),
                on(ggml_mul(ctx0, o_b, w_b))));
    } else {
        std::vector<ggml_tensor *> ps;
        for (const auto & p : parts) {
            GGML_ASSERT(p.p && "context-parallel N-way merge needs the packed partials (LLAMA_CP_PACK)");
            ps.push_back(p.p);
        }
        cur = on(ggml_flash_attn_merge(ctx0, ps.data(), N, false));
    }
    cb(cur, "cp_merged", il);
    ggml_build_forward_expand(gf, cur);

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // whole seq fits into batch in training mode
    const bool use_kv_cur = cparams.training && k_cur && v_cur;
    if (use_kv_cur) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);
    }

    const bool store_kv = !use_kv_cur || hparams.n_layer_kv_from_start >= 0;

    // optionally store to KV cache
    if (store_kv && k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (store_kv && v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows,
            int64_t   rs_pre_row) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

    // pre-ubatch plane (llama_memory_recurrent::rs_pre_pos): keep the input state of every seq of this ubatch,
    // so the caller can roll back to the start of the ubatch at any depth without a host checkpoint.
    // written on every ubatch, so the graph shape does not depend on the batch size.
    // cost: n_seqs rows of state_size per state tensor per layer, one cpy (plus one get_rows when a custom getter
    // reads the cache directly). Qwen3.8-27B: ~3 MB per GDN layer, ~150 MB D2D per decode over 48 layers.
    // the custom-getter gather is emitted before the getter, so it reads the rows before the state write that
    // follows the getter (node order; same base tensor, so ggml-vulkan graph_optimize keeps the order)
    if (rs_pre_row >= 0) {
        GGML_ASSERT(states->ne[1] >= rs_pre_row + (int64_t) rs_head + n_seqs);

        ggml_tensor * pre_src = get_state_rows ?
            ggml_get_rows(ctx0, states, state_copy_main) :
            ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);

        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, pre_src,
                ggml_view_2d(ctx0, states, state_size, n_seqs, states->nb[1], (rs_pre_row + rs_head)*states->nb[1])));
    }

    ggml_tensor * output_states = get_state_rows ?
        get_state_rows(ctx0, states, state_copy_main) :
        ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    inp->s_copy_main = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;

    return build_rs(s, state_copy, inp->s_copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows, kv_state->get_rs_pre_row());
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_MEAN) {
                    // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row to keep the single-output graph static regardless of active samplers
    // multi-output graphs can still vary with the number of output rows
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> dummy_row = { 0 };

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);

        // inactive samplers always work on the first row
        const bool active = it != sampling_rows.end();
        const auto & rows = active ? it->second : dummy_row;
        const int i_out   = active ? 1          : 0;

        for (uint32_t i = 0; i < rows.size(); ++i) {
            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], rows[i] * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[rows[i]] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[rows[i]] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[rows[i]] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[rows[i]] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
