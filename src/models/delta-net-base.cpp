#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

// conv-state snapshot slot index for the per-column set_rows writes in build_conv_state (2026-10-20)
//
// a static I32 [K] leaf (K = n_rs_seq + 1) holding {K-1, ..., 1, 0}. a layer needs {n_w-1, ..., 0} with
// n_w = min(n_seq_tokens, K), which is exactly the LAST n_w entries, so the content never depends on the
// ubatch and the input can always be reused. it is a host-side graph input like rs_s_copy (uploaded by
// the scheduler with the other inputs; an in-graph arange + scale + cast would be three dispatches per
// layer). one leaf per graph, shared by every GDN layer.
//
// each layer takes ggml_cont(leaf) and views its last n_w entries from that GPU-side copy (see
// build_conv_state). two shapes were run before and failed (2026-10-20, test-recurrent-state-rollback,
// qwen35-dense, test_multi_seq_split_replay, GGML_SCHED_DEBUG_REALLOC=1, "unexpected graph reallocation"):
//   - leaf sized n_w, and
//   - K leaf + a view of n_w entries handed straight to set_rows.
// in both the scheduler's cross-backend input copy is n_w-sized (it copies the view, ggml-backend.cpp
// pass 4 gives a view its view_src's CPU backend). the pp reserve plans it with n_w = n_ubatch / n_seqs
// (16 / 2 = 8 there) and the next 1-seq 16-token ubatch needs n_w = 9 with the same node count, so
// ggml-alloc re-plans a graph whose size did not change (the gallocr instrument showed this copy as the
// only node over plan). the cont makes the copy K-sized on every backend; its n_w view is free.
class llm_graph_input_conv_snap_idx : public llm_graph_input_i {
public:
    llm_graph_input_conv_snap_idx(int64_t K) : K(K) {}
    virtual ~llm_graph_input_conv_snap_idx() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);

        std::vector<int32_t> data(K);
        for (int64_t i = 0; i < K; ++i) {
            data[i] = (int32_t) (K - 1 - i);
        }
        ggml_backend_tensor_set(idx, data.data(), 0, ggml_nbytes(idx));
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return idx->ne[0] == K; // static content and shape
    }

    ggml_tensor * idx = nullptr; // I32 [K], leaf

    const int64_t K;
};

// LLAMA_CONV_SNAPSHOT_LOOP=1 restores the per-slot ggml_cpy loop in build_conv_state (A/B, and the escape
// hatch if a backend rejects the overlapping-window set_rows source). read once per process.
static bool llama_conv_snapshot_loop() {
    static const bool v = [] {
        const char * e = getenv("LLAMA_CONV_SNAPSHOT_LOOP");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend ggml_cumsum with axis parameter to avoid transpose
    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);
    cb(attn, "attn", il);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);
    cb(s, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);
        cb(v_t_p, "v_prime", il);

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);
        cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);
        cb(v_attn, "v_attn", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);
        cb(attn_inter, "attn_inter", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);
        cb(o_ch, "dnet_add_ch_attn_out", il);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // TODO: head broadcast might not work here - probably will need a transpose
        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
        cb(s, "dnet_add_ch_state", il);
    }

    // truncate padded tokens
    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);
    cb(s, "output_state", il);

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b, // beta
        ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);

    cb(s, "dnet_add_ar_state", il);

    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_fused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    if (n_tokens == 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, result, il});
    }

    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t n_seq_tokens = q->ne[2];

    if (n_seq_tokens == 1) {
        if (cparams.fused_gdn_ar) {
            return build_delta_net_fused(q, k, v, g, b, s, il);
        }
        return build_delta_net_autoregressive(q, k, v, g, b, s, il);
    }

    if (cparams.fused_gdn_ch) {
        return build_delta_net_fused(q, k, v, g, b, s, il);
    }

    return build_delta_net_chunking(q, k, v, g, b, s, il);
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head  = mctx_cur->get_head();
    const auto mem_size = mctx_cur->get_size();

    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    const int64_t row_count = (conv_kernel_size - 1) * conv_channels;

    const size_t row_size  = ggml_row_size(conv_states_all->type, row_count);

    if (cparams.n_rs_seq == 0) {
        const int64_t s_idx  = conv_input->ne[0] - conv_states->ne[0];
        const int64_t s_slot = 0;

        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx0, conv_input,
                    conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        cb(conv_state_last, "conv_state_last", il);

        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx0, conv_states_all,
                    row_count, n_seqs, conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size);
        cb(conv_state_update, "conv_state_update", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
        //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument

        const int64_t K            = (int64_t) cparams.n_rs_seq + 1;
        const int64_t n_seq_tokens = conv_input->ne[0] - conv_states->ne[0];

        // write only the last min(n_seq_tokens, K) snapshots, like the SSM states; rs_depth never selects the rest
        //
        // K = n_rs_seq + 1 conv snapshot slots exist for rollback, and slot s is selected by rs_depth, which
        // the recurrent memory resets to (tokens of the seq in this ubatch) - 1 after every ubatch
        // (llama_memory_recurrent_context apply(), set_rs_depth(seq_id, n_tokens - 1) in
        // src/llama-memory-recurrent.cpp ~1385). so after an N-token batch only slots [0, min(N, K)) can
        // ever be read; no rollback reaches the slots past that.
        // the SSM state write in build_recurrent_attn (n_written, ~603) already wrote min(n_seq_tokens, K).
        // the old loop wrote all K: after a 1-token decode 11 of the 12 copies were the same clamped
        // pre-batch window (s_idx clamped to 0), 96 KB each, on 24 layers per GPU (2026-10-18, 86ecf17dd1).
        //
        // the same write-all-K pattern exists, not measured and not changed, in glm5_conv1d
        // (glm5-next.cpp:215), kimi_k3_conv1d (kimi-k3.cpp:379) and build_conv_state_at
        // (qwen4exp.cpp:1329). mamba-base.cpp, lfm2.cpp and bailingmoe3.cpp already loop over n_written.
        //
        // ---- 2026-10-20: all n_w snapshots in ONE op per layer (set_rows), the per-slot loop is the fallback ----
        //
        // what the loop writes (derived from it, unchanged semantics): for t in [max(1, K-n+1), K],
        // s_slot = K - t and s_idx = n - K + t = n - s_slot, with n = n_seq_tokens. so slot s, for
        // s in [0, n_w) with n_w = min(n, K), receives conv_input columns [n - s, n - s + kernel - 1).
        // conv_input column (kernel - 1) + j is token j of this ubatch, so slot s holds the window that
        // ENDS at token n - 1 - s: slot 0 = the newest window, slot s = the state s tokens back, which is
        // what s_copy() reads after a rollback of s tokens (plane rs_idx * size + cell, rs_idx = s).
        //
        // why not one strided ggml_cpy, as the SSM write below does: the order is reversed. as the slot
        // index goes up the dst plane offset goes UP (plane stride mem_size * row_size) while the src
        // window offset goes DOWN (one token, conv_input->nb[0], per slot). one positive-stride view on
        // each side cannot express that, ggml strides are size_t (no negative stride on any backend, and
        // the Vulkan push constants carry them as uint32 element counts), and ggml has no flip/reverse op.
        // the SSM sibling gets away with a plain copy because ggml_gated_delta_net already emits its
        // snapshots in slot order. (a dead probe on paper, not run: changing the slot order would move
        // every reader - s_copy(), seq_rm's rs_idx composition, the SSM planes - so semantics stay put.)
        //
        // what is done instead: ggml_set_rows, whose row index tensor does the reversal, one op per
        // window column c in [0, kernel - 1) (3 ops per layer for the qwen35 kernel 4), each moving one
        // column of every slot:
        //   src  b_c = view of conv_input, ne [1, n_w, C, n_seqs],
        //          nb [esz, esz (one token per row), conv_input->nb[1], conv_input->nb[2]],
        //          offset (n - n_w + 1 + c) * esz                      -> b_c row i = column n - n_w + 1 + i + c
        //   dst  a_c = view of conv_states_all, ne [1, n_w, C, n_seqs],
        //          nb [esz, mem_size * row_size (one plane = one slot), (kernel-1) * esz (channel stride
        //              inside a row), conv_states_all->nb[1] (one cell = one seq)],
        //          offset kv_head * row_size + c * esz
        //   idx  = I32 [n_w] = {n_w-1, ..., 0} (last n_w entries of the static {K-1..0} input, via a
        //          per-layer cont), broadcast over C and n_seqs (ne11 = ne12 = 1)
        // b_c row i lands in slot n_w - 1 - i; with s = n_w - 1 - i, slot s column c gets conv_input column
        // n - s + c, i.e. slot s holds columns [n - s, n - s + kernel - 1), exactly the loop below.
        // ne0 == 1 satisfies ggml_set_rows' ggml_is_contiguous_rows assert; CPU and Vulkan set_rows index
        // every operand through its full nb[] (copy_to_quant.comp SET_ROWS: src0_idx / dst_idx with all
        // four strides), so the permuted destination strides are legal. the dst type is the cache type
        // (f32 here; set_rows also takes f16/bf16/quant dst like the cpy did).
        //
        // dead probe (2026-10-20, run): ONE set_rows over all columns, src = a self-overlapping view
        // ne [kernel-1, n_w, C, n_seqs] with nb1 = esz (the windows share kernel-2 columns). ggml_view_4d
        // aborts in ggml_new_tensor_impl (ggml.c ~1797): the view bound is checked as nelements * esz
        // + offset, not from the strides, so an overlapping view that claims more elements than its
        // source holds is rejected however legal its strides are (test-recurrent-state-rollback,
        // qwen35-dense: 3*9*768*4 = 82944 B claimed over a 36864 B conv_input at n = 9). patching ne[1]
        // after building a smaller view would get past the check, but anything that rebuilds views from
        // ne/nb (the meta / tensor-split backend) would trip the same assert, so the per-column split
        // keeps every view non-overlapping and costs kernel - 2 extra dispatches per layer.
        // n_w <= kernel (every plain 1-token decode, the MTP drafter's single-token steps, tiny batches)
        // stays on the loop: n_w copies are no more dispatches than (kernel - 1) set_rows + 1 cont.
        //
        // measured before (2026-10-20, Qwen3.8-27B hybrid, 48 GDN layers over 3 Vulkan GPUs, per-op
        // instrument GGML_VK_MMVQ_SLOPE_STATS on one 7900 XTX): the CPY bucket per verify step was 1.3 ms
        // at 12 rows and 16 ms at 65 rows, i.e. ~0.28 ms per extra row per card and ~0.85 ms per row over
        // the three cards, the third-largest per-row cost after matmuls and attention. each copy is a 96 KB
        // strided window, one dispatch, and there were min(n, K) of them per GDN layer (~16 layers per card).
        // after: min(n, K) dispatches per layer -> kernel (3 set_rows + 1 tiny cont). NOT yet measured
        // for speed.
        //
        // how to re-measure: GGML_VK_MMVQ_SLOPE_STATS=<file> on a seeded verify run with the n-gram cap 33
        // or 64, once as is and once with LLAMA_CONV_SNAPSHOT_LOOP=1. the CPY bucket should fall by the
        // conv share and n_cpy by min(n, K) - 1 per GDN layer (one cont stays); n_set_rows rises by
        // kernel - 1 per GDN layer
        // (the bucket also holds the KV-cache writes, so compare its delta, not its total).
        // correctness A/B: same seed, greedy, rollback-heavy run (speculative verify), with and without
        // LLAMA_CONV_SNAPSHOT_LOOP=1 - token streams must be identical.
        const int64_t n_w = std::min<int64_t>(n_seq_tokens, K);

        // set_rows path when it saves dispatches: (kernel - 1) set_rows + 1 cont vs n_w copies, so from
        // n_w > kernel (n_w >= 5 for kernel 4). n_w <= kernel keeps the loop (a 1-token decode is 1 copy).
        if (n_w > conv_kernel_size && !llama_conv_snapshot_loop()) {
            if (conv_snap_slot_idx == nullptr) {
                auto inp_idx = std::make_unique<llm_graph_input_conv_snap_idx>(K);

                inp_idx->idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, K);
                ggml_set_input(inp_idx->idx);
                ggml_set_name(inp_idx->idx, "conv_snap_slot_idx");

                conv_snap_slot_idx = inp_idx->idx;
                res->add_input(std::move(inp_idx));
            }

            // per layer, so the K-sized copy lands on this layer's GPU (a shared cont would sit on one
            // GPU and the other GPUs would get an n_w-sized copy of its view again, see the input class)
            ggml_tensor * slot_idx_dev = ggml_cont(ctx0, conv_snap_slot_idx);
            ggml_tensor * slot_idx     = ggml_view_1d(ctx0, slot_idx_dev, n_w,
                    (size_t) (K - n_w) * ggml_element_size(slot_idx_dev)); // = {n_w-1, ..., 0}
            cb(slot_idx, "conv_snap_slot_idx_view", il);

            const size_t esz = ggml_element_size(conv_input);

            for (int64_t c = 0; c < conv_kernel_size - 1; ++c) {
                ggml_tensor * conv_col =
                    ggml_view_4d(ctx0, conv_input,
                            1, n_w, conv_channels, n_seqs,
                            esz,
                            conv_input->nb[1],
                            conv_input->nb[2],
                            (size_t) (n_seq_tokens - n_w + 1 + c) * esz);
                cb(conv_col, "conv_state_col", il);

                ggml_tensor * conv_slots =
                    ggml_view_4d(ctx0, conv_states_all,
                            1, n_w, conv_channels, n_seqs,
                            (size_t) mem_size * row_size,
                            ggml_row_size(conv_states_all->type, conv_kernel_size - 1),
                            conv_states_all->nb[1],
                            (size_t) kv_head * row_size + (size_t) c * ggml_element_size(conv_states_all));
                cb(conv_slots, "conv_state_slots", il);

                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, conv_slots, conv_col, slot_idx));
            }

            return conv_input;
        }

        for (int64_t t = std::max<int64_t>(1, K - n_seq_tokens + 1); t <= K; ++t) {
            const int64_t s_idx  = n_seq_tokens - K + t;
            const int64_t s_slot = K - t;

            ggml_tensor * conv_state_last =
                ggml_view_3d(ctx0, conv_input,
                        conv_kernel_size - 1, conv_channels, n_seqs,
                        conv_input->nb[1], conv_input->nb[2],
                        ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * conv_state_update =
                ggml_view_2d(ctx0,
                        conv_states_all, row_count, n_seqs,
                        conv_states_all->nb[1],
                        (s_slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        }
    }

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    if (n_seq_tokens > 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, gdn_out, il});
    }

    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
