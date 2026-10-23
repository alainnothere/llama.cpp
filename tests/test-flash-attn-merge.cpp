// Context-parallel attention merge check (ggml_flash_attn_ext_set_emit_lse and ggml_flash_attn_merge, see ggml.h).
//
// Part 1 (FA partials). N FLASH_ATTN_EXT calls over the KV ranges [h_0 = 0, h_1), [h_1, h_2), ..., [h_{N-1}, n), each
// emitting its lse, must match one FLASH_ATTN_EXT over [0, n): NMSE and max abs error on O, argmax agreement over the
// D dimension (on the rows whose top-2 margin is above the kernel noise), and the merged lse against the full lse.
// Range i's mask is the column view mask[:, h_i:h_{i+1}] of the full mask, which exercises the strided-mask path.
// The partials are merged three ways:
//   - "ops":   the 2-way recipe build_attn_cp runs in production (concat the two lse -> soft_max -> 2 mul -> add), N = 2 only
//   - "nway":  ggml_flash_attn_merge over all N partials
//   - "fold":  ggml_flash_attn_merge pairwise, ((p0 + p1) + p2) + ..., each step with emit_lse
// nway must equal fold and (N = 2) ops within f32 rounding (rel_tol below), and must equal a host double-precision
// softmax over the N lse. Small cases also run a host double full attention as a direct reference for the full FA.
// As a negative control, every case also merges the partials on the host with equal weights (1/N), which must FAIL the
// same criteria, so the tolerances cannot hide a wrong merge.
//
// Part 2 (synthetic partials). Random O with lse spreads 0 .. 1e4 (weights from ~1/N down to exactly 0 in f32), one
// part empty (lse -inf, O = NaN: a helper whose KV range is beyond n_kv contributes nothing and its O must not be read
// into the sum), rows with every part empty (O = 0, lse -inf). nway vs fold vs host double, N = 2 .. 8 (Vulkan <= 4).
//
// CPU always runs and is the cross-backend reference; pass -b <device name> (e.g. -b Vulkan3) to add a device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct merge_case {
    int64_t hs;       // head size (K and V)
    int64_t nh_kv;    // kv heads
    int64_t gqa;      // q heads per kv head
    int64_t n_kv;     // full KV length
    int64_t nb;       // batch (q rows)
    std::vector<int64_t> h; // split points, strictly increasing in (0, n_kv): N = h.size() + 1 ranges
    int       empty;  // range index fully masked for the even rows (-1: none); with empty_all for every row
    bool      empty_all;
    ggml_type kv_type;

    int n_parts() const { return (int) h.size() + 1; }
    int64_t lo(int i) const { return i == 0 ? 0 : h[i - 1]; }
    int64_t hi(int i) const { return i == (int) h.size() ? n_kv : h[i]; }
};

struct merge_result {
    std::vector<float> full;               // [hs + 4, nh, nb]
    std::vector<float> ops;                // [hs, nh, nb], N = 2 only
    std::vector<float> nway;               // [hs + 4, nh, nb]
    std::vector<float> fold;               // [hs + 4, nh, nb]
    std::vector<std::vector<float>> parts; // [hs + 4, nh, nb] each
    std::vector<float> q, k, v, m;         // inputs as f32 (host reference)
    bool ok = false;
};

static bool all_supported(ggml_backend_t backend, ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, t)) {
            printf("    %s: %s (%s) not supported\n", ggml_backend_name(backend), ggml_op_desc(t), t->name);
            return false;
        }
    }
    return true;
}

static ggml_tensor * fold_merge(ggml_context * ctx, const std::vector<ggml_tensor *> & parts) {
    ggml_tensor * acc = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        ggml_tensor * two[2] = { acc, parts[i] };
        acc = ggml_flash_attn_merge(ctx, two, 2, true);
    }
    return acc;
}

static merge_result run_case(ggml_backend_t backend, const merge_case & c, uint32_t seed) {
    merge_result res;

    const int64_t nh = c.nh_kv * c.gqa;
    const int64_t D  = c.hs;
    const int     N  = c.n_parts();

    ggml_init_params ip = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, c.nb, nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, c.kv_type, D, c.n_kv, c.nh_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, c.kv_type, D, c.n_kv, c.nh_kv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.nb, 1, 1);

    const float scale = 1.0f / sqrtf((float) D);

    ggml_tensor * full = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(full, GGML_PREC_F32);
    ggml_flash_attn_ext_set_emit_lse(full, true);

    auto range = [&](int64_t i0, int64_t i1) {
        ggml_tensor * kr = ggml_view_4d(ctx, k, D, i1 - i0, c.nh_kv, 1, k->nb[1], k->nb[2], k->nb[3], i0 * k->nb[1]);
        ggml_tensor * vr = ggml_view_4d(ctx, v, D, i1 - i0, c.nh_kv, 1, v->nb[1], v->nb[2], v->nb[3], i0 * v->nb[1]);
        ggml_tensor * mr = ggml_view_4d(ctx, m, i1 - i0, c.nb, 1, 1, m->nb[1], m->nb[2], m->nb[3], i0 * m->nb[0]);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kr, vr, mr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        ggml_flash_attn_ext_set_emit_lse(fa, true);
        return fa;
    };
    std::vector<ggml_tensor *> parts;
    for (int i = 0; i < N; ++i) {
        parts.push_back(range(c.lo(i), c.hi(i)));
    }

    // the production 2-way recipe (llama-graph.cpp build_attn_cp), unchanged
    ggml_tensor * ops = nullptr;
    if (N == 2) {
        auto lse_of = [&](ggml_tensor * fa) {
            return ggml_cont(ctx, ggml_view_3d(ctx, fa, 1, nh, c.nb, fa->nb[1], fa->nb[2], D * sizeof(float)));
        };
        ggml_tensor * w  = ggml_soft_max(ctx, ggml_concat(ctx, lse_of(parts[0]), lse_of(parts[1]), 0)); // [2, nh, nb]
        ggml_tensor * w1 = ggml_cont(ctx, ggml_view_3d(ctx, w, 1, nh, c.nb, w->nb[1], w->nb[2], 0));
        ggml_tensor * w2 = ggml_cont(ctx, ggml_view_3d(ctx, w, 1, nh, c.nb, w->nb[1], w->nb[2], sizeof(float)));
        auto o_of = [&](ggml_tensor * fa) {
            return ggml_view_3d(ctx, fa, D, nh, c.nb, fa->nb[1], fa->nb[2], 0);
        };
        ops = ggml_add(ctx, ggml_mul(ctx, o_of(parts[0]), w1), ggml_mul(ctx, o_of(parts[1]), w2));
    }

    ggml_tensor * nway = ggml_flash_attn_merge(ctx, parts.data(), N, true);
    ggml_tensor * fold = fold_merge(ctx, parts);

    ggml_set_output(full);
    ggml_set_output(nway);
    ggml_set_output(fold);
    for (ggml_tensor * t : parts) {
        ggml_set_output(t);
    }
    if (ops) {
        ggml_set_output(ops);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, nway);
    ggml_build_forward_expand(gf, fold);
    if (ops) {
        ggml_build_forward_expand(gf, ops);
    }

    if (!all_supported(backend, gf)) {
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("    %s: allocation failed\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return res;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uq(-4.0f, 4.0f), ukv(-1.0f, 1.0f), um(-1.0f, 1.0f);

    res.q.resize(ggml_nelements(q));
    for (auto & x : res.q) { x = uq(rng); }
    ggml_backend_tensor_set(q, res.q.data(), 0, ggml_nbytes(q));

    // V carries a per-region spike (thirds of the KV range, one D column each) on top of the noise, so O has a
    // decidable argmax and the partials differ; the mask adds +2 to the last third's scores, so the partials'
    // true weights are far from equal (uniform data would make any merge weights look right)
    for (ggml_tensor * t : { k, v }) {
        std::vector<float> d(ggml_nelements(t));
        for (int64_t i = 0; i < (int64_t) d.size(); ++i) {
            const int64_t dd     = i % D;
            const int64_t j      = (i / D) % c.n_kv;
            const int64_t region = 3 * j / c.n_kv;
            d[i] = t == v ? 0.25f * ukv(rng) + (dd == (37 * region + 5) % D ? 1.0f : 0.0f) : ukv(rng);
        }
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(d.size());
            ggml_fp32_to_fp16_row(d.data(), h.data(), (int64_t) d.size());
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
            ggml_fp16_to_fp32_row(h.data(), d.data(), (int64_t) d.size()); // the host reference sees the rounded values
        } else {
            ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
        }
        (t == k ? res.k : res.v) = std::move(d);
    }

    // causal over the last nb positions (row i sees [0, n - nb + i]) plus a small random bias
    std::vector<ggml_fp16_t> md(ggml_nelements(m));
    res.m.resize(md.size());
    for (int64_t i = 0; i < c.nb; ++i) {
        for (int64_t j = 0; j < c.n_kv; ++j) {
            float val = j <= c.n_kv - c.nb + i ? um(rng) + (3 * j >= 2 * c.n_kv ? 2.0f : 0.0f) : -INFINITY;
            if (c.empty >= 0 && (c.empty_all || (i % 2) == 0) && j >= c.lo(c.empty) && j < c.hi(c.empty)) {
                val = -INFINITY;
            }
            md[i * c.n_kv + j] = ggml_fp32_to_fp16(val);
            res.m[i * c.n_kv + j] = ggml_fp16_to_fp32(md[i * c.n_kv + j]);
        }
    }
    ggml_backend_tensor_set(m, md.data(), 0, ggml_nbytes(m));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("    %s: compute failed\n", ggml_backend_name(backend));
    } else {
        auto get = [](ggml_tensor * t, std::vector<float> & out) {
            out.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        };
        get(full, res.full);
        get(nway, res.nway);
        get(fold, res.fold);
        if (ops) {
            get(ops, res.ops);
        }
        res.parts.resize(N);
        for (int i = 0; i < N; ++i) {
            get(parts[i], res.parts[i]);
        }
        res.ok = true;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

struct cmp_stats {
    double o_nmse       = 0.0;
    double o_max_abs    = 0.0;
    double lse_max_abs  = 0.0;
    double argmax_agree = 1.0; // over the decidable rows
    double decidable    = 0.0; // fraction of rows whose reference top-2 margin is above argmax_margin
    int    n_nonfinite  = 0;
};

static constexpr float argmax_margin = 1e-2f;

// O of `got` (row stride sg) vs O of `ref` (row stride sr); lse (column D) compared when both strides are D + 4
static cmp_stats compare(int64_t D, int64_t rows, const float * ref, int64_t sr, const float * got, int64_t sg) {
    cmp_stats st;
    int64_t n_dec = 0, n_agree = 0;
    double sq_err = 0.0, sq_ref = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        const float * f  = ref + r * sr;
        const float * mg = got + r * sg;
        int64_t af = 0, am = 0;
        for (int64_t d = 0; d < D; ++d) {
            if (!std::isfinite(mg[d]) || !std::isfinite(f[d])) {
                st.n_nonfinite++;
                continue;
            }
            sq_err += (double) (f[d] - mg[d]) * (f[d] - mg[d]);
            sq_ref += (double) f[d] * f[d];
            st.o_max_abs = std::max(st.o_max_abs, (double) std::fabs(f[d] - mg[d]));
            if (f[d]  > f[af])  { af = d; }
            if (mg[d] > mg[am]) { am = d; }
        }
        float second = -INFINITY;
        for (int64_t d = 0; d < D; ++d) {
            if (d != af) { second = std::max(second, f[d]); }
        }
        if (f[af] - second > argmax_margin) {
            n_dec++;
            n_agree += af == am;
        }
        if (sr == D + 4 && sg == D + 4) {
            if (std::isinf(f[D]) && std::isinf(mg[D]) && f[D] < 0 && mg[D] < 0) {
                continue;
            }
            if (!std::isfinite(f[D]) || !std::isfinite(mg[D])) {
                st.n_nonfinite++;
                continue;
            }
            st.lse_max_abs = std::max(st.lse_max_abs, (double) std::fabs(f[D] - mg[D]));
        }
    }
    st.o_nmse       = sq_err / std::max(sq_ref, 1e-30);
    st.decidable    = (double) n_dec / rows;
    st.argmax_agree = n_dec > 0 ? (double) n_agree / n_dec : 1.0;
    return st;
}

// f32 rounding agreement between two merges of the same partials: max over rows of max_d |a - b| / max_d |a| (row
// relative, so a near-zero element does not blow the ratio up), and max |lse_a - lse_b| / max(1, |lse_a|)
struct rel_stats {
    double o_rel   = 0.0;
    double lse_rel = 0.0;
    int    n_bad   = 0; // non-finite on one side only, or NaN
};

static rel_stats rel_compare(int64_t D, int64_t rows, const float * a, int64_t sa, const float * b, int64_t sb) {
    rel_stats st;
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = a + r * sa;
        const float * y = b + r * sb;
        double amax = 0.0, dmax = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            if (!std::isfinite(x[d]) || !std::isfinite(y[d])) {
                st.n_bad++;
                continue;
            }
            amax = std::max(amax, (double) std::fabs(x[d]));
            dmax = std::max(dmax, (double) std::fabs(x[d] - y[d]));
        }
        if (amax > 0.0) {
            st.o_rel = std::max(st.o_rel, dmax / amax);
        } else if (dmax > 0.0) {
            st.o_rel = std::max(st.o_rel, 1.0);
        }
        if (sa == D + 4 && sb == D + 4) {
            if (std::isinf(x[D]) || std::isinf(y[D])) {
                st.n_bad += !(std::isinf(x[D]) && std::isinf(y[D]) && x[D] < 0 && y[D] < 0);
                continue;
            }
            if (std::isnan(x[D]) || std::isnan(y[D])) {
                st.n_bad++;
                continue;
            }
            st.lse_rel = std::max(st.lse_rel, std::fabs((double) x[D] - y[D]) / std::max(1.0, std::fabs((double) x[D])));
        }
    }
    return st;
}

// host double softmax over the N lse: the definition the op implements, [D + 4] rows (lse -inf / O 0 when all empty)
static std::vector<float> host_merge(int64_t D, int64_t rows, const std::vector<const float *> & parts, int64_t sp) {
    std::vector<float> out(rows * (D + 4), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        double mx = -INFINITY;
        for (const float * p : parts) {
            mx = std::max(mx, (double) p[r * sp + D]);
        }
        float * o = out.data() + r * (D + 4);
        if (mx == -INFINITY) {
            o[D] = -INFINITY;
            continue;
        }
        std::vector<double> w(parts.size());
        double s = 0.0;
        for (size_t i = 0; i < parts.size(); ++i) {
            w[i] = exp((double) parts[i][r * sp + D] - mx);
            s += w[i];
        }
        for (int64_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (w[i] != 0.0) {
                    acc += w[i] / s * parts[i][r * sp + d];
                }
            }
            o[d] = (float) acc;
        }
        o[D] = (float) (mx + log(s));
    }
    return out;
}

// host double full attention (no split): the direct softmax reference for the full FA, small cases only
static std::vector<float> host_full_attn(const merge_case & c, const merge_result & r) {
    const int64_t D = c.hs, nh = c.nh_kv * c.gqa;
    const double scale = 1.0 / sqrt((double) D);
    std::vector<float> out(nh * c.nb * (D + 4), 0.0f);
    std::vector<double> s(c.n_kv), acc(D);
    for (int64_t t = 0; t < c.nb; ++t) {
        for (int64_t h = 0; h < nh; ++h) {
            const int64_t hk = h / c.gqa;
            const float * qr = r.q.data() + (h * c.nb + t) * D;
            double mx = -INFINITY;
            for (int64_t j = 0; j < c.n_kv; ++j) {
                const float mv = r.m[t * c.n_kv + j];
                if (mv == -INFINITY) { s[j] = -INFINITY; continue; }
                const float * kr = r.k.data() + (hk * c.n_kv + j) * D;
                double dot = 0.0;
                for (int64_t d = 0; d < D; ++d) { dot += (double) qr[d] * kr[d]; }
                s[j] = dot * scale + mv;
                mx = std::max(mx, s[j]);
            }
            float * o = out.data() + (t * nh + h) * (D + 4);
            if (mx == -INFINITY) { o[D] = -INFINITY; continue; }
            std::fill(acc.begin(), acc.end(), 0.0);
            double L = 0.0;
            for (int64_t j = 0; j < c.n_kv; ++j) {
                if (s[j] == -INFINITY) { continue; }
                const double e = exp(s[j] - mx);
                L += e;
                const float * vr = r.v.data() + (hk * c.n_kv + j) * D;
                for (int64_t d = 0; d < D; ++d) { acc[d] += e * vr[d]; }
            }
            for (int64_t d = 0; d < D; ++d) { o[d] = (float) (acc[d] / L); }
            o[D] = (float) (mx + log(L));
        }
    }
    return out;
}

// negative control: merge the partials on the host with equal weights instead of softmax(lse)
static std::vector<float> host_merge_equal(int64_t D, int64_t rows, const std::vector<std::vector<float>> & parts) {
    std::vector<float> out(rows * D, 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t d = 0; d < D; ++d) {
            for (const auto & p : parts) {
                out[r * D + d] += p[r * (D + 4) + d] / (float) parts.size();
            }
        }
    }
    return out;
}

// part 2: synthetic partials through ggml_flash_attn_merge, nway vs fold vs host double
struct synth_case {
    int64_t D, nh, nt;
    int     n_parts;
    float   spread;
    int     empty; // part with lse -inf (and O NaN) on every 3rd row, -1 none; every 17th row all parts empty
};

static bool run_synth(ggml_backend_t backend, const synth_case & c, uint32_t seed, int & n_run) {
    const int64_t rows = c.nh * c.nt, row = c.D + 4;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uo(-1.0f, 1.0f), ul(-c.spread, c.spread);
    std::vector<std::vector<float>> data(c.n_parts, std::vector<float>(rows * row));
    for (int i = 0; i < c.n_parts; ++i) {
        for (int64_t r = 0; r < rows; ++r) {
            float * x = data[i].data() + r * row;
            for (int64_t d = 0; d < c.D; ++d) { x[d] = uo(rng); }
            x[c.D] = ul(rng);
            x[c.D + 1] = x[c.D + 2] = x[c.D + 3] = 0.0f;
            if (r % 17 == 5 || (i == c.empty && r % 3 == 0)) {
                x[c.D] = -INFINITY;
                for (int64_t d = 0; d < c.D; ++d) { x[d] = NAN; }
            }
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> parts;
    for (int i = 0; i < c.n_parts; ++i) {
        parts.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, row, c.nh, c.nt));
    }
    ggml_tensor * nway = ggml_flash_attn_merge(ctx, parts.data(), c.n_parts, true);
    ggml_tensor * fold = fold_merge(ctx, parts);
    ggml_set_output(nway);
    ggml_set_output(fold);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, nway);
    ggml_build_forward_expand(gf, fold);

    printf("  synth D=%lld nh=%lld nt=%lld N=%d spread=%g empty=%d: ", (long long) c.D, (long long) c.nh, (long long) c.nt,
           c.n_parts, c.spread, c.empty);
    if (!all_supported(backend, gf)) {
        ggml_free(ctx);
        return true; // a backend without the op (Vulkan, N > 4) is not a failure
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int i = 0; i < c.n_parts; ++i) {
        ggml_backend_tensor_set(parts[i], data[i].data(), 0, ggml_nbytes(parts[i]));
    }
    bool pass = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> got_n(rows * row), got_f(rows * row);
    if (pass) {
        ggml_backend_tensor_get(nway, got_n.data(), 0, ggml_nbytes(nway));
        ggml_backend_tensor_get(fold, got_f.data(), 0, ggml_nbytes(fold));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    std::vector<const float *> ptrs;
    for (const auto & d : data) { ptrs.push_back(d.data()); }
    const std::vector<float> ref = host_merge(c.D, rows, ptrs, row);

    // empty rows: exactly O = 0, lse -inf
    int n_empty_bad = 0;
    for (int64_t r = 0; r < rows; r += 17) {
        if (r + 5 >= rows) { break; }
        const float * x = got_n.data() + (r + 5) * row;
        n_empty_bad += !(std::isinf(x[c.D]) && x[c.D] < 0);
        for (int64_t d = 0; d < c.D; ++d) { n_empty_bad += x[d] != 0.0f; }
    }
    const rel_stats vf = rel_compare(c.D, rows, got_n.data(), row, got_f.data(), row);
    const rel_stats vh = rel_compare(c.D, rows, ref.data(), row, got_n.data(), row);
    // [fork, 2026-10-23 12:39, Vulkan3 gate] the pairwise f32 fold is the less exact side at spread 200 (N=4 Vulkan3
    // O rel 1.67e-6, N=8 CPU 2.57e-6 vs fold, while both sit at 1.2e-7 / 1.8e-7 vs the f64 host merge): the fold
    // accumulates one rounding per merge, so it gets 1e-5; the f64 reference keeps 1e-6 and is the real criterion
    pass = pass && vf.n_bad == 0 && vh.n_bad == 0 && n_empty_bad == 0 && vf.o_rel <= 1e-5 && vf.lse_rel <= 1e-5 &&
           vh.o_rel <= 1e-6 && vh.lse_rel <= 1e-6;
    printf("%-8s nway vs fold: O rel %.2e lse rel %.2e | nway vs host f64: O rel %.2e lse rel %.2e | bad %d/%d/%d -> %s\n",
           ggml_backend_name(backend), vf.o_rel, vf.lse_rel, vh.o_rel, vh.lse_rel, vf.n_bad, vh.n_bad, n_empty_bad, pass ? "OK" : "FAIL");
    n_run++;
    return pass;
}

int main(int argc, char ** argv) {
    std::vector<std::string> extra;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            extra.push_back(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [-b <device name>]...\n", argv[0]);
            return 2;
        }
    }

    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    std::vector<ggml_backend_t> backends = { ggml_backend_dev_init(cpu_dev, nullptr) };
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_threads) {
            set_threads(backends[0], 8);
        }
    }
    for (const auto & name : extra) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) {
            fprintf(stderr, "device %s not found\n", name.c_str());
            return 1;
        }
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
    }

    std::vector<merge_case> cases;
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (int64_t n_kv : { 1024, 32768 }) {
            for (int64_t nb : { 1, 6, 12, 33 }) {
                cases.push_back({ 256, 4, 6, n_kv, nb, { n_kv / 2 }, -1, false, t });
            }
            // uneven split (unaligned range lengths) and a fully masked second range on half the rows
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 4 + 7 }, -1, false, t });
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 2 },      1,  false, t });
            // N = 3 (owner + two helpers, the production target) and N = 4
            for (int64_t nb : { 1, 12, 33 }) {
                cases.push_back({ 256, 4, 6, n_kv, nb, { n_kv / 3, 2 * n_kv / 3 }, -1, false, t });
            }
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 4, n_kv / 2, 3 * n_kv / 4 }, -1, false, t });
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 5 + 3, n_kv / 2 + 11 }, -1, false, t });
            // one helper empty: half the rows (middle range) and every row (last range: KV beyond n_kv for this ubatch)
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 3, 2 * n_kv / 3 }, 1, false, t });
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 3, 2 * n_kv / 3 }, 2, true,  t });
            cases.push_back({ 256, 4, 6, n_kv, 12, { n_kv / 4, n_kv / 2, 3 * n_kv / 4 }, 3, true, t });
        }
        cases.push_back({ 128, 8, 1, 4096, 1,  { 2048 }, -1, false, t });
        cases.push_back({ 128, 8, 1, 4096, 33, { 1000 },  1, false, t });
        cases.push_back({ 128, 8, 1, 4096, 33, { 1000, 3000 }, 1, false, t });
    }

    // tolerances vs the full FA, measured 2026-10-21 (CPU + Vulkan3 7900 XTX cm1): merged vs full O nmse <= 1e-12 on
    // CPU, <= 3e-6 on Vulkan (max abs 4.9e-3 at f32 K/V kv 32768 nb 33, P is f16 there); lse <= 1.1e-5. The
    // equal-weights control lands at nmse >= 0.24. Before the CPU FA accumulated f16 V in f32 (ops.cpp v_acc_f16)
    // the CPU reference itself drifted to nmse 3.6e-3 at kv 32768 (1.1e-1 against Vulkan).
    const double o_nmse_tol = 1e-4;
    const double o_tol      = 1e-2;
    const double lse_tol    = 5e-3;
    const double argmax_tol = 1.0;
    // merge vs merge of the same partials (nway / fold / ops / host f64): f32 rounding only, row-relative
    const double rel_tol    = 1e-6;

    int n_fail = 0, n_run = 0;
    uint32_t seed = 1234;
    for (const auto & c : cases) {
        std::string hs;
        for (int64_t x : c.h) { hs += (hs.empty() ? "" : ",") + std::to_string(x); }
        printf("case hs=%lld nh=%lldx%lld n_kv=%lld nb=%lld N=%d h=%s empty=%d%s kv=%s\n",
               (long long) c.hs, (long long) c.nh_kv, (long long) c.gqa, (long long) c.n_kv, (long long) c.nb,
               c.n_parts(), hs.c_str(), c.empty, c.empty_all ? "(all rows)" : "", ggml_type_name(c.kv_type));
        const int64_t D = c.hs, rows = c.nh_kv * c.gqa * c.nb;
        merge_result ref;
        for (size_t bi = 0; bi < backends.size(); ++bi) {
            merge_result r = run_case(backends[bi], c, seed);
            if (!r.ok) {
                n_fail++;
                continue;
            }
            auto o_ok = [&](const cmp_stats & x) {
                return x.n_nonfinite == 0 && x.o_nmse <= o_nmse_tol && x.o_max_abs <= o_tol && x.argmax_agree >= argmax_tol;
            };
            const cmp_stats self = compare(D, rows, r.full.data(), D + 4, r.nway.data(), D + 4);
            bool pass = o_ok(self) && self.lse_max_abs <= lse_tol;
            printf("    %-8s nway vs full: O nmse %.2e max abs %.2e, lse max abs %.2e, argmax %.4f of %.2f decidable, non-finite %d",
                   ggml_backend_name(backends[bi]), self.o_nmse, self.o_max_abs, self.lse_max_abs, self.argmax_agree, self.decidable, self.n_nonfinite);

            std::vector<const float *> ptrs;
            for (const auto & p : r.parts) { ptrs.push_back(p.data()); }
            const std::vector<float> hm = host_merge(D, rows, ptrs, D + 4);
            const rel_stats vf = rel_compare(D, rows, r.nway.data(), D + 4, r.fold.data(), D + 4);
            const rel_stats vh = rel_compare(D, rows, hm.data(), D + 4, r.nway.data(), D + 4);
            printf("\n             nway vs fold: O rel %.2e lse rel %.2e | nway vs host f64 merge: O rel %.2e lse rel %.2e",
                   vf.o_rel, vf.lse_rel, vh.o_rel, vh.lse_rel);
            pass = pass && vf.n_bad == 0 && vh.n_bad == 0 && vf.o_rel <= rel_tol && vf.lse_rel <= rel_tol &&
                   vh.o_rel <= rel_tol && vh.lse_rel <= rel_tol;
            if (!r.ops.empty()) {
                const rel_stats vo = rel_compare(D, rows, r.ops.data(), D, r.nway.data(), D + 4);
                const cmp_stats so = compare(D, rows, r.full.data(), D + 4, r.ops.data(), D);
                printf(" | ops vs nway: O rel %.2e | ops vs full: O nmse %.2e", vo.o_rel, so.o_nmse);
                pass = pass && vo.n_bad == 0 && vo.o_rel <= rel_tol && o_ok(so);
            }

            const cmp_stats ctrl = compare(D, rows, r.full.data(), D + 4, host_merge_equal(D, rows, r.parts).data(), D);
            // with an empty partial on half the rows or more the equal-weights control is not meaningful
            const bool ctrl_rejected = !o_ok(ctrl) || c.empty >= 0;
            printf("\n             equal-weight control: nmse %.2e%s", ctrl.o_nmse, ctrl_rejected ? "" : " NOT REJECTED");
            pass = pass && ctrl_rejected;

            if (bi == 0 && c.n_kv <= 4096) {
                const std::vector<float> hf = host_full_attn(c, r);
                const cmp_stats xf = compare(D, rows, hf.data(), D + 4, r.full.data(), D + 4);
                const cmp_stats xn = compare(D, rows, hf.data(), D + 4, r.nway.data(), D + 4);
                printf(" | host f64 full attn vs full: O nmse %.2e lse %.2e, vs nway: O nmse %.2e lse %.2e",
                       xf.o_nmse, xf.lse_max_abs, xn.o_nmse, xn.lse_max_abs);
                pass = pass && o_ok(xf) && o_ok(xn) && xf.lse_max_abs <= lse_tol && xn.lse_max_abs <= lse_tol;
            }
            if (bi == 0) {
                ref = r;
            } else if (ref.ok) {
                const cmp_stats x = compare(D, rows, ref.full.data(), D + 4, r.nway.data(), D + 4);
                printf(" | nway vs CPU full: O nmse %.2e max abs %.2e, argmax %.4f", x.o_nmse, x.o_max_abs, x.argmax_agree);
                pass = pass && o_ok(x);
            }
            printf(" -> %s\n", pass ? "OK" : "FAIL");
            n_fail += !pass;
            n_run++;
        }
        seed++;
    }

    printf("synthetic partials\n");
    std::vector<synth_case> synth;
    for (int n : { 2, 3, 4, 8 }) {
        for (float spread : { 0.0f, 1.0f, 30.0f, 200.0f, 1e4f }) {
            synth.push_back({ 256, 24, 12, n, spread, -1 });
        }
        for (int e = 0; e < n; ++e) {
            synth.push_back({ 256, 24, 12, n, 4.0f, e });
        }
        synth.push_back({ 128, 32, 33, n, 30.0f, n - 1 });
    }
    for (const auto & c : synth) {
        for (ggml_backend_t b : backends) {
            if (!run_synth(b, c, seed++, n_run)) {
                n_fail++;
            }
        }
    }

    for (ggml_backend_t b : backends) {
        ggml_backend_free(b);
    }

    printf("%d/%d passed\n", n_run - n_fail, n_run);
    return n_fail == 0 ? 0 : 1;
}
