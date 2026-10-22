// Context-parallel attention merge check (ggml_flash_attn_ext_set_emit_lse, see ggml.h).
//
// Two FLASH_ATTN_EXT calls over the KV ranges [0, h) and [h, n), each emitting its lse, are merged with plain ggml
// ops (concat the two lse columns -> soft_max over the range axis -> scale each partial O by its weight -> add) and
// must match one FLASH_ATTN_EXT over [0, n): NMSE and max abs error on O, argmax agreement over the D dimension
// (on the rows whose top-2 margin is above the kernel noise), and the host-side logsumexp of the two partial lse
// against the full lse. The second range's mask is the column view mask[:, h:n] of the full mask, which exercises
// the strided-mask path.
//
// As a negative control, every case also merges the partials on the host with equal weights (0.5 / 0.5), which must
// FAIL the same criteria, so the tolerances cannot hide a wrong merge (the data is built so that equal weights are
// far from the true ones, see run_case).
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
    int64_t h;        // split point
    bool    mask_tail; // fully mask [h, n) for the even rows: the second partial is -inf / O = 0 there
    ggml_type kv_type;
};

struct merge_result {
    std::vector<float> full;   // [hs + 4, nh, nb]
    std::vector<float> merged; // [hs, nh, nb]
    std::vector<float> p1;     // [hs + 4, nh, nb]
    std::vector<float> p2;
    bool ok = false;
};

static merge_result run_case(ggml_backend_t backend, const merge_case & c, uint32_t seed) {
    merge_result res;

    const int64_t nh = c.nh_kv * c.gqa;
    const int64_t D  = c.hs;

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
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
    ggml_tensor * p1 = range(0, c.h);
    ggml_tensor * p2 = range(c.h, c.n_kv);

    // lse_i: [1, nh, nb]; w = softmax over the 2 ranges
    auto lse_of = [&](ggml_tensor * fa) {
        return ggml_cont(ctx, ggml_view_3d(ctx, fa, 1, nh, c.nb, fa->nb[1], fa->nb[2], D * sizeof(float)));
    };
    ggml_tensor * w  = ggml_soft_max(ctx, ggml_concat(ctx, lse_of(p1), lse_of(p2), 0)); // [2, nh, nb]
    ggml_tensor * w1 = ggml_cont(ctx, ggml_view_3d(ctx, w, 1, nh, c.nb, w->nb[1], w->nb[2], 0));
    ggml_tensor * w2 = ggml_cont(ctx, ggml_view_3d(ctx, w, 1, nh, c.nb, w->nb[1], w->nb[2], sizeof(float)));

    auto o_of = [&](ggml_tensor * fa) {
        return ggml_view_3d(ctx, fa, D, nh, c.nb, fa->nb[1], fa->nb[2], 0);
    };
    ggml_tensor * merged = ggml_add(ctx, ggml_mul(ctx, o_of(p1), w1), ggml_mul(ctx, o_of(p2), w2));

    for (ggml_tensor * t : { full, p1, p2, merged }) {
        ggml_set_output(t);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, merged);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, t)) {
            printf("    %s: %s (%s) not supported\n", ggml_backend_name(backend), ggml_op_desc(t), t->name);
            ggml_free(ctx);
            return res;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("    %s: allocation failed\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return res;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uq(-4.0f, 4.0f), ukv(-1.0f, 1.0f), um(-1.0f, 1.0f);

    std::vector<float> qd(ggml_nelements(q));
    for (auto & x : qd) { x = uq(rng); }
    ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));

    // V carries a per-region spike (thirds of the KV range, one D column each) on top of the noise, so O has a
    // decidable argmax and the two partials differ; the mask adds +2 to the last third's scores, so the partials'
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
        } else {
            ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
        }
    }

    // causal over the last nb positions (row i sees [0, n - nb + i]) plus a small random bias
    std::vector<ggml_fp16_t> md(ggml_nelements(m));
    for (int64_t i = 0; i < c.nb; ++i) {
        for (int64_t j = 0; j < c.n_kv; ++j) {
            float val = j <= c.n_kv - c.nb + i ? um(rng) + (3 * j >= 2 * c.n_kv ? 2.0f : 0.0f) : -INFINITY;
            if (c.mask_tail && (i % 2) == 0 && j >= c.h) {
                val = -INFINITY;
            }
            md[i * c.n_kv + j] = ggml_fp32_to_fp16(val);
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
        get(merged, res.merged);
        get(p1, res.p1);
        get(p2, res.p2);
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
    double decidable    = 0.0; // fraction of rows whose full-O top-2 margin is above argmax_margin
    int    n_nonfinite  = 0;
};

static constexpr float argmax_margin = 1e-2f;

// merged O (row stride D) vs the O columns of a full FA output (row stride D + 4); lse: host logsumexp of the
// partials vs the full lse
static cmp_stats compare(const merge_case & c, const std::vector<float> & full, const std::vector<float> & merged,
                         const std::vector<float> * p1, const std::vector<float> * p2) {
    cmp_stats st;
    const int64_t D    = c.hs;
    const int64_t rows = c.nh_kv * c.gqa * c.nb;
    int64_t n_dec = 0, n_agree = 0;
    double sq_err = 0.0, sq_ref = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        const float * f  = full.data()   + r * (D + 4);
        const float * mg = merged.data() + r * D;
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
        if (p1 && p2) {
            const float l1 = (*p1)[r * (D + 4) + D];
            const float l2 = (*p2)[r * (D + 4) + D];
            const float mx = std::max(l1, l2);
            const float lse = mx == -INFINITY ? -INFINITY : mx + logf(expf(l1 - mx) + expf(l2 - mx));
            if (std::isinf(lse) && std::isinf(f[D])) {
                continue;
            }
            if (!std::isfinite(lse) || !std::isfinite(f[D])) {
                st.n_nonfinite++;
                continue;
            }
            st.lse_max_abs = std::max(st.lse_max_abs, (double) std::fabs(lse - f[D]));
        }
    }
    st.o_nmse       = sq_err / std::max(sq_ref, 1e-30);
    st.decidable    = (double) n_dec / rows;
    st.argmax_agree = n_dec > 0 ? (double) n_agree / n_dec : 1.0;
    return st;
}

// negative control: merge the partials on the host with equal weights instead of softmax(lse)
static std::vector<float> host_merge_equal(const merge_case & c, const std::vector<float> & p1, const std::vector<float> & p2) {
    const int64_t D    = c.hs;
    const int64_t rows = c.nh_kv * c.gqa * c.nb;
    std::vector<float> out(rows * D);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t d = 0; d < D; ++d) {
            out[r * D + d] = 0.5f * p1[r * (D + 4) + d] + 0.5f * p2[r * (D + 4) + d];
        }
    }
    return out;
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
                cases.push_back({ 256, 4, 6, n_kv, nb, n_kv / 2, false, t });
            }
            // uneven split (unaligned range lengths) and a fully masked second range on half the rows
            cases.push_back({ 256, 4, 6, n_kv, 12, n_kv / 4 + 7, false, t });
            cases.push_back({ 256, 4, 6, n_kv, 12, n_kv / 2,     true,  t });
        }
        cases.push_back({ 128, 8, 1, 4096, 1,  2048, false, t });
        cases.push_back({ 128, 8, 1, 4096, 33, 1000, true,  t });
    }

    // tolerances, measured 2026-10-21 (CPU + Vulkan3 7900 XTX cm1): merged vs full O nmse <= 1e-12 on CPU, <= 3e-6 on
    // Vulkan (max abs 4.9e-3 at f32 K/V kv 32768 nb 33, P is f16 there); lse <= 1.1e-5. The equal-weights control
    // lands at nmse >= 0.24. Before the CPU FA accumulated f16 V in f32 (ops.cpp v_acc_f16) the CPU reference itself
    // drifted to nmse 3.6e-3 at kv 32768 (1.1e-1 against Vulkan).
    const double o_nmse_tol = 1e-4;
    const double o_tol      = 1e-2;
    const double lse_tol    = 5e-3;
    const double argmax_tol = 1.0;

    int n_fail = 0, n_run = 0;
    uint32_t seed = 1234;
    for (const auto & c : cases) {
        printf("case hs=%lld nh=%lldx%lld n_kv=%lld nb=%lld h=%lld mask_tail=%d kv=%s\n",
               (long long) c.hs, (long long) c.nh_kv, (long long) c.gqa, (long long) c.n_kv, (long long) c.nb,
               (long long) c.h, (int) c.mask_tail, ggml_type_name(c.kv_type));
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
            const cmp_stats self = compare(c, r.full, r.merged, &r.p1, &r.p2);
            bool pass = o_ok(self) && self.lse_max_abs <= lse_tol;
            printf("    %-8s merged vs full: O nmse %.2e max abs %.2e, lse max abs %.2e, argmax %.4f of %.2f decidable, non-finite %d",
                   ggml_backend_name(backends[bi]), self.o_nmse, self.o_max_abs, self.lse_max_abs, self.argmax_agree, self.decidable, self.n_nonfinite);
            const cmp_stats ctrl = compare(c, r.full, host_merge_equal(c, r.p1, r.p2), nullptr, nullptr);
            const bool ctrl_rejected = !o_ok(ctrl) || c.mask_tail;
            printf(" | equal-weight control: nmse %.2e%s", ctrl.o_nmse, ctrl_rejected ? "" : " NOT REJECTED");
            pass = pass && ctrl_rejected;
            if (bi == 0) {
                ref = r;
            } else if (ref.ok) {
                const cmp_stats x = compare(c, ref.full, r.merged, nullptr, nullptr);
                printf(" | merged vs CPU full: O nmse %.2e max abs %.2e, argmax %.4f", x.o_nmse, x.o_max_abs, x.argmax_agree);
                pass = pass && o_ok(x);
            }
            printf(" -> %s\n", pass ? "OK" : "FAIL");
            n_fail += !pass;
            n_run++;
        }
        seed++;
    }

    for (ggml_backend_t b : backends) {
        ggml_backend_free(b);
    }

    printf("%d/%d passed\n", n_run - n_fail, n_run);
    return n_fail == 0 ? 0 : 1;
}
