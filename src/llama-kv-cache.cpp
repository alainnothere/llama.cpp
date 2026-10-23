#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag,
                   bool   kq_mask_inc) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer + 6)*ggml_tensor_overhead()), // +1: [KQ_MASK_INC] kq_mask_dev, +4: [CP] range 1 K/V + check reference K/V
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    // 2026-10-20: layer indices per buffer type, consumed by the CPU PLACEMENT warning and (2026-10-21) the
    // incremental KQ mask placement
    std::map<ggml_backend_buffer_type_t, std::vector<int>> buft_layers;

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    // [CP] context-parallel attention, range-split KV, see cp_state in llama-kv-cache.h. Decided before the layer loop
    // because the CP layer's range 0 replaces its K/V tensor. The helper is a FIXED device (LLAMA_CP_DEV, mandatory) and
    // range 1 is allocated there whenever the layer is in this cache, whatever device the layer itself landed on: the
    // fit probes this constructor with many placements; 2026-10-21 night the helper was "the other GPU with the owner's
    // description", so its share moved (or vanished) between probes: Vulkan1 was sized while layer 27 sat on the CPU,
    // then the owner moved to Vulkan3 and Vulkan1 grew by 1156 MiB after its share was fixed -> both XTX over budget,
    // GTT placement, every step ~3x slower. Constant memory per device = honest fit (memory_breakdown counts every range
    // tensor on its own buffer type in the no_alloc probes, common/fit.cpp needs no change).
    int32_t            cp_il   = -1;
    ggml_backend_dev_t cp_devb = nullptr;
    if (const char * env_il = getenv("LLAMA_CP_LAYER"); env_il && atoi(env_il) >= 0 && (uint32_t) atoi(env_il) < n_layer) {
        const int32_t il = atoi(env_il);

        const char * env_dev = getenv("LLAMA_CP_DEV");
        if (!env_dev || !*env_dev) {
            throw std::runtime_error("LLAMA_CP_LAYER is set but LLAMA_CP_DEV is not: set LLAMA_CP_DEV to the helper device name (e.g. Vulkan1), one of the model's devices, or CPU for the CPU test mode");
        }
        ggml_backend_dev_t dev_b = nullptr;
        if (strcmp(env_dev, "CPU") == 0) {
            // CPU test mode (2026-10-22 night): both ranges in host memory, the whole CP graph on the CPU backend
            dev_b = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else {
            std::string names;
            for (const auto & d : model.devices) {
                names += std::string(names.empty() ? "" : ", ") + ggml_backend_dev_name(d.dev);
                if (strcmp(ggml_backend_dev_name(d.dev), env_dev) == 0) {
                    dev_b = d.dev;
                }
            }
            if (!dev_b) {
                throw std::runtime_error(std::string("LLAMA_CP_DEV=") + env_dev + " is not a device of this model's split (" + (names.empty() ? std::string("none, CPU only; LLAMA_CP_DEV=CPU is the CPU test mode") : names) + ")");
            }
            if (ggml_backend_dev_type(dev_b) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                throw std::runtime_error(std::string("LLAMA_CP_DEV=") + env_dev + " is not a GPU");
            }
        }

        const char * why_not = nullptr;
        if (!offload)                             why_not = "KV offload off";
        else if (v_trans)                         why_not = "no flash attention";
        else if (n_stream != 1)                   why_not = "more than one stream";
        else if (swa_type != LLAMA_SWA_TYPE_NONE) why_not = "SWA cache";
        else if (is_mla)                          why_not = "MLA";
        else if (kv_size < 512)                   why_not = "kv_size < 512 (each range needs >= 256 cells)";

        if (const char * env_cell = getenv("LLAMA_CP_SPLIT_CELL"); !why_not && env_cell && (atoi(env_cell) <= 0 || (uint32_t) atoi(env_cell) >= kv_size)) {
            why_not = "LLAMA_CP_SPLIT_CELL outside (0, kv_size)";
        }
        // the CPU test mode runs the stock tests at n_ctx 256 with LLAMA_CP_SPLIT_CELL=128
        if (why_not && kv_size < 512 && getenv("LLAMA_CP_SPLIT_CELL") && atoi(getenv("LLAMA_CP_SPLIT_CELL")) > 0 && (uint32_t) atoi(getenv("LLAMA_CP_SPLIT_CELL")) < kv_size &&
                offload && !v_trans && n_stream == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !is_mla) {
            why_not = nullptr;
        }

        if (why_not) {
            LLAMA_LOG_WARN("%s: context-parallel attention off for layer %d: %s\n", __func__, il, why_not);
        } else {
            cp_il   = il;
            cp_devb = dev_b;

            const char * env_frac = getenv("LLAMA_CP_SPLIT_FRAC");
            const char * env_cell = getenv("LLAMA_CP_SPLIT_CELL");
            const char * env_chk  = getenv("LLAMA_CP_CHECK");
            const char * env_pack = getenv("LLAMA_CP_PACK");
            cp.frac  = env_frac ? std::clamp((float) atof(env_frac), 0.0f, 1.0f) : 0.5f;
            cp.check = env_chk && atoi(env_chk) != 0;
            cp.pack  = !env_pack || atoi(env_pack) != 0;
            if (env_cell) {
                cp.h = (uint32_t) atoi(env_cell);
            } else {
                // 256: n_kv is padded to max(n_pad, 256) (get_n_kv) and every FA KV tile divides 256, so both FA
                // lengths (h, n_kv - h) stay tile aligned, the invariant the 2026-10-21 cp_split had
                constexpr uint32_t pad = 256;
                cp.h = std::clamp((uint32_t) std::lround(cp.frac*kv_size/pad)*pad, pad, kv_size - pad);
            }
        }
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            // [CP] a CP source layer is split into ranges: a sharer would view range 0 as the whole cache
            if (il_share >= 0 && other->cp_active(il_share)) {
                LLAMA_LOG_WARN("%s: layer %3d: not sharing with layer %d, it is a context-parallel (range-split) layer\n", __func__, il, il_share);
            } else if (il_share >= 0) {
                // Check if backend supports cross-context tensor sharing
                auto * dev = model.dev_layer(il);
                ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
                const char * backend_name = ggml_backend_buft_name(buft);
                
                // Vulkan doesn't support cross-context pre-allocated tensors
                if (strstr(backend_name, "Vulkan") == nullptr) {
                    const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                    LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                            layer_share.k->data, layer_share.v->data);

                    map_layer_ids[il] = layers.size();

                    layers.push_back(layer_share);
                    layers.back().il = il;
                    layers.back().kq_mask_dev = nullptr; // the source cache's mask follows the source's uploads

                    continue;
                } else {
                    LLAMA_LOG_WARN("%s: layer %3d: skipping cross-context sharing on Vulkan backend\n", __func__, il);
                }
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }
        buft_layers[buft].push_back((int) il); // for the CPU PLACEMENT warning below

        const bool has_k = true;
        const bool has_v = !is_mla;

        // [CP] the CP layer's tensor is range 0 = cells [0, h) + the trash row (see cp_state); range 1 and the check
        // reference are allocated next to it below. kv_size here stays the cell count every other path uses.
        const bool     cp_here  = (int32_t) il == cp_il && has_k && has_v;
        const uint32_t n_rows_0 = cp_here ? cp.h + 1 : kv_size;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, n_rows_0, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, n_rows_0, n_stream) : nullptr;

        has_k && ggml_format_name(k, cp_here ? "cache_%sk_l%d_r0" : "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, cp_here ? "cache_%sv_l%d_r0" : "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, n_rows_0, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, n_rows_0, v->nb[1], s*v->nb[2]) : nullptr);
        }

        if (cp_here) {
            cp.il = il;

            cp_range r0;
            r0.c0   = 0;
            r0.c1   = cp.h;
            r0.dev  = offload ? model.dev_layer(il) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            r0.buft = buft;
            r0.k    = k;
            r0.v    = v;

            cp_range r1;
            r1.c0   = cp.h;
            r1.c1   = kv_size;
            r1.dev  = cp_devb;
            r1.buft = ggml_backend_dev_buffer_type(cp_devb);

            ggml_context * ctx_b = ctx_for_buft(r1.buft);
            if (!ctx_b) {
                throw std::runtime_error("failed to create ggml context for the kv cache helper range");
            }
            r1.k = ggml_new_tensor_3d(ctx_b, type_k, n_embd_k_gqa, r1.c1 - r1.c0 + 1, 1);
            r1.v = ggml_new_tensor_3d(ctx_b, type_v, n_embd_v_gqa, r1.c1 - r1.c0 + 1, 1);
            ggml_format_name(r1.k, "cache_%sk_l%d_r1", name_tag, il);
            ggml_format_name(r1.v, "cache_%sv_l%d_r1", name_tag, il);

            // B needs its own persistent mask even when it holds no other attention layer
            buft_layers[r1.buft];

            if (cp.check) {
                cp.ref_k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, 1);
                cp.ref_v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, 1);
                ggml_format_name(cp.ref_k, "cache_%sk_l%d_cpref", name_tag, il);
                ggml_format_name(cp.ref_v, "cache_%sv_l%d_cpref", name_tag, il);
            }

            cp.ranges = { r0, r1 };

            // the fit probes this constructor with no_alloc many times: the decision is loud only for the real cache
            auto mib = [](const ggml_tensor * a, const ggml_tensor * b) { return (ggml_nbytes(a) + ggml_nbytes(b))/1024.0/1024.0; };
            const ggml_log_level lvl = hparams.no_alloc ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_WARN;
            llama_log_internal(lvl, "%s: context-parallel attention on layer %d: range split at cell %u of %u (frac %.3f): owner %s cells [0, %u) %.2f MiB, helper %s cells [%u, %u) %.2f MiB%s; check %d%s, pack %d\n",
                    __func__, il, cp.h, kv_size, (double) cp.h/kv_size,
                    ggml_backend_dev_name(r0.dev), cp.h, mib(r0.k, r0.v),
                    ggml_backend_dev_name(r1.dev), cp.h, kv_size, mib(r1.k, r1.v),
                    r0.dev == r1.dev ? " (same device: no overlap)" : "",
                    cp.check, cp.check ? " (+ full reference on the owner)" : "", cp.pack);
            if (cp.h % 256 != 0 && (ggml_backend_dev_type(r0.dev) != GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_dev_type(r1.dev) != GGML_BACKEND_DEVICE_TYPE_CPU)) {
                llama_log_internal(lvl, "%s: context-parallel split cell %u is not a multiple of 256: the FA ranges are not tile aligned on the GPU\n", __func__, cp.h);
            }
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // [CP] a layer that reuses the CP layer's K/V would read it as one tensor (get_k) above h
    if (cp.active()) {
        for (const auto & [il_m, ikv] : map_layer_ids) {
            if (il_m != cp.il && ikv == map_layer_ids.at(cp.il)) {
                throw std::runtime_error("LLAMA_CP_LAYER=" + std::to_string(cp.il) + ": layer " + std::to_string(il_m) + " reuses its K/V, not supported with context-parallel attention");
            }
        }
    }

    // loud only when the layer is a K/V layer of this cache that could not be split (shared); a cache that does not hold
    // the layer at all (the MTP / draft cache next to the target) stays quiet
    if (cp_il >= 0 && !cp.active()) {
        const bool here = map_layer_ids.find(cp_il) != map_layer_ids.end();
        llama_log_internal(here ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_DEBUG, "%s: context-parallel attention off for layer %d: %s\n", __func__, cp_il,
                here ? "the layer shares or reuses another layer's K/V" : "not a K/V layer of this cache");
    }

    // [KQ_MASK_DEV] static gate of the device-built mask (see kq_build_state); decided before 6a so its
    // persistent tensors are not allocated when the op replaces them. The anchors are picked after allocation.
    {
        const char * env_dev = getenv("LLAMA_KQ_MASK_DEV");
        const char * env_chk = getenv("LLAMA_KQ_MASK_DEV_CHECK");

        const char * why_not = nullptr;
        if (!kq_mask_inc)                         why_not = "not read by the graphs of this cache (wrapper)";
        else if (env_dev && atoi(env_dev) == 0)   why_not = "LLAMA_KQ_MASK_DEV=0";
        else if (v_trans)                         why_not = "no flash attention (f32 mask)";
        else if (n_stream != 1)                   why_not = "more than one stream";
        else if (swa_type != LLAMA_SWA_TYPE_NONE) why_not = "SWA cache";
        else if (hparams.use_alibi)               why_not = "ALiBi";
        else if (n_seq_max > 32)                  why_not = "n_seq_max > 32 (one seq bitmap word per cell)";
        else if (layers.empty())                  why_not = "no layers";

        // every attention buffer type's device must run the op, else its layers would read a CPU-built mask
        // through a cross-device copy; probed with a throwaway node (supports_op reads types/shapes only)
        if (!why_not) {
            ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr pctx { ggml_init(ip) };
            ggml_tensor * pc = ggml_new_tensor_2d(pctx.get(), GGML_TYPE_I32, 2, 256);
            ggml_tensor * pr = ggml_new_tensor_2d(pctx.get(), GGML_TYPE_I32, 2, 1);
            ggml_tensor * pa = ggml_new_tensor_1d(pctx.get(), GGML_TYPE_F32, 1);
            ggml_tensor * op = ggml_kq_mask_build(pctx.get(), pc, pr, pa, 256, 256, 1, true);
            for (const auto & [buft, lids] : buft_layers) {
                ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                if (!dev || !ggml_backend_dev_supports_op(dev, op)) {
                    why_not = "a device holding attention layers does not support GGML_OP_KQ_MASK_BUILD";
                    break;
                }
            }
        }

        kq_build.enabled = why_not == nullptr;
        kq_build.check   = kq_build.enabled && env_chk && atoi(env_chk) != 0;

        const ggml_log_level lvl = hparams.no_alloc ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_WARN;
        if (kq_build.enabled) {
            llama_log_internal(lvl, "%s: device-built KQ mask on (%zu attention buffer types, check %d); LLAMA_KQ_MASK_DEV=0 for the host mask\n",
                    __func__, buft_layers.size(), kq_build.check);
        } else if (kq_mask_inc) {
            LLAMA_LOG_DEBUG("%s: device-built KQ mask off: %s\n", __func__, why_not);
        }
    }

    // [KQ_MASK_INC] static gate + one persistent mask per buffer type that holds attention layers
    // (see kq_mask_inc_state). !v_trans <=> flash attention, the only path with an f16 mask
    {
        const char * env_inc  = getenv("LLAMA_KQ_MASK_INCREMENTAL");
        const char * env_rows = getenv("LLAMA_KQ_MASK_INC_ROWS");
        const char * env_chk  = getenv("LLAMA_KQ_MASK_CHECK");

        const bool on = !(env_inc && atoi(env_inc) == 0);
        const int  n_rows = env_rows ? atoi(env_rows) : 64;

        const char * why_not = nullptr;
        if (!kq_mask_inc)                         why_not = "not read by the graphs of this cache (wrapper)";
        else if (kq_build.enabled)                why_not = "replaced by the device-built mask (LLAMA_KQ_MASK_DEV)"; // saves kv_size x 64 x 2 B per device
        else if (!on)                             why_not = "LLAMA_KQ_MASK_INCREMENTAL=0";
        else if (n_rows <= 0)                     why_not = "LLAMA_KQ_MASK_INC_ROWS <= 0";
        else if (v_trans)                         why_not = "no flash attention (f32 mask)";
        else if (n_stream != 1)                   why_not = "more than one stream";
        else if (swa_type != LLAMA_SWA_TYPE_NONE) why_not = "SWA cache";
        else if (hparams.use_alibi)               why_not = "ALiBi";
        else if (layers.empty())                  why_not = "no layers";

        if (!why_not) {
            kq_inc.slot = v_cells[0].dirty_add_slot();
            if (kq_inc.slot < 0) {
                why_not = "no free dirty slot in the shared cells";
            }
        }

        if (why_not) {
            LLAMA_LOG_DEBUG("%s: incremental KQ mask off: %s\n", __func__, why_not);
        } else {
            kq_inc.enabled = true;
            kq_inc.check   = env_chk && atoi(env_chk) != 0;
            kq_inc.n_rows  = (uint32_t) n_rows;
            kq_inc.row_lo.assign(kq_inc.n_rows, 0);

            for (auto & [buft, ctx] : ctx_map) {
                const auto it = buft_layers.find(buft);
                if (it == buft_layers.end()) {
                    continue;
                }

                ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, (int64_t) kv_size*kq_inc.n_rows);
                ggml_format_name(t, "cache_%skq_mask_dev", name_tag);
                kq_inc.dev.push_back(t);

                for (int il : it->second) {
                    layers[map_layer_ids.at(il)].kq_mask_dev = t;
                }
                if (cp.active() && buft == cp.ranges[1].buft) {
                    cp.kq_mask_dev = t;
                }
            }

            LLAMA_LOG_DEBUG("%s: incremental KQ mask on: %zu device tensors, %u rows, %.2f MiB each, check = %d\n", __func__,
                    kq_inc.dev.size(), kq_inc.n_rows, (double) kv_size*kq_inc.n_rows*sizeof(ggml_fp16_t)/1024.0/1024.0, kq_inc.check);
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_WARN("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        // CPU PLACEMENT warning #2 of 3 (2026-10-20), KV half (the recurrent-state half,
        // with the incident numbers, is in llama_memory_recurrent's constructor).
        //
        // What: offload is on, a GPU device exists, and this KV buffer still landed in a
        // host buffer type -> WARN with buft name, MiB and the layers it holds. The size
        // line above is already WARN but does not say "this is wrong" or which layers.
        //
        // Why: a layer that model.dev_layer() put on the CPU (fit lowering n_gpu_layers,
        // short -ts) drags its KV with it; attention for that layer then runs on the CPU
        // and splits the graph. 2026-10-20 the hybrid case cost 3.3 s -> 9.2 s per
        // 2048-token batch and ~50 -> ~30 t/s.
        //
        // Not warned: offload == false (-nkvo), CPU-only runs (model.n_devices() == 0),
        // no_alloc probes (--fit measures with dummy zero-size buffers).
        //
        // Re-measure: -lv 3, grep "CPU PLACEMENT:"; force with -ngl <n_layer>.
        if (offload && model.n_devices() > 0 && !hparams.no_alloc && ggml_backend_buft_is_host(buft)) {
            const auto it = buft_layers.find(buft);
            LLAMA_LOG_WARN("CPU PLACEMENT: %s: KV buffer %s = %.2f MiB is in host memory with offload on, "
                    "holding layers [%s]; a layer on the CPU in a GPU split costs seconds per prompt batch; "
                    "check -ts / --fit-target / -ngl\n",
                    __func__, ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buf)/1024.0/1024.0,
                    it != buft_layers.end() ? llama_format_index_ranges(it->second).c_str() : "?");
        }

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    // [KQ_MASK_DEV] anchor = the K tensor of the first layer in each buffer type (no_alloc fit probes: the dummy
    // buffer still carries the buft, so the sched places the op the same way and the reserve sizes it per device)
    if (kq_build.enabled) {
        std::map<ggml_backend_buffer_type_t, int> idx;
        for (auto & layer : layers) {
            if (!layer.k || !layer.k->buffer) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(layer.k->buffer);
            auto it = idx.find(buft);
            if (it == idx.end()) {
                it = idx.emplace(buft, (int) kq_build.anchor.size()).first;
                kq_build.anchor.push_back(layer.k);
            }
            layer.kq_build_dev = it->second;
        }
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        n_rot_k = other->n_rot_k;
        n_rot_v = other->n_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        bool attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        const bool attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        if (attn_rot_k) {
            n_rot_k = 64;

            // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
            while (n_embd_head_k_all % (2*n_rot_k) == 0) {
                n_rot_k *= 2;
            }
        }
        if (attn_rot_v) {
            n_rot_v = 64;
            // using smaller rotation matrices for V seems beneficial
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
            //while (hparams.n_embd_head_v() % (2*n_rot_v) == 0) {
            //    n_rot_v *= 2;
            //}
        }
    }

    LLAMA_LOG_INFO("%s: n_rot_k = %u, n_embd_head_k_all = %d\n", __func__, n_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: n_rot_v = %u, n_embd_head_k_all = %d\n", __func__, n_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (n_rot_k || n_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    if (kq_inc.enabled && (kq_inc.n_inc + kq_inc.n_full) > 0) {
        LLAMA_LOG_DEBUG("%s: incremental KQ mask: %" PRIu64 " incremental + %" PRIu64 " full uploads, %.2f of %.2f MiB per device (%.1f%%)\n", __func__,
                kq_inc.n_inc, kq_inc.n_full, kq_inc.bytes/1024.0/1024.0, kq_inc.bytes_full/1024.0/1024.0,
                kq_inc.bytes_full ? 100.0*kq_inc.bytes/kq_inc.bytes_full : 0.0);
    }
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }

        // [CP] the K-shift graph rotates every range (and the check reference) in place: nothing to resync
    }

    // [CP] LLAMA_CP_CHECK only: bring the reference up to the ranges after host writes (state restore / failed-restore
    // clear). Production (no check) has no copy here: the restore wrote the owning ranges directly.
    if (!cp.ref_dirty.empty()) {
        llama_synchronize(lctx);
        for (const auto & [c0, c1] : cp.ref_dirty) {
            cp_ref_from_ranges(c0, c1);
        }
        cp.ref_dirty.clear();
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);

                if (!is_embd) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_seq_max() const {
    return n_seq_max;
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    // [CP] a CP layer has no single K tensor (the hybrid-idx and dsv4 caches that use this never configure CP)
    GGML_ASSERT(!cp_active(il) && "get_k_storage: layer is context-parallel (range-split K/V)");
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_stream(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return seq_to_stream[seq_id];
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv_after(const slot_info & sinfo) const {
    uint32_t result = 0;

    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        uint32_t used_max_p1 = cells.used_max_p1();
        for (const uint32_t idx : sinfo.idxs[s]) {
            used_max_p1 = std::max(used_max_p1, idx + 1);
        }

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(used_max_p1, n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    // [CP] plain attention over a CP layer reads range 0, which holds [0, h): only valid while n_kv <= h (cp_split
    // returns 0 exactly then). Above h there is no contiguous K: build_attn_cp is the only reader, and the graph paths
    // that cannot take it (training, kq_b, sinks, no FA) end here.
    if (cp_active(il)) {
        GGML_ASSERT(n_kv <= cp.h && "context-parallel layer read as one tensor with n_kv > h (training, kq_b, sinks or no FA?)");
        return get_k_range(ctx, il, 0, n_kv, false);
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    // [CP] see get_k
    if (cp_active(il)) {
        GGML_ASSERT(n_kv <= cp.h && "context-parallel layer read as one tensor with n_kv > h (training, kq_b, sinks or no FA?)");
        return get_v_range(ctx, il, 0, n_kv, false);
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    // [CP] a CP layer is stored per range with cpy_k_range (the global indices would overrun range 0)
    GGML_ASSERT(!cp_active(il) && "cpy_k on a context-parallel layer: use cpy_k_range");

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    GGML_ASSERT(!cp_active(il) && "cpy_v on a context-parallel layer: use cpy_v_range");

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

// [CP] see cp_state in llama-kv-cache.h

bool llama_kv_cache::cp_active(int32_t il) const {
    return cp.active() && il == cp.il;
}

uint32_t llama_kv_cache::cp_split(int32_t il, uint32_t n_kv) const {
    // a function of n_kv only, so graph reuse holds (n_kv is pinned by can_reuse_kq_mask); n_kv <= h: every cell of the
    // ubatch is < used_max_p1 <= n_kv <= h (apply_ubatch runs before get_n_kv), range 1 has nothing to store or read
    if (!cp_active(il) || n_kv <= cp.h) {
        return 0;
    }
    return cp.h;
}

bool llama_kv_cache::cp_check() const {
    return cp.check;
}

ggml_backend_dev_t llama_kv_cache::cp_dev(bool helper) const {
    return cp.active() ? cp.ranges[helper ? 1 : 0].dev : nullptr;
}

ggml_tensor * llama_kv_cache::cp_kq_mask_dev() const {
    return kq_inc.enabled ? cp.kq_mask_dev : nullptr;
}

bool llama_kv_cache::cp_pack() const {
    return cp.pack;
}

int llama_kv_cache::cp_n_ranges() const {
    return (int) cp.ranges.size();
}

bool llama_kv_cache::cp_ref_dirty() const {
    return !cp.ref_dirty.empty();
}

// [CP] llama_io_write/read_device group their copies per buffer type and abort when the per-buft byte counts of the save
// and the restore differ; with ranges on two buffer types those counts depend on where the restored cells land
bool llama_kv_cache::cp_on_device_ok() const {
    for (const auto & r : cp.ranges) {
        if (r.buft != cp.ranges[0].buft) {
            return false;
        }
    }
    return true;
}

// [CP] the helper builds its own mask only when its device already has a KQ_MASK_BUILD op (an attention layer of this
// cache lives there): no new op, no new per-device reserve (kv_size x n_ubatch f16 per op, 876 MiB at 224k/2048), so
// the fit is unchanged. On the 2026-10-22 production split (Vulkan1 holds layers 43-63) it exists; -1 keeps the hop.
int llama_kv_cache::cp_kq_build_dev() const {
    if (!kq_build.enabled || !cp.active() || cp.ranges[0].dev == cp.ranges[1].dev) {
        return -1;
    }
    for (size_t i = 0; i < kq_build.anchor.size(); i++) {
        const ggml_tensor * a = kq_build.anchor[i];
        if (a && a->buffer && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(a->buffer)) == cp.ranges[1].dev) {
            return (int) i;
        }
    }
    return -1;
}

// [CP] cells [i0, i1) as a [head, n_head_kv, cells, 1] view: of the range holding them (local rows), or of the reference
ggml_tensor * llama_kv_cache::get_k_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const {
    GGML_ASSERT(cp_active(il) && i0 < i1 && i1 <= get_size());

    ggml_tensor * k = nullptr;
    uint32_t      c0 = 0;
    if (ref) {
        GGML_ASSERT(cp.ref_k && "context-parallel: the reference K exists only with LLAMA_CP_CHECK=1");
        k = cp.ref_k;
    } else {
        for (const auto & r : cp.ranges) {
            if (i0 >= r.c0 && i1 <= r.c1) {
                k  = r.k;
                c0 = r.c0;
            }
        }
        GGML_ASSERT(k && "context-parallel: a K read must lie inside one range");
    }

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), i1 - i0, 1,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            k->nb[1],
            k->nb[2],
            k->nb[1]*(i0 - c0));
}

ggml_tensor * llama_kv_cache::get_v_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const {
    GGML_ASSERT(cp_active(il) && !v_trans && i0 < i1 && i1 <= get_size());

    ggml_tensor * v = nullptr;
    uint32_t      c0 = 0;
    if (ref) {
        GGML_ASSERT(cp.ref_v && "context-parallel: the reference V exists only with LLAMA_CP_CHECK=1");
        v = cp.ref_v;
    } else {
        for (const auto & r : cp.ranges) {
            if (i0 >= r.c0 && i1 <= r.c1) {
                v  = r.v;
                c0 = r.c0;
            }
        }
        GGML_ASSERT(v && "context-parallel: a V read must lie inside one range");
    }

    return ggml_view_4d(ctx, v,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), i1 - i0, 1,
            ggml_row_size(v->type, hparams.n_embd_head_v(il)),
            v->nb[1],
            v->nb[2],
            v->nb[1]*(i0 - c0));
}

// [CP] the store of a CP layer: range r with its local indices (trash row for the other range's tokens), r = -1 the
// LLAMA_CP_CHECK reference with the global indices. k_cur may be a strided view (the packed Q|K|V on the helper): only
// dims 0 and 1 must merge (the same requirement as cpy_k).
ggml_tensor * llama_kv_cache::cpy_k_range(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il, int r) const {
    GGML_ASSERT(cp_active(il) && r >= -1 && r < (int) cp.ranges.size());
    GGML_ASSERT(ggml_row_size(k_cur->type, k_cur->ne[0]) == k_cur->nb[1]);

    ggml_tensor * dst = r < 0 ? cp.ref_k : cp.ranges[r].k;
    GGML_ASSERT(dst);

    k_cur = ggml_view_2d(ctx, k_cur, k_cur->ne[0]*k_cur->ne[1], k_cur->ne[2], k_cur->nb[2], 0);

    return ggml_set_rows(ctx, dst, k_cur, idxs);
}

ggml_tensor * llama_kv_cache::cpy_v_range(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il, int r) const {
    GGML_ASSERT(cp_active(il) && !v_trans && r >= -1 && r < (int) cp.ranges.size());
    GGML_ASSERT(ggml_row_size(v_cur->type, v_cur->ne[0]) == v_cur->nb[1]);

    ggml_tensor * dst = r < 0 ? cp.ref_v : cp.ranges[r].v;
    GGML_ASSERT(dst);

    v_cur = ggml_view_2d(ctx, v_cur, v_cur->ne[0]*v_cur->ne[1], v_cur->ne[2], v_cur->nb[2], 0);

    return ggml_set_rows(ctx, dst, v_cur, idxs);
}

// [CP] cell-indexed host access to a layer's K or V: splits [c0, c1) at the range bounds. The pieces come in cell order,
// so a caller that streams them (state write/read) produces/consumes exactly the bytes of a no-CP cache.
template <typename F>
void llama_kv_cache::cp_rows(const kv_layer & layer, bool is_v, uint32_t strm, uint32_t c0, uint32_t c1, F && fn) const {
    if (c0 >= c1) {
        return;
    }
    if (!cp_active((int32_t) layer.il)) {
        fn(is_v ? layer.v_stream[strm] : layer.k_stream[strm], c0, c1 - c0);
        return;
    }
    GGML_ASSERT(strm == 0 && c1 <= get_size());
    for (const auto & r : cp.ranges) {
        const uint32_t a = std::max(c0, r.c0);
        const uint32_t b = std::min(c1, r.c1);
        if (a < b) {
            fn(is_v ? r.v : r.k, a - r.c0, b - a);
        }
    }
}

// [CP] LLAMA_CP_CHECK only: after a host write of cells [c0, c1) of the CP layer (state restore), bring the reference up
// to the ranges. The ranges are what production reads; the round trip tests (test-save-load-state,
// test-state-restore-fragmented with the CP env) prove the host routing, the check then proves the graph side.
void llama_kv_cache::cp_ref_from_ranges(uint32_t c0, uint32_t c1) const {
    if (!cp.active() || !cp.ref_k || c0 >= c1) {
        return;
    }
    const auto & layer = layers[map_layer_ids.at(cp.il)];
    std::vector<uint8_t> buf;
    for (const bool is_v : { false, true }) {
        ggml_tensor * ref = is_v ? cp.ref_v : cp.ref_k;
        const size_t row = ref->nb[1];
        cp_rows(layer, is_v, 0, c0, c1, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
            buf.resize((size_t) n*row);
            ggml_backend_tensor_get(t, buf.data(), (size_t) r0*row, buf.size());
            // r0 is local to t; the global cell is found from the range that t belongs to
            uint32_t g0 = r0;
            for (const auto & r : cp.ranges) {
                if (t == r.k || t == r.v) {
                    g0 = r.c0 + r0;
                }
            }
            ggml_backend_tensor_set(ref, buf.data(), (size_t) g0*row, buf.size());
        });
    }
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "attn_inp_k_idxs");

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "attn_inp_v_idxs");

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_k) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_k, n_rot_k);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_v) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_v, n_rot_v);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

// [CP] one input per range, shared by K and V (FA: !v_trans, one index per token) and by every CP layer: the token's
// local row if its cell is in [c0, c1), else the range's trash row c1 - c0 (see cp_state)
ggml_tensor * llama_kv_cache::build_input_cp_idxs(ggml_context * ctx, const llama_ubatch & ubatch, int r) const {
    GGML_ASSERT(r >= 0 && r < (int) cp.ranges.size());

    ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ubatch.n_tokens);

    ggml_set_input(idxs);
    ggml_format_name(idxs, "attn_inp_cp_idxs_r%d", r);

    return idxs;
}

void llama_kv_cache::set_input_cp_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo, int r) const {
    GGML_ASSERT(r >= 0 && r < (int) cp.ranges.size());
    GGML_ASSERT(sinfo.n_stream() == 1 && ubatch->n_tokens == (int64_t) sinfo.size());
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto & rg = cp.ranges[r];
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t i = 0; i < sinfo.size(); ++i) {
        const uint32_t c = sinfo.idxs[0][i];
        data[i] = c >= rg.c0 && c < rg.c1 ? c - rg.c0 : rg.c1 - rg.c0;
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

// [KQ_MASK_DEV] see kq_build_state in llama-kv-cache.h for the measurements and the gate reasoning

bool llama_kv_cache::kq_build_causal(bool causal_attn) const {
    // same fold as set_input_kq_mask: only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }
    return causal_attn;
}

int llama_kv_cache::kq_build_dev(int32_t il) const {
    if (!kq_build.enabled) {
        return -1;
    }
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return -1;
    }
    return layers[it->second].kq_build_dev;
}

// Host cost: one pass over n_kv cells + n_tokens rows (vs the n_kv x n_tokens host build, set_inputs 58 ms at 60-75k / 2048
// rows before). Precondition: n_stream == 1 and every seq id < 32 (static gate), so seq_get_all() fits one word.
void llama_kv_cache::set_input_kq_build(ggml_tensor * cells, ggml_tensor * rows, const llama_ubatch * ubatch, int64_t n_kv) const {
    GGML_ASSERT(kq_build.enabled && n_stream == 1);
    const int64_t cs = cells->ne[0];
    GGML_ASSERT(cs == kq_build_cell_stride() && cells->ne[1] >= n_kv);
    GGML_ASSERT(rows->ne[0] == 2 && rows->ne[1] >= (int64_t) ubatch->n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(cells->buffer) && ggml_backend_buffer_is_host(rows->buffer));

    const auto & c = v_cells[0];
    GGML_ASSERT(n_kv <= (int64_t) c.size());

    int32_t * dc = (int32_t *) cells->data;
    if (cs == 1) {
        // single seq: a used cell belongs to seq 0 (the only one), so pos alone decides
        for (int64_t j = 0; j < n_kv; ++j) {
            dc[j] = c.is_empty(j) ? -1 : c.pos_get(j);
        }
    }
    for (int64_t j = 0; j < n_kv && cs == 2; ++j) {
        if (c.is_empty(j)) {
            dc[2*j + 0] = -1;
            dc[2*j + 1] = 0;
        } else {
            dc[2*j + 0] = c.pos_get(j);
            dc[2*j + 1] = (int32_t) (uint32_t) c.seq_get_all(j).to_ulong();
        }
    }

    int32_t * dr = (int32_t *) rows->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        GGML_ASSERT(ubatch->seq_id[i][0] >= 0 && ubatch->seq_id[i][0] < 32);
        dr[2*i + 0] = ubatch->pos[i];
        dr[2*i + 1] = ubatch->seq_id[i][0];
    }
}

void llama_kv_cache::kq_build_verify(const std::vector<ggml_tensor *> & outs, const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const {
    const int64_t n_tps = ubatch->n_tokens;
    const size_t  n     = (size_t) n_kv*n_tps;

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ 1,
        /*.n_tps            =*/ n_tps,
    };

    kq_build.host.resize(n);
    set_input_kq_mask_impl<ggml_fp16_t>(args, kq_build.host.data(), kq_build_causal(causal_attn));

    kq_build.dev.resize(n);
    for (ggml_tensor * t : outs) {
        GGML_ASSERT(t->buffer && ggml_nbytes(t) >= n*sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(t, kq_build.dev.data(), 0, n*sizeof(ggml_fp16_t));
        if (memcmp(kq_build.dev.data(), kq_build.host.data(), n*sizeof(ggml_fp16_t)) != 0) {
            for (size_t k = 0; k < n; ++k) {
                if (kq_build.dev[k] != kq_build.host[k]) {
                    LLAMA_LOG_ERROR("%s: LLAMA_KQ_MASK_DEV_CHECK: device mask %s (%s) differs from the host build: row %zu col %zu (n_kv %" PRId64 ", n_tokens %" PRId64 "): host %f device %f\n",
                            __func__, t->name, ggml_backend_buffer_name(t->buffer), k / n_kv, k % n_kv, n_kv, n_tps,
                            ggml_fp16_to_fp32(kq_build.host[k]), ggml_fp16_to_fp32(kq_build.dev[k]));
                    break;
                }
            }
            GGML_ABORT("LLAMA_KQ_MASK_DEV_CHECK failed");
        }
    }

    const uint64_t nc = ++kq_build.n_checked;
    if ((nc & (nc - 1)) == 0 || nc % 1024 == 0) {
        LLAMA_LOG_WARN("%s: LLAMA_KQ_MASK_DEV_CHECK: %" PRIu64 " ubatches match (last: %zu devices, n_kv %" PRId64 ", n_tokens %" PRId64 ")\n",
                __func__, kq_build.n_checked, outs.size(), n_kv, n_tps);
    }
}

// [KQ_MASK_INC] see kq_mask_inc_state in llama-kv-cache.h for the measurements and the gate reasoning

bool llama_kv_cache::kq_mask_dev_usable(const llama_ubatch & ubatch, int64_t n_stream, ggml_type type, bool causal_attn) const {
    return kq_inc.enabled && causal_attn && type == GGML_TYPE_F16 && n_stream == 1 && ubatch.n_tokens <= kq_inc.n_rows;
}

ggml_tensor * llama_kv_cache::get_kq_mask_dev(int32_t il) const {
    if (!kq_inc.enabled) {
        return nullptr;
    }

    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    return layers[it->second].kq_mask_dev;
}

void llama_kv_cache::set_input_kq_mask_dev(const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const {
    auto & st = kq_inc;

    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(st.enabled && causal_attn);
    GGML_ASSERT(n_tokens <= st.n_rows);
    GGML_ASSERT(n_kv > 0 && (uint32_t) n_kv <= v_cells[0].size());

    auto & cells = v_cells[0];

    // every cell change since the previous upload (incl. the cells apply_ubatch just wrote for this ubatch)
    const uint32_t d = cells.dirty_take(st.slot);
    for (auto & lo : st.row_lo) {
        lo = std::min(lo, d);
    }

    const llama_seq_id seq = ubatch->seq_id[0][0];

    llama_pos pos_min = std::numeric_limits<llama_pos>::max();
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pos_min = std::min(pos_min, ubatch->pos[i]);
    }

    // the cells whose value can still change for later rows: every cell of seq at or after pos_min. when the
    // positions are monotonic these are exactly the n_tokens cells of this ubatch
    uint32_t u   = UINT32_MAX;
    uint32_t cnt = 0;
    {
        const auto & sp = cells.seq_pos_get(seq);
        for (auto it = sp.lower_bound({ pos_min, 0 }); it != sp.end(); ++it) {
            if (++cnt > n_tokens) {
                break;
            }
            u = std::min(u, it->second);
        }
    }

    const char * reason = nullptr;
    if (ubatch->n_seqs_unq != 1) {
        reason = "more than one seq in the ubatch";
    } else if (n_kv != st.n_kv) {
        reason = "n_kv changed (row stride)";
    } else if (seq != st.seq) {
        reason = "seq changed";
    } else if (cnt != n_tokens) {
        reason = "retained cells at or after the ubatch positions";
    }

    if (reason) {
        std::fill(st.row_lo.begin(), st.row_lo.end(), 0);
        if (reason != st.last_reason) {
            LLAMA_LOG_DEBUG("%s: incremental KQ mask fallback (full upload): %s\n", __func__, reason);
        }
        st.last_reason = reason;
        st.n_kv = n_kv;
        st.seq  = seq;
        if (st.shadow.size() < (size_t) n_kv*st.n_rows) {
            st.shadow.resize((size_t) n_kv*st.n_rows);
        }
    }

    ggml_fp16_t * data = st.shadow.data();

    if (st.check && !reason) {
        st.tmp.assign(data, data + (size_t) n_kv*n_tokens);
    }

    // the host build, unchanged (cheap): full rows [0, n_tokens) x [0, n_kv) into the shadow
    {
        const args_set_input_kq_mask args = {
            /*.hparams          =*/ hparams,
            /*.ubatch           =*/ ubatch,
            /*.v_cells          =*/ v_cells,
            /*.seq_to_stream    =*/ seq_to_stream,
            /*.n_swa            =*/ n_swa,
            /*.swa_type         =*/ swa_type,
            /*.n_kv             =*/ n_kv,
            /*.n_stream         =*/ 1,
            /*.n_tps            =*/ n_tokens,
        };

        set_input_kq_mask_impl<ggml_fp16_t>(args, data, true);
    }

    if (st.check && !reason) {
        for (uint32_t r = 0; r < n_tokens; ++r) {
            const uint32_t lo = (uint32_t) std::min<int64_t>(st.row_lo[r], n_kv);
            const ggml_fp16_t * a = st.tmp.data() + (size_t) r*n_kv;
            const ggml_fp16_t * b = data          + (size_t) r*n_kv;
            if (memcmp(a, b, lo*sizeof(ggml_fp16_t)) != 0) {
                uint32_t c = 0;
                while (a[c] == b[c]) {
                    c++;
                }
                LLAMA_LOG_ERROR("%s: LLAMA_KQ_MASK_CHECK: settled region changed: device = all (shared shadow), row = %u, col = %u (row_lo = %u, n_kv = %" PRId64 "), expected = %f, got = %f\n",
                        __func__, r, c, lo, n_kv, ggml_fp16_to_fp32(b[c]), ggml_fp16_to_fp32(a[c]));
                GGML_ABORT("LLAMA_KQ_MASK_CHECK failed");
            }
        }
    }

    uint32_t lo = (uint32_t) n_kv;
    for (uint32_t r = 0; r < n_tokens; ++r) {
        lo = std::min(lo, st.row_lo[r]);
    }

    if (lo == 0 && !reason) {
        // not a gate failure: the cells changed from cell 0 (clear, state restore, seq_rm/cp at 0, first ubatch)
        reason = "cells changed from cell 0";
        if (reason != st.last_reason) {
            LLAMA_LOG_DEBUG("%s: incremental KQ mask full upload: %s\n", __func__, reason);
        }
        st.last_reason = reason;
    } else if (lo > 0 && (!st.logged_inc || st.last_reason != nullptr)) {
        LLAMA_LOG_DEBUG("%s: incremental KQ mask: incremental upload (n_kv = %" PRId64 ", n_tokens = %u, cols [%u, %" PRId64 "))\n", __func__, n_kv, n_tokens, lo, n_kv);
        st.logged_inc  = true;
        st.last_reason = nullptr;
    }

    const size_t es     = sizeof(ggml_fp16_t);
    const size_t stride = (size_t) n_kv*es;

    // one 2-D set per run of consecutive rows with the same row_lo: rows beyond the previous step's n_tokens are
    // usually stale since the last n_kv change (row_lo 0) and must not drag the fresh rows into a full upload.
    // 2026-10-21 session B: a 49-row verify after 6-row steps took t_set_inputs p50 6.8 ms with a single min(row_lo)
    // upload (~6 MB/card at 60k ctx through the XT's x1 link)
    for (uint32_t r0 = 0; r0 < n_tokens; ) {
        const uint32_t c0 = std::min<uint32_t>(st.row_lo[r0], (uint32_t) n_kv);
        uint32_t r1 = r0 + 1;
        while (r1 < n_tokens && std::min<uint32_t>(st.row_lo[r1], (uint32_t) n_kv) == c0) {
            r1++;
        }
        if (c0 < (uint32_t) n_kv) {
            for (ggml_tensor * t : st.dev) {
                if (c0 == 0) {
                    ggml_backend_tensor_set(t, data + (size_t) r0*n_kv, r0*stride, (size_t) (r1 - r0)*stride);
                } else {
                    ggml_backend_tensor_set_2d(t, data + (size_t) r0*n_kv + c0, r0*stride + c0*es, (n_kv - c0)*es, r1 - r0, stride, stride);
                }
            }
            st.bytes += (size_t) (n_kv - c0)*(r1 - r0)*es;
        }
        r0 = r1;
    }

    st.bytes_full += (size_t) n_tokens*stride;
    if (lo == 0) {
        st.n_full++;
    } else {
        st.n_inc++;
    }

    if (st.check) {
        st.tmp.resize((size_t) n_kv*n_tokens);
        for (ggml_tensor * t : st.dev) {
            ggml_backend_tensor_get(t, st.tmp.data(), 0, (size_t) n_tokens*stride);
            if (memcmp(st.tmp.data(), data, (size_t) n_tokens*stride) != 0) {
                size_t k = 0;
                while (st.tmp[k] == data[k]) {
                    k++;
                }
                LLAMA_LOG_ERROR("%s: LLAMA_KQ_MASK_CHECK: device tensor differs from a fresh build: device = %s, row = %zu, col = %zu (lo = %u, n_kv = %" PRId64 "), expected = %f, got = %f\n",
                        __func__, ggml_backend_buffer_name(t->buffer), k / n_kv, k % n_kv, lo, n_kv,
                        ggml_fp16_to_fp32(data[k]), ggml_fp16_to_fp32(st.tmp[k]));
                GGML_ABORT("LLAMA_KQ_MASK_CHECK failed");
            }
        }
    }

    // the cells of this ubatch carry the causal triangle -> they are refreshed on the next upload.
    // multi-seq ubatch: u and cnt describe seq_id[0][0] only, while the rows of the other seqs hold 0 (attend) at
    // their own history columns, which can sit below u in a unified cache. the next single-seq step of seq_id[0][0]
    // with the same n_kv passes the dynamic gate, so a u_lo > 0 here left those columns stale on the device
    // (review 2026-10-21, -np 2 -kvu: before this fix LLAMA_KQ_MASK_CHECK aborted with "settled region changed: row = 1,
    // col = 0 (row_lo = 27)" in tools/server/tests/unit/test_kq_mask_inc.py and on falcon-h1-dense in the "kq mask" case of
    // tests/test-recurrent-state-rollback.cpp). keep every row unsettled -> the next step does a full upload
    const uint32_t u_lo = (ubatch->n_seqs_unq == 1 && cnt == n_tokens) ? std::min<uint32_t>(u, (uint32_t) n_kv) : 0;
    for (uint32_t r = 0; r < n_tokens; ++r) {
        st.row_lo[r] = u_lo;
    }
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    // same for a mixed ubatch
    const bool by_order = !ubatch.token || ubatch.is_mixed();

    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (by_order) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (by_order) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    // [CP] layer.k is range 0; the other ranges and the check reference
    for (size_t r = 1; r < cp.ranges.size(); ++r) {
        size_k_bytes += ggml_nbytes(cp.ranges[r].k);
    }
    size_k_bytes += cp.ref_k ? ggml_nbytes(cp.ref_k) : 0;

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    for (size_t r = 1; r < cp.ranges.size(); ++r) {
        size_v_bytes += ggml_nbytes(cp.ranges[r].v);
    }
    size_v_bytes += cp.ref_v ? ggml_nbytes(cp.ref_v) : 0;

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // [CP] a CP layer rotates each range (cells [c0, c1), not the trash row) with the shift input viewed at c0, and
        // the LLAMA_CP_CHECK reference whole: the K-shift is a device-side write, nothing to resync afterwards
        auto shift_view = [&](ggml_tensor * t, uint32_t c0, uint32_t n) {
            ggml_tensor * k =
                ggml_view_3d(ctx, t,
                    n_rot, n_head_kv, n,
                    ggml_row_size(t->type, n_embd_head_k),
                    ggml_row_size(t->type, n_embd_k_gqa),
                    ggml_row_size(t->type, n_embd_nope));
            ggml_tensor * shift = c0 == 0 && n == inp->k_shift->ne[0] ? inp->k_shift
                : ggml_view_1d(ctx, inp->k_shift, n, (size_t) c0*ggml_element_size(inp->k_shift));

            ggml_tensor * cur = build_rope_shift(cparams, ctx, k, shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

            ggml_build_forward_expand(gf, cur);
        };

        if (cp_active((int32_t) il)) {
            for (const auto & r : cp.ranges) {
                shift_view(r.k, r.c0, r.c1 - r.c0);
            }
            if (cp.ref_k) {
                shift_view(cp.ref_k, 0, get_size());
            }
        } else {
            shift_view(layer.k, 0, get_size()*n_stream);
        }
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    state_write_range(io, seq_id, -1, -1, flags);
}

void llama_kv_cache::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // [CP] see state_read_sinfo
    if (!cp_on_device_ok() && (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        throw std::runtime_error("LLAMA_STATE_SEQ_FLAGS_ON_DEVICE is not supported with context-parallel attention across two devices (LLAMA_CP_LAYER)");
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            if (add_cell && (p0 >= 0 || p1 >= 0)) {
                const llama_pos pos = cells.pos_get(i);

                add_cell = (p0 < 0 || pos >= p0) && (p1 < 0 || pos < p1);
            }

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    // [CP] state_read_data routes every cell run to the range that holds it (cp_rows): nothing to resync. ON_DEVICE is
    // refused: llama_io_read_device groups the copies per buffer type and aborts when the per-buft byte counts of the
    // save and the restore differ, which with ranges depends on where the restored cells land
    if (!cp_on_device_ok() && (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        throw std::runtime_error("LLAMA_STATE_SEQ_FLAGS_ON_DEVICE is not supported with context-parallel attention across two devices (LLAMA_CP_LAYER)");
    }

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, flags, sinfos_in ? &(*sinfos_in)[s] : nullptr);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            state_clear(seq_id, strm, sinfo);
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_rot_k, sizeof(n_rot_k));
    io.write(&n_rot_v, sizeof(n_rot_v));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        // [CP] split at the range bounds (cp_rows), same bytes in the same order
        for (const auto & range : cr.data) {
            cp_rows(layer, false, cr.strm, range.first, range.second, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                io.write_tensor(t, (size_t) r0 * k_size_row, (size_t) n * k_size_row);
            });
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            // [CP] split at the range bounds (cp_rows), same bytes in the same order
            for (const auto & range : cr.data) {
                cp_rows(layer, true, cr.strm, range.first, range.second, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                    io.write_tensor(t, (size_t) r0 * v_size_row, (size_t) n * v_size_row);
                });
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, llama_state_seq_flags flags, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // with APPEND the incoming cells are merged into the existing sequence state
        // instead of replacing it - used to apply pos-range blobs in order
        if (!(flags & LLAMA_STATE_SEQ_FLAGS_APPEND)) {
            seq_rm(dest_seq_id, -1, -1);
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;
    uint32_t n_rot_k_ref;
    uint32_t n_rot_v_ref;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_rot_k_ref, sizeof(n_rot_k_ref));
    io.read(&n_rot_v_ref, sizeof(n_rot_v_ref));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    if (n_rot_k_ref != n_rot_k) {
        LLAMA_LOG_ERROR("%s: incompatible key rotation (%u instead of %u)\n", __func__, n_rot_k_ref, n_rot_k);
        return false;
    }

    if (n_rot_v_ref != n_rot_v) {
        LLAMA_LOG_ERROR("%s: incompatible value rotation (%u instead of %u)\n", __func__, n_rot_v_ref, n_rot_v);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        // [CP] each destination run goes to the range(s) holding it: the restore writes the owning device directly
        // (the 2026-10-21 mirror needed cp_resync after this: 410 MiB / 233 ms per layer per restore)
        for (const auto & r : runs) {
            cp_rows(layer, false, strm, r.from, r.to, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                io.read_tensor(t, (size_t) r0 * k_size_row, (size_t) n * k_size_row);
            });
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                cp_rows(layer, true, strm, r.from, r.to, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                    io.read_tensor(t, (size_t) r0 * v_size_row, (size_t) n * v_size_row);
                });
            }
        }

        // [CP] LLAMA_CP_CHECK: the reference follows the ranges at the next update (the io may defer its tensor writes to
        // its destruction: llama_io_read_buffer/_file/_device all do, so the ranges are not written yet here)
        if (cp.ref_k) {
            for (const auto & r : runs) {
                cp.ref_dirty.emplace_back(r.from, r.to);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

void llama_kv_cache::state_clear(llama_seq_id seq_id) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    const auto & cells = v_cells[strm];

    slot_info sinfo;
    sinfo.s0 = strm;
    sinfo.s1 = strm;
    sinfo.resize(1);
    sinfo.strm[0] = strm;

    // a cell that another sequence still uses keeps its data
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            sinfo.idxs[0].push_back(i);
        }
    }

    state_clear(seq_id, strm, sinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    const auto & cells = v_cells[strm];

    const uint32_t cell_count = sinfo.size();

    const bool is_contiguous = sinfo.is_contiguous();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        GGML_UNUSED(k);
        // [CP] cp_rows routes each cell to its range
        auto clear_k = [&](uint32_t c0, uint32_t c1) {
            cp_rows(layer, false, strm, c0, c1, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                llama_clear_tensor_data(t, (size_t) r0 * k_size_row, (size_t) n * k_size_row);
            });
        };
        if (is_contiguous) {
            clear_k(sinfo.head(), sinfo.head() + cell_count);
        } else {
            for (uint32_t i = 0; i < cell_count; ++i) {
                clear_k(sinfo.idxs[0][i], sinfo.idxs[0][i] + 1);
            }
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            auto clear_v = [&](uint32_t c0, uint32_t c1) {
                cp_rows(layer, true, strm, c0, c1, [&](ggml_tensor * t, uint32_t r0, uint32_t n) {
                    llama_clear_tensor_data(t, (size_t) r0 * v_size_row, (size_t) n * v_size_row);
                });
            };
            if (is_contiguous) {
                clear_v(sinfo.head(), sinfo.head() + cell_count);
            } else {
                for (uint32_t i = 0; i < cell_count; ++i) {
                    clear_v(sinfo.idxs[0][i], sinfo.idxs[0][i] + 1);
                }
            }
        } else {
            const size_t v_size_el = ggml_type_size(v->type);

            if (is_contiguous) {
                const uint32_t h = sinfo.head();

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_clear_tensor_data(v, (h + j * cells.size()) * v_size_el, cell_count * v_size_el);
                }
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        llama_clear_tensor_data(v, (sinfo.idxs[0][i] + j * cells.size()) * v_size_el, v_size_el);
                    }
                }
            }
        }
    }

    // [CP] LLAMA_CP_CHECK: the reference follows the cleared ranges at the next update
    if (cp.ref_k) {
        for (uint32_t i = 0; i < cell_count; ++i) {
            cp.ref_dirty.emplace_back(sinfo.idxs[0][i], sinfo.idxs[0][i] + 1);
        }
    }
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty() && !kv->cp_ref_dirty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

bool llama_kv_cache_context::prepare_graph() {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || ubatches.empty() || i_cur >= ubatches.size()) {
        return false;
    }

    n_kv = kv->get_n_kv_after(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

bool llama_kv_cache_context::cp_active(int32_t il) const {
    return kv->cp_active(il);
}

uint32_t llama_kv_cache_context::cp_split(int32_t il) const {
    return kv->cp_split(il, n_kv);
}

bool llama_kv_cache_context::cp_check() const {
    return kv->cp_check();
}

ggml_backend_dev_t llama_kv_cache_context::cp_dev(bool helper) const {
    return kv->cp_dev(helper);
}

ggml_tensor * llama_kv_cache_context::cp_kq_mask_dev() const {
    return kv->cp_kq_mask_dev();
}

bool llama_kv_cache_context::cp_pack() const {
    return kv->cp_pack();
}

int llama_kv_cache_context::cp_kq_build_dev() const {
    return kv->cp_kq_build_dev();
}

int llama_kv_cache_context::cp_n_ranges() const {
    return kv->cp_n_ranges();
}

ggml_tensor * llama_kv_cache_context::cpy_k_range(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il, int r) const {
    return kv->cpy_k_range(ctx, k_cur, idxs, il, r);
}

ggml_tensor * llama_kv_cache_context::cpy_v_range(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il, int r) const {
    return kv->cpy_v_range(ctx, v_cur, idxs, il, r);
}

ggml_tensor * llama_kv_cache_context::build_input_cp_idxs(ggml_context * ctx, const llama_ubatch & ubatch, int r) const {
    return kv->build_input_cp_idxs(ctx, ubatch, r);
}

void llama_kv_cache_context::set_input_cp_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, int r) const {
    kv->set_input_cp_idxs(dst, ubatch, sinfos[i_cur], r);
}

ggml_tensor * llama_kv_cache_context::get_k_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const {
    return kv->get_k_range(ctx, il, i0, i1, ref);
}

ggml_tensor * llama_kv_cache_context::get_v_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const {
    return kv->get_v_range(ctx, il, i0, i1, ref);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

bool llama_kv_cache_context::kq_mask_dev_usable(const llama_ubatch & ubatch, int64_t n_stream, ggml_type type, bool causal_attn) const {
    return kv->kq_mask_dev_usable(ubatch, n_stream, type, causal_attn);
}

ggml_tensor * llama_kv_cache_context::get_kq_mask_dev(int32_t il) const {
    return kv->get_kq_mask_dev(il);
}

void llama_kv_cache_context::set_input_kq_mask_dev(const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const {
    kv->set_input_kq_mask_dev(ubatch, n_kv, causal_attn);
}

bool llama_kv_cache_context::kq_build_enabled() const {
    return kv->kq_build_enabled();
}

bool llama_kv_cache_context::kq_build_check() const {
    return kv->kq_build_check();
}

bool llama_kv_cache_context::kq_build_ext_2d() const {
    return kv->kq_build_ext_2d();
}

int64_t llama_kv_cache_context::kq_build_cell_stride() const {
    return kv->kq_build_cell_stride();
}

bool llama_kv_cache_context::kq_build_causal(bool causal_attn) const {
    return kv->kq_build_causal(causal_attn);
}

int llama_kv_cache_context::kq_build_dev(int32_t il) const {
    return kv->kq_build_dev(il);
}

ggml_tensor * llama_kv_cache_context::kq_build_anchor(int idx) const {
    return kv->kq_build_anchor(idx);
}

uint32_t llama_kv_cache_context::kq_build_kv_size() const {
    return kv->get_size();
}

void llama_kv_cache_context::set_input_kq_build(ggml_tensor * cells, ggml_tensor * rows, const llama_ubatch * ubatch, int64_t n_kv) const {
    kv->set_input_kq_build(cells, rows, ubatch, n_kv);
}

void llama_kv_cache_context::kq_build_verify(const std::vector<ggml_tensor *> & outs, const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const {
    kv->kq_build_verify(outs, ubatch, n_kv, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
