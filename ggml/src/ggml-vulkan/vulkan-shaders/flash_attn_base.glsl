
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;
// ggml_type enumerant for K/V
layout (constant_id = 12) const uint32_t FaTypeK = 0;
layout (constant_id = 13) const uint32_t FaTypeV = 0;
// sizeof(decode buffer): quants -> ggml block size; F32 -> 16 (decodeBufF32 vec4).
layout (constant_id = 14) const uint32_t FaBlockBytesK = 2;
layout (constant_id = 15) const uint32_t FaBlockBytesV = 2;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;
// Sparse: gather binding-7 indices instead of scanning [0,KV); p.split_kv = n_kv_max.
const bool USE_SPARSE      = (Flags & 16) != 0;
// Ablation bits for profiling (GGML_VK_FA_ABLATE): results are wrong when set, timing is the point.
const bool ABL_MASK        = (Flags & 32)  != 0; // skip mask load and add
const bool ABL_SOFTMAX     = (Flags & 64)  != 0; // skip row max / exp / rescale (P = S)
const bool ABL_PV          = (Flags & 128) != 0; // skip the P.V product and O accumulation
const bool ABL_QK          = (Flags & 256) != 0; // skip the K loads and Q.K^T product (S = 0)
// 2026-10-20: P.V operands loaded in the per-lane-contiguous layout (P as A RowMajor, V as B ColumnMajor)
// over exactly the same addresses and bytes as the real loads. RDNA3 WMMA wants an A lane to hold a row and a
// B lane a column, both along K; Q.K gets that (K A RowMajor, Q B ColumnMajor = 2 b128 per lane per tile), P.V
// gets the opposite on both operands (16 scalar d16 loads per lane per tile). t(0) - t(512) is the most a layout
// fix can win. cm1 aligned f16 only (the staged-V kvsh loads are not swapped). On the PV_REG path (steps A+B)
// only the global V load is swapped (P is already contiguous); in mode 3 (step C) it is a no-op for f16 V; for q8_0 V
// under GGML_VK_FA_PV_REG_QUANT it restores the C2 (unrotated, 4-way bank conflict) step-C store mapping for A/B
// (flash_attn_cm1.comp V_ROT, 2026-10-24; results stay correct). See
// contextFiles/aiInstructionsAndResponses/fa-cm1-pv-partition-design.md sections 2 and 5.
const bool ABL_PV_LAYOUT   = (Flags & 512) != 0;
// Not an ablation: GGML_VK_FA_PV_REG=1|2 requests the cm1 persistent P.V accumulator path (flash_attn_cm1.comp, PV_REG).
// The shader re-checks its own gate, so a stray bit on an ineligible variant runs the old code.
const bool PV_REG_REQ      = (Flags & 1024) != 0;
// GGML_VK_FA_PV_REG=2 adds bit 2048: f16 (O_TYPE) persistent accumulators instead of ACC_TYPE, perf probe only.
const bool PV_REG_F16_REQ  = (Flags & 2048) != 0;
// GGML_VK_FA_PV_REG=3 adds bit 4096 (with 1024, ACC_TYPE accumulators): step C, V staged transposed through kvsh.
const bool PV_REG_C_REQ    = (Flags & 4096) != 0;
// ggml_flash_attn_ext_set_emit_lse (context-parallel merge of partial attentions over disjoint KV ranges): the
// final dst rows are HSV + 4 floats, [HSV] = lse = Mf + log(Lf) (natural log, -inf when Lf == 0), [HSV+1..3] = 0.
// Mf carries FATTN_KQ_MAX_OFFSET and Lf is relative to that same Mf, so the offset cancels in the lse.
// Implemented in flash_attn.comp (scalar) and flash_attn_cm1.comp; split_k partials keep the HSV stride and
// flash_attn_split_k_reduce.comp writes the lse. cm2 is refused in ggml_backend_vk_device_supports_op.
const bool EMIT_LSE        = (Flags & 8192) != 0;
// GGML_VK_FA_PV_REG_QUANT=1 adds bit 32768 (with 1024|4096, mode 3 only): q8_0 K and/or V allowed into the cm1 PV_REG
// path, dequantized per subgroup into its private kvsh region (flash_attn_cm1.comp PV_REG_TYPES). Off by default.
const bool PV_REG_Q8_REQ   = (Flags & 32768) != 0;
const uint32_t HSV_out     = HSV + (EMIT_LSE ? 4 : 0);

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

layout (binding = 7) readonly buffer SP {int32_t data_sparse[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1

#include "fa_types.glsl"

#if defined(BFLOAT16)
#define O_TYPE float
#define O_TYPEV4 vec4
#else
#define O_TYPE FLOAT_TYPE
#define O_TYPEV4 FLOAT_TYPEV4
#endif

// These can't be `const` globals because GLSL forbids function calls in global
// const initializers, even when the spec constants would let the driver fold
// them. Macros expand at the use site and fold after specialization.
#define BLOCK_SIZE_K fa_block_elems(FaTypeK)
#define BLOCK_SIZE_V fa_block_elems(FaTypeV)
// F16 reads f16 elements directly from the binding; everything else routes
// through dequantize4 / the MMQ helpers to unpack from the packed block layout.
#define USE_DECODE_K (FaTypeK != GGML_TYPE_F16)
#define USE_DECODE_V (FaTypeV != GGML_TYPE_F16)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, gqa_T, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride, sparse_base;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    // Grouped query attention: a workgroup covers gqa_T consecutive tokens x gqa_ratio heads
    // that share one KV head, as N = gqa_T * gqa_ratio rows: row r -> token gqa_iq1 + r / gqa_ratio,
    // head iq2 + r % gqa_ratio. gqa_T == 1 is the classic single-token fold.
    gqa_T = p.gqa_ratio > 1 ? N / p.gqa_ratio : 1;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = (gl_WorkGroupID.x / p.k_num) * gqa_T;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x * gqa_T;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;

    // Sparse: the tile shares one mask row (gqa heads, or Br==1). split_k
    // partitions the n_kv_max blocks.
    if (USE_SPARSE) {
        uint32_t qrow = (p.gqa_ratio > 1) ? gqa_iq1 : (i * Br);
        sparse_base = (((iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * p.nem1 + qrow) * p.split_kv;

        uint32_t total_blocks = CEIL_DIV(p.split_kv, Bc);
        uint32_t per_blocks   = CEIL_DIV(total_blocks, p.k_num);
        start_j = min(split_k_index * per_blocks, total_blocks);
        end_j   = min((split_k_index + 1) * per_blocks, total_blocks);
    }
}

// Resolve a linear KV slot to a real column; false for inactive (sparse padding/-1, or dense OOB).
bool fa_kv_index(uint lin, out uint kv_col) {
    if (USE_SPARSE) {
        if (lin >= p.split_kv) {
            kv_col = 0;
            return false;
        }
        int idx = data_sparse[sparse_base + lin];
        kv_col = idx >= 0 ? uint(idx) : 0;
        return idx >= 0;
    }
    kv_col = lin;
    return !KV_bounds_check || lin < KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
// row_stride: HSV for split_k partials, HSV_out for the final dst.
void gqaStore(const in uint32_t r, const in uint32_t c, const in O_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N, const in uint32_t row_stride)
{
    uint32_t offset = (iq2 + r) * row_stride / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}

// EMIT_LSE tail of one dst row: o_vec4 indexes data_ov4 at the row's first O element.
void storeLse(const in uint32_t o_vec4, const in float M, const in float L)
{
    const float lse = (L == 0.0) ? uintBitsToFloat(0xFF800000) : M + log(L);
    data_ov4[o_vec4 + HSV / 4] = D_TYPEV4(lse, 0.0, 0.0, 0.0);
}
