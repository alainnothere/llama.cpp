// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } k_packed_iq4_nl;
layout (binding = 2) readonly buffer V_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } v_packed_iq4_nl;

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_DEQUANT4_IQ4_NL(BUF) {                                                                 \
    const uint shift = (iqs & 0x10) >> 2;                                                         \
    const uint qs_i = (iqs & 0xC) >> 1;                                                           \
    const uint qsw = uint(BUF.data[a_offset + ib].qs[qs_i])                                       \
                   | (uint(BUF.data[a_offset + ib].qs[qs_i + 1u]) << 16);                         \
    const FLOAT_TYPE d = FLOAT_TYPE(BUF.data[a_offset + ib].d);                                   \
    const u8vec4 q = unpack8((qsw >> shift) & 0x0F0F0F0Fu);                                       \
    return d * FLOAT_TYPEV4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],                               \
                            kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);                              \
}

#define FA_DEQUANT4_BF16(BUF) \
    return FLOAT_TYPEV4(bf16_to_fp32(uvec4(BUF.data[(a_offset + ib) / 4])));

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(k_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
        }
    } else {
        switch (FaTypeV) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(v_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
        }
    }
    return FLOAT_TYPEV4(0);
}

#if defined(FA_WANT_DEQUANT8_Q8_0)
// GGML_VK_FA_PV_REG_QUANT C2 (2026-10-24): 8 contiguous q8_0 elements per call for the cm1 PV_REG quant path
// (flash_attn_cm1.comp only, which always has the f16 arithmetic extension). One scale load per 8 elements
// instead of per 4, and no int -> float converts: each int16 holds two int8 quants; XOR 0x80 biases them to
// u8 = q + 128, placing u8 in the low mantissa of f16 0x6400 (= 1024) gives the f16 value 1024 + u8 exactly,
// so (packed - 1152) is q exactly and one packed f16 multiply by d finishes. Bit-identical to dequantize4
// (int8 -> f16 is exact there too, and both round once, in the f16 multiply).
// iqs must be a multiple of 8 (the 8 elements never straddle a 32-element block).
// Measured 2026-10-24, 7900 XTX, 27B shape, knob on, vs the C1 dequantize4 version: 0.90-0.91x at 4-16 tokens
// (229k nb 4/6 1359.7 -> 1231.6, 2007.3 -> 1809.8 us), 0.96-0.97x at 1-2, prefill unchanged. Per extra 16-row tile
// at 131k (GGML_VK_FA_ABLATE): Q.K-only 186 -> 146 us (-21%), P.V-only 222 -> 210 us (-5%).
// C2b: the V side calls dequantize8_q8_0_rot (rotated word order for conflict-free LDS stores): P.V-only per
// extra tile 208.8 -> 198.8 us, full kernel 0.98x at 4-16 tokens; see flash_attn_cm1.comp step C.
#define FA_Q8_0_PAIR(W) unpackFloat2x16((((W) & 0xFFu) | (((W) & 0xFF00u) << 8)) ^ 0x64806480u)
// ROT (0..3) rotates the word order: register slot w holds word (w + ROT) & 3, i.e. element e of (lo, hi) is
// d = (e + 2 * ROT) & 7 of the 8. ROT 0 is the plain loader (K side); the step-C V loop passes d8 & 3 so its
// transposed stores land on distinct LDS banks (C2b, flash_attn_cm1.comp). Pure address arithmetic, no selects.
#define FA_DEQUANT8_Q8_0(BUF, ROT) {                                                               \
    const uint w0 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + ((ROT)     & 3u)]);                  \
    const uint w1 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + ((ROT + 1) & 3u)]);                  \
    const uint w2 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + ((ROT + 2) & 3u)]);                  \
    const uint w3 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + ((ROT + 3) & 3u)]);                  \
    const f16vec2 dd = f16vec2(BUF.data[a_offset + ib].d);                                         \
    const f16vec2 bias = f16vec2(1152.0);                                                          \
    lo = FLOAT_TYPEV4((FA_Q8_0_PAIR(w0) - bias) * dd, (FA_Q8_0_PAIR(w1) - bias) * dd);             \
    hi = FLOAT_TYPEV4((FA_Q8_0_PAIR(w2) - bias) * dd, (FA_Q8_0_PAIR(w3) - bias) * dd);             \
}

void dequantize8_q8_0(uint ib, uint iqs, uint a_offset, uint binding_idx, out FLOAT_TYPEV4 lo, out FLOAT_TYPEV4 hi) {
    if (binding_idx == BINDING_IDX_K) {
        FA_DEQUANT8_Q8_0(k_packed_q8_0, 0u)
    } else {
        FA_DEQUANT8_Q8_0(v_packed_q8_0, 0u)
    }
}

void dequantize8_q8_0_rot(uint ib, uint iqs, uint a_offset, uint binding_idx, uint rot, out FLOAT_TYPEV4 lo, out FLOAT_TYPEV4 hi) {
    if (binding_idx == BINDING_IDX_K) {
        FA_DEQUANT8_Q8_0(k_packed_q8_0, rot)
    } else {
        FA_DEQUANT8_Q8_0(v_packed_q8_0, rot)
    }
}
#endif
