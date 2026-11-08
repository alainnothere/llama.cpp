#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
    // [fork, 2026-10-27] the part of compute that is backend scratch outside the scheduler's buffers (ggml-vulkan
    // prealloc_x/y/split_k/mask, sized at reserve by ggml_backend_sched_get_scratch_size). ALREADY INCLUDED in
    // compute, so total() and every existing sum stay right; kept apart only so the fit summary can print it.
    // Measured gap it closes: prealloc_x 1368 MiB per XTX at -c 350000 (one layer's K+V copy above ~262k cells).
    size_t scratch = 0;
    // [fork, 2026-10-30] fit step 3 B (design-fit-lazy-reserve-and-warmup-gate.md): what the backend allocates LAZILY on
    // this device after the reserve (D2D rings, sync staging, descriptor/pipeline arenas), as reported by the backend's
    // GGML_BACKEND_FORK_DEV_LAZY_RESERVE_PROC for this context's shape; 0/0 when the backend has no such proc.
    // NOT inside compute (it is an estimate, nothing is allocated for it at reserve time), but inside total(), so every
    // fit sum charges it against the device's free exactly like compute. vram = device-local, gtt = host-visible memory
    // the device maps (charged against VRAM free too: pessimistic by design, the kernel's eviction pressure is one pool).
    // Why: 2026-10-30 Vulkan1 placed 22961 of 24511 MiB, 1446 MiB free after init, first request moved ~1 GB to GTT with
    // no SCRATCH GROW line in between (rings, staging and gallocr re-plans were unpriced).
    // Counted once per device, on the buffer type of the backend that runs on it (like scratch).
    size_t lazy_vram = 0;
    size_t lazy_gtt  = 0;
    // the same figure as the INCREMENT this context adds when it is the SECOND context on the device (draft / MTP):
    // proc(params, n_contexts = 2) - proc(params, n_contexts = 1). The fit's add_extra_memory charges this instead of
    // lazy_vram/lazy_gtt for the extra context on a main-model device, so per-device singletons (device sync staging)
    // are counted once, by the target context; per-context parts (context staging, rings, arenas) are counted per context.
    // NOT in total().
    size_t lazy_vram_2nd = 0;
    size_t lazy_gtt_2nd  = 0;

    size_t total() const {
        return model + context + compute + lazy_vram + lazy_gtt;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

// [MTP_HEAD] Overwrite the NextN/MTP block tensors with the ones in the GGUF file at path (same names, types and
// shapes, only blk.<n_layer>.. tensors allowed). All tensors are checked before the first write. Call it after
// load and before decoding. Returns the number of replaced tensors, or < 0 on any mismatch (model unchanged).
LLAMA_API int32_t llama_model_load_mtp_head(struct llama_model * model, const char * path);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

// [GRAPH_REUSE_PP] the graph-reuse rule of process_ubatch, pure for tests/test-graph-reuse-pp.cpp: false when
// pipeline_parallel and the decode call has more than one ubatch, unless env_reuse_pp (LLAMA_GRAPH_REUSE_PP=1)
LLAMA_API bool llama_graph_reuse_allowed(bool pipeline_parallel, uint32_t n_ubatches_in_batch, bool env_reuse_pp);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a joint decision head (clef) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// [VOCAB_SUBSET] Restrict the LM head of this context's draft graphs (the ones built with
// llm_graph_context::build_head_logits, today the qwen35 / qwen35moe MTP heads) to the rows of model.output listed
// in ids. Logits of other tokens become -INF: the host logits row is still n_vocab wide, and backend samplers
// return vocab ids. The rows are gathered on the device that holds model.output (n_pad = n_ids rounded up to
// 256, ~92 MB at 17k rows of the 27B's Q8_0 head). n_ids == 0 or ids == NULL turns it off. Synchronizes the
// context. Returns 0 on success, < 0 when unsupported (no plain model.output, row size % 4 != 0, CPU-repacked
// head, device without a backend in this context, out-of-range ids, allocation failure); the head is then full.
// The list is taken as is: keep it append-only so rows keep their place, and call it between requests, not per
// step - a new n_pad (or the first call) re-creates the tensors and every graph after it is rebuilt.
LLAMA_API int32_t llama_set_logits_subset(struct llama_context * ctx, const llama_token * ids, int32_t n_ids);

//
// [MTP_CIRCUIT] (fork, 2026-11-07) runtime-reconfigurable pass list of the NextN/MTP draft graph (qwen35 dense)
//
// The draft graph is: enorm/hnorm -> concat -> eh_proj (once), then the passes in order over the residual stream,
// then the head norm + LM head (once). Grammar (whitespace ignored, case-insensitive letters):
//   circuit := pass ("," pass)*            "" or "default" = "M0" (the graph before 2026-11-07, bit-identical)
//   pass    := unit [sub] ["*" N] [":" mode]  "*N" repeats the pass N times (1..16), each copy gets the mode
//   unit    := "M" k                       NextN block k (model.layers[n_layer() + k]); qwen35 has k = 0 only
//            | "T" il                      trunk layer il of the TARGET used as a pass (full-attention layers only;
//                                          a gated-delta-net layer is rejected: its recurrent state is not in ctx_dft)
//   sub     := ".attn"                     the attention sub-layer only: x = x + attn(attn_norm(x)); takes a K/V
//                                          slot like a full pass, same modes
//            | ".ffn"                      the FFN sub-layer only: x = x + ffn(post_attn_norm(x)); no K/V, no mode;
//                                          allowed on every trunk layer, gated-delta-net ones included
//   (no sub = the full block = ".attn" then ".ffn" of the same layer, one pass, one slot)
//   mode    := "o"                         own K/V slot (the default)
//            | "b" j                       borrow the K/V slot of pass j (j < this pass): attend over that slot's
//                                          stored K/V of earlier positions, with this pass's K/V for the current
//                                          tokens, and store nothing (the slot keeps pass j's rows)
//            | "t"                         borrow the TARGET's K/V: NOT supported (ctx_tgt and ctx_dft are separate
//                                          contexts with their own cell numbering), rejected with a message
// At most 16 passes in total, and at most mtp_circuit_max_passes (llama_context_params) own-K/V passes (full blocks
// and .attn sub-blocks without ':b<j>'; .ffn passes are free).
struct llama_mtp_pass {
    char    kind   = 'M'; // 'M' NextN block, 'T' target trunk layer
    int32_t idx    = 0;   // M: block offset k, T: trunk layer il
    char    part   = 'B'; // 'B' full block, 'A' attention sub-layer (.attn), 'F' FFN sub-layer (.ffn)
    int32_t borrow = -1;  // -1: own K/V slot, else the index of the pass whose slot is borrowed (resolved to an owner)
    int32_t slot   = 0;   // K/V slot the pass reads: own passes are numbered 0.. in circuit order; -1 for .ffn
};

// Grammar only, no model: fills passes (expanded) and canonical ("M0,M0:b0"), returns false + err on error.
LLAMA_API bool llama_mtp_circuit_parse(const char * circuit, std::vector<llama_mtp_pass> & passes, std::string & canonical, std::string & err);

// Set the pass list of an MTP draft context (ctx_type LLAMA_CONTEXT_TYPE_MTP). Parsed and checked against the model
// and the preallocated own-K/V slots right here, so a bad circuit returns < 0 (message: llama_mtp_circuit_last_error)
// and the active one stays. Thread-safe w.r.t. decode: the new list takes effect at the start of the next decode on
// ctx_dft (the decode thread swaps it in, rebuilds the graph, and clears the draft memory unless every own-K/V pass of
// the new list is in the common prefix with the old one - see llama_context::mtp_circuit_apply). Returns 0 on success.
LLAMA_API int32_t llama_mtp_set_circuit(struct llama_context * ctx_dft, const char * circuit);

// Canonical form of the last accepted circuit (applied at the next decode if not yet), snprintf semantics: returns the
// full length, writes at most buf_size - 1 chars + NUL.
LLAMA_API int32_t llama_mtp_get_circuit(const struct llama_context * ctx_dft, char * buf, size_t buf_size);

// The own-K/V passes the draft memory was allocated for (0 for a non-MTP context).
LLAMA_API int32_t llama_mtp_circuit_max_passes(const struct llama_context * ctx_dft);

// The message of the last rejected llama_mtp_set_circuit ("" if none), snprintf semantics.
LLAMA_API int32_t llama_mtp_circuit_last_error(const struct llama_context * ctx_dft, char * buf, size_t buf_size);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

//
// memory fit (common/fit.cpp)
//

// [fork, 2026-10-22] Probe-only switch for the memory fit: while true, a new llama_context decides pipeline
// parallelism as if every layer were offloaded (the "n_gpu_layers > n_layer_all" condition is skipped; split mode,
// offload_kqv, tensor overrides and async/event support still gate it). The fit's layer-fill probes leave the
// not-yet-filled devices at 0 layers, so without this they measure the scheduler with 1 input copy while the final
// all-GPU context runs with GGML_SCHED_MAX_COPIES (4); see the comment at the use site in common/fit.cpp.
// Global state, not thread safe, same as the logger swap the fit already does. Returns the previous value.
LLAMA_API bool llama_fit_set_assume_full_offload(bool value);
