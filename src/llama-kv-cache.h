#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
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
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
        // [KQ_MASK_INC] false: no persistent mask, for wrappers whose graphs never read get_kq_mask_dev()
                       bool   kq_mask_inc = true);

    ~llama_kv_cache();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    void state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags = 0) const override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    // [CP] context-parallel attention, range-split KV, see cp_state
    bool               cp_active(int32_t il) const;            // layer il's K/V is split into ranges
    uint32_t           cp_split(int32_t il, uint32_t n_kv) const; // boundary cell h when n_kv > h, 0: attend on range 0 only
    bool               cp_check() const;
    ggml_backend_dev_t cp_dev(bool helper) const;               // device of range 1 (helper) or range 0 (owner)
    ggml_tensor *      cp_kq_mask_dev() const;                 // the helper's persistent mask (nullptr: graph input)
    bool               cp_pack() const;                        // LLAMA_CP_PACK (default 1), see cp_state::pack
    int                cp_kq_build_dev() const;                // kq_build anchor index of the helper's device, -1: none
    int                cp_n_ranges() const;                    // 0: no CP layer in this cache
    bool               cp_ref_dirty() const;                   // LLAMA_CP_CHECK reference waits for the next update
    bool               cp_on_device_ok() const;                // ON_DEVICE state works (all ranges on one buffer type)
    // K/V cells [i0, i1) of layer il: from the range that holds them (must be one range), or from the
    // LLAMA_CP_CHECK reference (ref = true, global cells)
    ggml_tensor * get_k_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const;
    ggml_tensor * get_v_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const;
    // store into range r with that range's local indices (build_input_cp_idxs), r = -1: the reference, global indices
    ggml_tensor * cpy_k_range(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il, int r) const;
    ggml_tensor * cpy_v_range(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il, int r) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    // [CP] per-range local store indices, shared by K and V and by every CP layer (see cp_state)
    ggml_tensor * build_input_cp_idxs(ggml_context * ctx, const llama_ubatch & ubatch, int r) const;
    void          set_input_cp_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo, int r) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // [KQ_MASK_INC] persistent per-device KQ mask, see the comment at kq_mask_inc_state
    // true if a graph for this ubatch can read the mask from get_kq_mask_dev() instead of a graph input
    bool          kq_mask_dev_usable(const llama_ubatch & ubatch, int64_t n_stream, ggml_type type, bool causal_attn) const;
    // the device-resident mask tensor for model layer il (nullptr: use the graph input)
    ggml_tensor * get_kq_mask_dev(int32_t il) const;
    // build the mask on the host and upload what changed since the last upload to every device tensor
    void          set_input_kq_mask_dev(const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const;

    // [KQ_MASK_DEV] device-built mask (GGML_OP_KQ_MASK_BUILD), see the comment at kq_build_state
    bool          kq_build_enabled() const { return kq_build.enabled; }
    bool          kq_build_check()   const { return kq_build.check; }
    // ints per cell in the cells input: 1 = pos only (n_seq_max == 1, -np 1: 4 B/cell), 2 = pos + seq bitmap
    int64_t       kq_build_cell_stride() const { return n_seq_max == 1 ? 1 : 2; }
    // a cell holds a 2-D M-RoPE position that the 1-D op cannot order (see llama_kv_cells::ext_2d_any)
    bool          kq_build_ext_2d()  const { return v_cells[0].ext_2d_any(); }
    // the causal flag the host build would apply (non_causal_type SWA_ONLY folds into it)
    bool          kq_build_causal(bool causal_attn) const;
    // index of the attention device of model layer il into kq_build_anchor() (-1: no op for this layer)
    int           kq_build_dev(int32_t il) const;
    ggml_tensor * kq_build_anchor(int idx) const { return kq_build.anchor[idx]; }
    // fills cells [0, n_kv) and rows [0, n_tokens) of the two fixed-size graph inputs
    void          set_input_kq_build(ggml_tensor * cells, ggml_tensor * rows, const llama_ubatch * ubatch, int64_t n_kv) const;
    // LLAMA_KQ_MASK_DEV_CHECK=1: host build of the same mask vs each device's op output (dense [n_kv, n_tokens]), aborts on a mismatch
    void          kq_build_verify(const std::vector<ggml_tensor *> & outs, const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // [KQ_MASK_INC] the persistent mask in the same buffer as k (nullptr for shared layers)
        ggml_tensor * kq_mask_dev = nullptr;

        // [KQ_MASK_DEV] index into kq_build.anchor of this layer's buffer type (-1: op off)
        int kq_build_dev = -1;
    };

    // [KQ_MASK_DEV] 2026-10-22: the KQ mask built on each attention device by GGML_OP_KQ_MASK_BUILD
    // (design-kq-mask-on-device.md, ~/Documents/contextFiles).
    //
    // Why (Qwen3.8-27B Q8, 3 cards layer split, -b 8192 -ub 2048, -fa on, LLAMA_DECODE_STATS): the host-built f16 mask
    // [n_kv, n_tokens] is a graph input the sched copies to every attention device with 4 rotating copies: per 8192 batch
    // at 60-75k, 12 copies of 251-330 MB, 150-220 ms each, sched.t_copy 1.9-2.0 s, sched.bytes 4.0-4.4 GB; the 4 copies
    // are reserved at kv_size x n_ubatch x 2 B (876 MiB at 224k) each, 3.5 GB per card the fit leaves out of n_ctx. Worse,
    // the mask grows with n_kv every ubatch, so after the first pp<->tg switch the gallocr plan is too small for every pp
    // ubatch: ggml_backend_sched_alloc_splits drains every backend, then re-plans. Step-0 measure 2026-10-22 (46.8k-token
    // prompt, port 8090): fresh server n_realloc 0; same prompt after one generation 22 re-plans, t_sched_alloc 27.2 s of
    // which 26.3 s the drain, prompt 78.0 s -> 94.3 s; realloc reason every time "src 0 attn_inp_kq_mask ... > planned".
    //
    // What: two fixed-size I32 graph inputs, cells [2, cap] {pos or -1, seq bitmap} ([1, cap] {pos} when n_seq_max == 1)
    // and rows [2, n_ubatch] {pos, seq},
    // and one op per attention buffer type, anchored on that buffer type's first K tensor so the sched places it there
    // (ggml-backend.cpp, rule "1.anchor"). The op output is a fixed 1-D f16 of kv_size*n_ubatch elements (one graph
    // intermediate per device, sized at reserve), FA reads a dense [n_kv, n_tokens] view of its start. Nothing in the
    // graph grows with n_kv except cells, whose cap is kv_size for pp ubatches and n_kv for decode (llama-graph.cpp
    // kq_build_cells_cap has the measured reasons).
    //
    // Gate (static, constructor): LLAMA_KQ_MASK_DEV != 0 (default on), FA (!v_trans, f16 mask), one stream, no SWA, no
    // ALiBi, n_seq_max <= 32 (one bitmap word), every attention buffer type's device supports the op. When on, lever 6a
    // (kq_mask_inc_state) is not allocated. Gate (per graph, llama-graph.cpp kq_build_decide): not training, the ubatch
    // has no 2-D (M-RoPE) position the 1-D compare cannot order: either the ubatch is not 2-D, or every cell's ext is
    // (pos, pos) (M-RoPE text: Qwen3.5/3.8 make EVERY ubatch is_pos_2d(), so a plain !is_pos_2d() gate would exclude the
    // production model; llama_kv_cells::ext_2d_any). Else the host graph input exactly as before.
    // Semantics = set_input_kq_mask_impl<f16, causal, swa=false, is_2d=false, alibi=false>, first seq id and main pos of
    // each token. LLAMA_KQ_MASK_DEV_CHECK=1 compares the host build against every device output after each ubatch.
    //
    // Measured 2026-10-22 (same 46.8k prompt twice, the second after a generation, i.e. after a pp<->tg switch):
    //   host mask (LLAMA_KQ_MASK_DEV=0, fit n_ctx 233728): 2nd prompt 88.4 s, n_realloc 22, drain 35.7 s, sched 11.65 GB,
    //     last 8192 batch 1.93 ms/token; compute buffers 4700/4340/4340 MiB + Vulkan_Host 4052 MiB
    //   device mask (-c 270336): 2nd prompt 55.7 s, n_realloc 1, drain 0, sched 4.87 GB, t_set_inputs 58-220 ms -> 0.1 ms
    //     per 8192 batch, last batch 1.08 ms/token; compute buffers 2109/1748/1748 MiB + Vulkan_Host 405 MiB
    //   fresh server (no re-plan either way) 1.09 vs 1.08 ms/token at the last batch: the win is the removed drain.
    //   The fit then picks n_ctx 289024 and the first prompt ran 319 s (GTT spill, the known fit overcommit at ub 2048;
    //   -c 270336 runs 57 s). Decode p50 per step 25.3 -> 26.5 ms (sched 718 -> 1023 KB/step; 6 samples, split differs).
    //   ppl gate (-c 8192 -b 8192 -ub 2048, 12 chunks): identical per chunk (2.8989 +/- 0.02573), 1159 -> 1233 t/s.
    struct kq_build_state {
        bool enabled = false;
        bool check   = false;

        std::vector<ggml_tensor *> anchor; // one K tensor per attention buffer type

        mutable std::vector<ggml_fp16_t> host; // LLAMA_KQ_MASK_DEV_CHECK scratch
        mutable std::vector<ggml_fp16_t> dev;
        mutable uint64_t n_checked = 0;
    };

    kq_build_state kq_build;

    // [KQ_MASK_INC] 2026-10-21: persistent per-device KQ mask with incremental upload.
    //
    // Why: the mask was a graph input built on the host and copied by the scheduler to every device
    // that holds attention layers, every decode step. LLAMA_DECODE_STATS --sched, Q8 Qwen3.8-27B, 3 cards
    // (Vulkan2 = 7900 XT on PCIe x1, Vulkan3/Vulkan1 = XTX), 2008 verify decodes at 1-12 rows: CPU->Vulkan2
    // 9.7 input copies/step, 366 KB, per-step sum p50 2.39 ms p90 3.95 (single copy p90 1.11 ms);
    // CPU->Vulkan3 5.7 copies 559 KB 1.34 ms; CPU->Vulkan1 6.7 copies 476 KB 0.29 ms; ~4 ms of an ~87 ms
    // step. The mask is n_kv x n_tokens f16 (~1.2 MB/card at 100k ctx, 6 rows) and only its tail changes.
    //
    // What: one 1-D f16 tensor of kv_size*n_rows per buffer type holding attention layers, owned like K/V;
    // graphs view it as [n_kv, n_tokens] (contiguous, row stride n_kv, the same layout as the graph input)
    // so the scheduler inserts no copy. A host shadow holds what every device holds (all devices get the
    // same bytes every upload, so one shadow serves all). row_lo[r] = lowest column of shadow row r that may
    // differ from a fresh build; each step uploads columns [min row_lo, n_kv) of the n_tokens rows with one
    // 2-D set per device.
    //
    // Gate (static, else the graph input path exactly as before): LLAMA_KQ_MASK_INCREMENTAL != 0, FA (f16
    // mask, !v_trans), one stream, no SWA, no ALiBi, causal, not training, n_tokens <= n_rows
    // (LLAMA_KQ_MASK_INC_ROWS, default 64; prompt ubatches take the graph input). Soft-cap does not enter the
    // mask (it is applied to KQ), so it needs no gate.
    // Gate (dynamic, else a full build + full upload into the persistent tensors): one seq in the ubatch, the
    // same seq and n_kv as the previous upload, and every cell of that seq at or after the ubatch's min
    // position is a ubatch cell (positions did not go backwards over retained cells). a multi-seq ubatch also leaves
    // every row unsettled (row_lo 0) for the next step: its rows of the other seqs hold their own history.
    //
    // Why the watermark is exact: a column j left of row_lo was, at its upload, empty (-inf), not in the seq
    // (-inf) or in the seq with pos < the ubatch min pos (0 for every row). Any later change of cell j goes
    // through a llama_kv_cells mutator and lowers row_lo via the dirty slot (seq_rm of a rejected draft
    // tail, find_slot/apply_ubatch reuse, seq_cp/keep/add/div, state restore, clear); a retained cell with
    // pos < the old min stays below every new row's pos while the dynamic gate holds. The cells of the
    // uploaded ubatch (all cells with pos >= its min) carry the causal triangle and flip next step, so after
    // an upload row_lo = their lowest index, not n_used. Rollback: the server seq_rm's the rejected tail and
    // the next step reuses those cells; both lower row_lo to the lowest removed cell. n_kv growth (padding
    // by 256) changes the row stride -> full upload. LLAMA_KQ_MASK_CHECK=1 verifies the settled region
    // against a fresh build and reads every device tensor back after the upload; aborts on a mismatch.
    struct kq_mask_inc_state {
        bool     enabled = false;
        bool     check   = false;
        int      slot    = -1;  // dirty slot in v_cells[0]
        uint32_t n_rows  = 0;

        std::vector<ggml_tensor *> dev; // one per buffer type

        std::vector<ggml_fp16_t> shadow; // [n_rows][n_kv], row stride n_kv
        std::vector<ggml_fp16_t> tmp;    // LLAMA_KQ_MASK_CHECK scratch
        std::vector<uint32_t>    row_lo; // [n_rows]

        int64_t      n_kv = -1;
        llama_seq_id seq  = -1;

        bool        logged_inc  = false;
        const char * last_reason = nullptr;

        uint64_t n_inc   = 0; // incremental uploads
        uint64_t n_full  = 0; // full uploads (dynamic gate)
        uint64_t bytes   = 0; // bytes uploaded per device
        uint64_t bytes_full = 0; // what full uploads would have been
    };

    mutable kq_mask_inc_state kq_inc;

    // [CP] context-parallel attention for ONE layer (env LLAMA_CP_LAYER=<il>, default -1 = off), RANGE-SPLIT KV
    // (2026-10-22 night, design-cp-range-split-kv.md in ~/Documents/contextFiles; replaces the 2026-10-21 full mirror).
    //
    // What: layer il's K/V cells are split at h: range 0 = cells [0, h) on the owner A (model.dev_layer(il), whatever
    // device the fit gives it), range 1 = cells [h, kv_size) on the helper B (LLAMA_CP_DEV=<device name>, mandatory, a
    // fixed device; "CPU" = the CPU test mode). Nothing is duplicated. h = LLAMA_CP_SPLIT_FRAC (default 0.5) of kv_size
    // rounded to 256 (every FA KV tile divides 256 and n_kv is padded to 256, so h and n_kv - h stay tile aligned),
    // fixed at construction; LLAMA_CP_SPLIT_CELL=<h> overrides it (tests). While n_kv <= h the graph is the plain one
    // over range 0; above, FA_A runs over [0, h) on A and FA_B over [h, n_kv) on B, A merges (build_attn_cp).
    // Why: the mirror cost one layer's full K/V on B (1056 MiB per layer at -c 270336, 16 layers = 17 GB, Vulkan1 has
    // ~3 GB), the production fit summary printed est. free -1658 MiB on Vulkan1, and cp_resync re-copied it through the
    // host after every host K/V write (410 MiB / 233 ms per layer, ~4 s per restore at 16 layers).
    // Layout: each range tensor has (c1 - c0) + 1 rows; the last row is the TRASH row: a ubatch's tokens are stored by
    // one fixed-shape set_rows per range (build_input_cp_idxs), tokens of the other range go to the trash row (the CPU
    // set_rows asserts 0 <= idx < ne1, and a per-ubatch token split would change the graph shape every step). FA never
    // reads it (reads are [0, n_kv - c0) of a range, n_kv <= kv_size).
    // Host side: every cell-indexed K/V access (state write/read/clear, K-shift) goes through cp_rows, which splits a cell
    // range at the bounds; the blob bytes are identical to a no-CP cache's. ON_DEVICE state is refused across two buffer types (per-buft
    // byte counts depend on where the restored cells land). The helper's range is allocated on B in every fit probe
    // whatever device A is (2026-10-21: a helper share that moved between probes was sized out of Vulkan1 -> GTT).
    // LLAMA_CP_CHECK=1 adds a full reference copy on A (ref_k / ref_v, global set_rows) that cp_fa_full reads.
    // Not supported (cp stays off, logged): n_stream > 1, v_trans (no FA), SWA, MLA, offload off, kv_size < 512, a shared
    // layer. LLAMA_CP_DEV missing or not a device of the split: throws.
    struct cp_range {
        uint32_t c0 = 0;  // global cells [c0, c1)
        uint32_t c1 = 0;
        ggml_backend_dev_t         dev  = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_tensor * k = nullptr; // [n_embd_k_gqa, c1 - c0 + 1]
        ggml_tensor * v = nullptr; // [n_embd_v_gqa, c1 - c0 + 1]
    };

    struct cp_state {
        int32_t  il    = -1;
        float    frac  = 0.5f;
        uint32_t h     = 0;
        bool     check = false;             // LLAMA_CP_CHECK=1, see llama_context::cp_check_step
        // LLAMA_CP_PACK (default 1; 0 = the 2026-10-21 graph). 2026-10-22 evening: one D2D hop each way instead of
        // 4 + 2 (Q|K|V packed on the owner, mask built on the helper, the helper's partial as one tensor), because each
        // hop costs ~0.13 ms of queue time on the card that submits it. Measurements: llm_graph_context::build_attn_cp.
        bool     pack  = true;

        std::vector<cp_range> ranges;       // empty: no CP layer in this cache

        ggml_tensor * ref_k = nullptr;      // LLAMA_CP_CHECK reference [n_embd_k_gqa, kv_size] on A
        ggml_tensor * ref_v = nullptr;
        ggml_tensor * kq_mask_dev = nullptr; // B's [KQ_MASK_INC] tensor

        // LLAMA_CP_CHECK: cells written from the host since the last update, copied ranges -> reference in update()
        mutable std::vector<std::pair<uint32_t, uint32_t>> ref_dirty;

        bool active() const { return !ranges.empty(); }
    };

    cp_state cp;

    // calls fn(tensor, local row0, n_rows) for each piece of cells [c0, c1) of layer (K or V); one call with the stream
    // tensor for a non-CP layer
    template <typename F>
    void cp_rows(const struct kv_layer & layer, bool is_v, uint32_t strm, uint32_t c0, uint32_t c1, F && fn) const;

    // LLAMA_CP_CHECK: copy cells [c0, c1) of the CP layer from the ranges into the reference after a host write
    void cp_ref_from_ranges(uint32_t c0, uint32_t c1) const;

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, llama_state_seq_flags flags = 0, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // [CP] see llama_kv_cache::cp_state
    bool               cp_active(int32_t il) const;
    uint32_t           cp_split(int32_t il) const;
    bool               cp_check() const;
    ggml_backend_dev_t cp_dev(bool helper) const;
    ggml_tensor *      cp_kq_mask_dev() const;
    bool               cp_pack() const;
    int                cp_kq_build_dev() const;
    int                cp_n_ranges() const;
    ggml_tensor *      get_k_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const;
    ggml_tensor *      get_v_range(ggml_context * ctx, int32_t il, uint32_t i0, uint32_t i1, bool ref) const;
    ggml_tensor *      cpy_k_range(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il, int r) const;
    ggml_tensor *      cpy_v_range(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il, int r) const;
    ggml_tensor *      build_input_cp_idxs(ggml_context * ctx, const llama_ubatch & ubatch, int r) const;
    void               set_input_cp_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, int r) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // [KQ_MASK_INC] see llama_kv_cache::kq_mask_inc_state
    bool          kq_mask_dev_usable(const llama_ubatch & ubatch, int64_t n_stream, ggml_type type, bool causal_attn) const;
    ggml_tensor * get_kq_mask_dev(int32_t il) const;
    void          set_input_kq_mask_dev(const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const;

    // [KQ_MASK_DEV] see llama_kv_cache::kq_build_state
    bool          kq_build_enabled() const;
    bool          kq_build_check() const;
    bool          kq_build_ext_2d() const;
    int64_t       kq_build_cell_stride() const;
    bool          kq_build_causal(bool causal_attn) const;
    int           kq_build_dev(int32_t il) const;
    ggml_tensor * kq_build_anchor(int idx) const;
    uint32_t      kq_build_kv_size() const;
    void          set_input_kq_build(ggml_tensor * cells, ggml_tensor * rows, const llama_ubatch * ubatch, int64_t n_kv) const;
    void          kq_build_verify(const std::vector<ggml_tensor *> & outs, const llama_ubatch * ubatch, int64_t n_kv, bool causal_attn) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
