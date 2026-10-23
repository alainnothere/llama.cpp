#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups,
    // plus one pre-ubatch group (index n_rs_seq + 1) when n_rs_seq > 0, see rs_pre_pos
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // per-seq bound on how far back the snapshot groups hold usable states: exactly what the LAST
    // decode of that seq wrote. a decode of N tokens (re)writes groups [0, N) relative to its own
    // tail and leaves deeper groups untouched (see ggml-vulkan gated_delta_net.comp: "older slots
    // are caller-owned"), so those deeper groups still describe positions relative to an OLDER tail
    // and must never be selected - the bound is therefore overwritten, not raised, on every decode.
    // reset to 0 whenever the groups are invalidated wholesale (clear/state_read/full seq_rm), since
    // state_read only restores group 0 - without this a partial rewind right after a checkpoint or
    // disk-cache restore would select a group still holding another prompt's state.
    std::vector<uint32_t> rs_depth;

    // per-seq position of the state held by the pre-ubatch group (-1 = invalid).
    // every ubatch copies each seq's input state (the row s_copy() selected, i.e. the state before this
    // ubatch) into group n_rs_seq + 1, so a rollback to exactly the start of the last ubatch of a seq is a
    // group select at any depth, also past n_rs_seq. this replaces the host checkpoint of the server for
    // spec drafts longer than the window (2026-10-20, 27B: 157 MB blob, 66-110 ms per long verify step).
    std::vector<llama_pos> rs_pre_pos;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // reset the pre-ubatch position of seq_id (all seqs when seq_id < 0)
    void reset_rs_pre_pos(llama_seq_id seq_id);

    // true when the pre-ubatch group is allocated and written by every graph
    bool has_rs_pre() const;

    // the rs_idx a partial rollback of seq_id to p0 (p0 > 0, p0 <= tail pos) would select, or -1 if none can serve it.
    // no state changes - seq_rm() and can_rollback_to() share it so they cannot disagree
    int32_t rollback_idx(llama_seq_id seq_id, llama_pos p0) const;

    // true when seq_rm(seq_id, p0, -1) would succeed now
    bool can_rollback_to(llama_seq_id seq_id, llama_pos p0) const;

    // set the valid rollback depth of seq_id to what the decode that just ran wrote
    void set_rs_depth(llama_seq_id seq_id, uint32_t depth);

    // reset the valid rollback depth of seq_id (all seqs when seq_id < 0)
    void reset_rs_depth(llama_seq_id seq_id);

    // how far back seq_id may currently be rolled back
    uint32_t get_rs_budget(llama_seq_id seq_id) const;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // [fork, 2026-10-22 night] head / n / rs_z are read from the memory at build time and are those of the last
    // find_slot(): for a seq continuing its decode find_slot keeps them, and llm_graph_input_rs::can_reuse rejects the
    // candidate when the real apply moved them
    bool prepare_graph() override { return status == LLAMA_MEMORY_STATUS_SUCCESS && !ubatches.empty(); }

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    // first row of the pre-ubatch group in the state tensors, -1 when it does not exist
    int64_t get_rs_pre_row() const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
