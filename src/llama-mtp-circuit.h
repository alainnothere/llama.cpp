#pragma once

// [MTP_CIRCUIT] (fork, 2026-11-07) pass list of the NextN/MTP draft graph, see llama_mtp_set_circuit in llama-ext.h
// for the grammar. This header is the internal side: the parsed list the graph builder reads, and the model checks.

#include "llama-ext.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_hparams;

// at most this many passes per circuit (own + borrowed). Bounds the graph size; the draft graph's node budget
// (graph_max_nodes) is sized for the whole model, a pass is ~40 nodes.
#define LLAMA_MTP_CIRCUIT_MAX_LEN 16

struct llama_mtp_circuit {
    std::vector<llama_mtp_pass> passes; // empty = default (one own M0 pass)
    std::string                 str = "M0";
    int32_t                     n_own = 1;

    // true when the graph is the pre-circuit one: a single own-K/V full-block M0 pass
    bool is_default() const {
        return passes.empty() || (passes.size() == 1 && passes[0].kind == 'M' && passes[0].idx == 0 && passes[0].part == 'B' &&
                passes[0].borrow < 0);
    }
};

// a pass that writes its own K/V slot: a full block or an .attn sub-block without ':b<j>' (.ffn has no attention)
inline bool llama_mtp_pass_owns_kv(const llama_mtp_pass & p) {
    return p.part != 'F' && p.borrow < 0;
}

// grammar + model checks in one: arch_ok = the draft graph of this arch reads the circuit (qwen35 only); hp = the
// model's hparams; max_passes = own-K/V slots allocated. On error returns false and sets err.
bool llama_mtp_circuit_build(const char * str, bool arch_ok, const llama_hparams & hp, uint32_t max_passes,
        llama_mtp_circuit & out, std::string & err);

// K/V layer id of an own-K/V slot in the draft memory: slot 0 is the NextN block's own layer (n_layer()), slot s >= 1
// is the clone layer n_layer_all + s - 1 that llama_kv_cache allocates when mtp_circuit_max_passes > 1
int32_t llama_mtp_circuit_slot_il(const llama_hparams & hp, int32_t slot);

// true when switching from a to b keeps the draft K/V valid: every own-K/V pass of b sits in the common prefix of
// the two lists, so each slot b writes was written by the same pass (same residual input) under a
bool llama_mtp_circuit_keeps_kv(const llama_mtp_circuit & a, const llama_mtp_circuit & b);
