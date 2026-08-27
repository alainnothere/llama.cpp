#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

// stochastic (rejection sampling) variant of common_sampler_sample_and_accept_n
//
// same contract as the greedy version: returns the accepted prefix of the draft plus one final token,
// and calls common_sampler_accept() exactly once for every returned token
//
// `dists[i]` is the distribution that the draft sampled `draft[i]` from. positions with an empty (or
// missing) distribution fall back to exact-match verification, which is what a deterministic proposal
// reduces to anyway. if no position has a distribution, or if a grammar is in use (the residual is not
// grammar-constrained), this simply forwards to the greedy version
//
// per-call accounting of which verification rule each drafted position actually went through -
// this is the only way to tell from the outside whether the stochastic path is doing anything
struct common_spec_verify_stats {
    uint32_t n_pos_stoch = 0; // positions verified by rejection sampling
    uint32_t n_acc_stoch = 0; // ... of which accepted
    uint32_t n_pos_exact = 0; // positions verified by exact match (no proposal distribution / grammar / no probs)
    uint32_t n_acc_exact = 0; // ... of which accepted
    uint32_t n_fallback  = 0; // 1 if the whole call was forwarded to the greedy version
};

// requires: idxs.size() == draft.size() + 1
std::vector<llama_token> common_sampler_sample_and_accept_n_stochastic(
        struct common_sampler * gsmpl,
        struct llama_context  * ctx,
        const std::vector<int> & idxs,
        const llama_tokens     & draft,
        const common_draft_dists & dists,
        bool grammar_first = false,
        common_spec_verify_stats * stats = nullptr);

// verify a single drafted token against the target distribution using rejection sampling
// (Leviathan et al., arXiv:2211.17192):
//
//   accept  id_dft  with probability  min(1, p_tgt(id_dft) / p_dft(id_dft))
//   else    sample a replacement from the normalized residual  max(0, p_tgt - p_dft)
//
// the returned token is distributed exactly as p_tgt, for any proposal p_dft
//
// note: for a deterministic proposal (p_dft is a point mass on id_dft - argmax drafting, the ngram-*
//       implementations, ...) this reduces to "accept with probability p_tgt(id_dft), otherwise sample
//       from p_tgt", i.e. it is exactly equivalent to the greedy sample-and-compare scheme. all of the
//       gain comes from proposals with a real spread
//
// cur_p_tgt: target candidates *after* the sampler chain has been applied, with normalized `p`
//            modified in place (turned into the unnormalized residual when the draft token is rejected)
// dft, n_dft: the proposal distribution
// r_accept, r_resample: two independent uniforms from [0, 1)
// accepted: set to true iff the drafted token was accepted (optional)
llama_token common_spec_verify_token(
        llama_token_data_array   * cur_p_tgt,
        const llama_token_data   * dft,
        size_t                     n_dft,
        llama_token                id_dft,
        float                      r_accept,
        float                      r_resample,
        bool                     * accepted);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
