#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // cap for the n-gram drafters only (ngram-simple, ngram-map-k*, ngram-mod, ngram-cache), <= 0 = use n_max
    // lets n-gram drafts, which are free to build, run longer than the draft-model cap
    // note: last member, so the positional initializers in the server and the examples stay valid
    int32_t n_max_ngram = -1;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
//
// `keep`, when non-empty, has one entry per batch row and marks the rows that the target model has
// committed to: a 0 entry is a drafted token that the target has just rejected and whose KV is about
// to be dropped from the target memory, so there is no point in mirroring it into the draft state.
// implementations that can skip such rows do so, the rest simply replay the whole batch
bool common_speculative_process(common_speculative * spec, const common_batch & batch, const std::vector<int8_t> & keep = {});

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// true for the n-gram drafters (no draft model involved): ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache
bool common_speculative_type_is_ngram(enum common_speculative_type type);

// the implementation that produced the draft of the last common_speculative_draft() call for seq_id
// COMMON_SPECULATIVE_TYPE_COUNT if no implementation drafted
enum common_speculative_type common_speculative_type_last(const common_speculative * spec, llama_seq_id seq_id);

// length of that draft as built by the implementation, before the n_max / n_max_ngram truncation
// and after it (= what was handed to the target for verification). 0 if no implementation drafted
int32_t common_speculative_n_built_last  (const common_speculative * spec, llama_seq_id seq_id);
int32_t common_speculative_n_offered_last(const common_speculative * spec, llama_seq_id seq_id);

// wall time of all the draft() calls tried for seq_id in the last common_speculative_draft() call
int64_t common_speculative_t_draft_last_us(const common_speculative * spec, llama_seq_id seq_id);

//
// shadow chains (log only): what a draft would have landed if the target had verified more of it
//

enum common_speculative_shadow_kind {
    COMMON_SPECULATIVE_SHADOW_NGRAM_FULL, // n-gram chain before the cap truncation
    COMMON_SPECULATIVE_SHADOW_MTP_EXT,    // draft-model draft + ngram-mod extension of it
    COMMON_SPECULATIVE_SHADOW_NGRAM_ALT,  // chain of the ngram-mod shadow table when the main one did not draft
};

struct common_speculative_shadow_chain {
    common_speculative_shadow_kind kind = COMMON_SPECULATIVE_SHADOW_NGRAM_FULL;

    size_t       pos_first = 0; // index in the committed token vector of tokens[0]
    llama_tokens tokens;
    int32_t      n_prefix  = 0; // MTP_EXT: leading tokens that are the offered draft
};

struct common_speculative_shadow_stats {
    // NGRAM_FULL, matched buckets: 0, 1-4, 5-8, 9-16, 17-32, 33-48, 49-64, 65+
    int32_t ngram_chains      = 0; // resolved
    int32_t ngram_censored    = 0; // unresolved at request end (getter adds the still pending ones)
    int32_t ngram_full        = 0;
    int64_t ngram_matched_sum = 0;
    int32_t ngram_hist[8]     = {};

    // MTP_EXT, landed buckets: 0, 1-4, 5-8, 9-16, 17-32, 33-48, 49+; built buckets: 1-4, 5-8, 9-16, 17-32, 33-64, 65+
    int32_t ext_attempts      = 0;
    int32_t ext_hits          = 0;
    int32_t ext_resolved      = 0;
    int32_t ext_censored      = 0;
    int32_t ext_tested        = 0; // resolved with the whole draft landed
    int64_t ext_landed_sum    = 0; // over tested
    int32_t ext_landed_hist[7] = {};
    int32_t ext_built_hist[6]  = {};

    int32_t n_ext_last        = 0; // extension built by the last draft call, 0 if none

    // NGRAM_ALT, same buckets as NGRAM_FULL / ext built
    int32_t alt_n             = 0; // n_match of the shadow table
    int32_t alt_attempts      = 0;
    int32_t alt_hits          = 0;
    int32_t alt_chains        = 0; // resolved
    int32_t alt_censored      = 0;
    int64_t alt_matched_sum   = 0;
    int32_t alt_hist[8]       = {};
    int32_t alt_built_hist[6] = {};
};

// matched length of the chain against the committed tokens prompt ++ [id_last], -1 while unresolved
int32_t common_speculative_shadow_resolve(const common_speculative_shadow_chain & chain, const llama_tokens & prompt, llama_token id_last);

// per seq pending chains and stats
struct common_speculative_shadow_seq {
    static constexpr size_t N_PENDING_MAX = 8;

    std::vector<common_speculative_shadow_chain> pending; // oldest first
    common_speculative_shadow_stats stats;

    void push(common_speculative_shadow_chain chain); // drops the oldest as censored on overflow
    void resolve(const llama_tokens & prompt, llama_token id_last);
    void censor_all();
    void record(const common_speculative_shadow_chain & chain, int32_t matched);
    void censor(const common_speculative_shadow_chain & chain);
};

common_speculative_shadow_stats common_speculative_get_shadow_stats(const common_speculative * spec, llama_seq_id seq_id);

//
// draft-mtp confidence (log only): landing rate by the drafter's probability of each offered token
//

enum common_speculative_mtp_stop {
    COMMON_SPECULATIVE_MTP_STOP_CAP,   // reached n_max or the caller's cap
    COMMON_SPECULATIVE_MTP_STOP_PMIN,  // top candidate below p_min
    COMMON_SPECULATIVE_MTP_STOP_OTHER, // anything else (decode failure)
    COMMON_SPECULATIVE_MTP_STOP_COUNT,
};

struct common_speculative_mtp_stats {
    static constexpr int N_PBIN = 6;  // [0,0.5) [0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9) [0.9,1]
    static constexpr int N_POS  = 16; // 15 = 15+

    int32_t stop_steps   [COMMON_SPECULATIVE_MTP_STOP_COUNT] = {};
    int64_t stop_accepted[COMMON_SPECULATIVE_MTP_STOP_COUNT] = {};

    int32_t pbin_tested[N_PBIN] = {};
    int32_t pbin_landed[N_PBIN] = {};
    int32_t na_tested = 0; // tokens without a probability (p <= 0 or not finite)
    int32_t na_landed = 0;

    int32_t pos_tested[N_POS] = {};
    int32_t pos_landed[N_POS] = {};
};

// account one verified step: p[k] is the drafter's probability of offered token k (< 0 = n/a),
// tokens [0, n_accepted) landed, token n_accepted (if offered) was rejected, the rest were not tested
void common_speculative_mtp_stats_add(common_speculative_mtp_stats & st, const std::vector<float> & p,
        int32_t n_offered, int32_t n_accepted, common_speculative_mtp_stop stop);

// zeros if there is no draft-mtp implementation
common_speculative_mtp_stats common_speculative_get_mtp_stats(const common_speculative * spec, llama_seq_id seq_id);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
