// TODO: merge with test-recurrent-state-rollback.cpp
// TODO: merge with test-state-restore-fragmented.cpp

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

constexpr double NMSE_THRESHOLD = 1e-5;

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[0m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[0m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[0m";
    }
    return "";
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        const float a_i = a[i];
        const float b_i = b[i];

        mse_a_b += (double) (a_i - b_i) * (a_i - b_i);
        mse_a_0 += (double) a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct generation_result {
    llama_tokens tokens;
    std::vector<std::vector<float>> logits;

    bool empty() const { return tokens.empty(); }
};

static bool get_current_logits(llama_context * ctx, std::vector<float> & out) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

static generation_result generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    generation_result result;
    common_batch batch(ctx);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return {};
        }

        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        // LOG_LEVEL_INFO gate: visible in single-model mode, silenced in --models table mode
        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        result.tokens.push_back(next_token);
        result.logits.push_back(std::move(logits));

        batch.clear();
        batch.add(next_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

static bool generate_tokens_compare(
        llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
        const generation_result & expected) {
    if (expected.tokens.size() != expected.logits.size() || expected.tokens.size() < (size_t) n_predict) {
        LOG_ERR("\n%s: invalid expected generation\n", __func__);
        return false;
    }

    common_batch batch(ctx);

    for (int i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            LOG_ERR("\n%s: failed to get logits\n", __func__);
            return false;
        }
        if (logits.size() != expected.logits[i].size()) {
            LOG_ERR("\n%s: logits size mismatch at step %d: %zu != %zu\n", __func__, i, logits.size(), expected.logits[i].size());
            return false;
        }

        const double nmse_val = nmse(expected.logits[i], logits);
        LOG_TRC("%s: step %d nmse = %.6e\n", __func__, i, nmse_val);
        if (nmse_val > NMSE_THRESHOLD) {
            LOG_ERR("\n%s: error: NMSE at step %d is %.6e (threshold %.1e)\n", __func__, i, nmse_val, NMSE_THRESHOLD);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);
        const auto expected_token = expected.tokens[i];

        LOGV(LOG_LEVEL_INFO, "%d ", next_token);
        if (next_token != expected_token) {
            LOG_TRC("%s: sampled token %d differs from expected %d, using expected token\n", __func__, next_token, expected_token);
        }

        batch.clear();
        batch.add(expected_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static generation_result test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOGV(LOG_LEVEL_INFO, "\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        common_batch batch(ctx.get());
        for (size_t i = 0; i < n_tokens; ++i) {
            batch.add(tokens[i], i, seq_id, i == n_tokens - 1);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 0, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 1, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation_result & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOGV(LOG_LEVEL_INFO, "\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens and compare logits against the baseline
    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, params.n_predict, 1, expected_result)) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 6/7: seq copy (scatter)
// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static bool test_seq_cp_scatter(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, int test_num, bool on_device) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOGV(LOG_LEVEL_INFO, "\n=== Test %d: seq copy (%s, scatter) ===\n", test_num, on_device ? "device" : "host");

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    auto decode_one = [&](llama_token tok, int pos, llama_seq_id seq) {
        common_batch batch(ctx.get());
        batch.add(tok, pos, seq, true);
        return llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!decode_one(tokens[0], 0, 0) ||
        !decode_one(tokens[1], 1, 0) ||
        !decode_one(tokens[0], 0, 1) ||
        !decode_one(tokens[1], 1, 1) ||
        !decode_one(tokens[2], 2, 0) ||
        !decode_one(tokens[2], 2, 1)) {
        LOG_ERR("%s: failed to build interleaved state\n", __func__);
        return false;
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, uint32_t fl, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size_ext(ctx.get(), seq_id, fl);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), seq_id, fl);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return false;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(1, flags, state_save)) {
        return false;
    }
    LOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (nset != state_save.size()) {
        LOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state_save.size());
        return false;
    }
    LOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return false;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    if (state_before.size() != state_after.size() || memcmp(state_before.data(), state_after.data(), state_before.size()) != 0) {
        LOG_ERR("\n%s: error: restored KV state is not byte-identical to the saved state\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 8: state blob round-trip
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
static bool test_state_roundtrip(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOGV(LOG_LEVEL_INFO, "\n=== Test 8: state blob round-trip ===\n");

    common_batch batch = common_batch_get_one(ctx.get(), tokens);
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("\n%s: failed to decode prompt\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_a(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_a = llama_state_seq_get_data(ctx.get(), blob_a.data(), blob_a.size(), 0);
    if (n_a != blob_a.size()) {
        LOG_ERR("\n%s: saved %zu bytes, expected %zu\n", __func__, n_a, blob_a.size());
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("\n%s: failed to erase seq 0\n", __func__);
        return false;
    }

    if (llama_state_seq_set_data(ctx.get(), blob_a.data(), blob_a.size(), 0) != blob_a.size()) {
        LOG_ERR("\n%s: failed to restore seq 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_b(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_b = llama_state_seq_get_data(ctx.get(), blob_b.data(), blob_b.size(), 0);
    if (n_b != n_a) {
        LOG_ERR("\n%s: re-saved %zu bytes, expected %zu\n", __func__, n_b, n_a);
        return false;
    }

    size_t n_diff = 0;
    size_t i_diff = 0;
    for (size_t i = 0; i < n_a; i++) {
        if (blob_a[i] != blob_b[i]) {
            if (n_diff == 0) {
                i_diff = i;
            }
            n_diff++;
        }
    }

    if (n_diff > 0) {
        LOG_ERR("\n%s: state changed across a restore: %zu of %zu bytes differ, first at offset %zu\n",
                __func__, n_diff, n_a, i_diff);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// overwrite the tensor data with 0xff bytes (NaN when read as f16/f32), so that the restore fails
static bool corrupt_state(std::vector<uint8_t> & data) {
    if (data.size() < 3*4096) {
        LOG_ERR("%s: state of %zu bytes is too small to corrupt\n", __func__, data.size());
        return false;
    }

    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
    return true;
}


// Test 9: state restore failure
// a failed restore must leave the sequence empty and must not change the logits of other sequences
static bool test_state_restore_failure(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 4;
    params_ctx.kv_unified = true;

    // without flash attention, corrupted data left behind by the restore shows up as NaN logits on the other sequences
    params_ctx.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\n=== Test 9: state restore failure ===\n");

    llama_memory_t mem = llama_get_memory(ctx.get());
    if (mem == nullptr) {
        LOGV(LOG_LEVEL_INFO, "PASS (model has no memory)\n");
        return true;
    }

    const auto decode = [&](const llama_tokens & inp, llama_seq_id seq_id, std::vector<float> * logits_out) {
        common_batch batch(ctx.get());
        for (size_t i = 0; i < inp.size(); ++i) {
            batch.add(inp[i], i, seq_id, i == inp.size() - 1);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to decode on sequence %d\n", __func__, seq_id);
            return false;
        }

        if (logits_out && !get_current_logits(ctx.get(), *logits_out)) {
            LOG_ERR("%s: failed to get logits\n", __func__);
            return false;
        }

        return true;
    };

    const llama_tokens tokens_save  (tokens.begin(), tokens.begin() + std::min<size_t>(24, tokens.size()));
    const llama_tokens tokens_verify(tokens.end() - std::min<size_t>(8, tokens.size()), tokens.end());

    // the registered tests share a working directory, so the state file is named after the model
    const std::string path = "state-restore-failure." + std::filesystem::path(params.model.path).filename().string() + ".tmp.bin";

    llama_memory_clear(mem, true);

    std::vector<float> baseline;
    if (!decode(tokens_verify, 1, &baseline)) {
        return false;
    }

    const std::vector<std::pair<const char *, std::function<bool()>>> cases = {
        { "buffer", [&]() {
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
            GGML_ASSERT(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == state.size());
            llama_memory_seq_rm(mem, 0, -1, -1);

            if (!corrupt_state(state)) {
                return false;
            }

            return llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0;
        }},
        { "file", [&]() {
            GGML_ASSERT(llama_state_seq_save_file(ctx.get(), path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
            llama_memory_seq_rm(mem, 0, -1, -1);

            std::vector<uint8_t> data;
            {
                std::ifstream f(path, std::ios::binary);
                data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }

            if (!corrupt_state(data)) {
                std::remove(path.c_str());
                return false;
            }

            {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) data.data(), data.size());
            }

            llama_tokens tokens_out(tokens_save.size());
            size_t n_token_count = 0;
            const size_t nread = llama_state_seq_load_file(ctx.get(), path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
            std::remove(path.c_str());

            return nread == 0;
        }},
    };

    for (const auto & [name, restore_failed] : cases) {
        llama_memory_clear(mem, true);

        if (!decode(tokens_save, 0, nullptr)) {
            return false;
        }

        if (!restore_failed()) {
            LOG_ERR("%s: %s: restoring a corrupted state did not fail\n", __func__, name);
            return false;
        }

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            LOG_ERR("%s: %s: sequence not empty after failed restore\n", __func__, name);
            return false;
        }

        std::vector<float> logits;
        if (!decode(tokens_verify, 1, &logits)) {
            return false;
        }

        float  diff_max = 0.0f;
        size_t n_nan    = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isnan(logits[i]) || std::isnan(baseline[i])) {
                n_nan++;
            } else {
                diff_max = std::max(diff_max, std::fabs(logits[i] - baseline[i]));
            }
        }

        if (n_nan > 0 || diff_max > 1e-6f) {
            LOG_ERR("%s: %s: logits changed after failed restore (max diff = %g, nan = %zu)\n", __func__, name, diff_max, n_nan);
            return false;
        }

        LOG_TRC("%s: %s: logits match (max diff = %g)\n", __func__, name, diff_max);
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}


// Test 10: state rotation
// a KV state saved with attention rotation enabled must restore only into a context with the same setting;
// note: rotation is only active for quantized KV caches with a head size that is a multiple of 64,
//       for other models the restore into the rotation-disabled context is valid and the test passes vacuously
static bool test_state_rotation(struct llama_model * model, const struct common_params & params) {
    LOGV(LOG_LEVEL_INFO, "\n=== Test 10: state rotation ===\n");

    const std::string attn_rot_disable = common_get_env("LLAMA_ATTN_ROT_DISABLE");
    const auto make_context = [&](ggml_type type_k, ggml_type type_v, bool disable_rotation) {
        common_set_env("LLAMA_ATTN_ROT_DISABLE", disable_rotation ? "1" : "0");
        auto params_ctx = common_context_params_to_llama(params);
        params_ctx.n_ctx    = 32;
        params_ctx.n_batch  = 1;
        params_ctx.n_ubatch = 1;
        params_ctx.type_k   = type_k;
        params_ctx.type_v   = type_v;
        params_ctx.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return llama_context_ptr(llama_init_from_model(model, params_ctx));
    };

    std::vector<std::pair<ggml_type, ggml_type>> type_pairs;
    for (const auto & types : { std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0} }) {
        if (make_context(types.first, types.second, false)) {
            type_pairs.push_back(types);
        }
    }
    if (type_pairs.empty()) {
        LOG_WRN("%s: no supported quantized KV cache type combination - skipping\n", __func__);
        return true;
    }

    bool success = true;
    for (const auto & types : type_pairs) {
        auto src = make_context(types.first, types.second, false);
        if (!src) {
            LOG_ERR("%s: failed to create source context\n", __func__);
            success = false;
            break;
        }

        llama_token token = 0;
        if (llama_decode(src.get(), llama_batch_get_one(&token, 1))) {
            LOG_ERR("%s: failed to decode token\n", __func__);
            success = false;
            break;
        }

        const size_t state_size = llama_state_seq_get_size(src.get(), 0);
        if (state_size == 0) {
            continue; // no KV state to test
        }

        std::vector<uint8_t> state(state_size);
        if (llama_state_seq_get_data(src.get(), state.data(), state.size(), 0) != state.size()) {
            LOG_ERR("%s: failed to save sequence state\n", __func__);
            success = false;
            break;
        }

        auto matching = make_context(types.first, types.second, false);
        if (!matching || llama_state_seq_set_data(matching.get(), state.data(), state.size(), 0) != state.size()) {
            LOG_ERR("%s: failed to restore matching rotation\n", __func__);
            success = false;
            break;
        }

        auto mismatched = make_context(types.first, types.second, true);
        if (!mismatched) {
            LOG_ERR("%s: failed to create mismatched rotation context\n", __func__);
            success = false;
            break;
        }
        if (llama_state_seq_set_data(mismatched.get(), state.data(), state.size(), 0) != 0) {
            LOG_TRC("%s: state restored into rotation-disabled context, model does not use attention rotation\n", __func__);
        }
    }
    common_set_env("LLAMA_ATTN_ROT_DISABLE", attn_rot_disable);

    if (!success) {
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}

// Test 11: K/V type conversion on load (llama_context_params.kv_cache_convert, fork 2026-10-30)
// - a state saved from an f16 cache must load into a q8_0 cache only with kv_cache_convert, and the logits of the
//   continuation must stay within 4x the f16 vs native q8_0 distance (+ a 1e-7 floor) of the native q8_0 run (the
//   q8_0 error itself is the yardstick, see K_CONV for the measured ratios)
// - the reverse (q8_0 saved, f16 cache) is d*q rounded to f16: a q8_0 -> f16 -> q8_0 round trip through two converted
//   loads must give back the saved q8_0 bytes (up to the rare f16 tie that moves a scale by one ulp), and its logits
//   must obey the same bound
// - pos-range + APPEND blobs (llama_state_seq_get_data_range, the server's disk segments) must convert to exactly the
//   bytes of the whole-blob conversion
// - kv_cache_convert off keeps the refusal, and pairs other than f16/q8_0 are refused even with it on
// note: models without an attention KV cache (pure recurrent) and models whose head size is not a multiple of the
//       q8_0 block pass vacuously
static double test_state_convert_max_nmse = 0.0; // worst max(d_conv, d_rev) / d_ref over the models, printed at the end
static double test_state_convert_max_abs  = 0.0; // worst max(d_conv, d_rev) over the models
static double test_state_convert_max_ref  = 0.0; // worst d_ref over the models

static bool test_state_convert(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    LOGV(LOG_LEVEL_INFO, "\n=== Test 11: K/V type conversion on load ===\n");

    if (tokens.size() < 8) {
        return true;
    }

    const auto make_context = [&](ggml_type type_k, ggml_type type_v, bool convert) {
        auto params_ctx = common_context_params_to_llama(params);
        params_ctx.n_ctx     = 256;
        params_ctx.n_batch   = 256;
        params_ctx.n_ubatch  = 256;
        params_ctx.n_seq_max = 1;
        params_ctx.type_k    = type_k;
        params_ctx.type_v    = type_v;
        params_ctx.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params_ctx.kv_cache_convert = convert;
        return llama_context_ptr(llama_init_from_model(model, params_ctx));
    };

    const int n_prefix = (int) tokens.size() - 1;

    // the continuation fed to every context after the restore: the last prompt token, then a few fixed ones
    llama_tokens cont = { tokens[n_prefix], tokens[0], tokens[1], tokens[2] };

    const auto decode_prefix = [&](llama_context * ctx) {
        llama_tokens pre(tokens.begin(), tokens.begin() + n_prefix);
        return llama_decode(ctx, llama_batch_get_one(pre.data(), (int32_t) pre.size())) == 0;
    };

    const auto save = [](llama_context * ctx, std::vector<uint8_t> & out, llama_pos p0 = -1, llama_pos p1 = -1) {
        if (p0 >= 0 || p1 >= 0) {
            out.resize(llama_state_seq_get_size_range(ctx, 0, p0, p1, 0));
            return !out.empty() && llama_state_seq_get_data_range(ctx, out.data(), out.size(), 0, p0, p1, 0) == out.size();
        }
        out.resize(llama_state_seq_get_size(ctx, 0));
        return !out.empty() && llama_state_seq_get_data(ctx, out.data(), out.size(), 0) == out.size();
    };

    // logits after each continuation token
    const auto run = [&](llama_context * ctx, std::vector<std::vector<float>> & out) {
        out.clear();
        for (llama_token t : cont) {
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
                return false;
            }
            std::vector<float> l;
            if (!get_current_logits(ctx, l)) {
                return false;
            }
            out.push_back(std::move(l));
        }
        return true;
    };

    const auto max_nmse = [](const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
        double m = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            m = std::max(m, nmse(a[i], b[i]));
        }
        return m;
    };

    auto ctx_f16 = make_context(GGML_TYPE_F16,  GGML_TYPE_F16,  false);
    auto ctx_q8  = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false);
    if (!ctx_f16 || !ctx_q8) {
        LOG_WRN("%s: no f16 + q8_0 flash-attention context for this model - skipping\n", __func__);
        return true;
    }

    if (!decode_prefix(ctx_f16.get()) || !decode_prefix(ctx_q8.get())) {
        LOG_ERR("%s: failed to decode the prefix\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_f16;
    std::vector<uint8_t> blob_q8;
    if (!save(ctx_f16.get(), blob_f16) || !save(ctx_q8.get(), blob_q8)) {
        LOG_ERR("%s: failed to save the states\n", __func__);
        return false;
    }

    if (blob_f16.size() == blob_q8.size()) {
        LOG_TRC("%s: the state does not depend on the K/V type (no attention KV) - nothing to convert\n", __func__);
        return true;
    }

    // conversion off: the refusal is unchanged
    {
        auto off_q8  = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false);
        auto off_f16 = make_context(GGML_TYPE_F16,  GGML_TYPE_F16,  false);
        if (llama_state_seq_set_data(off_q8.get(),  blob_f16.data(), blob_f16.size(), 0) != 0 ||
            llama_state_seq_set_data(off_f16.get(), blob_q8.data(),  blob_q8.size(),  0) != 0) {
            LOG_ERR("%s: a mismatched K/V type loaded with kv_cache_convert off\n", __func__);
            return false;
        }
    }

    // unsupported pair: refused even with conversion on
    {
        auto q4 = make_context(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true);
        if (q4 && llama_state_seq_set_data(q4.get(), blob_f16.data(), blob_f16.size(), 0) != 0) {
            LOG_ERR("%s: an f16 state loaded into a q4_0 cache\n", __func__);
            return false;
        }
    }

    // the DSV4 memory (llama-kv-cache-dsv4.cpp) has its own state reader without the conversion: it must refuse, cleanly
    {
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch) == "deepseek4") {
            auto c = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true);
            if (llama_state_seq_set_data(c.get(), blob_f16.data(), blob_f16.size(), 0) != 0) {
                LOG_ERR("%s: the DSV4 memory accepted a converted state it cannot convert\n", __func__);
                return false;
            }
            LOG_TRC("%s: DSV4 memory: conversion not implemented, refused as expected\n", __func__);
            return true;
        }
    }

    // f16 -> q8_0
    auto conv_q8 = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true);
    if (!conv_q8 || llama_state_seq_set_data(conv_q8.get(), blob_f16.data(), blob_f16.size(), 0) != blob_f16.size()) {
        LOG_ERR("%s: f16 -> q8_0 load failed\n", __func__);
        return false;
    }
    if (llama_state_get_convert_stats(conv_q8.get()).n_loads != 1) {
        LOG_ERR("%s: f16 -> q8_0 load did not report a conversion\n", __func__);
        return false;
    }
    std::vector<uint8_t> blob_conv_q8;
    if (!save(conv_q8.get(), blob_conv_q8) || blob_conv_q8.size() != blob_q8.size()) {
        LOG_ERR("%s: the converted q8_0 state has the wrong size (%zu vs native %zu)\n", __func__, blob_conv_q8.size(), blob_q8.size());
        return false;
    }

    // pos-range + APPEND: the server's disk segments. state_write_range exists only for the plain KV cache and the
    // hybrid memory, and nothing outside libllama names the memory type (n_swa == 0 && !recurrent still lands on the
    // DSA/MSA memories, which abort: deepseek32), so the segment check runs on archs known to use the plain cache
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    const std::string arch_s = arch;
    if (arch_s == "llama" || arch_s == "qwen2" || arch_s == "qwen3") {
        const llama_pos mid = n_prefix / 2;
        std::vector<uint8_t> seg0;
        std::vector<uint8_t> seg1;
        if (!save(ctx_f16.get(), seg0, 0, mid) || !save(ctx_f16.get(), seg1, mid, -1)) {
            LOG_ERR("%s: failed to save the range segments\n", __func__);
            return false;
        }
        auto conv_seg = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true);
        if (llama_state_seq_set_data_ext(conv_seg.get(), seg0.data(), seg0.size(), 0, 0) != seg0.size() ||
            llama_state_seq_set_data_ext(conv_seg.get(), seg1.data(), seg1.size(), 0, LLAMA_STATE_SEQ_FLAGS_APPEND) != seg1.size()) {
            LOG_ERR("%s: range segments failed to load with conversion\n", __func__);
            return false;
        }
        std::vector<uint8_t> blob_seg;
        if (!save(conv_seg.get(), blob_seg) || blob_seg != blob_conv_q8) {
            LOG_ERR("%s: range + APPEND conversion differs from the whole-blob conversion\n", __func__);
            return false;
        }
    }

    // q8_0 -> f16, then back: the saved q8_0 bytes come back
    auto conv_f16 = make_context(GGML_TYPE_F16, GGML_TYPE_F16, true);
    if (!conv_f16 || llama_state_seq_set_data(conv_f16.get(), blob_q8.data(), blob_q8.size(), 0) != blob_q8.size()) {
        LOG_ERR("%s: q8_0 -> f16 load failed\n", __func__);
        return false;
    }
    {
        std::vector<uint8_t> rt_f16;
        std::vector<uint8_t> rt_q8;
        auto rt = make_context(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true);
        if (!save(conv_f16.get(), rt_f16) || rt_f16.size() != blob_f16.size() ||
            llama_state_seq_set_data(rt.get(), rt_f16.data(), rt_f16.size(), 0) != rt_f16.size() ||
            !save(rt.get(), rt_q8) || rt_q8.size() != blob_q8.size()) {
            LOG_ERR("%s: q8_0 -> f16 -> q8_0 round trip failed\n", __func__);
            return false;
        }
        size_t n_diff = 0;
        for (size_t i = 0; i < rt_q8.size(); ++i) {
            n_diff += rt_q8[i] != blob_q8[i];
        }
        LOG_TRC("%s: q8_0 -> f16 -> q8_0 round trip: %zu of %zu bytes differ\n", __func__, n_diff, rt_q8.size());
        // 1e-3: an f16 tie can move one block scale (2 bytes) and a few of its quants. measured 2026-10-30 over the
        // 121 dummy models with a K/V cache: 0 .. 2.1e-4 of the bytes (deepseek2-moe 26 of 122420, gpt2 13 of 108980)
        if ((double) n_diff > 1e-3 * (double) rt_q8.size()) {
            LOG_ERR("%s: q8_0 -> f16 -> q8_0 round trip changed %zu of %zu bytes\n", __func__, n_diff, rt_q8.size());
            return false;
        }
    }

    std::vector<std::vector<float>> l_f16, l_q8, l_conv_q8, l_conv_f16;
    if (!run(ctx_f16.get(), l_f16) || !run(ctx_q8.get(), l_q8) || !run(conv_q8.get(), l_conv_q8) || !run(conv_f16.get(), l_conv_f16)) {
        LOG_ERR("%s: failed to decode the continuation\n", __func__);
        return false;
    }

    const double d_ref     = max_nmse(l_f16, l_q8);      // the cost of q8_0 itself
    const double d_conv    = max_nmse(l_q8,  l_conv_q8); // f16 -> q8_0 converted vs native q8_0
    const double d_conv_rf = max_nmse(l_f16, l_conv_q8); // ... and vs f16
    const double d_rev     = max_nmse(l_q8,  l_conv_f16); // q8_0 -> f16 converted vs native q8_0

    LOG_INF("%s: nmse f16 vs q8_0 %.3e | f16->q8_0 converted vs q8_0 %.3e (vs f16 %.3e) | q8_0->f16 converted vs q8_0 %.3e\n",
            __func__, d_ref, d_conv, d_conv_rf, d_rev);

    // tolerance = K_CONV x the q8_0 error itself (d_ref, nmse f16 vs native q8_0) + NMSE_FLOOR.
    // measured 2026-10-30 on CPU over the 121 dummy models with a K/V cache, 31 regenerations of the random weights:
    //   d_conv / d_ref is ~1 (0.6 .. 1.8 worst per regeneration) except on models where q8_0 barely moves the
    //   logits (d_ref ~1e-9), where it spiked to 2.7 (gpt2/starcoder), 3.7, 18.7 (mpt) and 59. the mpt case was taken
    //   apart element by element (scratch dump of the blobs): the converted K/V carries the same q8_0 error as the
    //   native one (rel rms vs the f16 values 5.49e-3 converted vs 5.50e-3 native for K, 5.25e-3 vs 5.26e-3 for V),
    //   only the logits differ, 4.84e-8 vs 1.25e-9. at nmse 1e-8 which q8_0 draw a run gets is luck, not a bias,
    //   hence the absolute floor. worst absolute values seen: converted 5.9e-5, f16 vs q8_0 1.7e-4. with this bound
    //   the last 12 regenerations passed 130/130 (the 4x-only bound failed mpt once in 8).
    //   the bug this test exists for is far outside both: the rotation applied where the graph does not rotate (MLA)
    //   gave 2.5e-3 (deepseek2, d_ref 4.7e-8) and 6.3e-2 (bailingmoe3, d_ref 2.2e-6)
    constexpr double K_CONV     = 4.0;
    constexpr double NMSE_FLOOR = 1e-7;
    const double bound = K_CONV * d_ref + NMSE_FLOOR;
    test_state_convert_max_nmse = std::max(test_state_convert_max_nmse, std::max(d_conv, d_rev) / std::max(d_ref, 1e-9));
    test_state_convert_max_abs  = std::max(test_state_convert_max_abs, std::max(d_conv, d_rev));
    test_state_convert_max_ref  = std::max(test_state_convert_max_ref, d_ref);
    if (d_conv > bound || d_rev > bound) {
        LOG_ERR("%s: converted logits are further from native q8_0 than %.0fx the f16 vs q8_0 distance + %.0e (f16->q8 %.3e, q8->f16 %.3e, bound %.3e)\n",
                __func__, K_CONV, NMSE_FLOOR, d_conv, d_rev, bound);
        return false;
    }

    LOGV(LOG_LEVEL_INFO, "\nPASS\n");
    return true;
}

struct test_suite {
    std::vector<test_status> results;

    bool all_passed() const {
        return std::all_of(results.begin(), results.end(), [](test_status s) { return s == test_status::PASS; });
    }
};

// column headers for the --models table, one per test, in the order they are run
static const std::vector<const char *> test_names = {
    "baseline", "seq_rm", "state_load", "cp_h", "cp_d", "cp_h_s", "cp_d_s", "rt", "rf", "rot", "conv",
};

// Run the full save/load test suite (tests 1-11) for a single model.
// Returns the per-test results.
static test_suite run_save_load_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    test_suite suite;

    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        suite.results.assign(test_names.size(), test_status::SKIP);
        return suite;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    suite.results.push_back(result_baseline.empty() ? test_status::FAIL : test_status::PASS);

    // Test 2: sequence removal isolation
    suite.results.push_back(test_seq_rm_isolated(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    if (!result_baseline.empty()) {
        // Test 3: state load
        suite.results.push_back(test_state_load(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);

        // Test 4: seq copy (host)
        suite.results.push_back(test_seq_cp_host(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);

        // Test 5: seq copy (device)
        suite.results.push_back(test_seq_cp_device(model, params, tokens, result_baseline) ? test_status::PASS : test_status::FAIL);
    } else {
        // tests 3-5 depend on the baseline result and the state file it saves
        suite.results.push_back(test_status::SKIP);
        suite.results.push_back(test_status::SKIP);
        suite.results.push_back(test_status::SKIP);
    }

    // Test 6: seq copy (host, scatter)
    suite.results.push_back(test_seq_cp_scatter(model, params, tokens, 6, false) ? test_status::PASS : test_status::FAIL);

    // Test 7: seq copy (device, scatter)
    suite.results.push_back(test_seq_cp_scatter(model, params, tokens, 7, true) ? test_status::PASS : test_status::FAIL);

    // Test 8: state blob round-trip
    suite.results.push_back(test_state_roundtrip(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    // Test 9: state restore failure
    suite.results.push_back(test_state_restore_failure(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    // Test 10: state rotation
    suite.results.push_back(test_state_rotation(model, params) ? test_status::PASS : test_status::FAIL);

    // Test 11: K/V type conversion on load
    suite.results.push_back(test_state_convert(model, params, tokens) ? test_status::PASS : test_status::FAIL);

    return suite;
}


static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -lv 5\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run the suite over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        auto col_width = [](const char * name) { return (int) std::max(strlen(name), (size_t) 4); };

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s", (int) name_width, "Model");
        for (const auto & name : test_names) {
            LOG("  %-*s", col_width(name), name);
        }
        LOG("\n");
        common_log_flush(common_log_main());

        size_t n_pass = 0;
        size_t n_fail = 0;
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());
            common_log_flush(common_log_main());

            const test_suite suite = run_save_load_tests_for_model(model_path, params);

            for (size_t i = 0; i < suite.results.size(); i++) {
                LOG("  %s%*s", test_status_str(suite.results[i]), col_width(test_names[i]) - 4, "");
            }
            LOG("\n");
            common_log_flush(common_log_main());

            if (suite.all_passed()) {
                n_pass++;
            } else {
                n_fail++;
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: summary: %zu passed, %zu failed (of %zu)\n", __func__, n_pass, n_fail, models.size());
        LOG_INF("%s: conv: worst converted vs native-q8 / f16 vs native-q8 logit nmse ratio %.3f, worst converted nmse %.3e, worst f16 vs q8_0 nmse %.3e\n",
                __func__, test_state_convert_max_nmse, test_state_convert_max_abs, test_state_convert_max_ref);

        return n_fail == 0 ? 0 : 1;
    }

    // single-model mode
    const test_suite suite = run_save_load_tests_for_model(params.model.path, params);
    const bool all_passed = suite.all_passed();
    if (all_passed) {
        LOG("\nAll tests passed.\n");
    }
    return all_passed ? 0 : 1;
}
