#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0 - 1) prefill. ctx_ref then decodes only
    // [p0 - 1, p0); ctx_roll decodes [p0 - 1, n_prompt) - n_rollback + 1 tokens - and rolls
    // back n_rollback of them so its restore is pending at replay. the tail decode must be one
    // token longer than the rollback: a decode of N tokens only writes snapshot groups
    // [0, N) (see ggml_ssm_scan / ggml_gated_delta_net), so group N is not a portable target
    // (the DSV4 cache additionally saturates plane N, but this test only relies on the common bound)
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0 - 1; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        common_batch_add(batch, tok(s, p0 - 1), p0 - 1, { (llama_seq_id) s }, false);
        ok = ok && llama_decode(ctx_ref, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0 - 1; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal composes with the pending one, but only while the sum stays
        // within the rollback budget - this one exceeds it and must be refused without disturbing
        // the pending rollback that the replay below depends on
        const llama_pos p_over = p0 - (llama_pos) (llama_n_rs_seq(ctx_roll) - n_rollback + 1);
        ok = ok && p_over > 0;
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p_over, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = logit_diff(l_roll[t], l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

// Two partial removals on the same sequence with no decode in between. The pending rollback must
// compose with the new one instead of being refused (the server drops rejected draft tokens, stops
// on EOG without decoding again, and the next request then rewinds the tail once more), while a
// composed rollback that would exceed n_rs_seq must still be refused.
static bool test_composed_partial_rollback(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t n_prompt = 12;
    constexpr uint32_t n_replay = 3;

    const auto make = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = 8;
        cparams.n_ctx     = 128;
        cparams.n_batch   = 128;
        cparams.n_ubatch  = 32;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make();
    llama_context * ctx_ref  = make();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_roll);
    if (n_rs_seq < 2 || n_rs_seq + 2 > n_prompt) {
        fprintf(stderr, "%s : skipping because n_rs_seq (%u) does not fit the test\n", __func__, n_rs_seq);
        cleanup();
        return true;
    }

    const auto tok = [&](llama_pos pos) {
        return (llama_token) ((5*(uint32_t) pos + 17) % (uint32_t) n_vocab);
    };

    std::vector<llama_token> tokens(n_prompt + n_replay);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = tok((llama_pos) i);
    }

    // the rolled-back context decodes the whole prompt, the reference only its surviving prefix
    const llama_pos p_keep = (llama_pos) n_prompt - (llama_pos) n_rs_seq;

    bool ok = decode_tokens(ctx_roll, tokens, n_prompt);
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < p_keep; ++pos) {
            common_batch_add(batch, tokens[pos], pos, { 0 }, false);
        }
        ok = ok && llama_decode(ctx_ref, batch) == 0;
        llama_batch_free(batch);
    }
    if (!ok) {
        fprintf(stderr, "%s : prompt decode failed\n", __func__);
        cleanup();
        return false;
    }

    // rewind one token at a time, with no decode in between: every step composes with the pending one
    for (uint32_t i = 1; i <= n_rs_seq; ++i) {
        const llama_pos p0 = (llama_pos) n_prompt - (llama_pos) i;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, p0, -1)) {
            fprintf(stderr, "%s : composed rollback #%u (p0 = %d) was refused\n", __func__, i, p0);
            cleanup();
            return false;
        }
    }

    // one more would exceed the budget
    if (llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, p_keep - 1, -1)) {
        fprintf(stderr, "%s : rollback past the budget (p0 = %d) was accepted\n", __func__, p_keep - 1);
        cleanup();
        return false;
    }

    // replay the removed tokens on both contexts - they must agree
    constexpr float eps = 1e-5f;
    float diff_max = 0.0f;
    for (uint32_t i = 0; i < n_replay; ++i) {
        const llama_pos pos = p_keep + (llama_pos) i;
        if (!decode_one(ctx_roll, tokens[pos], pos) || !decode_one(ctx_ref, tokens[pos], pos)) {
            fprintf(stderr, "%s : replay failed at position %d\n", __func__, pos);
            cleanup();
            return false;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing logits at position %d\n", __func__, pos);
            cleanup();
            return false;
        }

        for (int t = 0; t < n_vocab; ++t) {
            diff_max = std::max(diff_max, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : composed rollback replay mismatch (max diff %g)\n", __func__, (double) diff_max);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : composed partial rollbacks matched (max diff %g)\n", __func__, (double) diff_max);
    cleanup();
    return true;
}

// A state restored with llama_state_seq_set_data only carries the final state (snapshot group 0),
// so a partial rewind must be refused until decodes have refreshed the groups it would select.
static bool test_rollback_after_state_restore(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t n_prompt = 12;
    constexpr uint32_t n_tail   = 4;

    const auto make = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = 8;
        cparams.n_ctx     = 128;
        cparams.n_batch   = 128;
        cparams.n_ubatch  = 32;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_src  = make();
    llama_context * ctx_roll = make();
    llama_context * ctx_ref  = make();
    if (ctx_src == nullptr || ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_src);
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_src) < n_tail) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](llama_pos pos) {
        return (llama_token) ((11*(uint32_t) pos + 5) % (uint32_t) n_vocab);
    };

    std::vector<llama_token> tokens(n_prompt + n_tail);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = tok((llama_pos) i);
    }

    if (!decode_tokens(ctx_src, tokens, n_prompt)) {
        fprintf(stderr, "%s : prompt decode failed\n", __func__);
        cleanup();
        return false;
    }

    // restore the saved state into two fresh contexts
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    if (!ckpt.load_tgt(ctx_roll, 0, 0) || !ckpt.load_tgt(ctx_ref, 0, 0)) {
        fprintf(stderr, "%s : state restore failed\n", __func__);
        cleanup();
        return false;
    }

    // right after the restore only group 0 holds this sequence's state - any rewind must be refused
    if (llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, (llama_pos) n_prompt - 1, -1)) {
        fprintf(stderr, "%s : partial rewind right after a state restore was accepted\n", __func__);
        cleanup();
        return false;
    }

    // a decode of n_tail tokens refreshes groups [0, n_tail), so a rollback of n_tail - 1 is usable
    {
        llama_batch batch = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);
            common_batch_add(batch, tokens[pos], pos, { 0 }, false);
        }
        const bool ok = llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "%s : tail decode failed\n", __func__);
            cleanup();
            return false;
        }
    }

    const llama_pos p_keep = (llama_pos) n_prompt + 1; // rollback of n_tail - 1

    // deeper than the decode that refreshed the groups: no memory can serve this, since the state
    // it would need was never snapshotted after the restore
    if (llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, (llama_pos) n_prompt - 1, -1)) {
        fprintf(stderr, "%s : rollback of %u after a %u token decode was accepted\n", __func__, n_tail + 1, n_tail);
        cleanup();
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, p_keep, -1)) {
        fprintf(stderr, "%s : rollback of %u after a %u token decode was refused\n", __func__, n_tail - 1, n_tail);
        cleanup();
        return false;
    }

    // the reference walks to the same position without ever rolling back
    if (!decode_one(ctx_ref, tokens[n_prompt], (llama_pos) n_prompt)) {
        fprintf(stderr, "%s : reference decode failed\n", __func__);
        cleanup();
        return false;
    }

    constexpr float eps = 1e-5f;
    float diff_max = 0.0f;
    for (uint32_t i = 0; i < n_tail - 1; ++i) {
        const llama_pos pos = p_keep + (llama_pos) i;
        if (!decode_one(ctx_roll, tokens[pos], pos) || !decode_one(ctx_ref, tokens[pos], pos)) {
            fprintf(stderr, "%s : replay failed at position %d\n", __func__, pos);
            cleanup();
            return false;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing logits at position %d\n", __func__, pos);
            cleanup();
            return false;
        }

        for (int t = 0; t < n_vocab; ++t) {
            diff_max = std::max(diff_max, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : post-restore rollback replay mismatch (max diff %g)\n", __func__, (double) diff_max);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : post-restore rollback refused then honored after a decode (max diff %g)\n", __func__, (double) diff_max);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode, llama_pos from) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = from + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full", rollback_pos)) {
        return 1;
    }

    // keep the reference logits of the full replay: the partial replay below runs at other positions
    const std::vector<std::vector<float>> logits_full = logits_src_replay;

    // ctx_dst was rebuilt from a saved state, which only carries snapshot group 0 - it cannot roll
    // back until decodes have rewritten the groups. refresh them on both contexts with a single
    // multi-token decode, which makes a rollback of (n_refresh - 1) tokens usable again.
    const uint32_t  n_refresh     = n_rollback + 1;
    const llama_pos refresh_pos   = (llama_pos) n_tokens;
    const llama_pos rollback_pos2 = refresh_pos + (llama_pos) n_refresh - (llama_pos) n_rollback;

    for (uint32_t i = 0; i < n_refresh; ++i) {
        tokens.push_back(tokens[i % n_tokens]);
    }

    for (llama_context * ctx : { ctx_src, ctx_dst }) {
        llama_batch batch = llama_batch_init(n_refresh, 0, 1);
        for (uint32_t i = 0; i < n_refresh; ++i) {
            const llama_pos pos = refresh_pos + (llama_pos) i;
            common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == n_refresh);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "%s : refresh decode failed\n", __func__);
            return 1;
        }
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos2, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos2, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial", rollback_pos2)) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_full[i][token], logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_full[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    if (!test_composed_partial_rollback(params, model, n_vocab, fill)) {
        return 1;
    }

    if (!test_rollback_after_state_restore(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return 0;
}
