// synchronize() idempotency regression test
//
// llama_context::synchronize() skips the GPU sync when the generation counters say nothing is
// in flight (async_gen == synced_gen). The counters bump at every graph submission and again
// after each ubatch's async D2H output copies. This test pins the property that matters:
// after every decode()/encode() the caller's drain (getter or llama_synchronize) is a REAL
// sync, no matter which barriers fired inside the call (output buffer reallocation, embd_seq
// drain, a llama_synchronize() from cb_eval). Those were the "idempotent bool" bugs - see
// potential-bugs-with-idempotent.md.
//
// Needs a model: ctest picks up the tinyllamas fixture (-m <model>); any small model works.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "testing.h"

#include "../src/llama-context.h"

#include <cstdio>
#include <vector>

struct probe {
    uint64_t async_gen;
    uint64_t synced_gen;
    int64_t  n_real;

    explicit probe(const llama_context * ctx)
        : async_gen(ctx->sync_async_gen()), synced_gen(ctx->sync_synced_gen()), n_real(ctx->sync_n_real()) {}

    bool drained() const { return async_gen == synced_gen; }
};

static llama_context_params make_cparams(const common_params & params, bool embeddings) {
    auto cparams = common_context_params_to_llama(params);

    cparams.n_seq_max = 1; // output buffer starts sized for 1 output -> a larger batch reallocates it
    cparams.n_ctx     = 128;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    cparams.embeddings = embeddings;
    if (embeddings) {
        cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    }

    return cparams;
}

static llama_batch make_batch(const std::vector<llama_token> & tokens, llama_pos pos0, bool all_outputs) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool out = all_outputs || i + 1 == tokens.size();
        common_batch_add(batch, tokens[i], pos0 + (llama_pos) i, { 0 }, out);
    }
    return batch;
}

// decode + assert the drain after it is still pending, then read the output and assert the
// read performed exactly one real sync and left the context drained
static void expect_real_drain_after_decode(testing & t, llama_context * ctx, llama_batch & batch, bool via_embeddings) {
    const probe before(ctx);

    t.assert_equal("decode succeeds", 0, llama_decode(ctx, batch));

    const probe after_decode(ctx);
    t.assert_true("decode submitted work (async_gen advanced)", after_decode.async_gen > before.async_gen);
    t.assert_true("drain still pending after decode - no mid-decode barrier may cover the output copies", !after_decode.drained());

    if (via_embeddings) {
        t.assert_true("embeddings readable", llama_get_embeddings_seq(ctx, 0) != nullptr);
    } else {
        t.assert_true("logits readable", llama_get_logits_ith(ctx, -1) != nullptr);
    }

    const probe after_read(ctx);
    t.assert_equal("the read performed one real sync", after_decode.n_real + 1, after_read.n_real);
    t.assert_true("context drained after the read", after_read.drained());

    // and every further read/sync is a no-op
    if (via_embeddings) {
        llama_get_embeddings_seq(ctx, 0);
    } else {
        llama_get_logits_ith(ctx, -1);
    }
    llama_synchronize(ctx);
    llama_synchronize(ctx);
    t.assert_equal("repeated reads/syncs are no-ops", after_read.n_real, probe(ctx).n_real);
}

struct cb_eval_state {
    llama_context * ctx   = nullptr;
    int             calls = 0;
};

// a callback that drains the context mid-compute, the way the old comment in graph_compute()
// said was fine. it drains the graph but must not cover the output copies enqueued after it
static bool cb_eval_sync(ggml_tensor * /*t*/, bool ask, void * user_data) {
    auto * st = (cb_eval_state *) user_data;
    if (ask) {
        return true; // observe every tensor
    }
    if (st->ctx && st->calls == 0) {
        llama_synchronize(st->ctx);
    }
    st->calls++;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> tokens = common_tokenize(vocab, "The meaning of life is a question that", true, false);
    if (tokens.size() < 8) {
        fprintf(stderr, "prompt tokenized to %zu tokens, need >= 8\n", tokens.size());
        return 1;
    }
    const std::vector<llama_token> one(tokens.begin(), tokens.begin() + 1);
    const std::vector<llama_token> many(tokens.begin() + 1, tokens.begin() + 8);

    testing t;
    t.verbose = true;

    t.test("default_params", [&](testing & t) {
        t.assert_true("perf_instrumentation off by default", !llama_context_default_params().perf_instrumentation);
        auto cparams = make_cparams(params, false);
        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        t.assert_true("context created", ctx != nullptr);
        t.assert_true("fresh context is drained", probe(ctx.get()).drained());
        const int64_t n0 = probe(ctx.get()).n_real;
        llama_synchronize(ctx.get());
        t.assert_equal("sync on a fresh context is a no-op", n0, probe(ctx.get()).n_real);
    });

    t.test("single_decode_then_getter", [&](testing & t) {
        llama_context_ptr ctx(llama_init_from_model(model.get(), make_cparams(params, false)));
        llama_batch batch = make_batch(one, 0, true);
        expect_real_drain_after_decode(t, ctx.get(), batch, false);
        llama_batch_free(batch);
    });

    // the "reallocation iteration" bug: decode with an in-flight previous batch and a larger
    // output count. output_reserve() reallocates and syncs for real before the new batch's
    // copies are enqueued; the caller's drain must still be real afterwards
    t.test("output_buffer_reallocation_does_not_cover_new_copies", [&](testing & t) {
        llama_context_ptr ctx(llama_init_from_model(model.get(), make_cparams(params, false)));

        llama_batch b1 = make_batch(one, 0, true);
        t.assert_equal("first decode succeeds", 0, llama_decode(ctx.get(), b1));
        t.assert_true("first batch left in flight (no read)", !probe(ctx.get()).drained());
        const int64_t n_before = probe(ctx.get()).n_real;

        llama_batch b2 = make_batch(many, 1, true); // 7 outputs > n_seq_max -> reallocation
        expect_real_drain_after_decode(t, ctx.get(), b2, false);

        // the reallocation barrier itself must have been a real sync (it drained b1), so the
        // trap scenario was actually exercised: mid-decode real sync, then a real drain
        t.assert_true("reallocation barrier + drain = at least two real syncs", probe(ctx.get()).n_real >= n_before + 2);

        llama_batch_free(b1);
        llama_batch_free(b2);
    });

    // the embeddings bug: from the second decode on, embd_seq is non-empty and the drain at the
    // top of decode() is real. it must not cover this decode's embedding copies
    t.test("embd_seq_drain_does_not_cover_new_copies", [&](testing & t) {
        llama_context_ptr ctx(llama_init_from_model(model.get(), make_cparams(params, true)));

        llama_batch b1 = make_batch(one, 0, true);
        t.assert_equal("first decode succeeds", 0, llama_decode(ctx.get(), b1));
        t.assert_true("first batch left in flight (no read)", !probe(ctx.get()).drained());
        const int64_t n_before = probe(ctx.get()).n_real;

        llama_batch b2 = make_batch(many, 1, true);
        expect_real_drain_after_decode(t, ctx.get(), b2, true);
        t.assert_true("embd_seq drain + read = at least two real syncs", probe(ctx.get()).n_real >= n_before + 2);

        // steady state: read, decode, read - the top-of-decode drain is a no-op now (already
        // drained by the read), and the read after decode is still real
        llama_batch b3 = make_batch(one, 8, true);
        expect_real_drain_after_decode(t, ctx.get(), b3, true);

        llama_batch_free(b1);
        llama_batch_free(b2);
        llama_batch_free(b3);
    });

    // no per-decode reset: two decodes without a read need exactly one real sync to drain
    t.test("counter_carries_across_decodes", [&](testing & t) {
        llama_context_ptr ctx(llama_init_from_model(model.get(), make_cparams(params, false)));

        llama_batch b1 = make_batch(one, 0, true);
        llama_batch b2 = make_batch(one, 1, true);
        t.assert_equal(0, llama_decode(ctx.get(), b1));
        t.assert_equal(0, llama_decode(ctx.get(), b2));
        t.assert_true("both batches in flight", !probe(ctx.get()).drained());

        const int64_t n_before = probe(ctx.get()).n_real;
        llama_synchronize(ctx.get());
        t.assert_equal("one real sync drains both", n_before + 1, probe(ctx.get()).n_real);
        t.assert_true("drained", probe(ctx.get()).drained());
        llama_synchronize(ctx.get());
        t.assert_equal("second sync is a no-op", n_before + 1, probe(ctx.get()).n_real);

        llama_batch_free(b1);
        llama_batch_free(b2);
    });

    // a sync between graph submission and the output copies (cb_eval) drains the graph but
    // must not make the caller's drain a no-op
    t.test("sync_from_cb_eval_does_not_cover_output_copies", [&](testing & t) {
        cb_eval_state st;
        auto cparams = make_cparams(params, false);
        cparams.cb_eval           = cb_eval_sync;
        cparams.cb_eval_user_data = &st;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        st.ctx = ctx.get();

        // warm up so the graph and a previous batch exist, then read to drain
        llama_batch b1 = make_batch(one, 0, true);
        t.assert_equal(0, llama_decode(ctx.get(), b1));
        llama_get_logits_ith(ctx.get(), -1);

        st.calls = 0;
        const int64_t n_before = probe(ctx.get()).n_real;
        llama_batch b2 = make_batch(one, 1, true);
        expect_real_drain_after_decode(t, ctx.get(), b2, false);
        t.assert_true("cb_eval ran", st.calls > 0);
        t.assert_true("cb_eval sync + drain = at least two real syncs", probe(ctx.get()).n_real >= n_before + 2);

        llama_batch_free(b1);
        llama_batch_free(b2);
    });

    t.test("encode_then_getter", [&](testing & t) {
        if (!llama_model_has_encoder(model.get())) {
            t.skip("model has no encoder");
            return;
        }
        llama_context_ptr ctx(llama_init_from_model(model.get(), make_cparams(params, true)));
        llama_batch batch = make_batch(many, 0, true);
        const probe before(ctx.get());
        t.assert_equal("encode succeeds", 0, llama_encode(ctx.get(), batch));
        t.assert_true("encode submitted work", probe(ctx.get()).async_gen > before.async_gen);
        t.assert_true("drain pending after encode", !probe(ctx.get()).drained());
        const int64_t n = probe(ctx.get()).n_real;
        t.assert_true("embeddings readable", llama_get_embeddings_seq(ctx.get(), 0) != nullptr);
        t.assert_equal("read performed one real sync", n + 1, probe(ctx.get()).n_real);
        t.assert_true("drained", probe(ctx.get()).drained());
        llama_batch_free(batch);
    });

    llama_backend_free();

    return t.summary();
}
