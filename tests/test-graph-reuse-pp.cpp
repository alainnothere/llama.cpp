// [GRAPH_REUSE_PP, fork 2026-10-27] graph reuse inside multi-ubatch batches under pipeline parallelism.
//
// Part 1 (always, CPU, ctest): truth table of llama_graph_reuse_allowed, the rule process_ubatch applies.
// Part 2 (end-to-end, needs a model path argument): decode one batch of 4 x n_ubatch tokens and read the context's
// counters. Without pipeline parallelism (CPU-only, single device) it only proves the multi-ubatch decode works and
// that nothing is counted as skipped. With pipeline parallelism (2+ GPU devices, layer split) it asserts:
//   default:                n_reused == 0, n_reuse_skipped_pp == (n_reused of the LLAMA_GRAPH_REUSE_PP=1 run)
//   LLAMA_GRAPH_REUSE_PP=1: n_reused >= 1 (n_ubatches - 1 when every later ubatch passes can_reuse), skipped == 0
// The pipelined asserts are for the owner's rig (3 dGPUs over Vulkan); elsewhere they print SKIP.
//
// Usage: test-graph-reuse-pp [model.gguf]   (ctest passes build/tests/test-models/llama-dense.gguf; missing = SKIP)

#include "../src/llama-ext.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void test_truth_table() {
    //                         pipeline_parallel  n_ubatches  env_reuse_pp
    CHECK( llama_graph_reuse_allowed(false, 1,  false)); // single ubatch, no pipeline
    CHECK( llama_graph_reuse_allowed(true,  1,  false)); // single ubatch (decode step): reuse as before
    CHECK( llama_graph_reuse_allowed(true,  0,  false)); // degenerate count treated as single
    CHECK(!llama_graph_reuse_allowed(true,  2,  false)); // multi-ubatch + pipeline: rebuild
    CHECK(!llama_graph_reuse_allowed(true,  16, false)); // 8192 / 512
    CHECK( llama_graph_reuse_allowed(false, 16, false)); // multi-ubatch, no pipeline: reuse
    CHECK( llama_graph_reuse_allowed(true,  16, true));  // env override: old behaviour
    CHECK( llama_graph_reuse_allowed(false, 16, true));
    CHECK( llama_graph_reuse_allowed(true,  1,  true));
    printf("part 1: truth table done\n");
}

struct run_result {
    bool    ok        = false;
    bool    pipelined = false;
    int32_t n_reused  = 0;
    int32_t n_skipped = 0;
};

static run_result run_once(llama_model * model, uint32_t n_ubatch, uint32_t n_ubatches) {
    run_result r;

    const uint32_t n_tokens = n_ubatch * n_ubatches;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 2 * n_tokens;
    cparams.n_batch   = n_tokens;
    cparams.n_ubatch  = n_ubatch;
    cparams.n_seq_max = 1;
    cparams.no_perf   = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return r;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.token[i]     = (llama_token) ((i * 7919u + 13u) % (uint32_t) n_vocab);
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == n_tokens - 1;
    }
    batch.n_tokens = (int32_t) n_tokens;

    llama_perf_context_reset(ctx);
    const int ret = llama_decode(ctx, batch);
    llama_synchronize(ctx);

    r.ok        = ret == 0 && llama_get_logits_ith(ctx, -1) != nullptr;
    r.pipelined = llama_pipeline_parallel(ctx);
    r.n_reused  = llama_perf_context(ctx).n_reused;
    r.n_skipped = llama_perf_context_n_reuse_skipped_pp(ctx);

    printf("  decode %u tokens as %u x %u: ret %d pipelined %d n_reused %d n_reuse_skipped_pp %d\n",
            n_tokens, n_ubatches, n_ubatch, ret, (int) r.pipelined, r.n_reused, r.n_skipped);

    llama_batch_free(batch);
    llama_free(ctx);
    return r;
}

static void test_end_to_end(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        printf("part 2: SKIP (no model at %s)\n", path);
        return;
    }
    fclose(f);

    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            n_gpu++;
        }
    }

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load %s\n", path);
        n_fail++;
        return;
    }

    const uint32_t n_ubatch   = 32;
    const uint32_t n_ubatches = 4;

    unsetenv("LLAMA_GRAPH_REUSE_PP"); // read at context creation
    printf("part 2: default (gpu devices: %d)\n", n_gpu);
    const run_result a = run_once(model, n_ubatch, n_ubatches);
    CHECK(a.ok);

    setenv("LLAMA_GRAPH_REUSE_PP", "1", 1);
    printf("part 2: LLAMA_GRAPH_REUSE_PP=1\n");
    const run_result b = run_once(model, n_ubatch, n_ubatches);
    CHECK(b.ok);
    unsetenv("LLAMA_GRAPH_REUSE_PP");

    if (!a.pipelined || !b.pipelined) {
        // the rule is inert without pipeline parallelism: nothing may be counted as skipped
        CHECK(a.n_skipped == 0);
        CHECK(b.n_skipped == 0);
        CHECK(a.n_reused == b.n_reused);
        printf("part 2: pipelined asserts SKIP (pipeline parallelism off: needs 2+ GPU devices with full offload)\n");
    } else {
        // the old path's reuse count is the reference, not n_ubatches - 1: on CPU (2026-10-27, llama-dense, 4 x 32)
        // only 2 of 3 later ubatches pass can_reuse (one input changes shape inside the batch), so the rule must turn
        // exactly the old reuses into skips
        CHECK(a.n_reused  == 0);
        CHECK(b.n_skipped == 0);
        CHECK(b.n_reused  >= 1);
        CHECK(a.n_skipped == b.n_reused);
        if (b.n_reused != (int32_t) n_ubatches - 1) {
            printf("  note: old path reused %d of %u later ubatches (LLAMA_GRAPH_REUSE_LOG=1 names the failing input)\n",
                    b.n_reused, n_ubatches - 1);
        }
    }

    llama_model_free(model);
}

int main(int argc, char ** argv) {
    test_truth_table();

    if (argc > 1) {
        llama_backend_init();
        test_end_to_end(argv[1]);
        llama_backend_free();
    } else {
        printf("part 2: SKIP (no model argument)\n");
    }

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
