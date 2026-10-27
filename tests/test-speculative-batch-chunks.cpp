// [MTP_CHUNK, fork 2026-10-27] common_batch entries beyond n_batch must be decoded in chunks, not through batch.get().
//
// Production crash 2026-10-27 (-np 2, -b 8192 -ub 512, Qwen3.8-27B, draft-mtp,ngram-mod): slot 1 generating (~15 rows),
// slot 0 starting a 16571-token prompt, shared batch filled to 8177 + 15 = 8192, the MTP catch-up batch adds one seed
// row for slot 1 = 8193 entries, batch.get() = get_sub_batch(0, 8193) aborted in common_batch::get_sub_batch.
// The MTP process() now decodes through common_batch_decode_chunked(); this test exercises that helper on the CPU.
//
// Same shape at toy scale: n_batch = 32, two sequences, 32 + 7 = 39 entries, an output row on the last entry of each
// sequence (the seq 0 one lands in chunk 0 like slot 1's seed did, the seq 1 one in the last chunk).
// Checks:
//   1. get_sub_batch(off, n) on each chunk renders exactly n entries at idx == i - off (get_sub_batch asserts
//      idx == i - off itself; here the next free index after the chunk must be n, and a full chunk is at capacity)
//   2. common_batch_decode_chunked returns 0, submits 2 chunks, off_last == 32, both sequences' KV ends where the
//      batch says, and the last-chunk output row read at (batch index - off_last) matches a reference decode of the
//      same rows in a context with n_batch = 64 (one chunk)
//
// Usage: test-speculative-batch-chunks [model.gguf]   (ctest passes build/tests/test-models/llama-dense.gguf;
// missing = SKIP, like test-graph-reuse-pp)

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static const int32_t N_BATCH = 32;
static const int32_t N_EXTRA = 7;
static const int32_t N_SEQ0  = 30; // seq 0 rows first, like the generating slot in the crash
static const int32_t N_SEQ1  = N_BATCH + N_EXTRA - N_SEQ0;

static llama_context * make_ctx(llama_model * model, int32_t n_batch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = 16;
    cparams.n_seq_max = 2;
    cparams.kv_unified = true;
    return llama_init_from_model(model, cparams);
}

// returns the batch indices of the two output rows
static void fill(common_batch & batch, int32_t n_vocab, int32_t & i_out0, int32_t & i_out1) {
    batch.clear();
    for (int32_t p = 0; p < N_SEQ0; ++p) {
        const int32_t idx = batch.add((llama_token) ((p * 7919 + 13) % n_vocab), p, 0, p == N_SEQ0 - 1);
        if (p == N_SEQ0 - 1) {
            i_out0 = idx;
        }
    }
    for (int32_t p = 0; p < N_SEQ1; ++p) {
        const int32_t idx = batch.add((llama_token) ((p * 104729 + 7) % n_vocab), p, 1, p == N_SEQ1 - 1);
        if (p == N_SEQ1 - 1) {
            i_out1 = idx;
        }
    }
}

static void run(const char * path) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load %s\n", path);
        n_fail++;
        return;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // reference: everything in one decode
    std::vector<float> ref1;
    {
        llama_context * ctx = make_ctx(model, 2 * N_BATCH);
        CHECK(ctx != nullptr);
        common_batch batch(ctx);
        int32_t i0 = -1, i1 = -1;
        fill(batch, n_vocab, i0, i1);
        CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        const float * l = llama_get_logits_ith(ctx, i1);
        CHECK(l != nullptr);
        if (l) {
            ref1.assign(l, l + n_vocab);
        }
        llama_free(ctx);
    }

    llama_context * ctx = make_ctx(model, N_BATCH);
    CHECK(ctx != nullptr);
    CHECK((int32_t) llama_n_batch(ctx) == N_BATCH);

    common_batch batch(ctx);
    int32_t i_out0 = -1, i_out1 = -1;
    fill(batch, n_vocab, i_out0, i_out1);
    CHECK(batch.size() == N_BATCH + N_EXTRA);
    CHECK(i_out0 < N_BATCH && i_out1 >= N_BATCH);

    // 1. chunk rendering
    for (int32_t off = 0; off < batch.size(); off += N_BATCH) {
        const int32_t n = std::min(N_BATCH, batch.size() - off);
        llama_batch_ext * sub = batch.get_sub_batch(off, n); // asserts idx == i - off per entry
        const int32_t next = llama_batch_ext_add_token(sub, 0, 0);
        if (n < N_BATCH) {
            CHECK(next == n);
        } else {
            CHECK(next < 0); // the full chunk is exactly at capacity
        }
        llama_batch_ext_clear(sub);
    }
    printf("part 1: get_sub_batch chunks of %d over %d entries OK\n", N_BATCH, batch.size());

    // 2. chunked decode
    int32_t off_last = -1, n_chunks = -1;
    const int32_t rc = common_batch_decode_chunked(ctx, batch, N_BATCH, &off_last, &n_chunks);
    CHECK(rc == 0);
    CHECK(n_chunks == 2);
    CHECK(off_last == N_BATCH);

    auto * mem = llama_get_memory(ctx);
    CHECK(llama_memory_seq_pos_max(mem, 0) == N_SEQ0 - 1);
    CHECK(llama_memory_seq_pos_max(mem, 1) == N_SEQ1 - 1);

    // the seq 1 output row is in the last chunk: readable at batch index - off_last, same logits as one decode
    const float * l = llama_get_logits_ith(ctx, i_out1 - off_last);
    CHECK(l != nullptr);
    if (l && !ref1.empty()) {
        float max_diff = 0.0f;
        for (int32_t v = 0; v < n_vocab; ++v) {
            max_diff = std::max(max_diff, std::fabs(l[v] - ref1[v]));
        }
        printf("part 2: chunked vs single decode, seq 1 last row, max |dlogit| = %g\n", max_diff);
        CHECK(max_diff < 1e-3f);
    }
    printf("part 2: decoded %d entries in %d chunks (off_last %d), rc %d\n", batch.size(), n_chunks, off_last, rc);

    // the n_batch argument is clamped to the context capacity, never past it
    llama_memory_clear(mem, true);
    off_last = -1;
    CHECK(common_batch_decode_chunked(ctx, batch, 1 << 20, &off_last, &n_chunks) == 0);
    CHECK(off_last == N_BATCH && n_chunks == 2);

    llama_free(ctx);
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("SKIP (no model argument)\n");
        return 0;
    }
    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        printf("SKIP (no model at %s)\n", argv[1]);
        return 0;
    }
    fclose(f);

    llama_backend_init();
    run(argv[1]);
    llama_backend_free();

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
