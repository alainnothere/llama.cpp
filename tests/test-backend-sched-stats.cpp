// [fork, 2026-10-21] CPU-only check of ggml_backend_sched_stats (ggml-backend.h), the source of the
// "sched"/"copies"/"splits" fields of LLAMA_DECODE_STATS. A fake non-host "GPU" device (malloc memory,
// compute delegated to a CPU backend) next to the real CPU backend gives a two-backend graph with one
// graph-input copy (CPU -> fake, path INPUT) and one bounce copy (fake -> CPU: the CPU backend has no
// cpy_tensor_async, the same fallback Vulkan takes cross-device on this tree).

#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static ggml_backend_t g_cpu_inner = nullptr; // computes the fake device's splits

// fake buffer: plain malloc, reported as non-host so the CPU backend cannot read it in place
static void   fake_buf_free(ggml_backend_buffer_t b) { free(b->context); }
static void * fake_buf_base(ggml_backend_buffer_t b) { return b->context; }
static void   fake_buf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t n) { memset((char *) t->data + off, v, n); }
static void   fake_buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * d, size_t off, size_t n) { memcpy((char *) t->data + off, d, n); }
static void   fake_buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * d, size_t off, size_t n) { memcpy(d, (const char *) t->data + off, n); }
static void   fake_buf_clear(ggml_backend_buffer_t b, uint8_t v) { memset(b->context, v, b->size); }

static ggml_backend_buffer_i fake_buf_iface = {
    /* .free_buffer   = */ fake_buf_free,
    /* .get_base      = */ fake_buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ fake_buf_memset,
    /* .set_tensor    = */ fake_buf_set,
    /* .get_tensor    = */ fake_buf_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ fake_buf_clear,
    /* .reset         = */ nullptr,
};

static const char * fake_buft_name(ggml_backend_buffer_type_t) { return "FakeGPU_buft"; }
static ggml_backend_buffer_t fake_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * p = aligned_alloc(64, GGML_PAD(size + 64, 64));
    return ggml_backend_buffer_init(buft, fake_buf_iface, p, size);
}
static size_t fake_buft_align(ggml_backend_buffer_type_t) { return 64; }
static bool   fake_buft_is_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type fake_buft;

static const char * fake_dev_name(ggml_backend_dev_t) { return "FakeGPU"; }
static void fake_dev_memory(ggml_backend_dev_t, size_t * f, size_t * t) { *f = 0; *t = 0; }
static enum ggml_backend_dev_type fake_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static ggml_backend_buffer_type_t fake_dev_buft(ggml_backend_dev_t) { return &fake_buft; }
static bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool fake_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return b == &fake_buft; }

static ggml_backend_device fake_dev;

static const char * fake_name(ggml_backend_t) { return "FakeGPU"; }
static enum ggml_status fake_compute(ggml_backend_t, ggml_cgraph * gf) { return ggml_backend_graph_compute(g_cpu_inner, gf); }

static ggml_guid fake_guid = { 0xfa, 0x4e, 0x00, 0x01 };

static void run(bool stats) {
    fake_buft = { { fake_buft_name, fake_buft_alloc, nullptr, fake_buft_align, nullptr, nullptr, nullptr, fake_buft_is_host }, &fake_dev, nullptr };
    ggml_backend_device_i di = {};
    di.get_name        = fake_dev_name;
    di.get_description = fake_dev_name;
    di.get_memory      = fake_dev_memory;
    di.get_type        = fake_dev_type;
    di.get_buffer_type = fake_dev_buft;
    di.supports_op     = fake_dev_supports_op;
    di.supports_buft   = fake_dev_supports_buft;
    fake_dev = { di, nullptr, nullptr };

    ggml_backend_i bi = {};
    bi.get_name      = fake_name;
    bi.graph_compute = fake_compute;
    ggml_backend fake = { &fake_guid, bi, &fake_dev, nullptr };

    g_cpu_inner        = ggml_backend_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { &fake, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);
    CHECK(ggml_backend_sched_get_last_stats(sched) == nullptr);
    ggml_backend_sched_set_stats_enabled(sched, stats);

    const int N = 1000;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_name(a, "a");
    ggml_set_input(a);
    ggml_tensor * b = ggml_add(ctx, a, a);
    ggml_set_name(b, "b");
    ggml_tensor * c = ggml_scale(ctx, b, 2.0f);
    ggml_set_name(c, "c");
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_sched_set_tensor_backend(sched, b, &fake);
    ggml_backend_sched_set_tensor_backend(sched, c, cpu);
    CHECK(ggml_backend_sched_alloc_graph(sched, gf));

    std::vector<float> in(N), out(N);
    for (int i = 0; i < N; ++i) {
        in[i] = (float) i;
    }
    ggml_backend_tensor_set(a, in.data(), 0, N * sizeof(float));
    CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(c, out.data(), 0, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        CHECK(out[i] == 4.0f * in[i]);
    }

    const ggml_backend_sched_stats * st = ggml_backend_sched_get_last_stats(sched);
    if (!stats) {
        CHECK(st == nullptr);
    } else {
        CHECK(st != nullptr);
        printf("n_splits %d n_copies %d bytes %zu dst %lld src %lld copy %lld split_sync %lld wall %lld us\n",
               st->n_splits, st->n_copies, st->nbytes, (long long) st->t_dst_wait_us, (long long) st->t_src_sync_us,
               (long long) st->t_copy_us, (long long) st->t_split_sync_us, (long long) st->t_wall_us);
        for (int i = 0; i < st->n_copies; ++i) {
            const auto & cp = st->copies[i];
            printf("  copy %s split %d %d -> %d path %d %zu B dst %lld src %lld copy %lld\n", cp.name, cp.split_id,
                   cp.src_backend, cp.dst_backend, cp.path, cp.nbytes,
                   (long long) cp.t_dst_wait_us, (long long) cp.t_src_sync_us, (long long) cp.t_copy_us);
        }
        for (int i = 0; i < st->n_splits; ++i) {
            const auto & sp = st->splits[i];
            printf("  split %d backend %d n_nodes %d n_inputs %d issue %lld next_wait %lld\n", i, sp.backend, sp.n_nodes,
                   sp.n_inputs, (long long) sp.t_issue_us, (long long) sp.t_issue_to_next_wait_us);
        }
        CHECK(st->n_splits == 2);
        CHECK(st->n_copies == 2);
        CHECK(st->nbytes == 2 * N * sizeof(float));
        CHECK(strcmp(st->copies[0].name, "a") == 0);
        CHECK(st->copies[0].path == GGML_BACKEND_SCHED_COPY_PATH_INPUT);
        CHECK(st->copies[0].src_backend == 1 && st->copies[0].dst_backend == 0);
        CHECK(strcmp(st->copies[1].name, "b") == 0);
        CHECK(st->copies[1].path == GGML_BACKEND_SCHED_COPY_PATH_BOUNCE);
        CHECK(st->copies[1].src_backend == 0 && st->copies[1].dst_backend == 1);
        CHECK(st->copies[1].nbytes == N * sizeof(float));
        CHECK(st->splits[0].backend == 0 && st->splits[1].backend == 1);
        CHECK(st->splits[0].t_issue_to_next_wait_us >= 0);
        CHECK(st->splits[1].t_issue_to_next_wait_us == -1);
        CHECK(st->t_wall_us >= st->t_dst_wait_us + st->t_src_sync_us + st->t_copy_us);
        ggml_backend_sched_set_stats_enabled(sched, false);
        CHECK(ggml_backend_sched_get_last_stats(sched) == nullptr);
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_free(g_cpu_inner);
}

// [fork, 2026-10-22 night] GGML_SCHED_SKIP_DEAD_INPUTS=1: an inactive ggml_build_forward_select branch (no COMPUTE flag)
// reading a CPU input on the fake device must not be copied (llama's unused build_inp_embd branches); with the knob
// off, or when the dead node is a graph OUTPUT, it is copied as before. The live result must not change.
static void run_dead(bool knob, bool dead_is_output) {
    setenv("GGML_SCHED_SKIP_DEAD_INPUTS", knob ? "1" : "0", 1);
    ggml_backend_i bi = {};
    bi.get_name      = fake_name;
    bi.graph_compute = fake_compute;
    ggml_backend fake = { &fake_guid, bi, &fake_dev, nullptr };

    g_cpu_inner        = ggml_backend_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[2] = { &fake, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, false);
    ggml_backend_sched_set_stats_enabled(sched, true);

    const int N = 1000;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_name(a, "a");
    ggml_set_input(a);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_name(x, "x_dead");
    ggml_set_input(x);
    ggml_tensor * b = ggml_add(ctx, a, a);
    ggml_set_name(b, "b");
    ggml_tensor * dd = ggml_scale(ctx, x, 3.0f);
    ggml_set_name(dd, "dd");
    if (dead_is_output) {
        ggml_set_output(dd);
    }
    ggml_tensor * c = ggml_scale(ctx, b, 2.0f);
    ggml_set_name(c, "c");
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * sel[2] = { c, dd };
    ggml_build_forward_select(gf, sel, 2, 0);
    CHECK((dd->flags & GGML_TENSOR_FLAG_COMPUTE) == 0);

    ggml_backend_sched_set_tensor_backend(sched, b, &fake);
    ggml_backend_sched_set_tensor_backend(sched, dd, &fake);
    ggml_backend_sched_set_tensor_backend(sched, c, cpu);
    CHECK(ggml_backend_sched_alloc_graph(sched, gf));

    std::vector<float> in(N), out(N);
    for (int i = 0; i < N; ++i) {
        in[i] = (float) i;
    }
    // inputs are re-set per compute: gallocr may reuse an input's memory once its copy is done
    for (int rep = 0; rep < 2; ++rep) {
        ggml_backend_tensor_set(a, in.data(), 0, N * sizeof(float));
        ggml_backend_tensor_set(x, in.data(), 0, N * sizeof(float));
        CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(c, out.data(), 0, N * sizeof(float));
        for (int i = 0; i < N; ++i) {
            CHECK(out[i] == 4.0f * in[i]);
        }
        const ggml_backend_sched_stats * st = ggml_backend_sched_get_last_stats(sched);
        const bool skip = knob && !dead_is_output;
        printf("dead knob %d output %d: n_copies %d n_copies_skipped %d bytes_skipped %zu\n", knob, dead_is_output,
               st->n_copies, st->n_copies_skipped, st->nbytes_skipped);
        CHECK(st->n_copies == (skip ? 2 : 3));
        CHECK(st->n_copies_skipped == (skip ? 1 : 0));
        CHECK(st->nbytes_skipped == (skip ? N * sizeof(float) : 0));
        for (int i = 0; i < st->n_copies; ++i) {
            CHECK(!skip || strcmp(st->copies[i].name, "x_dead") != 0);
        }
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_free(g_cpu_inner);
    unsetenv("GGML_SCHED_SKIP_DEAD_INPUTS");
}

int main() {
    run(false);
    run(true);
    run_dead(true, false);
    run_dead(false, false);
    run_dead(true, true);
    printf("OK\n");
    return 0;
}
