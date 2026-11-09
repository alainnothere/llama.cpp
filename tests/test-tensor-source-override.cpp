// [fork, 2026-11-09] --override-tensor-source (llama_model_params.tensor_source_overrides), CPU only.
//
// Model A = the generated dummy llama-dense.gguf (test-generate-models fixture; missing model = SKIP, like
// test-graph-reuse-pp). Source B is written here with gguf: blk.0.* only, no KV at all (the overlay ignores the
// source's metadata, which is what a head-only GGUF needs), with
//   blk.0.attn_q.weight     = A's values + 1, stored as F16 (type change, half the bytes)
//   blk.0.attn_norm.weight  = 7.0 everywhere (same type)
// and every other blk.0 tensor copied as is (present in B but not matched, so it must be ignored).
// Asserts: overlaid tensors carry B's type and data, the rest A's data, llama_model_size moves by the byte delta,
// a no_alloc load (what every fit probe does) sees the F16 type, a decode runs, and the error cases fail the load:
// pattern matching nothing, shape mismatch, model tensor missing from B, B tensor missing from the model,
// one tensor claimed by two overrides.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"

#include "../src/llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static const char * Q_NAME    = "blk.0.attn_q.weight";
static const char * NORM_NAME = "blk.0.attn_norm.weight";
static const char * K_NAME    = "blk.0.attn_k.weight";
static const char * Q1_NAME   = "blk.1.attn_q.weight";

static std::vector<float> to_f32(const ggml_tensor * t, const void * data) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, out.data(), n);
    } else {
        fprintf(stderr, "unexpected type %s for %s\n", ggml_type_name(t->type), t->name);
        exit(1);
    }
    return out;
}

// reads a model tensor back through the backend (works for mmap and read loads alike)
static std::vector<float> model_tensor_f32(const llama_model * model, const char * name, ggml_type * type_out) {
    const ggml_tensor * t = nullptr;
    for (const auto & it : llama_internal_get_tensor_map(model)) {
        if (it.first == name) {
            t = it.second;
        }
    }
    CHECK(t != nullptr);
    *type_out = t->type;
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    return to_f32(t, raw.data());
}

// writes a GGUF holding only the tensors `pick` returns, edited by `edit` (which may change type/shape/data)
struct src_tensor {
    std::string        name;
    ggml_type          type;
    std::vector<int64_t> ne;
    std::vector<float> data;
};

static void write_source(const std::string & path, const std::vector<src_tensor> & ts) {
    size_t mem = ggml_tensor_overhead() * (ts.size() + 1);
    for (const auto & t : ts) {
        mem += t.data.size() * sizeof(float) + 64;
    }
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context_ptr ctx { ggml_init(ip) };
    gguf_context_ptr gguf { gguf_init_empty() };
    for (const auto & t : ts) {
        ggml_tensor * cur = ggml_new_tensor(ctx.get(), t.type, (int) t.ne.size(), t.ne.data());
        ggml_set_name(cur, t.name.c_str());
        CHECK((int64_t) t.data.size() == ggml_nelements(cur));
        if (t.type == GGML_TYPE_F32) {
            memcpy(cur->data, t.data.data(), ggml_nbytes(cur));
        } else {
            CHECK(t.type == GGML_TYPE_F16);
            ggml_fp32_to_fp16_row(t.data.data(), (ggml_fp16_t *) cur->data, ggml_nelements(cur));
        }
        gguf_add_tensor(gguf.get(), cur);
    }
    CHECK(gguf_write_to_file(gguf.get(), path.c_str(), false));
}

static llama_model_params cpu_params(const llama_model_tensor_source_override * tso, llama_load_mode mode, bool no_alloc) {
    llama_model_params mp = llama_model_default_params();
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu     = -1; // no devices: CPU only, the cards stay untouched
    mp.n_gpu_layers = 0;
    mp.load_mode    = mode;
    mp.no_alloc     = no_alloc;
    mp.tensor_source_overrides = tso;
    return mp;
}

static bool load_fails(const std::string & path_a, const std::vector<llama_model_tensor_source_override> & tso, const char * what) {
    std::vector<llama_model_tensor_source_override> list = tso;
    list.push_back({nullptr, nullptr});
    llama_model * m = llama_model_load_from_file(path_a.c_str(), cpu_params(list.data(), LLAMA_LOAD_MODE_NONE, true));
    if (m != nullptr) {
        fprintf(stderr, "FAIL: load succeeded but should fail: %s\n", what);
        llama_model_free(m);
        return false;
    }
    fprintf(stderr, "ok: load fails as expected: %s\n", what);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <llama-dense.gguf> <scratch dir>\n", argv[0]);
        return 1;
    }
    const std::string path_a = argv[1];
    const std::string dir    = argv[2];

    FILE * f = fopen(path_a.c_str(), "rb");
    if (f == nullptr) {
        printf("SKIP: %s not found (run the test-generate-models fixture)\n", path_a.c_str());
        return 0;
    }
    fclose(f);

    llama_backend_init();

    // A's tensors with data
    ggml_context * ctx_a_raw = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_a_raw };
    gguf_context_ptr gguf_a { gguf_init_from_file(path_a.c_str(), gp) };
    CHECK(gguf_a && ctx_a_raw);
    ggml_context_ptr ctx_a { ctx_a_raw };

    auto a_f32 = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(ctx_a.get(), name);
        CHECK(t != nullptr);
        return to_f32(t, t->data);
    };
    auto a_tensor = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(ctx_a.get(), name);
        CHECK(t != nullptr);
        CHECK(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
        src_tensor s { name, t->type, std::vector<int64_t>(t->ne, t->ne + ggml_n_dims(t)), to_f32(t, t->data) };
        if (s.ne.empty()) { s.ne.push_back(1); }
        return s;
    };

    CHECK(ggml_get_tensor(ctx_a.get(), Q1_NAME) != nullptr); // need a second block for the "not overlaid" check
    const ggml_tensor * q_a = ggml_get_tensor(ctx_a.get(), Q_NAME);
    CHECK(q_a != nullptr && q_a->type == GGML_TYPE_F32); // the F32 -> F16 change below depends on it

    // B: all blk.0.* tensors, Q edited + F16, norm = 7
    std::vector<src_tensor> b;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_a.get()); t; t = ggml_get_next_tensor(ctx_a.get(), t)) {
        if (strncmp(t->name, "blk.0.", 6) != 0) {
            continue;
        }
        src_tensor s = a_tensor(t->name);
        if (s.name == Q_NAME) {
            for (float & v : s.data) { v += 1.0f; }
            s.type = GGML_TYPE_F16;
        } else if (s.name == NORM_NAME) {
            for (float & v : s.data) { v = 7.0f; }
        }
        b.push_back(s);
    }
    const std::string path_b = dir + "/tso-source-b.gguf";
    write_source(path_b, b);

    // C: blk.0.attn_norm with the wrong shape
    src_tensor norm_bad = a_tensor(NORM_NAME);
    norm_bad.ne[0] += 1;
    norm_bad.data.resize(norm_bad.data.size() + 1, 1.0f);
    const std::string path_c = dir + "/tso-source-c-shape.gguf";
    write_source(path_c, { norm_bad });

    // D: a matching tensor the model does not have
    src_tensor extra = a_tensor(NORM_NAME);
    extra.name = "blk.0.attn_extra.weight";
    const std::string path_d = dir + "/tso-source-d-extra.gguf";
    write_source(path_d, { a_tensor(NORM_NAME), extra });

    const char * pat = "^blk\\.0\\.attn_(q|norm)\\.weight$";

    // ---- positive: mmap and read loads
    for (llama_load_mode mode : { LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE }) {
        const char * mode_name = mode == LLAMA_LOAD_MODE_MMAP ? "mmap" : "read";

        llama_model * plain = llama_model_load_from_file(path_a.c_str(), cpu_params(nullptr, mode, false));
        CHECK(plain != nullptr);
        const uint64_t size_plain = llama_model_size(plain);
        llama_model_free(plain);

        llama_model_tensor_source_override tso[] = { { pat, path_b.c_str() }, { nullptr, nullptr } };
        llama_model * m = llama_model_load_from_file(path_a.c_str(), cpu_params(tso, mode, false));
        CHECK(m != nullptr);

        ggml_type type;
        // (a) overlaid tensors come from B
        {
            const std::vector<float> got = model_tensor_f32(m, Q_NAME, &type);
            const std::vector<float> ref = a_f32(Q_NAME);
            CHECK(type == GGML_TYPE_F16);
            CHECK(got.size() == ref.size());
            for (size_t i = 0; i < got.size(); i++) {
                CHECK(std::fabs(got[i] - (ref[i] + 1.0f)) < 2e-3f);
            }
            const std::vector<float> norm = model_tensor_f32(m, NORM_NAME, &type);
            CHECK(type == GGML_TYPE_F32);
            for (float v : norm) {
                CHECK(v == 7.0f);
            }
        }
        // (b) everything else comes from A, including blk.0 tensors B has but the pattern does not match
        for (const char * name : { K_NAME, Q1_NAME }) {
            const std::vector<float> got = model_tensor_f32(m, name, &type);
            const std::vector<float> ref = a_f32(name);
            CHECK(got == ref);
        }
        // the size the server reports (and the loader's n_bytes) moves by the F32 -> F16 delta of attn_q
        const uint64_t delta = ggml_nelements(q_a) * (sizeof(float) - sizeof(ggml_fp16_t));
        CHECK(llama_model_size(m) == size_plain - delta);

        // a decode runs on the mixed-source model
        {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx   = 64;
            cp.n_batch = 16;
            cp.n_ubatch = 16;
            llama_context * lctx = llama_init_from_model(m, cp);
            CHECK(lctx != nullptr);
            const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
            std::vector<llama_token> toks = { 1 % n_vocab, 2 % n_vocab, 3 % n_vocab, 4 % n_vocab };
            CHECK(llama_decode(lctx, llama_batch_get_one(toks.data(), (int32_t) toks.size())) == 0);
            const float * logits = llama_get_logits_ith(lctx, -1);
            CHECK(logits != nullptr);
            for (int32_t i = 0; i < n_vocab; i++) {
                CHECK(std::isfinite(logits[i]));
            }
            llama_free(lctx);
        }
        llama_model_free(m);
        fprintf(stderr, "ok: %s load: overlay data from B, rest from A, size -%llu B, decode ok\n",
                mode_name, (unsigned long long) delta);
    }

    // ---- no_alloc load (every fit probe): the overlaid type is what gets measured
    {
        llama_model_tensor_source_override tso[] = { { pat, path_b.c_str() }, { nullptr, nullptr } };
        llama_model * m = llama_model_load_from_file(path_a.c_str(), cpu_params(tso, LLAMA_LOAD_MODE_NONE, true));
        CHECK(m != nullptr);
        bool found = false;
        for (const auto & it : llama_internal_get_tensor_map(m)) {
            if (it.first == Q_NAME) {
                CHECK(it.second->type == GGML_TYPE_F16);
                found = true;
            }
        }
        CHECK(found);
        llama_model_free(m);
        fprintf(stderr, "ok: no_alloc load sees F16 %s\n", Q_NAME);
    }

    // ---- negative cases
    bool ok = true;
    ok &= load_fails(path_a, { { "^nothing\\.matches\\.this$", path_b.c_str() } }, "(c) pattern matches no tensor");
    ok &= load_fails(path_a, { { "^blk\\.0\\.attn_norm\\.weight$", path_c.c_str() } }, "(d) shape mismatch");
    ok &= load_fails(path_a, { { "^blk\\.1\\.attn_q\\.weight$", path_b.c_str() } }, "model tensor missing from the source");
    ok &= load_fails(path_a, { { "^blk\\.0\\.attn_", path_d.c_str() } }, "source tensor missing from the model");
    ok &= load_fails(path_a, { { "^blk\\.0\\.attn_q\\.", path_b.c_str() }, { "^blk\\.0\\.attn_q\\.weight$", path_b.c_str() } },
            "one tensor claimed by two overrides");
    ok &= load_fails(path_a, { { "^blk\\.0\\.attn_q\\.", (dir + "/tso-does-not-exist.gguf").c_str() } }, "missing source file");
    ok &= load_fails(path_a, { { "(", path_b.c_str() } }, "invalid regex");

    std::remove(path_b.c_str());
    std::remove(path_c.c_str());
    std::remove(path_d.c_str());

    llama_backend_free();

    if (!ok) {
        return 1;
    }
    printf("PASS\n");
    return 0;
}
