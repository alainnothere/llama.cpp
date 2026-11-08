// [MTP_CIRCUIT] (fork, 2026-11-07) the drafter-circuit pass list of the MTP draft graph: grammar, shorthand, borrow
// resolution, model checks against a synthetic qwen35-27B-shaped hparams (64 trunk layers, full attention every 4th,
// one NextN block), the keep-or-clear rule of a runtime switch, the K/V slot ids and the hparams clone layers.
// No model, CPU only, runs in ctest. The graph itself is covered by the 27B CPU smoke in
// agent-reports/2026-11-07-mtp-circuit-engine.md (logits of M0 bit-identical to the pre-circuit build).

#include "../src/llama-ext.h"
#include "../src/llama-hparams.h"
#include "../src/llama-mtp-circuit.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;
static int n_ok   = 0;

#define CHECK(cond, ...) do { if (cond) { n_ok++; } else { n_fail++; fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// canonical form or "ERR"
static std::string parse(const char * s, std::vector<llama_mtp_pass> * out = nullptr, std::string * err_out = nullptr) {
    std::vector<llama_mtp_pass> passes;
    std::string canonical, err;
    const bool ok = llama_mtp_circuit_parse(s, passes, canonical, err);
    if (out) {
        *out = passes;
    }
    if (err_out) {
        *err_out = err;
    }
    return ok ? canonical : "ERR";
}

static std::string slots(const std::vector<llama_mtp_pass> & p) {
    std::string r;
    for (size_t i = 0; i < p.size(); ++i) {
        r += (i ? "," : "") + std::to_string(p[i].slot);
    }
    return r;
}

static llama_hparams make_hparams() {
    llama_hparams hp = {};
    hp.n_layer_all        = 65;
    hp.n_layer_nextn      = 1;
    hp.n_embd             = 5120;
    hp.n_embd_head_k_full = 256;
    hp.n_embd_head_v_full = 256;
    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        hp.n_head_arr[il]    = 24;
        hp.n_head_kv_arr[il] = 4;
        hp.n_ff_arr[il]      = 17408;
        hp.is_recr_impl[il]  = il < 64 && (il + 1) % 4 != 0;
    }
    return hp;
}

int main() {
    // --- grammar
    std::vector<llama_mtp_pass> p;
    std::string err;

    CHECK(parse("") == "M0", "%s", parse("").c_str());
    CHECK(parse(nullptr) == "M0", "null");
    CHECK(parse(" DeFault ") == "M0", "default keyword");
    CHECK(parse("M0", &p) == "M0" && p.size() == 1 && p[0].kind == 'M' && p[0].borrow < 0 && p[0].slot == 0, "M0");
    CHECK(parse("M0*2", &p) == "M0,M0" && slots(p) == "0,1", "%s", slots(p).c_str());
    CHECK(parse("M0,M0:b0", &p) == "M0,M0:b0" && slots(p) == "0,0" && p[1].borrow == 0, "borrow");
    CHECK(parse(" m0 , m0 : B0 ") == "M0,M0:b0", "case + whitespace");
    CHECK(parse("M0,M0*3:b0", &p) == "M0,M0:b0,M0:b0,M0:b0" && slots(p) == "0,0,0,0", "repeat with mode");
    CHECK(parse("M0,M0:b0*2", &p) == "M0,M0:b0,M0:b0", "mode before repeat");
    CHECK(parse("M0,M0:b0,M0,M0:b2", &p) == "M0,M0:b0,M0,M0:b2" && slots(p) == "0,0,1,1", "%s", slots(p).c_str());
    CHECK(parse("M0,M0:b0,M0:b1", &p) == "M0,M0:b0,M0:b1" && slots(p) == "0,0,0", "borrow of a borrow resolves to the owner");
    CHECK(parse("T63,M0:o", &p) == "T63,M0" && p[0].kind == 'T' && p[0].idx == 63 && slots(p) == "0,1", "trunk + explicit own");
    CHECK(parse("M0:own") == "M0", "own spelled out");
    CHECK(parse("M0*16") != "ERR", "16 passes");

    // sub-block passes (2026-11-07 follow-up): .attn takes a K/V slot like a full pass, .ffn none (slot -1)
    CHECK(parse("M0.attn,M0.attn,M0.ffn", &p) == "M0.attn,M0.attn,M0.ffn" && slots(p) == "0,1,-1" && p[0].part == 'A' && p[2].part == 'F', "%s", slots(p).c_str());
    CHECK(parse("M0.attn,M0.ffn,M0.ffn", &p) == "M0.attn,M0.ffn,M0.ffn" && slots(p) == "0,-1,-1", "%s", slots(p).c_str());
    CHECK(parse("M0,M0.ffn", &p) == "M0,M0.ffn" && slots(p) == "0,-1" && p[0].part == 'B', "%s", slots(p).c_str());
    CHECK(parse("M0,T63.ffn", &p) == "M0,T63.ffn" && p[1].kind == 'T' && p[1].part == 'F', "trunk ffn");
    CHECK(parse("M0.attn,M0.attn:b0,M0.ffn", &p) == "M0.attn,M0.attn:b0,M0.ffn" && slots(p) == "0,0,-1", "%s", slots(p).c_str());
    CHECK(parse(" m0 . ATTN * 2 , M0.FFN ", &p) == "M0.attn,M0.attn,M0.ffn" && slots(p) == "0,1,-1", "case + whitespace + repeat");
    CHECK(parse("M0.attn,M0.ffn,M0.attn:b0") == "M0.attn,M0.ffn,M0.attn:b0", "borrow across an ffn pass");
    for (const char * bad : { "M0.ffn:b0", "M0.ffn:o", "M0.ffn,M0:b0", "M0.ffn,M0.attn:b0", "M0.foo", "M0.", "M0.attn.ffn" }) {
        CHECK(parse(bad, nullptr, &err) == "ERR" && !err.empty(), "'%s' should fail", bad);
    }

    // errors, each with a message
    for (const char * bad : { "X0", "M", "T", "M0*0", "M0*17", "M0*16,M0", "M0:b0", "M0,M0:b1", "M0:t", "M0:z", "M0,,M0",
                              "M0 M0", "M0:b", "M0,", "M0*", "M0*2*2", "M0:b0:b0", "-1" }) {
        CHECK(parse(bad, nullptr, &err) == "ERR" && !err.empty(), "'%s' should fail", bad);
    }
    parse("M0:t", nullptr, &err);
    CHECK(err.find("not supported") != std::string::npos, "%s", err.c_str());

    // --- model checks (qwen35 27B shape)
    const llama_hparams hp = make_hparams();
    llama_mtp_circuit c;
    CHECK(llama_mtp_circuit_build("M0", false, hp, 1, c, err) && c.is_default() && c.n_own == 1, "default ok on any arch");
    CHECK(!llama_mtp_circuit_build("M0,M0:b0", false, hp, 2, c, err), "non-default needs a graph that reads it");
    CHECK(llama_mtp_circuit_build("M0,M0:b0", true, hp, 1, c, err) && c.n_own == 1 && !c.is_default(), "borrow needs no slot: %s", err.c_str());
    CHECK(!llama_mtp_circuit_build("M0*2", true, hp, 1, c, err) && err.find("own-K/V") != std::string::npos, "%s", err.c_str());
    CHECK(llama_mtp_circuit_build("M0*2", true, hp, 2, c, err) && c.n_own == 2, "%s", err.c_str());
    CHECK(llama_mtp_circuit_build("T63", true, hp, 1, c, err), "T63 is full attention: %s", err.c_str());
    CHECK(llama_mtp_circuit_build("M0,T3,T7:b1", true, hp, 2, c, err), "%s", err.c_str());
    CHECK(!llama_mtp_circuit_build("T62", true, hp, 1, c, err) && err.find("gated-delta-net") != std::string::npos, "%s", err.c_str());
    CHECK(!llama_mtp_circuit_build("T64", true, hp, 1, c, err), "T64 is the NextN block, not a trunk layer");
    CHECK(!llama_mtp_circuit_build("M1", true, hp, 1, c, err), "one NextN block only");
    CHECK(llama_mtp_circuit_build("M0.attn,M0.attn,M0.ffn", true, hp, 2, c, err) && c.n_own == 2, "%s", err.c_str());
    CHECK(!llama_mtp_circuit_build("M0.attn,M0.attn,M0.ffn", true, hp, 1, c, err), "two .attn slots need max passes 2");
    CHECK(llama_mtp_circuit_build("M0.attn,M0.ffn,M0.ffn", true, hp, 1, c, err) && c.n_own == 1, "%s", err.c_str());
    CHECK(llama_mtp_circuit_build("M0,M0.ffn", true, hp, 1, c, err) && c.n_own == 1 && !c.is_default(), "%s", err.c_str());
    CHECK(llama_mtp_circuit_build("M0,T63.ffn", true, hp, 1, c, err), "%s", err.c_str());
    CHECK(llama_mtp_circuit_build("M0,T62.ffn", true, hp, 1, c, err), "the FFN of a GDN layer needs no state: %s", err.c_str());
    CHECK(!llama_mtp_circuit_build("T62.attn", true, hp, 1, c, err), "GDN attention");
    CHECK(llama_mtp_circuit_build("M0.attn,M0.attn:b0,M0.ffn", true, hp, 1, c, err) && c.n_own == 1, "%s", err.c_str());
    CHECK(!llama_mtp_circuit_build("M0.ffn", false, hp, 1, c, err), "sub-blocks are non-default");
    {
        llama_hparams hp2 = hp;
        hp2.n_head_kv_arr[63] = 8;
        CHECK(!llama_mtp_circuit_build("T63", true, hp2, 1, c, err) && err.find("attention shape") != std::string::npos, "%s", err.c_str());
    }

    // --- keep-or-clear on a runtime switch
    auto keeps = [&](const char * a, const char * b) {
        llama_mtp_circuit ca, cb;
        std::string e;
        const bool ok = llama_mtp_circuit_build(a, true, hp, 4, ca, e) && llama_mtp_circuit_build(b, true, hp, 4, cb, e);
        CHECK(ok, "%s / %s: %s", a, b, e.c_str());
        return llama_mtp_circuit_keeps_kv(ca, cb);
    };
    CHECK( keeps("M0", "M0,M0:b0"), "add a borrow pass");
    CHECK( keeps("M0,M0:b0", "M0"), "drop a borrow pass");
    CHECK(!keeps("M0", "M0*2"), "new own pass has no history");
    CHECK( keeps("M0*2", "M0"), "prefix");
    CHECK( keeps("M0,M0", "M0,M0:b0"), "slot 0 history is valid for the borrower");
    CHECK(!keeps("T63", "M0"), "pass 0 changed");
    CHECK(!keeps("M0,T63", "M0,T3"), "pass 1 changed");
    CHECK( keeps("M0,T63", "M0,T63,M0:b1"), "extend with a borrow");
    CHECK( keeps("M0", "M0,M0.ffn"), "an ffn pass adds no K/V");
    CHECK( keeps("M0,M0.ffn", "M0"), "drop an ffn pass");
    CHECK(!keeps("M0", "M0.attn,M0.ffn"), "pass 0 is a different part: cleared (conservative)");
    CHECK(!keeps("M0.attn,M0.ffn", "M0.attn,M0.attn,M0.ffn"), "new own .attn slot");
    CHECK( keeps("M0.attn,M0.ffn", "M0.attn,M0.ffn,M0.attn:b0"), "borrowing .attn after the prefix");

    // --- K/V slot ids: slot 0 = the NextN block's layer, then the clone layers after n_layer_all
    CHECK(llama_mtp_circuit_slot_il(hp, 0) == 64, "slot 0");
    CHECK(llama_mtp_circuit_slot_il(hp, 1) == 65, "slot 1");
    CHECK(llama_mtp_circuit_slot_il(hp, 3) == 67, "slot 3");

    // --- hparams clones (what llama_kv_cache builds the extra slots from)
    {
        const llama_hparams hc = hp.with_layer_clones(64, 3);
        CHECK(hc.n_layer_all == 68 && hc.n_layer() == 64 && hc.n_layer_nextn == 4, "counts");
        bool same = true;
        for (uint32_t il = 65; il < 68; ++il) {
            same &= hc.n_embd_k_gqa(il) == hp.n_embd_k_gqa(64) && hc.n_embd_v_gqa(il) == hp.n_embd_v_gqa(64) &&
                    hc.n_head(il) == hp.n_head(64) && !hc.is_recr(il) && !hc.is_swa(il) && hc.has_kv(il);
        }
        CHECK(same, "clone layers are copies of the source layer");
        CHECK(hc.n_embd_k_gqa(65) == 1024, "27B K row: 4 heads x 256");
    }

    printf("test-mtp-circuit: %d ok, %d failed\n", n_ok, n_fail);
    return n_fail == 0 ? 0 : 1;
}
