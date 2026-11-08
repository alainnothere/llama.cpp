#include "llama-mtp-circuit.h"

#include "llama-hparams.h"

#include <cctype>
#include <cstdlib>

// [MTP_CIRCUIT] (fork, 2026-11-07) parser for the draft-graph pass list. Pure string work so the grammar has a ctest
// without a model (tests/test-mtp-circuit.cpp); the model checks are in llama_mtp_circuit_build below.

namespace {

struct cursor {
    const char * s;
    size_t       i = 0;

    void skip_ws() {
        while (s[i] == ' ' || s[i] == '\t') {
            i++;
        }
    }
    char peek() {
        skip_ws();
        return s[i];
    }
    // unsigned decimal, false if there is none; caps at 1e6 so a long digit run cannot overflow
    bool num(int32_t & out) {
        skip_ws();
        if (!isdigit((unsigned char) s[i])) {
            return false;
        }
        long v = 0;
        while (isdigit((unsigned char) s[i])) {
            v = v*10 + (s[i] - '0');
            if (v > 1000000) {
                v = 1000000;
            }
            i++;
        }
        out = (int32_t) v;
        return true;
    }
};

std::string pass_str(const llama_mtp_pass & p) {
    std::string r(1, p.kind);
    r += std::to_string(p.idx);
    if (p.part == 'A') {
        r += ".attn";
    } else if (p.part == 'F') {
        r += ".ffn";
    }
    if (p.borrow >= 0) {
        r += ":b" + std::to_string(p.borrow);
    }
    return r;
}

} // namespace

bool llama_mtp_circuit_parse(const char * circuit, std::vector<llama_mtp_pass> & passes, std::string & canonical, std::string & err) {
    passes.clear();
    canonical.clear();
    err.clear();

    std::string s = circuit ? circuit : "";
    // "" / "default" = the pre-circuit graph
    {
        std::string t;
        for (char c : s) {
            if (c != ' ' && c != '\t') {
                t += (char) tolower((unsigned char) c);
            }
        }
        if (t.empty() || t == "default") {
            s = "M0";
        }
    }

    cursor cur { s.c_str() };
    while (true) {
        const size_t at = cur.i;
        const char   u  = (char) toupper((unsigned char) cur.peek());
        if (u != 'M' && u != 'T') {
            err = "pass " + std::to_string(passes.size()) + " at offset " + std::to_string(at) +
                  ": expected M<k> (NextN block) or T<il> (trunk layer)";
            return false;
        }
        cur.i++;
        int32_t idx = 0;
        if (!cur.num(idx)) {
            err = std::string("pass ") + std::to_string(passes.size()) + ": '" + u + "' needs a number (M0, T63)";
            return false;
        }

        // [MTP_CIRCUIT] 2026-11-07 sub-block passes: ".attn" / ".ffn" (no whitespace inside the word)
        char part = 'B';
        if (cur.peek() == '.') {
            cur.i++;
            cur.skip_ws();
            std::string w;
            while (isalpha((unsigned char) cur.s[cur.i])) {
                w += (char) tolower((unsigned char) cur.s[cur.i++]);
            }
            if (w == "attn") {
                part = 'A';
            } else if (w == "ffn") {
                part = 'F';
            } else {
                err = "pass " + std::to_string(passes.size()) + ": unknown sub-block '." + w + "', expected '.attn' or '.ffn'";
                return false;
            }
        }

        int32_t rep    = 1;
        int32_t borrow = -1;
        bool    has_rep = false, has_mode = false;
        while (true) {
            const char c = cur.peek();
            if (c == '*' && !has_rep) {
                cur.i++;
                if (!cur.num(rep) || rep < 1 || rep > LLAMA_MTP_CIRCUIT_MAX_LEN) {
                    err = "pass " + std::to_string(passes.size()) + ": '*' needs a count in 1.." + std::to_string(LLAMA_MTP_CIRCUIT_MAX_LEN);
                    return false;
                }
                has_rep = true;
            } else if (c == ':' && !has_mode) {
                cur.i++;
                const char m = (char) tolower((unsigned char) cur.peek());
                if (m == 'b') {
                    cur.i++;
                    if (!cur.num(borrow)) {
                        err = "pass " + std::to_string(passes.size()) + ": ':b' needs the index of the pass to borrow from (M0:b0)";
                        return false;
                    }
                } else if (m == 'o') {
                    cur.i++;
                    // accept "own" spelled out
                    if (tolower((unsigned char) cur.s[cur.i]) == 'w' && tolower((unsigned char) cur.s[cur.i + 1]) == 'n') {
                        cur.i += 2;
                    }
                    borrow = -1;
                } else if (m == 't') {
                    err = "pass " + std::to_string(passes.size()) + ": ':t' (borrow the target's K/V) is not supported: ctx_tgt and ctx_dft "
                          "are separate contexts with separate cell numbering and the target holds no K/V for the draft positions";
                    return false;
                } else {
                    err = "pass " + std::to_string(passes.size()) + ": unknown K/V mode, expected ':o' (own) or ':b<j>' (borrow pass j)";
                    return false;
                }
                has_mode = true;
            } else {
                break;
            }
        }

        for (int32_t r = 0; r < rep; ++r) {
            if ((int32_t) passes.size() >= LLAMA_MTP_CIRCUIT_MAX_LEN) {
                err = "more than " + std::to_string(LLAMA_MTP_CIRCUIT_MAX_LEN) + " passes";
                return false;
            }
            llama_mtp_pass p;
            p.kind   = u;
            p.idx    = idx;
            p.part   = part;
            p.borrow = borrow;
            if (part == 'F' && has_mode) {
                err = "pass " + std::to_string(passes.size()) + " (" + pass_str(p) + "): an FFN sub-block has no K/V, it takes no ':o' / ':b<j>' mode";
                return false;
            }
            if (borrow >= 0 && borrow < (int32_t) passes.size() && passes[borrow].part == 'F') {
                err = "pass " + std::to_string(passes.size()) + " (" + pass_str(p) + "): pass " + std::to_string(borrow) + " is an FFN sub-block, it has no K/V to borrow";
                return false;
            }
            if (borrow >= 0 && borrow >= (int32_t) passes.size()) {
                err = "pass " + std::to_string(passes.size()) + " (" + pass_str(p) + "): can only borrow an earlier pass (index < " +
                      std::to_string(passes.size()) + ")";
                return false;
            }
            passes.push_back(p);
        }

        const char c = cur.peek();
        if (c == 0) {
            break;
        }
        if (c != ',') {
            err = std::string("unexpected '") + c + "' at offset " + std::to_string(cur.i) + ", expected ',' between passes";
            return false;
        }
        cur.i++;
    }

    // resolve the slots: own passes are numbered in order, a borrow takes its target's slot (a borrow of a borrow
    // ends at the same owner, so "b<j>" always names the K/V that pass j attended over)
    int32_t n_own = 0;
    for (auto & p : passes) {
        if (p.part == 'F') {
            p.slot = -1; // no attention, no K/V
        } else if (p.borrow < 0) {
            p.slot = n_own++;
        } else {
            p.slot = passes[p.borrow].slot;
        }
    }

    for (size_t i = 0; i < passes.size(); ++i) {
        canonical += (i ? "," : "") + pass_str(passes[i]);
    }

    return true;
}

int32_t llama_mtp_circuit_slot_il(const llama_hparams & hp, int32_t slot) {
    return slot == 0 ? (int32_t) hp.n_layer() : (int32_t) hp.n_layer_all + slot - 1;
}

bool llama_mtp_circuit_build(const char * str, bool arch_ok, const llama_hparams & hp, uint32_t max_passes,
        llama_mtp_circuit & out, std::string & err) {
    llama_mtp_circuit c;
    if (!llama_mtp_circuit_parse(str, c.passes, c.str, err)) {
        return false;
    }

    c.n_own = 0;
    for (const auto & p : c.passes) {
        c.n_own += llama_mtp_pass_owns_kv(p) ? 1 : 0;
    }

    if (!c.is_default() && !arch_ok) {
        err = "'" + c.str + "': this model's MTP draft graph does not read the circuit (implemented for qwen35 dense only), only M0 is accepted";
        return false;
    }

    if ((uint32_t) c.n_own > max_passes) {
        err = "'" + c.str + "' needs " + std::to_string(c.n_own) + " own-K/V passes but the draft memory holds " + std::to_string(max_passes) +
              " (start with --spec-mtp-circuit-max-passes / LLAMA_MTP_CIRCUIT_MAX_PASSES >= " + std::to_string(c.n_own) + ")";
        return false;
    }

    const uint32_t il_mtp = hp.n_layer();
    for (size_t i = 0; i < c.passes.size(); ++i) {
        const auto & p = c.passes[i];
        if (p.kind == 'M') {
            if (p.idx < 0 || (uint32_t) p.idx >= hp.n_layer_nextn) {
                err = "pass " + std::to_string(i) + ": M" + std::to_string(p.idx) + " but the model has " + std::to_string(hp.n_layer_nextn) + " NextN block(s)";
                return false;
            }
            if (p.idx != 0) {
                // the qwen35 MTP graph asserts n_layer_nextn == 1; kept as a check in case that assert goes
                err = "pass " + std::to_string(i) + ": only M0 is implemented";
                return false;
            }
            continue;
        }
        // T<il>
        if (p.idx < 0 || (uint32_t) p.idx >= hp.n_layer()) {
            err = "pass " + std::to_string(i) + ": T" + std::to_string(p.idx) + " is out of range (trunk layers 0.." + std::to_string(hp.n_layer() - 1) + ")";
            return false;
        }
        // an FFN sub-block of any trunk layer is a plain dense FFN (every Qwen3.5 trunk layer has attn_post_norm +
        // ffn_gate/up/down, GDN layers included): no recurrent state, no K/V, so no further check
        if (p.part == 'F') {
            continue;
        }
        if (hp.is_recr(p.idx)) {
            std::string full;
            for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                if (!hp.is_recr(il)) {
                    full += (full.empty() ? "" : ",") + std::to_string(il);
                }
            }
            err = "pass " + std::to_string(i) + ": T" + std::to_string(p.idx) + " is a gated-delta-net (linear attention) layer, its recurrent "
                  "state lives in the target context only (T" + std::to_string(p.idx) + ".ffn is allowed); full-attention trunk layers: " + full;
            return false;
        }
        // the pass writes / reads a draft K/V slot shaped like the NextN block's layer
        if (hp.n_embd_k_gqa(p.idx) != hp.n_embd_k_gqa(il_mtp) || hp.n_embd_v_gqa(p.idx) != hp.n_embd_v_gqa(il_mtp) ||
            hp.n_head_kv(p.idx) != hp.n_head_kv(il_mtp) || hp.n_head(p.idx) != hp.n_head(il_mtp)) {
            err = "pass " + std::to_string(i) + ": T" + std::to_string(p.idx) + " has a different attention shape than the NextN block (heads / K/V width)";
            return false;
        }
    }

    out = std::move(c);
    return true;
}

bool llama_mtp_circuit_keeps_kv(const llama_mtp_circuit & a, const llama_mtp_circuit & b) {
    // the default circuit is stored as an empty pass list; compare on expanded lists
    auto expand = [](const llama_mtp_circuit & c) {
        if (!c.passes.empty()) {
            return c.passes;
        }
        return std::vector<llama_mtp_pass>(1);
    };
    const auto pa = expand(a);
    const auto pb = expand(b);

    size_t n_common = 0;
    while (n_common < pa.size() && n_common < pb.size() &&
           pa[n_common].kind == pb[n_common].kind && pa[n_common].idx == pb[n_common].idx && pa[n_common].part == pb[n_common].part &&
           pa[n_common].borrow == pb[n_common].borrow) {
        n_common++;
    }
    for (size_t i = n_common; i < pb.size(); ++i) {
        if (llama_mtp_pass_owns_kv(pb[i])) {
            return false;
        }
    }
    return true;
}
