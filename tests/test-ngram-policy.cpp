// 2026-10-20: tests for the ngram-mod chain policies --spec-ngram-auto (A) and --spec-ngram-resume N (B), see the
// comment block at AUTO_STEP in common/speculative.cpp. drives the public draft/accept API with synthetic tokens: S is
// 200 distinct tokens, the prompt holds one copy, generation reproduces it, optionally with substituted tokens (X, Y)
// that the table has never seen - exactly the "file rewrite with renames" case that motivated (B)

#include "speculative.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <vector>

static constexpr llama_token X = 5000;
static constexpr llama_token Y = 5001;

static llama_tokens make_s() {
    llama_tokens s;
    for (int i = 0; i < 200; ++i) {
        s.push_back(1000 + i);
    }
    return s;
}

struct driver {
    common_speculative * spec = nullptr;

    llama_tokens target; // the text the target model "generates", index = position in hist
    llama_tokens hist;   // committed tokens, the last one is id_last
    llama_tokens prompt_buf;
    llama_tokens draft;

    driver(bool auto_on, int resume_n, llama_tokens target_) : target(std::move(target_)) {
        common_params_speculative params;
        params.types             = { COMMON_SPECULATIVE_TYPE_NGRAM_MOD };
        params.ngram_mod.n_match = 4;
        params.ngram_mod.n_min   = 4;
        params.ngram_mod.n_max   = 32;
        params.ngram_mod.size    = 1 << 16;
        params.ngram_draft_n_max = 32;
        params.ngram_auto        = auto_on;
        params.ngram_resume      = resume_n;

        spec = common_speculative_init(params, 1);
        assert(spec != nullptr);

        // one copy of S plus the first 10 tokens of the second copy
        hist.assign(target.begin(), target.begin() + 210);
        common_speculative_begin(spec, 0, llama_tokens(hist.begin(), hist.end() - 1));
    }

    ~driver() {
        common_speculative_free(spec);
    }

    const llama_tokens & draft_once() {
        prompt_buf.assign(hist.begin(), hist.end() - 1);
        draft.clear();

        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp = {};
        dp.drafting    = true;
        dp.n_max       = 32;
        dp.pos0        = (llama_pos) hist.size() - 1;
        dp.id_last     = hist.back();
        dp.prompt      = &prompt_buf;
        dp.result      = &draft;
        dp.n_max_ngram = 32;

        common_speculative_draft(spec);
        return draft;
    }

    // greedy verification against `target`: accepted prefix, then the target's own token
    int verify() {
        const size_t base = hist.size();
        int k = 0;
        while (k < (int) draft.size() && draft[k] == target[base + k]) {
            k++;
        }
        common_speculative_accept(spec, 0, (uint16_t) k);
        for (int i = 0; i < k; ++i) {
            hist.push_back(draft[i]);
        }
        hist.push_back(target[hist.size()]);
        return k;
    }

    common_speculative_ngram_policy_stats st() const {
        return common_speculative_get_ngram_policy_stats(spec, 0);
    }
};

static llama_tokens target_with(std::vector<std::pair<size_t, llama_token>> subs) {
    const llama_tokens s = make_s();
    llama_tokens t = s;
    t.insert(t.end(), s.begin(), s.end());
    t.insert(t.end(), s.begin(), s.end());
    for (auto [pos, tok] : subs) {
        t[pos] = tok;
    }
    return t;
}

// offset of S token i in the second copy
static llama_token s_at(int i) { return 1000 + i; }

int main() {
    // (B) off: after the substitution at 220 the key holds X and the table goes quiet
    {
        driver d(false, 0, target_with({{220, X}}));
        const auto & c = d.draft_once();
        assert(c.size() == 32 && c[0] == s_at(10));
        assert(d.verify() == 10);
        assert(d.draft_once().empty());
        d.verify();
    }

    // (B) on: the old chain is resumed one past the corrected token and lands whole
    {
        driver d(false, 1, target_with({{220, X}}));
        d.draft_once();
        assert(d.verify() == 10);

        const auto & c = d.draft_once();
        assert(c.size() == 21);
        assert(c.front() == s_at(21) && c.back() == s_at(41));
        assert(d.verify() == 21);

        auto st = d.st();
        assert(st.resume_offered == 1 && st.resume_tok_off == 21 && st.resume_tok_land == 21 && st.resume_whole == 1);
        assert(st.resume_dead == 0 && st.resume_denied == 0);

        // back on the table: the target sampled S[42] after the whole landing, key S[39..43) -> fresh chain from S[43]
        const auto & c2 = d.draft_once();
        assert(!c2.empty() && c2[0] == s_at(43));
        d.verify();
        assert(d.st().resume_offered == 1);
    }

    // (B): the resumed tail is rejected at its first token -> dead, no further resume
    {
        driver d(false, 1, target_with({{220, X}, {221, Y}}));
        d.draft_once();
        assert(d.verify() == 10);
        assert(d.draft_once().size() == 21);
        assert(d.verify() == 0);
        assert(d.draft_once().empty());
        d.verify();

        const auto st = d.st();
        assert(st.resume_offered == 1 && st.resume_dead == 1 && st.resume_tok_land == 0);
    }

    // (B) with N = 2: two partial landings in a row drop the landed-whole history below 0.5 -> denied
    {
        driver d(false, 2, target_with({{220, X}, {225, Y}}));
        d.draft_once();
        assert(d.verify() == 10);
        assert(d.draft_once().size() == 21); // prior ema 0.5 -> allowed
        assert(d.verify() == 4);
        assert(d.draft_once().empty());      // prior ema 0.375 -> denied
        d.verify();

        const auto st = d.st();
        assert(st.resume_offered == 1 && st.resume_denied == 1 && st.resume_tok_land == 4 && st.resume_tok_off == 21);
    }

    // (A): start at 32/2 = 16, raise x1.5 on a whole landing the cap cut (16 -> 24 -> 32), hold at the ceiling,
    // drop to accepted + 4 rounded up to 4 on a partial landing (5 -> 12)
    {
        driver d(true, 0, target_with({{290, X}})); // hist 210 -> 227 -> 252 -> 285, the 4th chain hits X at index 5

        assert(d.st().cap_cur == 16);

        assert(d.draft_once().size() == 16);
        assert(d.verify() == 16);
        assert(d.st().cap_cur == 24);

        assert(d.draft_once().size() == 24);
        assert(d.verify() == 24);
        assert(d.st().cap_cur == 32);

        assert(d.draft_once().size() == 32);
        assert(d.verify() == 32);
        assert(d.st().cap_cur == 32);

        assert(d.draft_once().size() == 32);
        assert(d.verify() == 5);

        const auto st = d.st();
        assert(st.cap_cur == 12);
        assert(st.cap_n == 4 && st.cap_min == 16 && st.cap_median == 24 && st.cap_max == 32);
        assert(st.cap_bound == 2);

        assert(d.draft_once().empty());
        d.verify();
    }

    printf("test-ngram-policy: OK\n");
    return 0;
}
