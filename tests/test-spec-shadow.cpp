// tests for the speculative shadow chains (log only): resolution against the committed tokens,
// the MTP_EXT prefix arithmetic and the pending overflow

#include "speculative.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <vector>

using chain_t = common_speculative_shadow_chain;

static chain_t make_chain(common_speculative_shadow_kind kind, size_t pos_first, llama_tokens tokens, int32_t n_prefix = 0) {
    chain_t c;
    c.kind      = kind;
    c.pos_first = pos_first;
    c.tokens    = std::move(tokens);
    c.n_prefix  = n_prefix;
    return c;
}

int main() {
    const auto NG  = COMMON_SPECULATIVE_SHADOW_NGRAM_FULL;
    const auto EXT = COMMON_SPECULATIVE_SHADOW_MTP_EXT;

    // drafted at prompt.size() == 4 with id_last == 50: chain starts at index 5
    const chain_t c = make_chain(NG, 5, {60, 70, 80});

    // full match, the last token is the id_last of the later draft call
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4, 50, 60, 70}, 80) == 3);
    // full match, all of it in prompt
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4, 50, 60, 70, 80, 90}, 91) == 3);
    // mismatch at k = 1
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4, 50, 60}, 99) == 1);
    // mismatch at k = 0
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4, 50}, 99) == 0);
    // partially committed: two tokens matched, the third is not known yet
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4, 50, 60}, 70) == -1);
    // nothing committed past the chain start yet
    assert(common_speculative_shadow_resolve(c, {1, 2, 3, 4}, 50) == -1);

    // NGRAM_FULL stats
    {
        common_speculative_shadow_seq sh;
        sh.push(make_chain(NG, 5, {60, 70, 80}));
        sh.resolve({1, 2, 3, 4, 50, 60}, 70); // unresolved
        assert(sh.pending.size() == 1 && sh.stats.ngram_chains == 0);
        sh.resolve({1, 2, 3, 4, 50, 60, 70}, 80);
        assert(sh.pending.empty());
        assert(sh.stats.ngram_chains == 1 && sh.stats.ngram_full == 1 && sh.stats.ngram_matched_sum == 3);
        assert(sh.stats.ngram_hist[1] == 1);

        sh.push(make_chain(NG, 8, {1, 2, 3, 4, 5, 6}));
        sh.resolve({1, 2, 3, 4, 50, 60, 70, 80, 1, 2, 3, 4, 5, 9}, 0);
        assert(sh.stats.ngram_chains == 2 && sh.stats.ngram_full == 1 && sh.stats.ngram_matched_sum == 8);
        assert(sh.stats.ngram_hist[2] == 1);
    }

    // MTP_EXT: draft {60, 70} + extension {80, 90, 100}
    {
        common_speculative_shadow_seq sh;

        // prefix only: the draft itself missed at k = 1 -> not tested
        sh.push(make_chain(EXT, 5, {60, 70, 80, 90, 100}, 2));
        sh.resolve({1, 2, 3, 4, 50, 60}, 71);
        assert(sh.stats.ext_resolved == 1 && sh.stats.ext_tested == 0 && sh.stats.ext_landed_sum == 0);

        // draft landed, extension missed at once -> tested, landed 0
        sh.push(make_chain(EXT, 5, {60, 70, 80, 90, 100}, 2));
        sh.resolve({1, 2, 3, 4, 50, 60, 70}, 81);
        assert(sh.stats.ext_tested == 1 && sh.stats.ext_landed_sum == 0 && sh.stats.ext_landed_hist[0] == 1);

        // draft landed, two extension tokens landed
        sh.push(make_chain(EXT, 5, {60, 70, 80, 90, 100}, 2));
        sh.resolve({1, 2, 3, 4, 50, 60, 70, 80, 90}, 7);
        assert(sh.stats.ext_tested == 2 && sh.stats.ext_landed_sum == 2 && sh.stats.ext_landed_hist[1] == 1);

        // whole chain landed
        sh.push(make_chain(EXT, 5, {60, 70, 80, 90, 100}, 2));
        sh.resolve({1, 2, 3, 4, 50, 60, 70, 80, 90}, 100);
        assert(sh.stats.ext_tested == 3 && sh.stats.ext_landed_sum == 5 && sh.stats.ext_resolved == 4);
        assert(sh.pending.empty());
    }

    // overflow drops the oldest as censored, censor_all flushes the rest
    {
        common_speculative_shadow_seq sh;
        for (size_t i = 0; i < common_speculative_shadow_seq::N_PENDING_MAX + 2; ++i) {
            sh.push(make_chain(i % 2 ? EXT : NG, 100 + i, {(llama_token) i}, i % 2 ? 1 : 0));
        }
        assert(sh.pending.size() == common_speculative_shadow_seq::N_PENDING_MAX);
        assert(sh.pending.front().pos_first == 102);
        assert(sh.stats.ngram_censored == 1 && sh.stats.ext_censored == 1);

        sh.censor_all();
        assert(sh.pending.empty());
        assert(sh.stats.ngram_censored + sh.stats.ext_censored == (int) common_speculative_shadow_seq::N_PENDING_MAX + 2);
        assert(sh.stats.ngram_chains == 0 && sh.stats.ext_resolved == 0);
    }

    // mixed resolution keeps the unresolved ones in order
    {
        common_speculative_shadow_seq sh;
        sh.push(make_chain(NG, 2, {3, 4}));       // resolves full
        sh.push(make_chain(NG, 3, {4, 5, 6, 7})); // unresolved
        sh.push(make_chain(NG, 4, {9}));          // resolves at 0
        sh.push(make_chain(NG, 5, {6, 7, 8}));    // unresolved
        sh.resolve({1, 2, 3, 4}, 5);
        assert(sh.stats.ngram_chains == 2 && sh.stats.ngram_full == 1 && sh.stats.ngram_matched_sum == 2);
        assert(sh.pending.size() == 2 && sh.pending[0].pos_first == 3 && sh.pending[1].pos_first == 5);
    }

    // NGRAM_ALT resolves into its own stats
    {
        const auto ALT = COMMON_SPECULATIVE_SHADOW_NGRAM_ALT;

        common_speculative_shadow_seq sh;
        sh.push(make_chain(ALT, 5, {60, 70, 80, 90}));
        sh.push(make_chain(ALT, 5, {61}));
        sh.push(make_chain(ALT, 6, {70, 80, 90, 95})); // stays pending
        sh.resolve({1, 2, 3, 4, 50, 60, 70, 80}, 90);
        assert(sh.stats.alt_chains == 2 && sh.stats.alt_matched_sum == 4);
        assert(sh.stats.alt_hist[0] == 1 && sh.stats.alt_hist[1] == 1);
        assert(sh.stats.ngram_chains == 0 && sh.stats.ext_resolved == 0);
        assert(sh.pending.size() == 1);
        sh.censor_all();
        assert(sh.stats.alt_censored == 1 && sh.stats.ngram_censored == 0 && sh.stats.ext_censored == 0);
    }

    // draft-mtp confidence bins
    {
        common_speculative_mtp_stats st;

        // 4 offered (5th cut by the cap), 2 landed, token 2 rejected, token 3 not tested
        common_speculative_mtp_stats_add(st, {0.95f, 0.85f, 0.55f, 0.3f, 0.99f}, 4, 2, COMMON_SPECULATIVE_MTP_STOP_CAP);
        assert(st.stop_steps[COMMON_SPECULATIVE_MTP_STOP_CAP] == 1 && st.stop_accepted[COMMON_SPECULATIVE_MTP_STOP_CAP] == 2);
        assert(st.pbin_tested[5] == 1 && st.pbin_landed[5] == 1);
        assert(st.pbin_tested[4] == 1 && st.pbin_landed[4] == 1);
        assert(st.pbin_tested[1] == 1 && st.pbin_landed[1] == 0);
        assert(st.pbin_tested[0] == 0);
        assert(st.pos_tested[0] == 1 && st.pos_landed[0] == 1 && st.pos_tested[2] == 1 && st.pos_landed[2] == 0 && st.pos_tested[3] == 0);

        // everything landed, one token without a probability
        common_speculative_mtp_stats_add(st, {0.6f, -1.0f}, 2, 2, COMMON_SPECULATIVE_MTP_STOP_PMIN);
        assert(st.stop_steps[COMMON_SPECULATIVE_MTP_STOP_PMIN] == 1 && st.stop_accepted[COMMON_SPECULATIVE_MTP_STOP_PMIN] == 2);
        assert(st.pbin_tested[2] == 1 && st.pbin_landed[2] == 1);
        assert(st.na_tested == 1 && st.na_landed == 1);

        // bin edges: 0.5 -> [0.5,0.6), 0.7 -> [0.7,0.8), 0.9 and 1.0 -> [0.9,1]; first token rejected
        common_speculative_mtp_stats st2;
        common_speculative_mtp_stats_add(st2, {0.5f}, 1, 0, COMMON_SPECULATIVE_MTP_STOP_OTHER);
        common_speculative_mtp_stats_add(st2, {0.7f}, 1, 0, COMMON_SPECULATIVE_MTP_STOP_OTHER);
        common_speculative_mtp_stats_add(st2, {0.9f}, 1, 0, COMMON_SPECULATIVE_MTP_STOP_OTHER);
        common_speculative_mtp_stats_add(st2, {1.0f}, 1, 0, COMMON_SPECULATIVE_MTP_STOP_OTHER);
        assert(st2.pbin_tested[1] == 1 && st2.pbin_tested[3] == 1 && st2.pbin_tested[5] == 2);
        assert(st2.pbin_landed[1] + st2.pbin_landed[3] + st2.pbin_landed[5] == 0);
        assert(st2.stop_steps[COMMON_SPECULATIVE_MTP_STOP_OTHER] == 4 && st2.stop_accepted[COMMON_SPECULATIVE_MTP_STOP_OTHER] == 0);

        // positions past 15 fold into 15+
        common_speculative_mtp_stats st3;
        common_speculative_mtp_stats_add(st3, std::vector<float>(20, 0.95f), 20, 20, COMMON_SPECULATIVE_MTP_STOP_CAP);
        assert(st3.pos_tested[15] == 5 && st3.pos_landed[15] == 5 && st3.pos_tested[14] == 1);
        assert(st3.pbin_tested[5] == 20 && st3.pbin_landed[5] == 20);
    }

    printf("test-spec-shadow: OK\n");
    return 0;
}
