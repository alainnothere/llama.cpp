// tests for the stochastic (rejection sampling) verification of speculative drafts
//
// the property under test is the one that makes the scheme usable at all: whatever the proposal
// distribution p_dft is, the token returned by common_spec_verify_token() is distributed exactly as
// the target distribution p_tgt

#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

static std::vector<llama_token_data> make_dist(const std::vector<std::pair<llama_token, float>> & tp) {
    std::vector<llama_token_data> res;
    for (const auto & t : tp) {
        res.push_back({ t.first, logf(t.second), t.second });
    }
    return res;
}

static llama_token_data_array make_array(std::vector<llama_token_data> & data) {
    // `selected` mimics what the sampler chain leaves behind
    int32_t selected = 0;
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].p > data[selected].p) {
            selected = (int32_t) i;
        }
    }
    return { data.data(), data.size(), selected, false };
}

static float dist_p(const std::vector<llama_token_data> & d, llama_token id) {
    for (const auto & c : d) {
        if (c.id == id) {
            return c.p;
        }
    }
    return 0.0f;
}

// (a) the drafted token is accepted iff r_accept <= p_tgt/p_dft
static void test_accept_threshold() {
    const auto tgt_ref = make_dist({ {0, 0.5f}, {1, 0.3f}, {2, 0.2f} });
    const auto dft     = make_dist({ {0, 0.8f}, {1, 0.2f} });

    // p_tgt(0)/p_dft(0) = 0.5/0.8 = 0.625
    const float ratio = 0.625f;

    for (float r : { 0.0f, 0.1f, 0.5f, 0.6f, 0.62f, 0.63f, 0.7f, 0.9f, 0.999f }) {
        auto tgt   = tgt_ref;
        auto arr   = make_array(tgt);
        bool acc   = false;
        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), 0, r, 0.5f, &acc);

        assert(acc == (r <= ratio));
        if (acc) {
            assert(id == 0);
        }
    }

    // p_tgt >= p_dft  =>  always accepted
    for (float r : { 0.0f, 0.5f, 0.999999f }) {
        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = false;
        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), 1, r, 0.5f, &acc);
        assert(acc);
        assert(id == 1);
    }

    // a token the target's sampler chain removed (p_tgt = 0) is never accepted
    {
        const auto dft_oov = make_dist({ {7, 1.0f} });
        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = true;
        const llama_token id = common_spec_verify_token(&arr, dft_oov.data(), dft_oov.size(), 7, 0.0f, 0.5f, &acc);
        assert(!acc);
        assert(id != 7);
    }

    printf("%s: OK\n", __func__);
}

// (b) on rejection, the replacement comes from the normalized residual max(0, p_tgt - p_dft)
static void test_residual_resample() {
    const auto tgt_ref = make_dist({ {0, 0.5f}, {1, 0.3f}, {2, 0.2f} });
    const auto dft     = make_dist({ {0, 0.8f}, {1, 0.2f} });

    // residual: max(0, 0.5-0.8) = 0, max(0, 0.3-0.2) = 0.1, max(0, 0.2-0.0) = 0.2 -> sum 0.3
    std::map<llama_token, double> expected = { {0, 0.0}, {1, 0.1/0.3}, {2, 0.2/0.3} };

    const int n = 100000;

    std::map<llama_token, int> hist;

    for (int i = 0; i < n; ++i) {
        const float r2 = (i + 0.5f)/n; // sweep the resampling uniform over [0,1)

        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = true;

        // r_accept = 1.0 forces a rejection (p_tgt(0) < p_dft(0))
        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), 0, 1.0f, r2, &acc);

        assert(!acc);
        hist[id]++;

        // the residual must be left in place, unnormalized, and never negative
        double sum = 0.0;
        for (const auto & c : tgt) {
            assert(c.p >= 0.0f);
            assert(std::fabs(c.p - std::max(0.0f, dist_p(tgt_ref, c.id) - dist_p(dft, c.id))) < 1e-6f);
            sum += c.p;
        }
        assert(std::fabs(sum - 0.3) < 1e-5);
    }

    // token 0 has zero residual mass -> it can never be returned
    assert(hist[0] == 0);

    for (const auto & e : expected) {
        const double f = (double) hist[e.first] / n;
        printf("%s: token %d: %.4f (expected %.4f)\n", __func__, e.first, f, e.second);
        assert(std::fabs(f - e.second) < 1e-3);
    }

    printf("%s: OK\n", __func__);
}

// (c) a point-mass proposal (argmax drafting, ngram-*) reduces to "accept with probability p_tgt(d)"
//     and, on rejection, to sampling the target with the drafted token removed - which is exactly what
//     the greedy sample-and-compare path does
static void test_point_mass_draft() {
    const auto tgt_ref = make_dist({ {0, 0.5f}, {1, 0.3f}, {2, 0.2f} });
    const auto dft     = make_dist({ {0, 1.0f} });

    const int n = 100000;

    int n_acc = 0;

    // acceptance probability: sweep r_accept
    for (int i = 0; i < n; ++i) {
        const float r1 = (i + 0.5f)/n;

        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = false;

        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), 0, r1, 0.5f, &acc);

        if (acc) {
            n_acc++;
            assert(id == 0);
        } else {
            assert(id != 0); // residual of the drafted token is max(0, 0.5 - 1.0) = 0
        }
    }

    // rejection distribution: force a rejection and sweep r_resample
    std::map<llama_token, int> hist_rej;

    for (int i = 0; i < n; ++i) {
        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = true;

        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), 0, 0.9f, (i + 0.5f)/n, &acc);

        assert(!acc);
        assert(id != 0);
        hist_rej[id]++;
    }

    const double p_acc = (double) n_acc / n;
    printf("%s: accept rate %.4f (expected %.4f)\n", __func__, p_acc, 0.5);
    assert(std::fabs(p_acc - 0.5) < 1e-3);

    // rejections are distributed as p_tgt conditioned on != 0: 0.3/0.5 and 0.2/0.5
    const double f1 = (double) hist_rej[1] / n;
    const double f2 = (double) hist_rej[2] / n;
    printf("%s: rejection split %.4f / %.4f (expected %.4f / %.4f)\n", __func__, f1, f2, 0.6, 0.4);
    assert(std::fabs(f1 - 0.6) < 1e-3);
    assert(std::fabs(f2 - 0.4) < 1e-3);

    printf("%s: OK\n", __func__);
}

// the whole point: the returned token is distributed as p_tgt, for any proposal
static void test_distribution_preserved() {
    const auto tgt_ref = make_dist({ {0, 0.40f}, {1, 0.25f}, {2, 0.20f}, {3, 0.10f}, {4, 0.05f} });

    // a truncated + renormalized proposal, as produced by a top-k draft sampler; note that it puts mass
    // on token 5, which the target's chain removed entirely
    const auto dft = make_dist({ {0, 0.30f}, {2, 0.35f}, {3, 0.20f}, {5, 0.15f} });

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    const int n = 400000;

    std::map<llama_token, int> hist;

    int n_acc = 0;

    for (int i = 0; i < n; ++i) {
        // draw the drafted token from the proposal, as the draft model would
        const float rd = u(rng);

        llama_token id_dft = dft.back().id;
        {
            float sum = 0.0f;
            for (const auto & c : dft) {
                sum += c.p;
                if (rd < sum) {
                    id_dft = c.id;
                    break;
                }
            }
        }

        auto tgt = tgt_ref;
        auto arr = make_array(tgt);
        bool acc = false;

        const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), id_dft, u(rng), u(rng), &acc);

        n_acc += acc;
        hist[id]++;
    }

    // acceptance rate must be sum_x min(p_tgt, p_dft) = 0.30 + 0.20 + 0.10 = 0.60
    const double p_acc = (double) n_acc / n;
    printf("%s: accept rate %.4f (expected %.4f)\n", __func__, p_acc, 0.60);
    assert(std::fabs(p_acc - 0.60) < 0.005);

    for (const auto & c : tgt_ref) {
        const double f = (double) hist[c.id] / n;
        printf("%s: token %d: %.4f (expected %.4f)\n", __func__, c.id, f, (double) c.p);
        assert(std::fabs(f - c.p) < 0.005);
    }

    assert(hist[5] == 0); // never returns a token outside the target's support

    printf("%s: OK\n", __func__);
}

// (d) the shape of the returned continuation: accepted prefix + exactly one final token
//     this mirrors the loop of common_sampler_sample_and_accept_n_stochastic() (which needs a live
//     model, so it is exercised end-to-end by the server tests instead)
static void test_result_shape() {
    const auto tgt_ref = make_dist({ {0, 0.5f}, {1, 0.3f}, {2, 0.2f} });

    const std::vector<llama_token> draft = { 0, 0, 0 };

    for (int n_expected = 0; n_expected <= (int) draft.size(); ++n_expected) {
        std::vector<llama_token> result;

        size_t i = 0;
        for (; i < draft.size(); i++) {
            // p_dft is chosen so that the token is accepted for the first `n_expected` positions
            const auto dft = make_dist({ {0, (int) i < n_expected ? 0.4f : 1.0f} });

            auto tgt = tgt_ref;
            auto arr = make_array(tgt);
            bool acc = false;

            const llama_token id = common_spec_verify_token(&arr, dft.data(), dft.size(), draft[i], 0.99f, 0.5f, &acc);

            result.push_back(id);

            if (!acc) {
                break;
            }
        }

        if (i == draft.size()) {
            result.push_back(2); // stands in for the extra token sampled from the target
        }

        assert((int) result.size() == n_expected + 1);
        for (int k = 0; k < n_expected; ++k) {
            assert(result[k] == draft[k]);
        }
    }

    printf("%s: OK\n", __func__);
}

int main() {
    test_accept_threshold();
    test_residual_resample();
    test_point_mass_draft();
    test_distribution_preserved();
    test_result_shape();

    printf("%s: all tests passed\n", __func__);

    return 0;
}
