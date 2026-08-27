// tests for the n-gram lookup cache used by `--spec-type ngram-cache`
//
// the properties under test are the ones the persistence of the dynamic cache (-lcd) depends on:
// updating from a token stream, drafting the continuation of a repeated pattern, a save -> load
// roundtrip that preserves the drafts, and merge semantics (counts accumulate, patterns are gained)

#include "llama.h"
#include "ngram-cache.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static common_ngram_cache make_cache(const std::vector<llama_token> & tokens) {
    common_ngram_cache cache;
    std::vector<llama_token> inp = tokens;
    common_ngram_cache_update(cache, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size(), false);
    return cache;
}

static std::vector<llama_token> repeat(const std::vector<llama_token> & pattern, int n) {
    std::vector<llama_token> res;
    for (int i = 0; i < n; ++i) {
        res.insert(res.end(), pattern.begin(), pattern.end());
    }
    return res;
}

// draft a continuation the way common_speculative does it: `inp` holds everything generated so far and
// the draft is seeded with the last token of `inp`, which is stripped from the result again
static std::vector<llama_token> draft_from(
        const std::vector<llama_token> & tokens,
        int n_draft,
        common_ngram_cache & nc_context,
        common_ngram_cache & nc_dynamic,
        common_ngram_cache & nc_static) {
    std::vector<llama_token> inp = tokens;
    std::vector<llama_token> draft = { inp.back() };

    common_ngram_cache_draft(inp, draft, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, nc_context, nc_dynamic, nc_static);

    draft.erase(draft.begin());

    return draft;
}

static int32_t count_of(common_ngram_cache & cache, const std::vector<llama_token> & ngram_tokens, llama_token next) {
    const common_ngram ngram(ngram_tokens.data(), (int) ngram_tokens.size());

    auto part_it = cache.find(ngram);
    if (part_it == cache.end()) {
        return 0;
    }

    auto count_it = part_it->second.find(next);
    if (count_it == part_it->second.end()) {
        return 0;
    }

    return count_it->second;
}

static bool caches_equal(const common_ngram_cache & a, const common_ngram_cache & b) {
    if (a.size() != b.size()) {
        return false;
    }

    for (const auto & entry : a) {
        auto it = b.find(entry.first);
        if (it == b.end() || it->second.size() != entry.second.size()) {
            return false;
        }
        for (const auto & count : entry.second) {
            auto count_it = it->second.find(count.first);
            if (count_it == it->second.end() || count_it->second != count.second) {
                return false;
            }
        }
    }

    return true;
}

static void print_tokens(const char * label, const std::vector<llama_token> & tokens) {
    printf("%s: [", label);
    for (size_t i = 0; i < tokens.size(); ++i) {
        printf("%s%d", i == 0 ? "" : ", ", tokens[i]);
    }
    printf("]\n");
}

// updating from a token stream records the empirical continuation counts
static void test_update() {
    const std::vector<llama_token> pattern = { 10, 11, 12, 13, 14 };
    const std::vector<llama_token> tokens  = repeat(pattern, 8);

    common_ngram_cache cache = make_cache(tokens);

    // 12 follows the 4-gram (13, 14, 10, 11) in every repetition but the first, which has no predecessor
    assert(count_of(cache, { 13, 14, 10, 11 }, 12) == 7);

    // the unigram 11 is always followed by 12
    assert(count_of(cache, { 11 }, 12) == 8);
    assert(count_of(cache, { 11 }, 13) == 0);

    // an n-gram that never occurred has no entry at all
    assert(count_of(cache, { 99, 98, 97, 96 }, 95) == 0);

    printf("%s: OK\n", __func__);
}

// a repeated pattern is drafted forward from the context cache
static void test_draft_repeated_pattern() {
    const std::vector<llama_token> pattern = { 10, 11, 12, 13, 14 };
    const std::vector<llama_token> tokens  = repeat(pattern, 8);

    common_ngram_cache context = make_cache(tokens);
    common_ngram_cache empty_dynamic;
    common_ngram_cache empty_static;

    const std::vector<llama_token> draft = draft_from(tokens, 7, context, empty_dynamic, empty_static);

    print_tokens("draft", draft);

    // the stream ends on 14, so the continuation is the pattern from the top, twice over
    const std::vector<llama_token> expected = { 10, 11, 12, 13, 14, 10, 11 };
    assert(draft == expected);

    // an empty cache drafts nothing
    common_ngram_cache empty_context;
    assert(draft_from(tokens, 7, empty_context, empty_dynamic, empty_static).empty());

    printf("%s: OK\n", __func__);
}

// save -> load is lossless: the loaded copy holds the same counts and produces the same draft
static void test_save_load_roundtrip() {
    const std::vector<llama_token> pattern = { 20, 21, 22, 23 };
    const std::vector<llama_token> tokens  = repeat(pattern, 10);

    common_ngram_cache saved = make_cache(tokens);

    const std::string path = (std::filesystem::temp_directory_path() /
            ("test-ngram-cache-roundtrip-" +
             std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin")).string();

    common_ngram_cache_save(saved, path);
    common_ngram_cache loaded = common_ngram_cache_load(path);
    std::filesystem::remove(path);

    assert(caches_equal(saved, loaded));

    // the loaded cache is used as the *dynamic* one, which is what -lcd feeds - the stricter dynamic
    // thresholds still accept a pattern this regular
    common_ngram_cache empty_context;
    common_ngram_cache empty_static;
    common_ngram_cache empty_dynamic;

    const std::vector<llama_token> draft_saved  = draft_from(tokens, 6, empty_context, saved,  empty_static);
    const std::vector<llama_token> draft_loaded = draft_from(tokens, 6, empty_context, loaded, empty_static);

    print_tokens("draft (saved) ", draft_saved);
    print_tokens("draft (loaded)", draft_loaded);

    const std::vector<llama_token> expected = { 20, 21, 22, 23, 20, 21 };
    assert(draft_saved == expected);
    assert(draft_loaded == draft_saved);

    // loading a file that does not exist throws rather than returning an empty cache
    bool threw = false;
    try {
        common_ngram_cache_load(path + ".does-not-exist");
    } catch (...) {
        threw = true;
    }
    assert(threw);

    printf("%s: OK\n", __func__);
}

// merging accumulates counts and hands the target the patterns it did not have
static void test_merge() {
    const std::vector<llama_token> pattern_a = { 30, 31, 32, 33 };
    const std::vector<llama_token> pattern_b = { 40, 41, 42, 43 };

    const std::vector<llama_token> tokens_a = repeat(pattern_a, 10);
    const std::vector<llama_token> tokens_b = repeat(pattern_b, 10);

    // counts accumulate when the same information is merged twice
    common_ngram_cache cache_a = make_cache(tokens_a);
    common_ngram_cache doubled = make_cache(tokens_a);
    const int32_t count_before = count_of(cache_a, { 30, 31 }, 32);
    assert(count_before == 10);

    common_ngram_cache_merge(doubled, cache_a);
    assert(doubled.size() == cache_a.size()); // no new n-grams, only bigger counts
    assert(count_of(doubled, { 30, 31 }, 32) == 2*count_before);

    // a pattern present only in A is drafted after merging A into B
    common_ngram_cache cache_b = make_cache(tokens_b);
    common_ngram_cache empty_context;
    common_ngram_cache empty_static;

    assert(draft_from(tokens_a, 4, empty_context, cache_b, empty_static).empty());

    common_ngram_cache_merge(cache_b, cache_a);

    const std::vector<llama_token> draft_merged = draft_from(tokens_a, 4, empty_context, cache_b, empty_static);
    print_tokens("draft (merged)", draft_merged);
    assert((draft_merged == std::vector<llama_token>{ 30, 31, 32, 33 }));

    // B kept what it already knew
    const std::vector<llama_token> draft_b = draft_from(tokens_b, 4, empty_context, cache_b, empty_static);
    print_tokens("draft (b)     ", draft_b);
    assert((draft_b == std::vector<llama_token>{ 40, 41, 42, 43 }));

    // merging into an empty cache is a copy
    common_ngram_cache into_empty;
    common_ngram_cache_merge(into_empty, cache_a);
    assert(caches_equal(into_empty, cache_a));

    printf("%s: OK\n", __func__);
}

int main() {
    test_update();
    test_draft_repeated_pattern();
    test_save_load_roundtrip();
    test_merge();

    printf("all tests passed\n");

    return 0;
}
