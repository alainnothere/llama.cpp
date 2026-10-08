// tests for the persistence of the ngram-mod hash table used by `--spec-ngram-mod-cache`
//
// a save -> load roundtrip reproduces the table exactly, and a file built with another n or size,
// or a truncated file, is rejected without touching the table it was loaded into

#include "ngram-mod.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using entry_t = common_ngram_mod::entry_t;

static constexpr uint16_t N_MATCH = 8;
static constexpr size_t   SIZE    = 64*1024;

static std::string tmp_path(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            ("test-ngram-mod-" + std::string(tag) + "-" + std::to_string(stamp) + ".bin")).string();
}

static std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// a pseudo-random token stream, long enough for a few hundred ngrams
static std::vector<entry_t> make_tokens(size_t n) {
    std::vector<entry_t> res(n);
    uint32_t x = 12345;
    for (auto & t : res) {
        x = x*1664525u + 1013904223u;
        t = (entry_t) ((x >> 8) % 32000);
    }
    return res;
}

static void fill(common_ngram_mod & mod, const std::vector<entry_t> & tokens) {
    for (size_t i = 0; i + mod.get_n() < tokens.size(); ++i) {
        mod.add(tokens.data() + i);
    }
}

static void test_roundtrip() {
    const auto tokens = make_tokens(500);

    common_ngram_mod a(N_MATCH, SIZE);
    fill(a, tokens);
    assert(a.get_used() > 300);

    const std::string path   = tmp_path("a");
    const std::string path_b = tmp_path("b");

    assert(a.save(path));
    assert(!std::filesystem::exists(path + ".tmp"));

    common_ngram_mod b(N_MATCH, SIZE);
    assert(b.load(path) == common_ngram_mod::LOAD_OK);
    assert(b.get_used() == a.get_used());

    // every learned continuation comes back
    for (size_t i = 0; i + N_MATCH < tokens.size(); ++i) {
        assert(b.get(tokens.data() + i) == a.get(tokens.data() + i));
    }

    // and the whole table is byte-identical
    assert(b.save(path_b));
    assert(read_file(path) == read_file(path_b));

    std::filesystem::remove(path);
    std::filesystem::remove(path_b);

    printf("%s: OK (used = %zu/%zu)\n", __func__, a.get_used(), a.size());
}

static void test_mismatch() {
    common_ngram_mod a(N_MATCH, SIZE);
    fill(a, make_tokens(500));

    const std::string path = tmp_path("mismatch");
    assert(a.save(path));

    common_ngram_mod other_n(N_MATCH + 1, SIZE);
    assert(other_n.load(path) == common_ngram_mod::LOAD_MISMATCH);
    assert(other_n.get_used() == 0);

    common_ngram_mod other_size(N_MATCH, SIZE*2);
    assert(other_size.load(path) == common_ngram_mod::LOAD_MISMATCH);
    assert(other_size.get_used() == 0);

    std::filesystem::remove(path);

    printf("%s: OK\n", __func__);
}

static void test_truncated() {
    const auto tokens = make_tokens(500);

    common_ngram_mod a(N_MATCH, SIZE);
    fill(a, tokens);

    const std::string path = tmp_path("trunc");
    assert(a.save(path));

    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 7);

    // a table that already holds data must keep it when the load fails
    common_ngram_mod b(N_MATCH, SIZE);
    fill(b, tokens);
    const size_t used_before = b.get_used();

    assert(b.load(path) == common_ngram_mod::LOAD_CORRUPT);
    assert(b.get_used() == used_before);
    assert(b.get(tokens.data()) == a.get(tokens.data()));

    // header only
    std::filesystem::resize_file(path, 10);
    assert(b.load(path) == common_ngram_mod::LOAD_CORRUPT);

    std::filesystem::remove(path);

    assert(b.load(path) == common_ngram_mod::LOAD_MISSING);
    assert(b.get_used() == used_before);

    printf("%s: OK\n", __func__);
}

int main() {
    test_roundtrip();
    test_mismatch();
    test_truncated();

    printf("all tests passed\n");

    return 0;
}
