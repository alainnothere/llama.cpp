#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    // a tentative slot (replicated copy from an extending load, see load()) is not counted in used, a real write claims it
    if (entries[i] < 0) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    // a tentative slot still answers with its token - that is what keeps an extended table's lookups exact
    const entry_t v = entries[i];

    return v <= TENTATIVE_MAX ? untentative(v) : v;
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}

//
// persistence
//
// file format (native endianness):
//   char[4]  magic   = "NGMD"
//   uint32   version = 1
//   uint32   n
//   uint64   size
//   uint64   used    (informational - recounted from the entries on load)
//   int32    entries[size]  token id >= 0, EMPTY = -1, or a tentative copy -(token + 2) <= -2 (see load())
//
// tentative values only appear in files written by builds with the extending load (2026-10-22, fork-only);
// older builds reject such files as LOAD_CORRUPT. one-way, accepted.
//

static constexpr char     NGRAM_MOD_MAGIC[4]   = { 'N', 'G', 'M', 'D' };
static constexpr uint32_t NGRAM_MOD_VERSION    = 1;
static constexpr size_t   NGRAM_MOD_HEADER_LEN = 4 + 4 + 4 + 8 + 8;

bool common_ngram_mod::save(const std::string & path) const {
    const std::string path_tmp = path + ".tmp";

    {
        std::ofstream f(path_tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }

        const uint32_t version = NGRAM_MOD_VERSION;
        const uint32_t n_u32   = (uint32_t) n;
        const uint64_t size_64 = (uint64_t) entries.size();
        const uint64_t used_64 = (uint64_t) used;

        f.write(NGRAM_MOD_MAGIC,              sizeof(NGRAM_MOD_MAGIC));
        f.write((const char *) &version,      sizeof(version));
        f.write((const char *) &n_u32,        sizeof(n_u32));
        f.write((const char *) &size_64,      sizeof(size_64));
        f.write((const char *) &used_64,      sizeof(used_64));
        f.write((const char *) entries.data(), size_bytes());

        f.close();
        if (!f) {
            std::remove(path_tmp.c_str());
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(path_tmp, path, ec);
    if (ec) {
        std::remove(path_tmp.c_str());
        return false;
    }

    return true;
}

common_ngram_mod::load_status common_ngram_mod::load(const std::string & path, uint64_t * size_file) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return LOAD_MISSING;
    }

    char     magic[4];
    uint32_t version = 0;
    uint32_t n_file  = 0;
    uint64_t size_64 = 0;
    uint64_t used_64 = 0;

    f.read(magic,                sizeof(magic));
    f.read((char *) &version,    sizeof(version));
    f.read((char *) &n_file,     sizeof(n_file));
    f.read((char *) &size_64,    sizeof(size_64));
    f.read((char *) &used_64,    sizeof(used_64));
    if (!f || memcmp(magic, NGRAM_MOD_MAGIC, sizeof(magic)) != 0 || version != NGRAM_MOD_VERSION) {
        return LOAD_CORRUPT;
    }

    if (size_file) {
        *size_file = size_64;
    }

    // a file of size S loads into a table of size S' = k*S by replication: a key that lived at slot i = h % S
    // has h % S' == i (mod S), i.e. it lands on one of i, i+S, ..., i+(k-1)S, so copying slot i to all k of them
    // keeps every real lookup exact. the other k-1 copies are phantoms - no worse than the collision hits the table
    // already serves (drafts are verified by the target), and real adds overwrite them over time.
    // motivating case 2026-10-22: a 3554432-slot, 67% full (14 MB) table must survive raising --spec-ngram-mod-size.
    // a configured size that is larger but not a multiple is rounded UP to the next multiple (the table is resized
    // here, the caller reads size() back). a smaller one has no mapping (the k-1 bands cannot be folded back
    // without the keys, the file stores none) -> mismatch, as does a different n
    const size_t size_req = entries.size();
    if (n_file != n || size_64 == 0 || size_64 > size_req) {
        return LOAD_MISMATCH;
    }
    const size_t S = (size_t) size_64;
    const size_t k = (size_req + S - 1) / S;

    // the payload must be exactly size entries - catches truncation and trailing garbage
    std::error_code ec;
    const auto file_len = std::filesystem::file_size(path, ec);
    if (ec || file_len != NGRAM_MOD_HEADER_LEN + S*sizeof(entry_t)) {
        return LOAD_CORRUPT;
    }

    std::vector<entry_t> tmp(k*S);
    f.read((char *) tmp.data(), S * sizeof(tmp[0]));
    if (!f) {
        return LOAD_CORRUPT;
    }

    // do not trust the header's used count - recount. every int32 is now a valid slot value (>= 0 token, -1 EMPTY,
    // <= -2 tentative), tentative slots are not used
    size_t used_new = 0;
    for (size_t i = 0; i < S; ++i) {
        used_new += tmp[i] >= 0;
    }

    // band 0 keeps the file's values as they are, bands 1..k-1 get tentative copies, so used stays the file's count
    // and occupancy drops by k (0.67 -> 0.067 at 10x) instead of reading phantoms as real entries.
    // which band is "real" is arbitrary: lookups decode tentative slots, it only matters for the count. the count
    // over-counts by one when a key whose true slot is in band j > 0 is re-added (band 0 still holds its plain copy)
    for (size_t j = 1; j < k; ++j) {
        for (size_t i = 0; i < S; ++i) {
            const entry_t v = tmp[i];
            tmp[j*S + i] = v >= 0 ? tentative(v) : v;
        }
    }

    entries.swap(tmp);
    used = used_new;

    return k > 1 ? LOAD_EXTENDED : LOAD_OK;
}
