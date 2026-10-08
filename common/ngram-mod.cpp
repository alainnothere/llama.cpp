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

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
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
//   int32    entries[size]
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

common_ngram_mod::load_status common_ngram_mod::load(const std::string & path) {
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

    if (n_file != n || size_64 != entries.size()) {
        return LOAD_MISMATCH;
    }

    // the payload must be exactly size entries - catches truncation and trailing garbage
    std::error_code ec;
    const auto file_len = std::filesystem::file_size(path, ec);
    if (ec || file_len != NGRAM_MOD_HEADER_LEN + size_bytes()) {
        return LOAD_CORRUPT;
    }

    std::vector<entry_t> tmp(entries.size());
    f.read((char *) tmp.data(), tmp.size() * sizeof(tmp[0]));
    if (!f) {
        return LOAD_CORRUPT;
    }

    // do not trust the header's used count - recount, and reject anything that cannot be a token id
    size_t used_new = 0;
    for (const entry_t e : tmp) {
        if (e < EMPTY) {
            return LOAD_CORRUPT;
        }
        used_new += e != EMPTY;
    }

    entries.swap(tmp);
    used = used_new;

    return LOAD_OK;
}
