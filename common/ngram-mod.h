#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <string>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    // persistence - see save()/load() in ngram-mod.cpp for the file format
    enum load_status {
        LOAD_OK,
        LOAD_MISSING,  // file does not exist or cannot be opened
        LOAD_CORRUPT,  // bad magic/version, truncated or trailing data, invalid entries
        LOAD_MISMATCH, // valid file, but built with a different n or size
    };

    // write via path + ".tmp" and rename, so an interrupted write never destroys the file on disk
    bool save(const std::string & path) const;

    // on any status other than LOAD_OK the table is left untouched
    load_status load(const std::string & path);

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
