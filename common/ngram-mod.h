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

    // replicated copy written by an extending load(): -(token + 2), always <= TENTATIVE_MAX. get() decodes it,
    // add() treats it as free for the used count
    static constexpr entry_t TENTATIVE_MAX = -2;
    static constexpr entry_t tentative  (entry_t token) { return -(token + 2); }
    static constexpr entry_t untentative(entry_t v)     { return -(v + 2); }

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
        LOAD_MISMATCH, // valid file, but built with a different n, or a larger size than the configured one
        LOAD_EXTENDED, // valid file of size S loaded into a table of k*S slots (k = ceil(configured / S) >= 2, the table
                       // is resized to that) by tentative copies of every slot, see load()
    };

    // write via path + ".tmp" and rename, so an interrupted write never destroys the file on disk
    bool save(const std::string & path) const;

    // on any status other than LOAD_OK/LOAD_EXTENDED the table is left untouched; LOAD_EXTENDED may change size()
    // size_file (optional) receives the file's table size once the header parsed, so a LOAD_MISMATCH can name it
    load_status load(const std::string & path, uint64_t * size_file = nullptr);

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
