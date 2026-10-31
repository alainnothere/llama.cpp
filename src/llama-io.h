#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);

    // [fork, 2026-10-30] K/V type conversion on load (cparams.kv_cache_convert). the flag rides on the io, not on the
    // memory modules: every cache a context restores (kv, iswa base+swa, hybrid attention child, mirrored indexers,
    // range/APPEND segments) reads through this one object, so one switch reaches all of them with no plumbing.
    // the context sets it only for the host and file readers: the ON_DEVICE reader never sees the tensor bytes (it
    // copies device to device, same process, same types), so a mismatch there stays the refusal it was
    bool kv_convert = false;

    // filled by llama_kv_cache::state_read_data when it converted, read back by the context for its one WARN per load
    struct kv_convert_stats {
        uint64_t n_tensors   = 0; // per-layer K or V tensors (per stream) that went through the conversion
        uint64_t n_rows      = 0; // cells converted, summed over those tensors
        uint64_t n_bytes_src = 0; // bytes consumed from the blob (saved type)
        uint64_t n_bytes_dst = 0; // bytes written to the cache (cache type)
        int32_t  type_src    = -1;
        int32_t  type_dst    = -1;
        bool     mixed       = false; // more than one (src, dst) pair in this load (e.g. K f16->q8_0, V q8_0->f16)
    } kv_conv;
};
