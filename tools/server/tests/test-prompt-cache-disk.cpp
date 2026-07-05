// Unit tests for the prompt-cache disk tier (checkpoint spill files).
//
// These exercise the REAL production code in server-task.cpp:
//   - server_prompt_cache::spill_checkpoint()      (writer)
//   - server_prompt_cache::merge_checkpoint_spills() (lazy reader)
//
// Both are host-only (no llama_context, no model, no GPU), so this is a pure
// file-I/O round trip that runs standalone — it does NOT start a server.
//
// Focus: the lazy / disk-backed checkpoint registration. merge() must register
// each spilled checkpoint as metadata + a file reference WITHOUT pulling the KV
// blobs into RAM, recording byte offsets that point at the correct bytes (the
// same offsets load_tgt/load_dft later fault in from).

#include "server-task.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

// header layout written by spill_checkpoint():
//   magic u32 | version u32 | arch u32 | vocab u32 | pos_min i32 | pos_max i32
//   | n_tokens i64 | tgt_sz u64 | <tgt bytes> | dft_sz u64 | <dft bytes>
// => tgt blob starts at byte 40; dft blob starts at byte 48 + sz_tgt.
static constexpr uint64_t HEADER_BEFORE_TGT = 40;

static int g_dir_counter = 0;

static std::string make_temp_dir(const std::string & tag) {
    const auto base = fs::temp_directory_path() /
        ("llama-pcd-" + tag + "-" + std::to_string(g_dir_counter++));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    return base.string();
}

// patterned bytes so an off-by-one offset is caught (not just a length match)
static common_prompt_checkpoint make_ckpt(int32_t pos_min, int32_t pos_max, int64_t n_tokens,
                                          size_t tgt_n, size_t dft_n, uint8_t seed) {
    common_prompt_checkpoint cp;
    cp.pos_min  = pos_min;
    cp.pos_max  = pos_max;
    cp.n_tokens = n_tokens;
    cp.data_tgt.resize(tgt_n);
    for (size_t i = 0; i < tgt_n; ++i) {
        cp.data_tgt[i] = static_cast<uint8_t>(seed + i * 3 + 1);
    }
    cp.data_dft.resize(dft_n);
    for (size_t i = 0; i < dft_n; ++i) {
        cp.data_dft[i] = static_cast<uint8_t>(seed * 2 + i * 5 + 7);
    }
    return cp;
}

static std::vector<uint8_t> read_file_range(const std::string & path, uint64_t off, uint64_t sz) {
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    std::ifstream f(path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(off));
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
    return buf;
}

static size_t count_bin_files(const std::string & dir) {
    size_t n = 0;
    for (const auto & e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".bin") {
            ++n;
        }
    }
    return n;
}

// build a cache-entry state with `n` bytes of host KV in data.main
static server_prompt make_state(size_t n, uint8_t fill) {
    server_prompt e;
    e.data.main.assign(n, fill);
    return e;
}

// the core test: a spilled checkpoint comes back as a LAZY entry (no blobs in
// RAM) whose recorded offsets point at the original bytes on disk.
static void test_lazy_registration_roundtrip() {
    std::printf("test_lazy_registration_roundtrip\n");
    const std::string dir = make_temp_dir("lazy");
    const uint32_t arch = 0xA0A0A0A0u, vocab = 0xB1B1B1B1u;
    server_prompt_cache cache(64, 0, dir, 16, -1, arch, vocab);

    const std::string cid = "conv_lazy";
    const common_prompt_checkpoint cp = make_ckpt(/*pos_min*/ 100, /*pos_max*/ 150, /*n_tokens*/ 151,
                                                  /*tgt_n*/ 2048, /*dft_n*/ 512, /*seed*/ 42);
    cache.spill_checkpoint(cp, cid);
    cache.wait_idle(); // spill is async - drain before reading the file back

    server_prompt prompt;
    prompt.conversation_id = cid;
    cache.merge_checkpoint_spills(prompt);

    CHECK(prompt.checkpoints.size() == 1);
    if (prompt.checkpoints.empty()) {
        fs::remove_all(dir);
        return;
    }
    const auto & got = prompt.checkpoints.front();

    // metadata preserved
    CHECK(got.pos_min == 100);
    CHECK(got.pos_max == 150);
    CHECK(got.n_tokens == 151);

    // LAZY: the KV blobs were NOT read into RAM
    CHECK(got.data_tgt.empty());
    CHECK(got.data_dft.empty());
    CHECK(got.size() == 0); // reports ~0 RAM footprint to cache-ram accounting

    // disk reference recorded
    CHECK(!got.src_path.empty());
    CHECK(fs::exists(got.src_path));
    CHECK(got.sz_tgt == 2048);
    CHECK(got.sz_dft == 512);

    // exact offsets for the fixed on-disk layout
    CHECK(got.off_tgt == HEADER_BEFORE_TGT);                       // 40
    CHECK(got.off_dft == HEADER_BEFORE_TGT + got.sz_tgt + 8);      // 48 + sz_tgt

    // the recorded offsets point at the original bytes (what load_tgt/load_dft read)
    CHECK(read_file_range(got.src_path, got.off_tgt, got.sz_tgt) == cp.data_tgt);
    CHECK(read_file_range(got.src_path, got.off_dft, got.sz_dft) == cp.data_dft);

    fs::remove_all(dir);
}

// a checkpoint written under a different arch/vocab identity must be skipped
static void test_hash_mismatch_rejected() {
    std::printf("test_hash_mismatch_rejected\n");
    const std::string dir = make_temp_dir("mismatch");
    {
        server_prompt_cache writer(64, 0, dir, 16, -1, 0x11111111u, 0x22222222u);
        writer.spill_checkpoint(make_ckpt(10, 20, 21, 256, 64, 1), "conv_hash_mm");
    }
    // reader has a different arch hash
    server_prompt_cache reader(64, 0, dir, 16, -1, 0x99999999u, 0x22222222u);
    server_prompt prompt;
    prompt.conversation_id = "conv_hash_mm";
    reader.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.empty());
    fs::remove_all(dir);
}

// a disk checkpoint whose pos_min already exists in the prompt is not duplicated
static void test_dup_pos_min_skipped() {
    std::printf("test_dup_pos_min_skipped\n");
    const std::string dir = make_temp_dir("dup");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0x5u, 0x6u);
    cache.spill_checkpoint(make_ckpt(200, 250, 251, 128, 0, 9), "conv_dup");
    cache.wait_idle();

    server_prompt prompt;
    prompt.conversation_id = "conv_dup";
    common_prompt_checkpoint existing;
    existing.pos_min  = 200;
    existing.pos_max  = 250;
    existing.n_tokens = 251;
    existing.data_tgt.assign(10, 0xFF); // resident; must be left untouched
    prompt.checkpoints.push_back(existing);

    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.size() == 1);
    CHECK(prompt.checkpoints.front().data_tgt.size() == 10);
    fs::remove_all(dir);
}

// merge only picks up files belonging to the prompt's conversation_id
static void test_conversation_filter() {
    std::printf("test_conversation_filter\n");
    const std::string dir = make_temp_dir("conv");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0x7u, 0x8u);
    cache.spill_checkpoint(make_ckpt(5, 9, 10, 64, 0, 3), "conv_a");
    cache.wait_idle();

    server_prompt other;
    other.conversation_id = "conv_b";
    cache.merge_checkpoint_spills(other);
    CHECK(other.checkpoints.empty());

    server_prompt same;
    same.conversation_id = "conv_a";
    cache.merge_checkpoint_spills(same);
    CHECK(same.checkpoints.size() == 1);
    fs::remove_all(dir);
}

// a corrupt / wrong-magic file is ignored without crashing
static void test_corrupt_file_skipped() {
    std::printf("test_corrupt_file_skipped\n");
    const std::string dir = make_temp_dir("corrupt");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0x1u, 0x2u);
    {
        std::ofstream f(fs::path(dir) / "cp_conv_corrupt_5.bin", std::ios::binary);
        const uint32_t junk[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        f.write(reinterpret_cast<const char *>(junk), sizeof(junk));
    }
    server_prompt prompt;
    prompt.conversation_id = "conv_corrupt";
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.empty());
    fs::remove_all(dir);
}

// merged checkpoints are inserted in ascending pos_min order
static void test_sorted_by_pos_min() {
    std::printf("test_sorted_by_pos_min\n");
    const std::string dir = make_temp_dir("sorted");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0x3u, 0x4u);
    cache.spill_checkpoint(make_ckpt(300, 310, 311, 32, 0, 1), "conv_sorted");
    cache.spill_checkpoint(make_ckpt(100, 110, 111, 32, 0, 2), "conv_sorted");
    cache.spill_checkpoint(make_ckpt(200, 210, 211, 32, 0, 3), "conv_sorted");
    cache.wait_idle();

    server_prompt prompt;
    prompt.conversation_id = "conv_sorted";
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.size() == 3);

    std::vector<int32_t> pos;
    for (const auto & c : prompt.checkpoints) {
        pos.push_back(c.pos_min);
    }
    CHECK((pos == std::vector<int32_t>{100, 200, 300}));
    fs::remove_all(dir);
}

// over_budget() reflects the host-RAM byte budget (the valley gate)
static void test_over_budget_reflects_ram_limit() {
    std::printf("test_over_budget_reflects_ram_limit\n");
    const std::string dir = make_temp_dir("budget");
    server_prompt_cache cache(/*limit MiB*/ 1, 0, dir, 16, -1, 0x1u, 0x2u);

    CHECK(!cache.over_budget()); // empty cache is within budget

    cache.states.push_back(make_state(2 * 1024 * 1024, 0x5A)); // 2 MiB > 1 MiB
    CHECK(cache.over_budget());

    fs::remove_all(dir);
}

// the valley mechanic: spill_all_to_disk() frees host RAM. with write-through
// (flush at prompt_save), anything resident in `states` is not serializable
// (mtmd), so freeing means dropping - nothing is written here.
static void test_spill_all_frees_ram() {
    std::printf("test_spill_all_frees_ram\n");
    const std::string dir = make_temp_dir("valley");
    server_prompt_cache cache(1, 0, dir, 16, -1, 0xC0FFEEu, 0xD00Du);

    cache.states.push_back(make_state(1024 * 1024, 0x11)); // 1 MiB
    cache.states.push_back(make_state(1024 * 1024, 0x22)); // 1 MiB  -> 2 MiB total
    CHECK(cache.states.size() == 2);
    CHECK(cache.over_budget());

    cache.spill_all_to_disk();

    CHECK(cache.states.empty());          // host RAM freed
    CHECK(!cache.over_budget());          // size() back to ~0
    CHECK(count_bin_files(dir) == 0);     // nothing written - these entries were not spillable

    fs::remove_all(dir);
}

// spill_all_to_disk() is a safe no-op when the disk tier is disabled
static void test_spill_all_noop_without_disk() {
    std::printf("test_spill_all_noop_without_disk\n");
    server_prompt_cache cache(1, 0, /*disk_path*/ "", 16, -1, 0x1u, 0x2u);
    cache.states.push_back(make_state(2 * 1024 * 1024, 0x07));
    cache.spill_all_to_disk();           // disk disabled -> must not drop the state or crash
    CHECK(cache.states.size() == 1);
}

int main() {
    test_lazy_registration_roundtrip();
    test_hash_mismatch_rejected();
    test_dup_pos_min_skipped();
    test_conversation_filter();
    test_corrupt_file_skipped();
    test_sorted_by_pos_min();
    test_over_budget_reflects_ram_limit();
    test_spill_all_frees_ram();
    test_spill_all_noop_without_disk();

    if (g_failures == 0) {
        std::printf("All prompt-cache disk tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
}
