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

#include <algorithm>
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
//   | n_tokens i64 | token_prefix_hash u64 | tgt_sz u64 | <tgt bytes> | dft_sz u64 | <dft bytes>
// => tgt blob starts at byte 48; dft blob starts at byte 56 + sz_tgt.
static constexpr uint64_t HEADER_BEFORE_TGT = 48;

// mirrors of the file-local constants in server-task.cpp
static constexpr uint32_t CP_MAGIC   = 0x43504B44; // 'CPKD'
static constexpr uint32_t CP_VERSION = 2;

static int g_dir_counter = 0;

// the token timeline every test checkpoint is bound to. all prefixes of it match, so a
// prompt built with make_token_ids(n) accepts any checkpoint with n_tokens <= n.
// TIMELINE_TOKENS is the upper bound on the n_tokens a test checkpoint may use.
static constexpr size_t TIMELINE_TOKENS = 16384;

static llama_tokens make_token_ids(size_t n) {
    llama_tokens ids(n);
    for (size_t i = 0; i < n; ++i) {
        ids[i] = static_cast<llama_token>(i * 7 + 1);
    }
    return ids;
}

static const server_tokens & timeline() {
    static const server_tokens t(make_token_ids(TIMELINE_TOKENS), false);
    return t;
}

static void bind_prompt(server_prompt & prompt, size_t n_tokens) {
    prompt.tokens = server_tokens(make_token_ids(n_tokens), false);
}

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
    cp.pos_min           = pos_min;
    cp.pos_max           = pos_max;
    cp.n_tokens          = n_tokens;
    cp.token_prefix_hash = hash_token_prefix(timeline(), static_cast<size_t>(n_tokens));
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
static server_prompt_cache_state make_state(size_t n, uint8_t fill) {
    server_prompt_cache_state e;
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
    bind_prompt(prompt, 512);
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
    CHECK(got.off_tgt == HEADER_BEFORE_TGT);                       // 48
    CHECK(got.off_dft == HEADER_BEFORE_TGT + got.sz_tgt + 8);      // 56 + sz_tgt

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
    bind_prompt(prompt, 512);
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
    bind_prompt(prompt, 512);
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
    bind_prompt(other, 512);
    cache.merge_checkpoint_spills(other);
    CHECK(other.checkpoints.empty());

    server_prompt same;
    same.conversation_id = "conv_a";
    bind_prompt(same, 512);
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
    bind_prompt(prompt, 512);
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.size() == 3);

    std::vector<int32_t> pos;
    for (const auto & c : prompt.checkpoints) {
        pos.push_back(c.pos_min);
    }
    CHECK((pos == std::vector<int32_t>{100, 200, 300}));
    fs::remove_all(dir);
}

// collect the pos_min values encoded in the cp_ filenames of one conversation
static std::vector<int32_t> cp_pos_min_on_disk(const std::string & dir, const std::string & cid) {
    const std::string prefix = "cp_" + cid + "_";
    std::vector<int32_t> pos;
    for (const auto & e : fs::directory_iterator(dir)) {
        const std::string fn = e.path().filename().string();
        if (e.path().extension() != ".bin") {
            continue;
        }
        if (fn.size() <= prefix.size() || fn.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        pos.push_back(static_cast<int32_t>(std::stoi(fn.substr(prefix.size()))));
    }
    std::sort(pos.begin(), pos.end());
    return pos;
}

// the per-conversation spill limit thins by gap: the file closest to its predecessor is the
// victim, and the newest n_protect files are never touched. that keeps deep-rewind coverage
// instead of collapsing the retained set onto the tip of the conversation.
// must not touch other conversations. regression guard: the prune parses pos_min out of the
// filename, where it is the last field and is terminated by ".bin", not by another underscore.
static void test_spill_prune_thins_by_gap() {
    std::printf("test_spill_prune_thins_by_gap\n");
    const std::string dir = make_temp_dir("prune");
    // limit 10 -> n_protect is the full 8
    server_prompt_cache cache(64, 0, dir, 16, /*checkpoint_spill_max*/ 10, 0xAAu, 0xBBu);

    // 5 sparse checkpoints deep in the conversation, then a dense cluster at the tip
    const std::vector<int32_t> written = {
        100, 1000, 2000, 3000, 4000,
        5000, 5010, 5020, 5030, 5040, 5050, 5060, 5070, 5080, 5090,
    };
    for (size_t i = 0; i < written.size(); ++i) {
        const int32_t p = written[i];
        cache.spill_checkpoint(make_ckpt(p, p + 5, p + 6, 64, 0, static_cast<uint8_t>(i + 1)), "conv_prune_a");
        cache.wait_idle(); // prune runs inside the writer, one file at a time
    }

    // conversation B: below the limit, must be left alone
    cache.spill_checkpoint(make_ckpt(10, 20, 21, 64, 0, 60), "conv_prune_b");
    cache.wait_idle();
    cache.spill_checkpoint(make_ckpt(20, 30, 31, 64, 0, 61), "conv_prune_b");
    cache.wait_idle();

    const std::vector<int32_t> pos_a = cp_pos_min_on_disk(dir, "conv_prune_a");
    const std::vector<int32_t> pos_b = cp_pos_min_on_disk(dir, "conv_prune_b");

    CHECK(pos_a.size() == 10);
    CHECK((pos_b == std::vector<int32_t>{10, 20}));

    // the 8 newest files are protected
    for (int32_t p : {5020, 5030, 5040, 5050, 5060, 5070, 5080, 5090}) {
        CHECK(std::find(pos_a.begin(), pos_a.end(), p) != pos_a.end());
    }

    // deep coverage survives: at least one file below the middle of the written range.
    // the old keep-the-highest-pos_min policy would have retained nothing below 4000.
    if (!pos_a.empty()) {
        CHECK(pos_a.front() < (written.front() + written.back()) / 2);
    }

    // exact victim order: 100 (gap to 0), then 1000, 3000, 2000 and 5010 as the set thins
    CHECK((pos_a == std::vector<int32_t>{2000, 4000, 5020, 5030, 5040, 5050, 5060, 5070, 5080, 5090}));

    // and the survivors are still loadable
    server_prompt prompt;
    prompt.conversation_id = "conv_prune_a";
    bind_prompt(prompt, 8192);
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.size() == 10);

    fs::remove_all(dir);
}

// with a limit at or below 8 the protected tail is clamped to limit/2 (min 1), so thinning
// still has something to work with
static void test_spill_prune_small_limit_clamp() {
    std::printf("test_spill_prune_small_limit_clamp\n");
    const std::string dir = make_temp_dir("prunesmall");
    server_prompt_cache cache(64, 0, dir, 16, /*checkpoint_spill_max*/ 3, 0xACu, 0xBCu);

    for (int32_t p : {100, 200, 300, 400, 500}) {
        cache.spill_checkpoint(make_ckpt(p, p + 5, p + 6, 64, 0, 1), "conv_small");
        cache.wait_idle();
    }

    const std::vector<int32_t> pos = cp_pos_min_on_disk(dir, "conv_small");
    CHECK(pos.size() == 3);
    CHECK((pos == std::vector<int32_t>{200, 400, 500}));

    fs::remove_all(dir);
}

// a cp_ file is bound to the token prefix it was taken at: a file left by an abandoned
// branch of the same conversation must not be registered
static void test_token_binding_rejects_diverged_prefix() {
    std::printf("test_token_binding_rejects_diverged_prefix\n");
    const std::string dir = make_temp_dir("tokbind");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0x31u, 0x32u);

    const std::string cid = "conv_tokbind";
    cache.spill_checkpoint(make_ckpt(400, 450, /*n_tokens*/ 451, 128, 0, 21), cid);
    cache.wait_idle();

    // divergence below n_tokens -> rejected
    {
        server_prompt diverged;
        diverged.conversation_id = cid;
        llama_tokens ids = make_token_ids(1024);
        ids[300] += 1;
        diverged.tokens = server_tokens(ids, false);
        cache.merge_checkpoint_spills(diverged);
        CHECK(diverged.checkpoints.empty());
    }

    // divergence above n_tokens -> accepted, the checkpoint only covers the prefix
    {
        server_prompt later;
        later.conversation_id = cid;
        llama_tokens ids = make_token_ids(1024);
        ids[600] += 1;
        later.tokens = server_tokens(ids, false);
        cache.merge_checkpoint_spills(later);
        CHECK(later.checkpoints.size() == 1);
    }

    // matching prefix -> accepted
    {
        server_prompt match;
        match.conversation_id = cid;
        bind_prompt(match, 1024);
        cache.merge_checkpoint_spills(match);
        CHECK(match.checkpoints.size() == 1);
    }

    // prompt shorter than the prefix the checkpoint covers -> rejected
    {
        server_prompt shorter;
        shorter.conversation_id = cid;
        bind_prompt(shorter, 100);
        cache.merge_checkpoint_spills(shorter);
        CHECK(shorter.checkpoints.empty());
    }

    fs::remove_all(dir);
}

// files written before the token-binding header bump have no hash to check, so they are
// skipped on the version check alone
static void test_old_version_file_skipped() {
    std::printf("test_old_version_file_skipped\n");
    const std::string dir = make_temp_dir("oldver");
    const uint32_t arch = 0x41u, vocab = 0x42u;
    server_prompt_cache cache(64, 0, dir, 16, -1, arch, vocab);

    // v1 layout: same as v2 but without the token_prefix_hash field
    {
        std::ofstream f(fs::path(dir) / "cp_conv_oldver_400.bin", std::ios::binary);
        auto put_u32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        auto put_u64 = [&f](uint64_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        put_u32(CP_MAGIC);
        put_u32(CP_VERSION - 1);
        put_u32(arch);
        put_u32(vocab);
        const int32_t pos_min = 400, pos_max = 450;
        const int64_t n_tokens = 451;
        f.write(reinterpret_cast<const char *>(&pos_min),  sizeof(pos_min));
        f.write(reinterpret_cast<const char *>(&pos_max),  sizeof(pos_max));
        f.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
        put_u64(64);
        const std::vector<uint8_t> tgt(64, 0x5C);
        f.write(reinterpret_cast<const char *>(tgt.data()), static_cast<std::streamsize>(tgt.size()));
        put_u64(0);
    }

    server_prompt prompt;
    prompt.conversation_id = "conv_oldver";
    bind_prompt(prompt, 1024);
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.empty());
    CHECK(fs::exists(fs::path(dir) / "cp_conv_oldver_400.bin")); // skipped, not consumed

    fs::remove_all(dir);
}

// a checkpoint with an empty draft blob (the normal case for a range-truncatable draft)
// must round-trip: written, lazified and merged with sz_dft == 0 and consistent offsets.
static void test_empty_draft_checkpoint() {
    std::printf("test_empty_draft_checkpoint\n");
    const std::string dir = make_temp_dir("emptydft");
    server_prompt_cache cache(64, 0, dir, 16, -1, 0xE1u, 0xE2u);

    const std::string cid = "conv_empty_dft";
    const common_prompt_checkpoint cp = make_ckpt(/*pos_min*/ 700, /*pos_max*/ 750, /*n_tokens*/ 751,
                                                  /*tgt_n*/ 1024, /*dft_n*/ 0, /*seed*/ 11);
    CHECK(cp.data_dft.empty());

    cache.spill_checkpoint(cp, cid);
    cache.wait_idle();

    const std::string path = (fs::path(dir) / ("cp_" + cid + "_700.bin")).string();
    CHECK(fs::exists(path));
    // no dft bytes on disk: header + tgt + the dft length field only
    CHECK(fs::file_size(path) == HEADER_BEFORE_TGT + 1024 + 8);

    // lazify converts the older resident checkpoint to disk-backed form with sz_dft == 0
    {
        server_prompt lazy;
        lazy.conversation_id = cid;
        lazy.checkpoints.push_back(cp);
        lazy.checkpoints.push_back(make_ckpt(800, 850, 851, 32, 0, 12)); // newest stays resident

        cache.lazify_checkpoints(lazy);

        const auto & older = lazy.checkpoints.front();
        CHECK(!older.src_path.empty());
        CHECK(older.data_tgt.empty());
        CHECK(older.data_dft.empty());
        CHECK(older.off_tgt == HEADER_BEFORE_TGT);
        CHECK(older.sz_tgt  == 1024);
        CHECK(older.off_dft == HEADER_BEFORE_TGT + 1024 + 8);
        CHECK(older.sz_dft  == 0);
    }

    // merge reads it back with the same offsets
    server_prompt prompt;
    prompt.conversation_id = cid;
    bind_prompt(prompt, 1024);
    cache.merge_checkpoint_spills(prompt);

    CHECK(prompt.checkpoints.size() == 1);
    if (prompt.checkpoints.empty()) {
        fs::remove_all(dir);
        return;
    }
    const auto & got = prompt.checkpoints.front();

    CHECK(got.pos_min == 700);
    CHECK(got.pos_max == 750);
    CHECK(got.n_tokens == 751);
    CHECK(got.sz_tgt == 1024);
    CHECK(got.sz_dft == 0);
    CHECK(got.off_tgt == HEADER_BEFORE_TGT);
    CHECK(got.off_dft == HEADER_BEFORE_TGT + 1024 + 8);
    CHECK(read_file_range(got.src_path, got.off_tgt, got.sz_tgt) == cp.data_tgt);

    // nothing for load_dft to restore: both the resident blob and the disk-backed guard
    // (sz_dft > 0) are empty, so it takes the "genuinely empty checkpoint" no-op path and
    // never reads at off_dft, which sits at EOF. no llama_context here, so this only
    // exercises the null-ctx short circuit - the state above is the real assertion.
    CHECK(got.data_dft.empty());
    CHECK(got.load_dft(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

    fs::remove_all(dir);
}

// a rollback drops this conversation's spill files bound past the divergence point,
// keeps the ones on the shared timeline and never touches another conversation
static void test_drop_stale_checkpoint_spills() {
    std::printf("test_drop_stale_checkpoint_spills\n");
    const std::string dir = make_temp_dir("dropstale");
    const uint32_t arch = 0x71u, vocab = 0x72u;
    server_prompt_cache cache(64, 0, dir, 16, -1, arch, vocab);

    const std::string cid = "conv_drop";
    for (int32_t p : {100, 500, 900, 1300}) {
        cache.spill_checkpoint(make_ckpt(p, p + 5, p + 6, 64, 0, static_cast<uint8_t>(p / 100)), cid);
    }
    cache.spill_checkpoint(make_ckpt(1000, 1010, 1011, 64, 0, 77), "conv_other");
    cache.wait_idle();

    // a file this build cannot read is dead weight in the spill budget as well
    {
        std::ofstream f(fs::path(dir) / "cp_conv_drop_2000.bin", std::ios::binary);
        auto put_u32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        put_u32(CP_MAGIC);
        put_u32(CP_VERSION - 1);
        put_u32(arch);
        put_u32(vocab);
    }

    // no conversation_id -> nothing to scan for
    {
        server_prompt anon;
        bind_prompt(anon, 2048);
        cache.drop_stale_checkpoint_spills(anon, 0);
        CHECK(cp_pos_min_on_disk(dir, cid).size() == 5);
        CHECK((cp_pos_min_on_disk(dir, "conv_other") == std::vector<int32_t>{1000}));
    }

    // divergence at 700 tokens: 906 and 1306 are on the abandoned branch
    server_prompt prompt;
    prompt.conversation_id = cid;
    bind_prompt(prompt, 2048);
    cache.drop_stale_checkpoint_spills(prompt, 700);

    CHECK((cp_pos_min_on_disk(dir, cid) == std::vector<int32_t>{100, 500}));
    CHECK((cp_pos_min_on_disk(dir, "conv_other") == std::vector<int32_t>{1000}));

    // the survivors still bind to the timeline
    cache.merge_checkpoint_spills(prompt);
    CHECK(prompt.checkpoints.size() == 2);

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
    test_spill_prune_thins_by_gap();
    test_spill_prune_small_limit_clamp();
    test_token_binding_rejects_diverged_prefix();
    test_old_version_file_skipped();
    test_empty_draft_checkpoint();
    test_drop_stale_checkpoint_spills();
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
