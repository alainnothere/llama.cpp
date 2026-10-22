import glob
import re
import struct
import time
import os
import tempfile
import pytest
from utils import *

# Integration tests for the prompt-cache DISK tier (--cache-disk-path).
#
# These spin up a real llama-server, so they are NOT run as part of the fast
# unit suite implicitly — they need a model loaded. They complement the
# model-free C++ test (test-prompt-cache-disk.cpp), which covers the on-disk
# checkpoint format; here we cover the end-to-end behaviour: a cached prompt
# state is evicted to disk and later restored from disk.
#
# NOTE: with write-through the state is flushed to disk at every prompt_save
# (slot switch), independent of `--cache-ram`. A single slot plus several
# distinct prompts guarantees save/load cycles.

server = ServerPreset.tinyllama2()

_disk_dir: str = ""


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


# the disk tier derives conversation ids from the first user message span, so
# every prompt carries a user delimiter and every request names it
# no trailing space in the delimiter: the tokenizer merges a trailing space into
# the next word, and the span search matches token sequences, not characters
DELIMITERS = [
    {"role": "user", "delimiter": "User:"},
]


def _long_prompt(theme: str) -> str:
    # distinct, reasonably long prompts so each produces its own cache entry
    return "User: " + (
        f"Tell me a detailed story about {theme}. "
        f"Describe {theme} across mountains, rivers, cities and the deep sea, "
        f"naming every companion {theme} meets and every trial {theme} endures, "
        f"so that the tale of {theme} grows long enough to fill many pages."
    ) * 3


PROMPTS = [
    _long_prompt(t)
    for t in ("a brave knight", "a curious robot", "an old fisherman",
              "a wandering scholar", "a small dragon", "a lost astronaut")
]


@pytest.fixture(autouse=True)
def create_server():
    global server, _disk_dir
    server = ServerPreset.tinyllama2()
    server.n_slots = 1                # single slot -> every new prompt forces a save/load cycle
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.cache_ram = 1              # 1 MiB: tiny RAM budget so entries spill to the disk tier
    server.debug = True
    _disk_dir = tempfile.mkdtemp(prefix="llama-disk-cache-test-")
    server.cache_disk_path = _disk_dir
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield


# the disk tier announces itself at startup with the configured path
def test_disk_cache_enabled_logs_startup():
    server.start()
    log = LogReader(server.log_path)
    out = log.drain()
    assert "disk cache: enabled at" in out
    assert _disk_dir in out


# a cached prompt state evicted to disk must be restorable from disk
def test_prompt_state_spills_and_restores_from_disk():
    server.start()
    log = LogReader(server.log_path)
    assert "disk cache: enabled at" in log.drain()

    # warm the cache with the first prompt and record its (cold) prompt token count
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPTS[0],
        "cache_prompt": True,
        "message_delimiters": DELIMITERS,
    })
    assert res.status_code == 200
    first_prompt_n = res.body["timings"]["prompt_n"]

    # drive several more distinct prompts through the single slot; each slot switch
    # flushes the outgoing conversation's state to disk (write-through)
    for p in PROMPTS[1:]:
        res = server.make_request("POST", "/completion", data={
            "prompt": p,
            "cache_prompt": True,
            "message_delimiters": DELIMITERS,
        })
        assert res.status_code == 200

    assert "disk cache: flushing" in log.drain()

    # re-send the first prompt: it is no longer resident, so it must come back
    # from the disk tier (load() Phase 3 -> load_from_disk)
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPTS[0],
        "cache_prompt": True,
        "message_delimiters": DELIMITERS,
    })
    assert res.status_code == 200
    assert "load_from_disk: restored" in log.drain()
    # restoring from disk means most of the prompt was cached, not recomputed
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < first_prompt_n


# saving should produce split cache files (.hdr, .tok, .kv, and .drft with a draft) per conversation
def test_disk_cache_writes_split_files():
    server.start()
    for p in PROMPTS:
        res = server.make_request("POST", "/completion", data={
            "prompt": p,
            "cache_prompt": True,
            "message_delimiters": DELIMITERS,
        })
        assert res.status_code == 200

    hdrs = glob.glob(os.path.join(_disk_dir, "*.hdr"))
    assert len(hdrs) >= 1, f"expected at least one flushed cache header in {_disk_dir}"


# a corrupt/foreign .bin in the directory must not crash the server or be matched
def test_disk_cache_ignores_foreign_file():
    # drop a junk file before startup; the server must scan past it safely
    os.makedirs(_disk_dir, exist_ok=True)
    with open(os.path.join(_disk_dir, "not_a_cache_entry.bin"), "wb") as f:
        f.write(b"\xde\xad\xbe\xef" * 16)

    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPTS[0],
        "cache_prompt": True,
        "message_delimiters": DELIMITERS,
    })
    assert res.status_code == 200  # server healthy despite the foreign file


# --- divergence below the high-water mark: truncate-then-append ---------------------------
#
# DISK_SEGMENT_TOKENS (8192) is fixed, so these tests build conversations of 3-4 real
# segments with the tiny model. the first user span ends at "Assistant:", so the long body
# can change without changing the conversation id.

SEG = 8192

DELIMITERS_UA = [
    {"role": "user",      "delimiter": "User:"},
    {"role": "assistant", "delimiter": "Assistant:"},
]

FLUSH_RE = re.compile(r"disk cache: flushing '(\w+)' tokens \[(\d+), (\d+)\) as (\d+) segment\(s\) \(([^)]*)\)")


def _start_long_ctx():
    server.n_ctx = 4 * SEG + 2048
    server.n_batch = 2048
    server.n_ubatch = 512
    # the server caps the slot context at n_ctx_train (2048 for this model)
    server.extra_args = ["--override-kv", f"llama.context_length=int:{4 * SEG + 2048}"]
    server.start()


def _tok(text: str, add_special: bool = False) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
    assert res.status_code == 200
    return res.body["tokens"]


def _conv_tokens(n_tok: int) -> list[int]:
    # built piecewise so each delimiter is the same token run the server searches for
    body = " ".join(
        f"On day {i} the {['knight', 'robot', 'fisher', 'scholar', 'dragon'][i % 5]} walked {i * 7 % 13} miles and found {i * 3 % 11} stones."
        for i in range(12000)
    )
    toks = _tok("User:", True) + _tok(" Tell me a long story.") + _tok("Assistant:") + _tok(body)
    assert len(toks) >= n_tok, f"body too short: {len(toks)} < {n_tok}"
    return toks[:n_tok]


def _diverge(toks: list[int], pos: int) -> list[int]:
    out = list(toks)
    out[pos] = toks[pos + 1] if toks[pos + 1] != toks[pos] else toks[pos + 2]
    return out


def _complete(prompt, n_predict: int = 4):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
        "n_predict": n_predict,
        "message_delimiters": DELIMITERS_UA,
    })
    assert res.status_code == 200
    return res.body


def _switch_away(i: int = 0):
    # a different conversation on the single slot forces prompt_save of the current one
    _complete(PROMPTS[i % len(PROMPTS)])


def _wait_log(log, pred, text: str = "", timeout: float = 120.0):
    # the server log and the disk writer are both async: poll until pred(text) holds
    deadline = time.time() + timeout
    while True:
        text += log.drain()
        res = pred(text)
        if res:
            return text, res
        assert time.time() < deadline, "timed out waiting for the server log"
        time.sleep(0.1)


def _wait_flush(log, conv: str | None = None, min_end: int = 0, text: str = ""):
    # wait for one flush of conv (or of any conv ending at >= min_end) and for the writer to
    # commit its last segment, so the .hdr/.tok/.kv files on disk match the flush
    def find_flush(t):
        for m in FLUSH_RE.finditer(t):
            if (conv is None or m.group(1) == conv) and int(m.group(3)) >= min_end:
                return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5))
        return None
    text, fl = _wait_log(log, find_flush, text)
    text, _ = _wait_log(log, lambda t: re.search(rf"wrote '{fl[0]}' segment pos \[\d+, {fl[2]}\)", t), text)
    return text, fl


def _read_hdr(conv_id: str):
    with open(os.path.join(_disk_dir, conv_id + ".hdr"), "rb") as f:
        hdr = f.read()
    (magic, version, fa, fv, fs, lt, n_tok, n_segs) = struct.unpack_from("<8I", hdr, 0)
    segs = []
    off = 56
    for _ in range(n_segs):
        kv_off, kv_sz, d_off, d_sz, p0, p1 = struct.unpack_from("<4Q2i", hdr, off)
        segs.append((kv_off, kv_sz, d_off, d_sz, p0, p1))
        off += 40
    with open(os.path.join(_disk_dir, conv_id + ".tok"), "rb") as f:
        toks = list(struct.unpack(f"<{n_tok}i", f.read(4 * n_tok)))
    return toks, segs


def _kv_snapshot(conv_id: str):
    path = os.path.join(_disk_dir, conv_id + ".kv")
    st = os.stat(path)
    with open(path, "rb") as f:
        data = f.read()
    return st.st_ino, data


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _check_kept(conv: str, segs_old, ino_old, kv_old, keep: int):
    # segments ending at or below keep: same table entries, same bytes, same file (no remove)
    toks_new, segs_new = _read_hdr(conv)
    kept = [s for s in segs_old if s[5] <= keep]
    assert len(kept) > 0
    assert segs_new[:len(kept)] == kept
    assert all(s[4] >= keep for s in segs_new[len(kept):])
    kept_bytes = kept[-1][0] + kept[-1][1]
    ino_new, kv_new = _kv_snapshot(conv)
    assert ino_new == ino_old
    assert kv_new[:kept_bytes] == kv_old[:kept_bytes]
    return toks_new, segs_new


def _run_divergence(n_tok: int, div_pos: int, restart: bool):
    _start_long_ctx()
    log = LogReader(server.log_path)

    toks = _conv_tokens(n_tok)
    _complete(toks)
    _switch_away(0)
    _, (conv, base, end, _, mode) = _wait_flush(log, min_end=n_tok)
    assert mode == "rewrite" and base == 0
    toks_old, segs_old = _read_hdr(conv)
    assert len(toks_old) == end and len(segs_old) >= 3
    ino_old, kv_old = _kv_snapshot(conv)

    if restart:
        server.stop()
        server.start()
        log = LogReader(server.log_path)

    div = _diverge(toks, div_pos)
    ref = _complete(div)
    _switch_away(1)
    text, _ = _wait_log(log, lambda t: f"load_from_disk: restored '{conv}'" in t)
    _, (_, base, end, n_seg, mode) = _wait_flush(log, conv=conv, text=text)

    # expected boundary from the real token lists: last committed segment end <= LCP
    toks_new, _ = _read_hdr(conv)
    assert len(toks_new) == end
    lcp = _lcp(toks_old, toks_new)
    assert lcp == div_pos
    keep = max(s[5] for s in segs_old if s[5] <= lcp and s[5] < end)
    assert base == keep
    assert mode == f"truncate from {len(toks_old)} to {keep}, append"
    assert n_seg == (end - keep + SEG - 1) // SEG
    _check_kept(conv, segs_old, ino_old, kv_old, keep)

    # the truncated + appended files restore and give the same answer
    if restart:
        server.stop()
        server.start()
        log = LogReader(server.log_path)
    again = _complete(div)
    _wait_log(log, lambda t: f"load_from_disk: restored '{conv}'" in t)
    assert again["content"] == ref["content"]
    assert again["timings"]["cache_n"] >= keep


# (a) divergence inside the last segment: only that segment is rewritten
def test_disk_cache_truncate_last_segment():
    _run_divergence(n_tok=3 * SEG + 1000, div_pos=3 * SEG + 500, restart=False)


# (b) divergence inside segment 2 of 4: segments 0-1 kept, 2-3 dropped, new tail appended;
# the server restarts in between so the segment list is seeded from the .hdr
def test_disk_cache_truncate_middle_segment():
    _run_divergence(n_tok=3 * SEG + 3000, div_pos=2 * SEG + 1234, restart=True)


# (c) a byte-identical replay writes nothing; a replay that extends it appends only the tail
def test_disk_cache_identical_replay_appends_tail():
    _start_long_ctx()
    log = LogReader(server.log_path)

    n_tok = 2 * SEG + 500
    toks = _conv_tokens(n_tok)
    _complete(toks)
    _switch_away(0)
    _, (conv, _, end, _, _) = _wait_flush(log, min_end=n_tok)
    saved, segs_old = _read_hdr(conv)
    assert len(saved) == end
    ino_old, kv_old = _kv_snapshot(conv)

    # replay exactly the saved list: with n_predict = 1 the sampled token is never decoded,
    # so the live list equals the saved list whatever the model picks -> nothing to write
    _complete(saved, n_predict=1)
    _switch_away(1)
    _wait_log(log, lambda t: f"disk cache: '{conv}' already flushed ({len(saved)} tokens)" in t)
    ino_mid, kv_mid = _kv_snapshot(conv)
    assert ino_mid == ino_old and kv_mid == kv_old

    # replay the saved list plus a new turn -> append from the old high-water mark
    ext = saved + _conv_tokens(n_tok + 300)[n_tok:]
    _complete(ext)
    _switch_away(2)
    _, fl = _wait_flush(log, conv=conv)
    toks_new, _ = _read_hdr(conv)
    assert fl == (conv, len(saved), len(toks_new), (len(toks_new) - len(saved) + SEG - 1) // SEG, "append")
    assert toks_new[:len(saved)] == saved
    _check_kept(conv, segs_old, ino_old, kv_old, segs_old[-1][5])
