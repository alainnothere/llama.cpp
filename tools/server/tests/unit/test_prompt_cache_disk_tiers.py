import glob
import os
import re
import shutil
import struct
import tempfile
import time
import pytest
from utils import *

# [TAG_DISK_ENOSPC] / [TAG_COLD_TIER] disk prompt cache: write failures, the hybrid .rec fallback, and the cold tier.
#
# LFM2-test-ci-80M is hybrid (conv-recurrent + attention) and delta-capable like the production 27B: the attention
# state goes to .kv as pos-range segments, the recurrent state to .rec, checkpoints to cp_<conv>_<pos_min>.bin.
#
# - a .hdr with rec_size 0 (what a rewrite that died under ENOSPC left behind on 2026-11-01 and 2026-11-04) must
#   restore the attention segments and take the recurrent state from the newest matching checkpoint, not reprocess
#   the whole prompt
# - a failed append (LLAMA_DISK_CACHE_FAIL_AFTER_BYTES fault injection) must leave the entry loadable at its last
#   committed segment, and a failed rewrite must leave the previous entry untouched
# - --cache-disk-path-cold: a cold entry is promoted back to hot by its restore, an idle hot entry is demoted

server = ServerPreset.tinyllama2()

_hot: str = ""
_cold: str = ""

DELIMITERS = [
    {"role": "user",      "delimiter": "<|im_start|>user\n"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant\n"},
]

FLUSH_RE    = re.compile(r"disk cache: flushing '(\w+)' tokens \[(\d+), (\d+)\) as (\d+) segment\(s\) \(([^)]*)\)")
WROTE_RE    = re.compile(r"disk cache: wrote '(\w+)' segment pos \[(\d+), (\d+)\)[^\n]*now (\d+) tokens")
RESTORED_RE = re.compile(r"load_from_disk: restored '(\w+)' tokens=(\d+)")


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def _sentences(tag: str, n: int) -> str:
    return "".join(
        f"Entry {tag}-{i}: the archivist recorded a curious object, its provenance, "
        f"its weight in grams, and the name of the donor who left it behind. "
        for i in range(n)
    )


def _conversation(tag: str, n_turns: int = 4, per_turn: int = 6) -> str:
    out = ""
    for t in range(n_turns):
        out += "<|im_start|>user\n"      + _sentences(f"{tag}u{t}", per_turn) + "<|im_end|>\n"
        out += "<|im_start|>assistant\n" + _sentences(f"{tag}a{t}", per_turn) + "<|im_end|>\n"
    return out + "<|im_start|>user\n" + _sentences(f"{tag}q", per_turn) + "<|im_end|>\n<|im_start|>assistant\n"


def _tok(text: str, add_special: bool = True) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
    assert res.status_code == 200
    return res.body["tokens"]


def _complete(prompt, n_predict: int = 2) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "temperature": 0.0,
        "seed": 42,
        "return_tokens": True,
        "message_delimiters": DELIMITERS,
    })
    assert res.status_code == 200
    return res.body


def _gen(body: dict) -> list[int]:
    return [t["id"] if isinstance(t, dict) else t for t in body["tokens"]]


def _wait_log(log, pred, text: str = "", timeout: float = 60.0):
    deadline = time.time() + timeout
    while True:
        text += log.drain()
        r = pred(text)
        if r:
            return text, r
        assert time.time() < deadline, "timed out waiting for the server log"
        time.sleep(0.1)


def _wait_written(log, text: str = "", conv: str | None = None, n_tok: int | None = None):
    """wait for a flush and for the writer to commit it; returns (text, conv_id, committed n_tok)"""
    def pred(t):
        for w in WROTE_RE.finditer(t):
            if (conv is None or w.group(1) == conv) and (n_tok is None or int(w.group(4)) == n_tok):
                return w
        return None
    text, w = _wait_log(log, pred, text)
    return text, w.group(1), int(w.group(4))


def _configure(cold: bool = False, demote_after: int | None = None, extra: list | None = None):
    global server
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = "ggml-org/LFM2-test-ci-80M"
    server.model_hf_file = "model-Q4_K_M.gguf"
    server.model_alias = "lfm2-80m"
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_ubatch = 512
    server.temperature = 0.0
    server.server_slots = True
    server.debug = True
    server.ctx_checkpoints = 32
    server.checkpoint_min_step = 128  # several cp_ files inside a ~700 token prompt
    server.cache_disk_path = _hot
    server.extra_args = list(extra or [])
    if cold:
        server.extra_args += ["--cache-disk-path-cold", _cold]
        if demote_after is not None:
            server.extra_args += ["--cache-disk-demote-after", str(demote_after)]
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


@pytest.fixture(autouse=True)
def dirs():
    global _hot, _cold
    _hot = tempfile.mkdtemp(prefix="llama-disk-tiers-hot-")
    _cold = tempfile.mkdtemp(prefix="llama-disk-tiers-cold-")
    yield
    shutil.rmtree(_hot, ignore_errors=True)
    shutil.rmtree(_cold, ignore_errors=True)


def _entry_files(d: str, conv: str) -> list[str]:
    names = [os.path.basename(p) for p in glob.glob(os.path.join(d, conv + ".*"))]
    names += [os.path.basename(p) for p in glob.glob(os.path.join(d, f"cp_{conv}_*.bin"))]
    return sorted(n for n in names if not n.endswith(".tmp"))


def _cp_files(d: str, conv: str) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, f"cp_{conv}_*.bin")))


def _read_hdr(path: str) -> dict:
    with open(path, "rb") as f:
        b = f.read()
    magic, version, arch, vocab, state, lt, n_tok, n_segs = struct.unpack_from("<8I", b, 0)
    kv_size, drft_size, rec_size = struct.unpack_from("<3Q", b, 32)
    return {"n_tok": n_tok, "n_segs": n_segs, "kv_size": kv_size, "drft_size": drft_size, "rec_size": rec_size}


def _set_hdr_rec_size(path: str, rec_size: int):
    """rewrite the header's rec_size and its xor checksum (mirror of disk_hdr_checksum)"""
    with open(path, "rb") as f:
        b = bytearray(f.read())
    struct.pack_into("<Q", b, 48, rec_size)
    body = bytes(b[:-8])
    pad = body + b"\0" * ((8 - len(body) % 8) % 8)
    s = 0
    for i in range(0, len(pad), 8):
        s ^= struct.unpack_from("<Q", pad, i)[0]
    struct.pack_into("<Q", b, len(b) - 8, s)
    with open(path, "wb") as f:
        f.write(bytes(b))


# (a) a hybrid entry whose .hdr says rec_size 0 and whose .rec is gone (an interrupted rewrite) restores the attention
# segments with APPEND and the recurrent state from the newest cp_ file, and only the tail after it is reprocessed
def test_hybrid_missing_rec_restores_from_checkpoint():
    _configure()
    server.start()
    log = LogReader(server.log_path)

    prompt = _conversation("A")
    ref = _complete(prompt)
    text, conv, n_tok = _wait_written(log)
    n_prompt = len(_tok(prompt))
    server.stop()  # drains the writer: every cp_ file is on disk

    hdr = os.path.join(_hot, conv + ".hdr")
    assert _read_hdr(hdr)["rec_size"] > 0
    assert _cp_files(_hot, conv), "the test needs checkpoint files"
    os.remove(os.path.join(_hot, conv + ".rec"))
    _set_hdr_rec_size(hdr, 0)

    _configure()
    server.start()
    log = LogReader(server.log_path)
    body = _complete(prompt)
    out = log.drain()

    assert "returned 0 of" not in out, "rec_size 0 header was applied as a full-state file"
    assert "recurrent state from checkpoint" in out
    m = RESTORED_RE.search(out)
    assert m and m.group(1) == conv
    n_restored = int(m.group(2))
    assert 0 < n_restored < n_tok
    # the restore is the checkpoint's position, not zero: only the tail is reprocessed
    assert body["timings"]["cache_n"] > 0
    assert body["timings"]["prompt_n"] < n_prompt
    # greedy: the restored state continues exactly like the original prefill
    assert _gen(body) == _gen(ref)
    server.stop()  # before the fixture removes the dirs: the shutdown flush still writes


# (a') no .rec and no checkpoint: the honest full reprocess, logged as such
def test_hybrid_missing_rec_without_checkpoint_reprocesses():
    _configure()
    server.start()
    log = LogReader(server.log_path)
    prompt = _conversation("B")
    _complete(prompt)
    text, conv, n_tok = _wait_written(log)
    server.stop()

    hdr = os.path.join(_hot, conv + ".hdr")
    os.remove(os.path.join(_hot, conv + ".rec"))
    _set_hdr_rec_size(hdr, 0)
    for p in _cp_files(_hot, conv):
        os.remove(os.path.join(_hot, p))

    _configure()
    server.start()
    log = LogReader(server.log_path)
    body = _complete(prompt)
    out = log.drain()
    assert "no matching checkpoint - the" in out
    assert "returned 0 of" not in out
    assert body["timings"]["cache_n"] == 0
    server.stop()


def _measure_first_flush_kv(prompt: str) -> int:
    """size of the .kv the first flush of `prompt` writes (fault-free run in a scratch dir)"""
    global _hot
    hot_saved = _hot
    _hot = tempfile.mkdtemp(prefix="llama-disk-tiers-measure-")
    try:
        _configure()
        server.start()
        log = LogReader(server.log_path)
        _complete(prompt)
        _, conv, _ = _wait_written(log)
        server.stop()
        return _read_hdr(os.path.join(_hot, conv + ".hdr"))["kv_size"]
    finally:
        shutil.rmtree(_hot, ignore_errors=True)
        _hot = hot_saved


# (b) LLAMA_DISK_CACHE_FAIL_AFTER_BYTES lets the first flush through and fails the next write half way, like a disk
# filling up. "append": the follow-up turn extends the flushed tokens, the append fails, the files are cut back to the
# committed header. "rewrite": the follow-up diverges inside the flushed segment, the staged rewrite fails and the old
# files must be untouched (2026-11-01: the old rewrite deleted them first). either way a fresh server restores the
# first turn from disk
@pytest.mark.parametrize("mode", ["append", "rewrite"])
def test_failed_write_keeps_entry_loadable(monkeypatch, mode: str):
    prompt = _conversation("C")
    s1 = _measure_first_flush_kv(prompt)
    assert s1 > 0

    # the fault fires a few KB into the second write: a partial tail, then the failure
    monkeypatch.setenv("LLAMA_DISK_CACHE_FAIL_AFTER_BYTES", str(s1 + 4096))
    _configure()
    server.start()
    log = LogReader(server.log_path)

    t1 = _tok(prompt)
    body = _complete(t1)
    text, conv, n1 = _wait_written(log)
    kv_path = os.path.join(_hot, conv + ".kv")
    assert os.path.getsize(kv_path) == s1

    tail = _tok("<|im_end|>\n<|im_start|>user\n" + _sentences("Cx", 3) + "<|im_end|>\n<|im_start|>assistant\n", add_special=False)
    if mode == "append":
        second = t1 + _gen(body) + tail          # extends the flushed tokens: append path
    else:
        second = t1[:len(t1) // 2] + tail          # diverges inside the only segment: keep = 0, staged rewrite
    _complete(second)
    text, _ = _wait_log(log, lambda t: "failed for '" + conv + "'" in t, text)
    if mode == "append":
        assert "rolled back to the committed" in text
    else:
        assert "staged rewrite" in text and "the previous entry" in text
    assert "marking conversation for rewrite" not in text
    server.stop()

    # the header is the truth and the files match it again
    h = _read_hdr(os.path.join(_hot, conv + ".hdr"))
    assert h["n_tok"] == n1
    assert os.path.getsize(kv_path) == h["kv_size"] == s1
    assert os.path.getsize(os.path.join(_hot, conv + ".rec")) == h["rec_size"]
    assert not glob.glob(os.path.join(_hot, conv + ".*.tmp"))

    monkeypatch.delenv("LLAMA_DISK_CACHE_FAIL_AFTER_BYTES")
    _configure()
    server.start()
    log = LogReader(server.log_path)
    body = _complete(t1 + _gen(body) + tail)
    out = log.drain()
    m = RESTORED_RE.search(out)
    assert m and m.group(1) == conv and int(m.group(2)) == n1, out[-4000:]
    assert body["timings"]["cache_n"] >= n1 - 1
    server.stop()


# (c) an entry in the cold dir restores, is promoted in the same pass, and afterwards lives only in hot,
# checkpoint files included
def test_cold_entry_is_promoted_on_restore():
    _configure(cold=True)
    server.start()
    log = LogReader(server.log_path)
    prompt = _conversation("D")
    ref = _complete(prompt)
    text, conv, n_tok = _wait_written(log)
    server.stop()

    files = _entry_files(_hot, conv)
    assert conv + ".hdr" in files and conv + ".rec" in files and _cp_files(_hot, conv)
    for f in files:
        shutil.move(os.path.join(_hot, f), os.path.join(_cold, f))
    assert not _entry_files(_hot, conv)

    _configure(cold=True)
    server.start()
    log = LogReader(server.log_path)
    body = _complete(prompt)
    out, _ = _wait_log(log, lambda t: "promoted '" + conv + "'" in t)

    assert re.search(r"disk cache: promoted '" + conv + r"' \d+ tokens [\d.]+ MiB from cold in \d+ ms", out)
    m = RESTORED_RE.search(out)
    assert m and m.group(1) == conv and int(m.group(2)) == n_tok
    assert body["timings"]["cache_n"] > 0
    assert _gen(body) == _gen(ref)

    hot_after = _entry_files(_hot, conv)
    assert set(files) <= set(hot_after), f"missing in hot: {set(files) - set(hot_after)}"
    assert not _entry_files(_cold, conv), "the entry must live in exactly one tier"
    assert not glob.glob(os.path.join(_hot, conv + ".*.tmp"))
    # the promoted header still describes the files
    h = _read_hdr(os.path.join(_hot, conv + ".hdr"))
    assert os.path.getsize(os.path.join(_hot, conv + ".kv")) == h["kv_size"]
    assert os.path.getsize(os.path.join(_hot, conv + ".rec")) == h["rec_size"]
    server.stop()


# (d) --cache-disk-demote-after 1: an entry no slot holds moves to cold, the one the slot holds stays hot, and the
# next request of the demoted conversation promotes it back
def test_idle_entry_is_demoted_and_promoted_back():
    _configure(cold=True, demote_after=1)
    server.start()
    log = LogReader(server.log_path)

    pa = _conversation("E")
    ref = _complete(pa)
    text, conv_a, n_a = _wait_written(log)
    files_a = _entry_files(_hot, conv_a)

    # the single slot moves on to another conversation: A is idle from here
    _complete(_conversation("F"))
    text, w = _wait_log(log, lambda t: next((w for w in WROTE_RE.finditer(t) if w.group(1) != conv_a), None), text)
    conv_b = w.group(1)

    text, _ = _wait_log(log, lambda t: "demoted '" + conv_a + "'" in t, text, timeout=30.0)
    assert not _entry_files(_hot, conv_a)
    assert set(_entry_files(_cold, conv_a)) == set(files_a)

    # the slot holds B: a few more scans must leave it hot
    time.sleep(3)
    text += log.drain()
    assert "demoted '" + conv_b + "'" not in text
    assert (conv_b + ".hdr") in _entry_files(_hot, conv_b)

    body = _complete(pa)
    text, _ = _wait_log(log, lambda t: "promoted '" + conv_a + "'" in t, text)
    assert body["timings"]["cache_n"] > 0
    assert _gen(body) == _gen(ref)
    assert set(files_a) <= set(_entry_files(_hot, conv_a))
    assert not _entry_files(_cold, conv_a)
    server.stop()
