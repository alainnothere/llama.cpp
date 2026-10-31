import os
import re
import shutil
import tempfile
import time
import pytest
from utils import *

# [fork, 2026-10-30] --cache-convert: a conversation flushed to the disk prompt cache by a server running one K/V
# type must restore into a server running the other (f16 <-> q8_0) instead of being re-prefilled.
#
# the owner switches -ctk/-ctv between f16 and q8_0; before this a switch made every .hdr a "kv-state identity
# mismatch" (the K/V types are part of the state hash) and a 98k conversation cost ~10 min of prefill.
#
# LFM2-test-ci-80M: hybrid (conv-recurrent + attention, head 64 -> q8_0-able, the q8_0 cache rotates K and V),
# delta-capable like the production 27B: .kv segments + .rec snapshot. the attention segments go through the
# converting state reader, the f32 recurrent snapshot is type-independent.

server = ServerPreset.tinyllama2()

_disk_dir: str = ""

DELIMITERS = [
    {"role": "user",      "delimiter": "<|im_start|>user\n"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant\n"},
]

FLUSH_RE   = re.compile(r"disk cache: flushing '(\w+)' tokens \[(\d+), (\d+)\) as (\d+) segment\(s\) \(([^)]*)\)")
WROTE_RE   = re.compile(r"disk cache: wrote '(\w+)' segment pos \[(\d+), (\d+)\)")
CONVERT_RE = re.compile(r"disk cache: restored '(\w+)' with K/V conversion (\S+)->(\S+) \(([\d.]+) MiB\)")


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


def _prompt(tag: str) -> str:
    return ("<|im_start|>user\n" + _sentences(tag, 12) + "<|im_end|>\n"
            + "<|im_start|>assistant\n")


def _completion(prompt: str, n_predict: int = 8):
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


def _wait_log(log, pred, text: str = "", timeout: float = 60.0):
    deadline = time.time() + timeout
    while True:
        text += log.drain()
        r = pred(text)
        if r:
            return text, r
        assert time.time() < deadline, "timed out waiting for the server log"
        time.sleep(0.1)


def _configure(ctk: str, convert: bool | None):
    global server
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = "ggml-org/LFM2-test-ci-80M"
    server.model_hf_file = "model-Q4_K_M.gguf"
    server.model_alias = "lfm2-80m"
    server.n_slots = 1                # every new conversation forces prompt_save (the disk flush) of the current one
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_ubatch = 512
    server.temperature = 0.0
    server.server_slots = True
    server.debug = True
    server.fa = "on"                  # quantized V needs flash attention; the fork runs -fa on
    server.ctk = ctk
    server.ctv = ctk
    server.cache_disk_path = _disk_dir
    server.extra_args = []
    if convert is not None:
        server.extra_args += ["--cache-convert", "on" if convert else "off"]
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


@pytest.fixture(autouse=True)
def create_dir():
    global _disk_dir
    _disk_dir = tempfile.mkdtemp(prefix="llama-cache-convert-test-")
    yield
    shutil.rmtree(_disk_dir, ignore_errors=True)


def _save_conversation(ctk: str) -> tuple[str, int, dict]:
    """first server: run conversation A, switch to B so A is flushed, wait for the write, stop"""
    _configure(ctk, convert=None)
    server.start()
    log = LogReader(server.log_path)

    body_a = _completion(_prompt("A"))
    n_a = body_a["timings"]["prompt_n"]
    _completion(_prompt("B"))

    # the flush of A happens at the slot switch; the writer is async
    text, m = _wait_log(log, lambda t: FLUSH_RE.search(t))
    conv_id = m.group(1)
    _wait_log(log, lambda t: any(w.group(1) == conv_id for w in WROTE_RE.finditer(t)), text)
    server.stop()
    return conv_id, n_a, body_a


@pytest.mark.parametrize("ctk_save,ctk_load", [("f16", "q8_0"), ("q8_0", "f16")])
def test_disk_cache_restores_across_kv_types(ctk_save: str, ctk_load: str):
    conv_id, n_a, body_a = _save_conversation(ctk_save)

    _configure(ctk_load, convert=True)
    server.start()
    log = LogReader(server.log_path)
    text = log.drain()
    assert "--cache-convert on, also accepting files saved with" in text

    body = _completion(_prompt("A"))
    text, m = _wait_log(log, lambda t: CONVERT_RE.search(t), text)
    assert m.group(1) == conv_id
    assert (m.group(2), m.group(3)) == (ctk_save, ctk_load)
    assert float(m.group(4)) > 0.0
    assert "load_from_disk: restored" in text
    # the library's own line, one per converted segment
    assert f"K/V type conversion on load: {ctk_save}->{ctk_load}" in text

    # restored, not re-prefilled, and generation continues
    assert body["timings"]["cache_n"] > n_a // 2
    assert body["timings"]["prompt_n"] < n_a // 2
    assert len(body["tokens"]) == 8

    # the next flush of A rewrites it in this server's type: never a file with mixed types
    _completion(_prompt("C"))
    text, m = _wait_log(log, lambda t: re.search(rf"disk cache: flushing '{conv_id}' tokens \[(\d+), (\d+)\) as \d+ segment\(s\) \(([^)]*)\)", t), text)
    assert m.group(1) == "0" and m.group(3) == "rewrite"
    assert f"disk cache: '{conv_id}' was saved with another K/V type" in text
    _wait_log(log, lambda t: re.search(rf"disk cache: wrote '{conv_id}' segment pos \[\d+, {m.group(2)}\)", t), text)
    server.stop()

    # a third server of the load type WITHOUT conversion now finds a native file
    _configure(ctk_load, convert=False)
    server.start()
    log = LogReader(server.log_path)
    body = _completion(_prompt("A"))
    text, _ = _wait_log(log, lambda t: "load_from_disk: restored" in t)
    assert "with K/V conversion" not in text
    assert body["timings"]["cache_n"] > n_a // 2


def test_disk_cache_type_switch_without_convert_is_a_miss():
    conv_id, n_a, _ = _save_conversation("f16")

    _configure("q8_0", convert=False)
    server.start()
    log = LogReader(server.log_path)

    body = _completion(_prompt("A"))
    text = log.drain()
    assert "kv-state identity mismatch" in text
    assert "load_from_disk: restored" not in text
    assert "with K/V conversion" not in text
    # re-prefilled from scratch, as before the change
    assert body["timings"]["cache_n"] == 0
    assert body["timings"]["prompt_n"] >= n_a


# /slots save + restore: the llama_state_seq_save_file / _load_file path (llama_io_read_file), no hash involved
@pytest.mark.parametrize("ctk_save,ctk_load,convert", [("f16", "q8_0", True), ("q8_0", "f16", True), ("f16", "q8_0", False)])
def test_slot_file_restores_across_kv_types(ctk_save: str, ctk_load: str, convert: bool):
    _configure(ctk_save, convert=None)
    server.slot_save_path = _disk_dir
    server.start()
    _completion(_prompt("S"))
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "conv.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0
    server.stop()

    _configure(ctk_load, convert=convert)
    server.slot_save_path = _disk_dir
    server.start()
    log = LogReader(server.log_path)
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "conv.bin"})
    text = log.drain()
    if not convert:
        # the refusal is unchanged: the type mismatch fails the restore
        assert res.status_code != 200
        # a q8_0 cache that rotates (head % 64 == 0) refuses at the rotation check, before the type check
        assert "mismatched key type" in text or "incompatible key rotation" in text
        return
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert f"state_seq_load_file: K/V type conversion on load: {ctk_save}->{ctk_load}" in text
    body = _completion(_prompt("S"))
    assert body["timings"]["cache_n"] > 0
    assert len(body["tokens"]) == 8
