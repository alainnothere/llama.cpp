import os
import tempfile
import pytest
from utils import *

# End-to-end tests for the dynamic n-gram lookup cache (-lcd / --lookup-cache-dynamic).
#
# Two things are covered here:
#   1. fresh start: pointing -lcd at a file that does not exist yet must not be fatal - the server
#      starts with an empty dynamic cache (it used to GGML_ABORT at startup).
#   2. persistence: the dynamic cache is written back when the server shuts down gracefully
#      (SIGTERM, which is what ServerProcess.stop() sends), and a second server started with the
#      same path loads it and keeps serving.

server = ServerPreset.tinyllama2()

_cache_dir: str = ""
_cache_path: str = ""


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


# heavily repetitive: the n-gram cache only drafts patterns it has already seen
REPETITIVE_PROMPT = "the little cat ran and ran " * 12


def _make_server() -> ServerProcess:
    s = ServerPreset.tinyllama2()
    s.n_slots = 1
    s.n_predict = 64
    s.temperature = 0.0
    s.spec_type = "ngram-cache"
    s.lookup_cache_dynamic = _cache_path
    fd, s.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return s


@pytest.fixture(autouse=True)
def create_server():
    global server, _cache_dir, _cache_path
    _cache_dir = tempfile.mkdtemp(prefix="llama-ngram-cache-test-")
    # deliberately NOT created: the fresh-start path is part of what is under test
    _cache_path = os.path.join(_cache_dir, "dyn.bin")
    server = _make_server()
    yield


def _completion(s: ServerProcess):
    res = s.make_request("POST", "/completion", data={
        "prompt": REPETITIVE_PROMPT,
        "n_predict": 64,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    return res


# a -lcd path that does not exist yet must start the server, not abort it
def test_missing_dynamic_cache_starts_with_empty_cache():
    global server
    assert not os.path.exists(_cache_path)

    server.start()
    log = LogReader(server.log_path)

    _completion(server)

    out = log.drain()
    assert "could not read dynamic lookup cache" in out
    assert _cache_path in out


# the dynamic cache is written back on graceful shutdown and reloaded by the next server
def test_dynamic_cache_persists_across_restart():
    global server
    server.start()
    log = LogReader(server.log_path)

    res = _completion(server)
    assert res.body["timings"]["predicted_n"] > 0

    server.stop()

    out = log.drain()
    assert "saved dynamic lookup cache" in out

    assert os.path.exists(_cache_path)
    assert os.path.getsize(_cache_path) > 0
    # the atomic write must not leave its temporary behind
    assert not os.path.exists(_cache_path + ".tmp")

    size_after_first_run = os.path.getsize(_cache_path)

    # a second server with the same path loads the file and keeps working
    server = _make_server()
    server.start()
    log2 = LogReader(server.log_path)

    res = _completion(server)
    assert res.body["timings"]["predicted_n"] > 0

    out2 = log2.drain()
    assert "could not read dynamic lookup cache" not in out2

    # best-effort: with the pattern already in the dynamic cache, drafting should fire.
    # a tiny CI model is not guaranteed to reproduce it, so this is only reported, not asserted
    draft_n = res.body["timings"].get("draft_n", 0)
    print(f"run 2: draft_n={draft_n}")

    server.stop()

    # the second shutdown merged this run's context into the loaded cache and rewrote the file
    assert "saved dynamic lookup cache" in log2.drain()
    assert os.path.getsize(_cache_path) >= size_after_first_run
