import glob
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


def _long_prompt(theme: str) -> str:
    # distinct, reasonably long prompts so each produces its own cache entry
    return (
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
    })
    assert res.status_code == 200
    first_prompt_n = res.body["timings"]["prompt_n"]

    # drive several more distinct prompts through the single slot; each slot switch
    # flushes the outgoing conversation's state to disk (write-through)
    for p in PROMPTS[1:]:
        res = server.make_request("POST", "/completion", data={
            "prompt": p,
            "cache_prompt": True,
        })
        assert res.status_code == 200

    assert "disk cache: flushing" in log.drain()

    # re-send the first prompt: it is no longer resident, so it must come back
    # from the disk tier (load() Phase 3 -> load_from_disk)
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPTS[0],
        "cache_prompt": True,
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
    })
    assert res.status_code == 200  # server healthy despite the foreign file
