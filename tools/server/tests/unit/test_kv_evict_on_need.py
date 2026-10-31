import os
import shutil
import tempfile
import pytest
from utils import *

# [TAG_EVICT_ON_NEED] with --kv-unified all slots share one KV pool and an idle slot keeps its cells after
# release. get_available_slot() evicts idle slots (LRU) only when the chosen slot needs their cells for the
# restore or the prefill. 2026-10-28 production: 420096-cell pool, 172070 parked, a 248289-token disk restore
# came up 263 short, streamed 15.4 GB and then re-prefilled from zero. here: a 512-cell pool, two ~300-token
# conversations.

server = ServerPreset.tinyllama2()

_disk_dir: str = ""

DELIMITERS_UA = [
    {"role": "user",      "delimiter": "User:"},
    {"role": "assistant", "delimiter": "Assistant:"},
]

N_POOL = 512
N_CONV = 300


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


@pytest.fixture(autouse=True)
def create_server():
    global server, _disk_dir
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = N_POOL
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.cache_ram = 1
    server.debug = True
    # scratch next to the test models, not on tmpfs
    tmp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    _disk_dir = tempfile.mkdtemp(prefix="evict-on-need-", dir=tmp_root)
    server.cache_disk_path = _disk_dir
    fd, server.log_path = tempfile.mkstemp(suffix=".log", dir=_disk_dir)
    os.close(fd)
    yield
    server.stop()
    shutil.rmtree(_disk_dir, ignore_errors=True)


def _tok(text: str, add_special: bool = False) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
    assert res.status_code == 200
    return res.body["tokens"]


# distinct wording per topic: the slot selection must not see the two conversations as similar (LCP > 10%)
BODIES = {
    "knight":    lambda i: f"On day {i} the knight walked {i * 7 % 13} miles and found {i * 3 % 11} stones.",
    "fisherman": lambda i: f"Fish number {i} weighed {i * 5 % 17} pounds, so the old fisherman sang {i % 4} songs.",
}


def _conv(topic: str, n_tok: int) -> list[int]:
    body = " ".join(BODIES[topic](i) for i in range(200))
    toks = _tok("User:", True) + _tok(f" {topic.capitalize()} story, please.") + _tok("Assistant:") + _tok(body)
    assert len(toks) >= n_tok
    return toks[:n_tok]


def _complete(prompt: list[int]):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
        "n_predict": 4,
        "message_delimiters": DELIMITERS_UA,
    })
    assert res.status_code == 200
    return res.body


def _next_turn(prev: list[int], body, text: str) -> list[int]:
    gen = [t["id"] if isinstance(t, dict) else t for t in body["tokens"]]
    return prev + gen + _tok("User:") + _tok(text)


@pytest.mark.parametrize("idle_slots", [False, True])
def test_evict_idle_slot_when_pool_is_short(idle_slots: bool):
    server.no_cache_idle_slots = not idle_slots
    server.start()
    log = LogReader(server.log_path)

    a1 = _conv("knight", N_CONV)
    b1 = _conv("fisherman", N_CONV)

    # turn 1 of A: the pool is empty, nothing to evict
    body_a1 = _complete(a1)
    text = log.drain()
    assert "evicting idle slot" not in text

    # B takes the other slot. A holds ~303 cells, B needs 300 of the 209 left: A is evicted before
    # token 0 (A was flushed to disk at release), not purged by the decode-retry path mid-prefill
    _complete(b1)
    text = log.drain()
    assert "failed to find free space in the KV cache" not in text
    assert "evicting idle slot" in text
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in text

    # turn 2 of A: its slot is empty, its state is on disk, and B holds ~303 cells. B is evicted first,
    # then the disk restore fits and is not refused
    a2 = _next_turn(a1, body_a1, " and then what happened?")
    body_a2 = _complete(a2)
    text = log.drain()
    assert "evicting idle slot" in text
    assert "load_from_disk: restored" in text
    assert "not restored" not in text
    assert "failed to find free space in the KV cache" not in text
    assert body_a2["timings"]["cache_n"] >= N_CONV

    # turn 3 of A: the slot holds A, reused in place. no eviction (the pool has room) and no disk reload
    a3 = _next_turn(a2, body_a2, " and after that?")
    body_a3 = _complete(a3)
    text = log.drain()
    assert "load_from_disk:" not in text
    assert "evicting idle slot" not in text
    assert body_a3["timings"]["cache_n"] >= len(a2)


# two conversations that fit side by side: no eviction, and alternating between them never touches the disk
def test_no_eviction_when_both_fit():
    server.start()
    log = LogReader(server.log_path)

    a1 = _conv("knight", 200)
    b1 = _conv("fisherman", 200)

    body_a1 = _complete(a1)
    body_b1 = _complete(b1)
    a2 = _next_turn(a1, body_a1, " go on")
    _complete(a2)
    b2 = _next_turn(b1, body_b1, " go on")
    _complete(b2)

    text = log.drain()
    assert "evicting idle slot" not in text
    # the first turn of B looks on disk and finds nothing ("no valid cache files"), that is not a reload
    assert "load_from_disk: restored" not in text


# per-slot KV streams: every slot has its own cells, nothing to evict
def test_non_unified_never_evicts():
    server.kv_unified = False
    server.n_ctx = 2 * N_POOL
    server.start()
    log = LogReader(server.log_path)

    _complete(_conv("knight", N_CONV))
    _complete(_conv("fisherman", N_CONV))

    text = log.drain()
    assert "evicting idle slot" not in text
    assert "failed to find free space in the KV cache" not in text
