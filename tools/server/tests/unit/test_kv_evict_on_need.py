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
        with open(self.path, errors="replace") as f:
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


# [TAG_POOL_TURNS] two conversations that do not fit the pool together take turns: the second one waits for the busy
# slot to release, then evicts it and restores from disk, instead of prefilling from token 0 into the cells that are
# left (2026-11-02 production: 343808-cell pool, 212085 held by a generating slot, a 250801-token conversation on
# disk got 131723 cells and prefilled from zero). here: a 512-cell pool, A = 250 tokens + 240 generated, B = 300.

import threading
import time
import requests


class StreamRun:
    """POST /completion with stream=True on a thread; `started` is set at the first chunk."""

    def __init__(self, data: dict):
        self.data = dict(data)
        self.data["stream"] = True
        self.started = threading.Event()
        self.content = ""
        self.final: dict | None = None
        self.error: dict | None = None
        self.status = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            r = requests.post(server.make_url("/completion"), json=self.data, stream=True, timeout=120)
            self.status = r.status_code
            if r.status_code != 200:
                self.error = r.json()
                return
            for raw in r.iter_lines():
                line = raw.decode("utf-8")
                if line.startswith("error: "):
                    self.error = json.loads(line[7:])
                    break
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if "error" in d:
                    self.error = d["error"]
                    break
                self.content += d.get("content", "")
                self.started.set()
                if d.get("stop"):
                    self.final = d
        finally:
            self.started.set()

    def join(self, timeout=120):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "stream did not finish"


def _gen_req(prompt: list[int], n_predict: int) -> dict:
    return {
        "prompt": prompt,
        "cache_prompt": True,
        "temperature": 0.0,
        "seed": 42,
        "n_predict": n_predict,
        "ignore_eos": True,
        "message_delimiters": DELIMITERS_UA,
    }


def test_busy_slot_second_conversation_waits_then_restores():
    server.start()
    log = LogReader(server.log_path)

    a1 = _conv("knight", 250)
    b1 = _conv("fisherman", 300)

    # reference output of A, alone in the pool
    ref = server.make_request("POST", "/completion", data=_gen_req(a1, 240))
    assert ref.status_code == 200
    assert ref.body["timings"]["predicted_n"] == 240

    # B turn 1 evicts A (490 cells, flushed at release) and is flushed to disk at its own release
    body_b1 = _complete(b1)
    log.drain()

    # A again: restores from disk (evicting B, which is then on disk only) and generates 240 tokens
    run_a = StreamRun(_gen_req(a1, 240)).start()
    assert run_a.started.wait(60)
    assert run_a.error is None

    # B turn 2 while A is generating: needs ~310 cells, the pool has <= 262 next to A. it must wait, not prefill
    b2 = _next_turn(b1, body_b1, " and then what happened?")
    t0 = time.time()
    body_b2 = _complete(b2)
    run_a.join()

    assert run_a.error is None, run_a.error
    assert run_a.final is not None
    assert run_a.final["timings"]["predicted_n"] == 240
    assert run_a.content == ref.body["content"]

    text = log.drain()
    assert "waits for a release" in text
    assert "launched after waiting" in text
    assert "load_from_disk: restored" in text
    assert "not restored" not in text
    assert "no idle slot left to evict" not in text
    assert "Context size has been exceeded" not in text
    assert "failed to find free space in the KV cache" not in text
    # restored, not prefilled from zero: only the new turn is processed
    assert body_b2["timings"]["cache_n"] >= 300
    assert body_b2["timings"]["prompt_n"] < len(b2) - 300
    assert time.time() - t0 < 60


# [TAG_POOL_TURNS] a request that can never fit (prompt >= n_ctx) errors right away, it does not wait for the busy slot
def test_request_larger_than_ctx_errors_while_slot_busy():
    server.start()
    log = LogReader(server.log_path)

    run_a = StreamRun(_gen_req(_conv("knight", 250), 240)).start()
    assert run_a.started.wait(60)

    t0 = time.time()
    res = server.make_request("POST", "/completion", data={
        "prompt": _conv("fisherman", N_POOL + 50),
        "cache_prompt": True,
        "n_predict": 4,
        "message_delimiters": DELIMITERS_UA,
    }, timeout=30)
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"
    run_a.join()
    assert run_a.error is None
    assert run_a.final["timings"]["predicted_n"] == 240

    text = log.drain()
    assert "waits for a release" not in text
    assert time.time() - t0 < 30


# [TAG_FLUSH_PREFIX_GUARD] a slot whose live tokens are a strict prefix of the conversation on disk (a cancelled
# prefill, or a request for an earlier point of the conversation) must not truncate the file on its next save.
# 2026-11-02: a cancelled 250801-token prefill at ~115k would have cut the 250744-token file to ~114688 and deleted
# its .rec. LFM2 (hybrid, .rec on disk) is slow enough on two CPU threads to cancel mid-prefill.
def _disk_n_tok(conv_dir: str) -> dict[str, int]:
    res = {}
    for f in os.listdir(conv_dir):
        if f.endswith(".tok"):
            res[f[:-4]] = os.path.getsize(os.path.join(conv_dir, f)) // 4
    return res


def test_cancelled_prefill_keeps_disk_file():
    global server
    disk_dir, log_path = server.cache_disk_path, server.log_path
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = "ggml-org/LFM2-test-ci-80M"
    server.model_hf_file = "model-Q4_K_M.gguf"
    server.model_alias = "lfm2-80m"
    server.n_slots = 2
    server.n_ctx = 8192
    server.n_batch = 64
    server.n_threads = 2
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.cache_ram = 1
    server.debug = True
    server.cache_disk_path = disk_dir
    server.log_path = log_path
    server.start()
    log = LogReader(server.log_path)

    def tok(text, add_special=False):
        r = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
        return r.body["tokens"]

    body = " ".join(BODIES["knight"](i) for i in range(400))
    d1 = (tok("User:", True) + tok(" Knight story, please.") + tok("Assistant:") + tok(body))[:3000]

    def comp(prompt, n_predict):
        r = server.make_request("POST", "/completion", data={
            "prompt": prompt, "cache_prompt": True, "temperature": 0.0, "seed": 42,
            "n_predict": n_predict, "message_delimiters": DELIMITERS_UA,
        })
        assert r.status_code == 200
        return r.body

    comp(d1, 1)
    # the release flush is written by the disk tier's worker thread
    for _ in range(100):
        files = _disk_n_tok(disk_dir)
        if files and os.path.exists(os.path.join(disk_dir, next(iter(files)) + ".rec")):
            break
        time.sleep(0.1)
    assert len(files) == 1
    conv, n_file = next(iter(files.items()))
    assert n_file >= 3000
    rec = os.path.join(disk_dir, conv + ".rec")
    assert os.path.exists(rec)
    log.drain()

    # a request for an earlier point of the conversation, no generation: the slot ends holding a strict prefix
    comp(d1[:400], 0)

    # continue on that slot (in place: 400/2900 > the 0.1 similarity threshold) and cancel mid-prefill: release()
    # keeps the partial tokens
    s = requests.Session()
    r = s.post(server.make_url("/completion"), json={
        "prompt": d1[:2900], "cache_prompt": True, "temperature": 0.0, "n_predict": 4, "stream": True,
        "return_progress": True, "message_delimiters": DELIMITERS_UA,
    }, stream=True, timeout=60)
    for raw in r.iter_lines():
        if raw.startswith(b"data: ") and b"prompt_progress" in raw:
            break
    r.close()
    s.close()

    # wait until the slot is released (cancelled), then make the server save it: a request on the other slot
    # runs the cache-idle-slots save over the idle slots, and the shutdown flush saves it again
    for _ in range(100):
        slots = server.make_request("GET", "/slots").body
        if not any(sl["is_processing"] for sl in slots):
            break
        time.sleep(0.1)
    text = log.drain()
    assert "prompt processing batch" in text
    comp(tok("User:", True) + tok(" Fisherman story, please.") + tok("Assistant:") + tok(" Fish."), 1)
    server.stop()

    text = log.drain()
    assert "truncate from" not in text
    assert _disk_n_tok(disk_dir)[conv] == n_file
    assert os.path.exists(rec)


# [TAG_WALL_ONE_SLOT] two generating slots outgrow the pool together (nothing at admission could see it): only one
# slot fails, the other one finishes, the server stays up
def test_context_wall_fails_one_slot():
    server.server_continuous_batching = True
    server.cache_disk_path = None
    server.cache_ram = 0
    server.start()
    log = LogReader(server.log_path)

    run_a = StreamRun(_gen_req(_conv("knight", 100), 400)).start()
    assert run_a.started.wait(60)
    run_b = StreamRun(_gen_req(_conv("fisherman", 200), 300)).start()
    run_a.join()
    run_b.join()

    text = log.drain()
    assert "waits for a release" not in text, "B was deferred, the wall was not reached"
    assert "context size has been exceeded at off" in text
    errs = [r for r in (run_a, run_b) if r.error is not None]
    oks = [r for r in (run_a, run_b) if r.error is None]
    assert len(errs) == 1 and len(oks) == 1
    assert "Context size has been exceeded" in json.dumps(errs[0].error)
    assert oks[0].final is not None
    assert oks[0].final["timings"]["predicted_n"] == oks[0].data["n_predict"]

    assert server.make_request("GET", "/health").status_code == 200
    _complete(_conv("knight", 50))
