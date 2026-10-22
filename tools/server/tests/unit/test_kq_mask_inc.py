import pytest
from utils import *

# [KQ_MASK_INC] review 2026-10-21: after a multi-seq ubatch the persistent KQ mask took its watermark from the first
# seq of the ubatch only, so the rows of the other seq stayed "settled" with that seq's history. the next single-seq
# ubatch of the first seq with the same n_kv and more than one row (here: a cached-prompt continuation) uploaded only
# the columns past the watermark. it needs the first seq of the ubatch exactly one position ahead of the other
# (cnt == n_tokens in llama_kv_cache::set_input_kq_mask_dev). the start skew of the two requests is not under the
# test's control, so several prompt length offsets are run. LLAMA_KQ_MASK_CHECK=1 aborts the server on a stale region.
# tests/test-recurrent-state-rollback.cpp ("kq mask" column) is the deterministic version on the dummy models.

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.fa = "on"
    server.n_ctx = 1024
    server.n_batch = 64
    server.n_ubatch = 64
    server.temperature = 0.0
    # no prompt cache: it clears the idle slot when a task starts (dirty from cell 0, full upload)
    server.extra_args = ["-dev", "none", "--cache-ram", "0"]


def _complete(id_slot: int, prompt: list[int], n_predict: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def test_kq_mask_two_slots_then_one(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_KQ_MASK_CHECK", "1")
    server.start()

    for rnd, off in enumerate([1, 0, 2, 1, 0, 2]):
        base = 10 + 40*rnd
        p1 = [1] + list(range(base, base + 12))
        p0 = [1] + list(range(base + 20, base + 32 + off))
        # slot 1 stops first, slot 0 keeps going alone with the same n_kv
        results = parallel_function_calls([
            (_complete, (0, p0, 24)),
            (_complete, (1, p1, 8)),
        ])
        for r in results:
            assert r["tokens_predicted"] > 0

        # a multi-row single-seq ubatch of slot 0: rows >= 1 held slot 1 in the multi-seq steps
        r = _complete(0, p0 + [base + 5, base + 6, base + 7], 4)
        assert r["tokens_predicted"] == 4

    res = server.make_request("GET", "/health")
    assert res.status_code == 200
