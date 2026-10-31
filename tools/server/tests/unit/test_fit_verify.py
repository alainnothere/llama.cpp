import os
import re
import tempfile
import time
import pytest
from utils import *

# [fork, 2026-10-30] --fit-verify, the server's warm-up gate (design-fit-lazy-reserve-and-warmup-gate.md, section E).
# 2026-10-30 09:50 the fit left Vulkan1 1446 MiB free and the first request moved ~1 GB to GTT (tg 34 -> 26 t/s); the
# gate runs one n_ubatch pp graph + one speculative step before is_ready and reloads with a raised --fit-target on an
# eviction. These tests run on the CPU (Vulkan devices hidden), where no device has the GTT_USED proc:
#   - the start must succeed with the "cannot verify" skip line (never fail a start on a missing proc)
#   - LLAMA_SERVER_FIT_VERIFY_FAKE_EVICT_MIB gates a fake device that evicts on the first warm-up only, which drives the
#     verdict, the destroy() + load_model_core() reload and the pass on attempt 2 without a GPU

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server(monkeypatch):
    global server
    # CPU only: an empty list hides every Vulkan device (ggml-vulkan GGML_VK_VISIBLE_DEVICES)
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "")
    monkeypatch.delenv("LLAMA_SERVER_FIT_VERIFY_FAKE_EVICT_MIB", raising=False)
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_predict = 4
    server.temperature = 0.0
    # an n-gram drafter so the gate's speculative step (begin, draft, verify 1 + 64 rows, catch-up) runs
    server.extra_args = ["--spec-type", "ngram-mod"]
    tmp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    fd, server.log_path = tempfile.mkstemp(prefix="fit-verify-", suffix=".log", dir=tmp_root)
    os.close(fd)
    yield
    server.stop()
    try:
        os.remove(server.log_path)
    except OSError:
        pass


def read_log() -> str:
    with open(server.log_path) as f:
        return f.read()


def complete_ok():
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 4})
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_fit_verify_skips_without_gtt_proc():
    server.extra_args += ["--fit-verify", "on"]
    server.start()
    complete_ok()
    log = read_log()
    assert "fit verify: cannot verify" in log
    assert "gate skipped" in log
    assert "fit verify: EVICTION" not in log
    assert "fit verify: pass" not in log


def test_fit_verify_off_is_silent():
    server.extra_args += ["--fit-verify", "off"]
    server.start()
    complete_ok()
    assert "fit verify" not in read_log()


def test_fit_verify_fake_eviction_reloads_and_passes(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_FIT_VERIFY_FAKE_EVICT_MIB", "1024")
    server.extra_args += ["--fit-verify", "on", "--fit-verify-mib", "256"]
    server.start()
    complete_ok()
    log = read_log()
    assert re.search(r"fit verify: EVICTION on fake \(attempt 1 of 3\): GTT grew 1024 MiB", log)
    assert "fit verify: attempt 1 of 3 found EVICTION on 1 device(s), reloading (retry 1 of 2)" in log
    assert re.search(r"fit verify: pass on attempt 2 of 3 \([0-9.]+ s: pp \d+ rows, verify \d+ rows, drafted \d+\)", log)
    # the model was loaded twice, the second load is the reload
    assert log.count("load_model_c: loading model") == 2


def test_fit_verify_fake_below_threshold_passes(monkeypatch):
    # 200 MiB of GTT growth is under --fit-verify-mib 256: no verdict, no reload
    monkeypatch.setenv("LLAMA_SERVER_FIT_VERIFY_FAKE_EVICT_MIB", "200")
    server.extra_args += ["--fit-verify", "on"]
    server.start()
    complete_ok()
    log = read_log()
    assert "fit verify: ok on fake (attempt 1 of 3)" in log
    assert "fit verify: pass on attempt 1 of 3" in log
    assert log.count("load_model_c: loading model") == 1


def test_fit_verify_gate_reruns_on_wake(monkeypatch):
    # the sleep path calls load_model() again: the gate re-runs (the fake evicts on the first run of the process only)
    monkeypatch.setenv("LLAMA_SERVER_FIT_VERIFY_FAKE_EVICT_MIB", "1024")
    server.extra_args += ["--fit-verify", "on"]
    server.sleep_idle_seconds = 1
    server.start()
    time.sleep(2)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["is_sleeping"] == True
    complete_ok()
    log = read_log()
    assert "fit verify: pass on attempt 2 of 3" in log
    assert "fit verify: pass on attempt 1 of 3" in log # the wake
