import json
import os
import re
import tempfile
import pytest
from utils import *

# 2026-10-21: --spec-rs-depth puts the recurrent rollback window below the n-gram cap. a draft deeper than the window
# skips the host checkpoint; a deep partial landing rolls back through the pre-ubatch plane of the recurrent memory
# and replays the accepted rows. the --spec-stats-file line says which rollback path ran.
# LFM2-test-ci-80M is a hybrid (conv-recurrent + attention) model.
#
# everything runs on the CPU (-dev none). with greedy sampling a draft cannot change the output in exact arithmetic,
# but the verify batches have other shapes than 1-token decodes, and on Vulkan that alone moves the near-tied argmax
# of this 80M model (2026-10-21: no-draft != every drafting arm on Vulkan, while all drafting arms agree; on the CPU
# the default window arm drifts the same way). so the reference is the run without drafts on the CPU, plus the
# arm that takes the old host checkpoint for every rollback (--spec-rs-depth 0), which drafts the same chains

# the 80M test model does not continue the repetition, so the n-gram drafts are mostly rejected early: n_match 1
# gives ~9 deep partial landings per request (2026-10-21), n_match 4 gives none
REPETITIVE_PROMPT = "Once upon a time, there was a little girl named Lily. She liked to play outside. " * 6 + "Once upon a time,"

NGRAM_ARGS = [
    "--spec-ngram-mod-n-match", "1", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "16",
    "--spec-ngram-draft-n-max", "16",
]

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = "ggml-org/LFM2-test-ci-80M"
    server.model_hf_file = "model-Q4_K_M.gguf"
    server.model_alias = "lfm2-80m"
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_batch = 64
    server.n_ubatch = 64
    server.temperature = 0.0


def _complete(prompt: str) -> str:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "cache_prompt": False,
    })
    assert res.status_code == 200
    return res.body["content"]


def _run_many(spec_args: list[str] | None, prompts: list[str], n_ubatch: int = 64):
    global server
    fd, stats = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    server.n_ubatch = n_ubatch
    server.extra_args = ["-dev", "none"]
    if spec_args is not None:
        server.spec_type = "ngram-mod"
        server.extra_args += NGRAM_ARGS + ["--spec-stats-file", stats] + spec_args
    server.start()
    contents = [_complete(p) for p in prompts]
    server.stop()
    with open(stats) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    os.unlink(stats)
    return contents, lines


def _run(spec_args: list[str] | None, n_ubatch: int = 64):
    contents, lines = _run_many(spec_args, [REPETITIVE_PROMPT], n_ubatch)
    return contents[0], (lines[0] if lines else None)


def test_prebatch_restore_keeps_output():
    base, _ = _run(None)

    # window 0: every rejected row is rolled back from a host checkpoint (the old path)
    ckpt, jc = _run(["--spec-rs-depth", "0"])
    assert jc["n_spec_ckpt_restore"] > 0 and jc["n_prebatch_restore"] == 0, jc

    # window 2, chains up to 16: deep partial landings go through the pre-ubatch plane, no host checkpoint at all
    content, j = _run(["--spec-rs-depth", "2"])
    assert j["rs_depth"] == 2
    assert j["n_draft_tokens"] > 0
    assert j["n_prebatch_restore"] > 0, j
    assert j["n_spec_ckpt"] == 0 and j["n_spec_ckpt_restore"] == 0, j
    assert content == ckpt, "pre-ubatch restore and host checkpoint restore disagree"
    assert content == base, "output changed with --spec-rs-depth 2"

    # the old sizing (window = cap): every rollback stays inside the window
    _, j = _run([])
    assert j["rs_depth"] == -1
    assert j["n_prebatch_restore"] == 0 and j["n_spec_ckpt"] == 0, j


def test_prebatch_falls_back_to_checkpoint():
    base, _ = _run(None, n_ubatch=8)
    ckpt, _ = _run(["--spec-rs-depth", "0"], n_ubatch=8)

    # verify rows longer than a ubatch cannot use the plane (it holds the start of the LAST ubatch only):
    # those steps take the host checkpoint, shorter deep drafts still use the plane
    content, j = _run(["--spec-rs-depth", "2"], n_ubatch=8)
    assert j["n_spec_ckpt"] > 0, j
    assert j["n_spec_ckpt_restore"] + j["n_prebatch_restore"] > 0, j
    assert content == ckpt, "pre-ubatch restore and host checkpoint restore disagree (-ub 8)"
    assert content == base, "output changed with --spec-rs-depth 2 and -ub 8"


OTHER_PROMPT = "The old man walked to the market. He bought bread and milk. " * 5 + "The old man"


# the rollback target of the plane path is recorded when the checkpoint is skipped (spec_prebatch_pos and
# spec_prebatch_n_tokens), never read from the checkpoint struct. two requests on one server, neither ever takes a
# host checkpoint: a wrong rollback position trips the GGML_ASSERT(can_rollback_to) in the restore (the plane only
# serves the exact batch start), a wrong keep_first length shows as a different context length at the end.
# the text is not compared: on OTHER_PROMPT every drafting arm (window 0, 2 and the default) leaves the no-draft text
# at char 166 on the CPU (2026-10-21), and the n-gram table carries over between requests, so the second request
# drafts other chains than on a fresh server
def test_prebatch_two_requests_without_checkpoint():
    global server
    prompts = [REPETITIVE_PROMPT, OTHER_PROMPT]
    _, lines_base = _run_many([], prompts)
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.debug = True
    try:
        _, lines = _run_many(["--spec-rs-depth", "2"], prompts)
        with open(server.log_path, errors="replace") as f:
            log = f.read()
    finally:
        os.remove(server.log_path)
    assert len(lines) == 2 and len(lines_base) == 2
    for j, jb in zip(lines, lines_base):
        assert j["n_spec_ckpt"] == 0, j
        assert j["n_prebatch_restore"] > 0, j
        assert j["n_decoded"] == jb["n_decoded"] == 128, (j["n_decoded"], jb["n_decoded"])
        assert j["n_ctx_used"] == jb["n_ctx_used"], (j["n_ctx_used"], jb["n_ctx_used"])
    assert "non-consecutive token position" not in log
    assert log.count("rolling back to the pre-ubatch plane") == lines[0]["n_prebatch_restore"] + lines[1]["n_prebatch_restore"]


def _auto_round(v: int, n_min: int, n_max: int) -> int:
    v = ((max(v, 1) + 3) // 4) * 4
    return max(n_min, min(n_max, v))


# --spec-ngram-auto scores each chain with its REAL landing on the restore step; the replay step does not call
# accept(). before 2026-10-21 the replay was the only accept(), one token longer than the real landing, so a chain
# rejected at its last token was scored whole and the cap jumped to the ceiling. every "ngram-mod auto:" line must
# carry the same (offered, accepted) as the "spec step:" line of that chain, and a partial landing sets the cap to
# round4(accepted + 4)
def test_prebatch_restore_scores_real_landing(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_TRACE", "1")  # ngram-mod logs its auto cap decisions only with LLAMA_TRACE
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    try:
        _, j = _run(["--spec-rs-depth", "2", "--spec-ngram-auto"])
        with open(server.log_path, errors="replace") as f:
            log = f.read()
    finally:
        os.remove(server.log_path)

    assert j["n_prebatch_restore"] > 0, j

    re_auto = re.compile(r"ngram-mod auto: seq 0 offered (\d+) accepted (\d+) built \d+(?: \(cap bound\))? -> cap (\d+) -> (\d+)")
    re_step = re.compile(r"spec step: impl=ngram-mod built=\d+ offered=(\d+) accepted=(\d+)")

    auto = [tuple(int(x) for x in m.groups()) for m in re_auto.finditer(log)]
    step = [tuple(int(x) for x in m.groups()) for m in re_step.finditer(log)]
    assert len(auto) > 0
    assert [(o, a) for o, a, _, _ in auto] == step, "auto cap scored a different landing than the verify step"

    # n_min 2 (min of --spec-ngram-mod-n-min and 12), ceiling 16 (--spec-ngram-mod-n-max / --spec-ngram-draft-n-max)
    n_deep = 0
    for o, a, _, cap in auto:
        if a < o:
            assert cap == _auto_round(a + 4, 2, 16), (o, a, cap)
            n_deep += 1 if o - a > 2 else 0
    assert n_deep > 0, "no landing past the --spec-rs-depth window"

    assert "rolling back to the pre-ubatch plane" in log
    assert "ngram policy: auto on" in log
    assert j["ngram_policy"]["cap_max"] < 16, j["ngram_policy"]
