import json
import os
import tempfile
import pytest
from utils import *

# 2026-10-20: end-to-end check of the ngram-mod chain policies --spec-ngram-auto and --spec-ngram-resume N
# (the policy logic itself is unit-tested in tests/test-ngram-policy.cpp). greedy verification means a draft can
# never change the output, so every arm must produce the same text as the base run, and the per-request
# --spec-stats-file line must carry the config fields spec_stats.py groups on plus the "ngram_policy" object

REPETITIVE_PROMPT = "the little cat ran and ran and the little dog sat and sat " * 8

NGRAM_ARGS = [
    "--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-min", "4", "--spec-ngram-mod-n-max", "32",
    "--spec-ngram-draft-n-max", "32",
]

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 512
    # the preset's -b 32 cannot hold a 32-token draft + the sampled token: the server asserts in add_sampled_tokens
    # ("batch must be large enough"), pre-existing for any draft >= n_batch, independent of these policies
    server.n_batch = 64
    server.spec_type = "ngram-mod"


def _run(extra: list[str]):
    global server
    fd, stats = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    server.extra_args = NGRAM_ARGS + ["--spec-stats-file", stats] + extra
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": REPETITIVE_PROMPT,
        "n_predict": 96,
        "temperature": 0.0,
        "top_k": 1,
    })
    assert res.status_code == 200
    server.stop()
    with open(stats) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    os.unlink(stats)
    assert len(lines) == 1
    return res.body["content"], lines[0]


def test_ngram_policies_keep_output_and_report():
    base, j0 = _run([])
    assert j0["ngram_auto"] is False and j0["ngram_resume"] == 0
    assert j0["ngram_policy"] is None
    assert j0["n_draft_tokens"] > 0

    for extra, auto, resume in [
        (["--spec-ngram-auto"], True, 0),
        (["--spec-ngram-resume", "1"], False, 1),
        (["--spec-ngram-auto", "--spec-ngram-resume", "1"], True, 1),
    ]:
        content, j = _run(extra)
        assert content == base, f"output changed with {extra}"
        assert j["ngram_auto"] is auto and j["ngram_resume"] == resume
        np = j["ngram_policy"]
        assert np is not None and np["auto"] is auto and np["resume_n"] == resume
        if auto:
            assert np["cap_n"] > 0
            assert 4 <= np["cap_min"] <= np["cap_median"] <= np["cap_max"] <= 32
        assert np["resume_tok_land"] <= np["resume_tok_off"]
