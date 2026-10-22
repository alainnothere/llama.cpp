import os
import re
import shutil
import tempfile
import pytest
from utils import *

# Recurrent-tail rewind on a hybrid model.
#
# A follow-up turn normally shares every cached token except the last one (the
# stop token of the previous answer) and then appends new tokens. On a hybrid /
# recurrent model the recurrent state lives only at the tail, so dropping that
# one token needs a checkpoint restore: pos_min == pos_next means the tail IS the
# first token to remove. Upstream #24110 skips the checkpoint search whenever new
# tokens follow (correct for a SWA window, wrong here), after which seq_rm refuses
# and the whole prompt is re-processed - on every turn.
#
# LFM2-test-ci-80M is a hybrid (conv-recurrent + attention) model with no
# per-token rollback budget in this configuration, so the rewind must come from a
# checkpoint.

server = ServerPreset.tinyllama2()

_disk_dir: str = ""

DELIMITERS = [
    {"role": "user",      "delimiter": "<|im_start|>user\n"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant\n"},
]


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


def _conversation(n_turns: int, per_turn: int = 6) -> str:
    out = ""
    for t in range(n_turns):
        out += "<|im_start|>user\n"      + _sentences(f"u{t}", per_turn) + "<|im_end|>\n"
        out += "<|im_start|>assistant\n" + _sentences(f"a{t}", per_turn) + "<|im_end|>\n"
    return out


def _ask(history: str, tag: str, per_turn: int = 6) -> str:
    return (history + "<|im_start|>user\n" + _sentences(tag, per_turn)
            + "<|im_end|>\n<|im_start|>assistant\n")


def _tokens(text: str) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return res.body["tokens"]


def _completion(prompt: str, n_predict: int) -> ServerResponse:
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "temperature": 0.0,
        "return_tokens": True,
        "message_delimiters": DELIMITERS,
    })


@pytest.fixture(autouse=True)
def create_server():
    global server, _disk_dir
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
    server.debug = True               # --verbose: the checkpoint logs are TRACE level
    server.ctx_checkpoints = 32
    server.checkpoint_min_step = 128  # several checkpoints inside a ~700 token prompt
    _disk_dir = tempfile.mkdtemp(prefix="llama-recurrent-rollback-test-")
    server.cache_disk_path = _disk_dir
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    shutil.rmtree(_disk_dir, ignore_errors=True)
    os.remove(server.log_path)


# the follow-up turn diverges exactly at the last cached token (the previous
# answer's single generated token) and appends new tokens: the recurrent tail
# must be rewound from a checkpoint, not thrown away
def test_follow_up_turn_rewinds_recurrent_tail():
    server.start()
    log = LogReader(server.log_path)

    first = _ask(_conversation(4), "q1")
    n_first = len(_tokens(first))
    assert n_first > 3 * server.checkpoint_min_step

    # two tokens: the first one is decoded (that is what moves the recurrent tail
    # onto it), the second is only sampled. n_predict=1 would leave the tail on
    # the last prompt token and the rewind below would be a no-op.
    res = _completion(first, n_predict=2)
    assert res.status_code == 200
    assert "created context checkpoint" in log.drain()

    gen = [t["id"] if isinstance(t, dict) else t for t in res.body["tokens"]]
    assert len(gen) == 2
    # the slot now caches n_first prompt tokens plus the two generated tokens, and
    # the recurrent state sits at position n_first (the first generated token)

    # the client renders the previous answer differently from what was sampled,
    # so the new prompt shares exactly the n_first tokens and continues from there
    for answer in ("Certainly.", "No."):
        second = _ask(first + answer + "<|im_end|>\n", "q2")
        toks = _tokens(second)
        assert toks[:n_first] == _tokens(first)
        if toks[n_first] != gen[0]:
            break
    else:
        pytest.skip("could not construct a prompt diverging at the generated token")

    res = _completion(second, n_predict=4)
    assert res.status_code == 200

    out = log.drain()
    assert "failed to truncate tokens" not in out, "recurrent tail was cleared instead of rewound"
    assert "restored context checkpoint" in out

    prompt_n = res.body["timings"]["prompt_n"]
    n_second = len(toks)
    # only the tokens after the newest checkpoint below the divergence are re-processed
    assert 0 < prompt_n < n_second // 2, f"prompt was re-processed: prompt_n = {prompt_n} of {n_second}"
    assert res.body["tokens_cached"] > n_first // 2
    # the rewind must have landed inside the first prompt, not at its end: the
    # divergence position itself is re-processed from the checkpoint
    assert res.body["timings"]["cache_n"] < n_first


def _tokens_raw(text: str, add_special: bool) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": add_special})
    assert res.status_code == 200
    return res.body["tokens"]


def _completion_tokens(tokens: list[int], n_predict: int) -> ServerResponse:
    return server.make_request("POST", "/completion", data={
        "prompt": tokens,
        "n_predict": n_predict,
        "cache_prompt": True,
        "temperature": 0.0,
        "return_tokens": True,
        "message_delimiters": DELIMITERS,
    })


def _follow_up_tokens(t1: list[int], gen: list[int], tag: str) -> list[int]:
    # the client echoes the previous answer token-exactly (prompt as token ids), so the
    # cached tokens are a strict prefix and the new turn starts right at the flushed position
    tail = "<|im_end|>\n<|im_start|>user\n" + _sentences(tag, 2) + "<|im_end|>\n<|im_start|>assistant\n"
    return t1 + gen + _tokens_raw(tail, add_special=False)


# [TAG_REC_REUSE] a short follow-up turn takes checkpoint A before its first prompt batch, at the
# position the previous turn ended. that is the state prompt_save() just read for the .rec file,
# so A must publish the .rec buffer instead of reading the recurrent state back a second time.
# LLAMA_SERVER_VERIFY_REC_REUSE makes the server read it back anyway and compare the bytes.
def test_follow_up_checkpoint_reuses_rec_snapshot(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_VERIFY_REC_REUSE", "1")
    # production runs --no-cache-idle-slots: the slot keeps its state between turns. with the
    # idle-slot clear the next turn is a disk restore, which rewrites the state (no reuse there)
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    t1 = _tokens_raw(_ask(_conversation(2), "q1"), add_special=True)
    res = _completion_tokens(t1, n_predict=4)
    assert res.status_code == 200
    gen = [t["id"] if isinstance(t, dict) else t for t in res.body["tokens"]]
    log.drain()

    t2 = _follow_up_tokens(t1, gen, "q2")
    res = _completion_tokens(t2, n_predict=4)
    assert res.status_code == 200
    out = log.drain()

    assert "reuses the .rec snapshot" in out, "checkpoint A read the recurrent state back from the device"
    assert "rec reuse verify: identical" in out
    assert "MISMATCH" not in out
    # the whole cached prefix was kept, only the new turn was processed
    assert res.body["timings"]["prompt_n"] < len(t2) - len(t1)


# a turn that diverges inside the cached prefix restores a checkpoint first: the state moved,
# so the stale .rec snapshot must not be published as the next checkpoint
def test_rec_snapshot_not_reused_after_state_change(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_VERIFY_REC_REUSE", "1")
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    t1 = _tokens_raw(_ask(_conversation(2), "q1"), add_special=True)
    res = _completion_tokens(t1, n_predict=4)
    assert res.status_code == 200
    gen = [t["id"] if isinstance(t, dict) else t for t in res.body["tokens"]]

    t2 = _follow_up_tokens(t1, gen, "q2")
    assert _completion_tokens(t2, n_predict=4).status_code == 200
    log.drain()

    # rewind into the first turn: diverge 40 tokens before its end
    t3 = t1[:-40] + _tokens_raw("<|im_start|>user\n" + _sentences("q3", 2) + "<|im_end|>\n<|im_start|>assistant\n", add_special=False)
    res = _completion_tokens(t3, n_predict=4)
    assert res.status_code == 200
    out = log.drain()

    assert "restored context checkpoint" in out
    assert "created context checkpoint" in out
    assert "MISMATCH" not in out
    assert "reuses the .rec snapshot" not in out


# [TAG_CP_INDEX] the cp_ files of a conversation are listed once per process; every later merge
# reads the writer-maintained index (no directory scan, no opens)
def test_checkpoint_index_seeded_once():
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    toks = _tokens_raw(_ask(_conversation(2), "q1"), add_special=True)
    for i in range(3):
        res = _completion_tokens(toks, n_predict=4)
        assert res.status_code == 200
        gen = [t["id"] if isinstance(t, dict) else t for t in res.body["tokens"]]
        toks = _follow_up_tokens(toks, gen, f"q{i + 2}")

    out = log.drain()
    seeded = re.findall(r"checkpoint index: seeded '(\w+)'", out)
    assert len(seeded) == len(set(seeded)) == 1, f"index seeded more than once: {seeded}"
    merges = [int(n) for n in re.findall(r"checkpoint merge: scanned (\d+)", out)]
    assert merges and max(merges) > 0
