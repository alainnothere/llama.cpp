import glob
import os
import re
import shutil
import struct
import tempfile
import time
import pytest
from utils import *

# Integration tests for rollback (rewind) into a context checkpoint.
#
# A rollback is a request whose prompt is a strict prefix of the slot's tokens
# plus a new tail (the user edits an earlier message). With a SWA model the KV
# window has already slid past the rewind point, so the slot must restore a
# context checkpoint or reprocess the whole prefix.
#
# The model is tinygemma3 (SWA window 4096), so prompts must be well over 4096
# tokens for the rewind to land behind the window. The prompts are raw text
# with gemma turn markers plus `message_delimiters`, so the server sees user
# message boundaries (that is what makes it checkpoint mid-conversation) and
# derives a conversation_id (that is what names the cp_ spill files).

server = ServerPreset.tinygemma3()

_disk_dir: str = ""

# header of a cp_ spill file: magic, version, arch hash, vocab hash, state hash,
# pos_min, pos_max, n_tokens, token_prefix_hash
CP_HEADER = "<IIIIIiiqQ"

DELIMITERS = [
    {"role": "user",      "delimiter": "<start_of_turn>user\n"},
    {"role": "assistant", "delimiter": "<start_of_turn>model\n"},
]


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


def _sentences(tag: str, n: int) -> str:
    return "".join(
        f"Entry {tag}-{i}: the archivist recorded a curious object, its provenance, "
        f"its weight in grams, and the name of the donor who left it behind. "
        for i in range(n)
    )


def _conversation(n_turns: int, per_turn: int = 8) -> str:
    # a finished user/model exchange is ~550 tokens
    out = ""
    for t in range(n_turns):
        out += "<start_of_turn>user\n"  + _sentences(f"u{t}", per_turn) + "<end_of_turn>\n"
        out += "<start_of_turn>model\n" + _sentences(f"a{t}", per_turn) + "<end_of_turn>\n"
    return out


def _ask(history: str, tag: str, per_turn: int = 8) -> str:
    return (history + "<start_of_turn>user\n" + _sentences(tag, per_turn)
            + "<end_of_turn>\n<start_of_turn>model\n")


def _reply(tag: str, per_turn: int = 8) -> str:
    return _sentences(tag, per_turn) + "<end_of_turn>\n"


def _n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def _completion(prompt: str) -> ServerResponse:
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 4,
        "cache_prompt": True,
        "temperature": 0.0,
        "message_delimiters": DELIMITERS,
    })


def _cp_files():
    # (pos_min, pos_max, n_tokens, path) of every checkpoint spill file
    out = []
    for path in glob.glob(os.path.join(_disk_dir, "cp_*.bin")):
        with open(path, "rb") as f:
            header = f.read(struct.calcsize(CP_HEADER))
        _, _, _, _, _, pos_min, pos_max, n_tokens, _ = struct.unpack(CP_HEADER, header)
        out.append((pos_min, pos_max, n_tokens, path))
    return out


def _cp_headers():
    # {path: (pos_min, n_tokens, token_prefix_hash)} - the hash identifies the branch,
    # so a file rewritten at the same pos_min by a later request is told apart
    out = {}
    for path in glob.glob(os.path.join(_disk_dir, "cp_*.bin")):
        with open(path, "rb") as f:
            header = f.read(struct.calcsize(CP_HEADER))
        if len(header) < struct.calcsize(CP_HEADER):
            continue
        _, _, _, _, _, pos_min, _, n_tokens, tok_hash = struct.unpack(CP_HEADER, header)
        out[path] = (pos_min, n_tokens, tok_hash)
    return out


@pytest.fixture(autouse=True)
def create_server():
    global server, _disk_dir
    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 512
    server.n_ubatch = 512
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.debug = True               # --verbose: the checkpoint logs are TRACE level
    server.ctx_checkpoints = 32
    server.checkpoint_spill_max = 64
    _disk_dir = tempfile.mkdtemp(prefix="llama-checkpoint-rollback-test-")
    server.cache_disk_path = _disk_dir
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    # the spill files are ~60 MiB each, do not leave them behind
    shutil.rmtree(_disk_dir, ignore_errors=True)
    os.remove(server.log_path)


# a rollback behind the SWA window restores a checkpoint instead of reprocessing the prefix
def test_rollback_restores_from_checkpoint():
    server.checkpoint_min_step = 512  # checkpoints spread over the conversation
    server.start()
    log = LogReader(server.log_path)

    first = _ask(_conversation(11), "q1")
    res = _completion(first)
    assert res.status_code == 200

    second = _ask(first + _reply("r1"), "q2")
    res = _completion(second)
    assert res.status_code == 200

    assert "created context checkpoint" in log.drain()

    # rollback: the first 9 turns are a strict prefix of the conversation above
    prefix = _conversation(9)
    n_prefix = _n_tokens(prefix)
    assert n_prefix > 4096  # the rewind must land behind the SWA window

    res = _completion(_ask(prefix, "q3"))
    assert res.status_code == 200

    out = log.drain()
    assert "restored context checkpoint" in out
    assert "forcing full prompt re-processing" not in out

    prompt_n = res.body["timings"]["prompt_n"]
    assert 0 < prompt_n < n_prefix // 4, f"prefix was reprocessed: prompt_n = {prompt_n}"


# a checkpoint whose spill file disappeared must not force a full reprocess
# while older, still readable checkpoints exist
def test_rollback_falls_back_when_file_missing():
    server.checkpoint_min_step = 512
    server.no_cache_idle_slots = True  # keep the slot's checkpoint list across requests
    server.start()
    log = LogReader(server.log_path)

    first = _ask(_conversation(11), "q1")
    assert _completion(first).status_code == 200

    second = _ask(first + _reply("r1"), "q2")
    assert _completion(second).status_code == 200

    prefix = _conversation(9)
    n_prefix = _n_tokens(prefix)

    # drop the newest checkpoint the rollback could use, keep the older ones
    usable = [cp for cp in _cp_files() if cp[1] <= n_prefix]
    assert len(usable) >= 2, f"expected several usable spill files, got {usable}"
    victim = max(usable)
    os.remove(victim[3])

    log.drain()

    res = _completion(_ask(prefix, "q3"))
    assert res.status_code == 200
    assert len(res.body["content"]) > 0

    out = log.drain()
    assert "falling back to an older checkpoint" in out

    prompt_n = res.body["timings"]["prompt_n"]
    if "restored context checkpoint" in out:
        # an older checkpoint took over: more tokens than the deleted one would
        # have needed, but far less than the whole prefix
        assert 0 < prompt_n < n_prefix // 2, f"unexpected prompt_n = {prompt_n}"
    else:
        assert "forcing full prompt re-processing" in out


# checkpoints thinned out of the slot's list are re-registered from disk at rollback time
def test_rollback_merges_disk_checkpoints():
    # default --checkpoint-min-step (8192): every checkpoint of a finished request is
    # thinned out of the RAM list by the next request, but its cp_ file stays on disk
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    history = _conversation(11)
    first = _ask(history, "q1")
    assert _completion(first).status_code == 200

    second = _ask(first + _reply("r1"), "q2")
    assert _completion(second).status_code == 200

    out = log.drain()
    assert "erasing context checkpoint too close to an earlier one" in out

    # rollback to the end of the same 11 turns, with a different question
    n_prefix = _n_tokens(history)
    assert n_prefix > 4096

    res = _completion(_ask(history, "q9"))
    assert res.status_code == 200

    out = log.drain()
    assert "restored context checkpoint" in out
    assert "forcing full prompt re-processing" not in out

    prompt_n = res.body["timings"]["prompt_n"]
    assert 0 < prompt_n < n_prefix // 4, f"prefix was reprocessed: prompt_n = {prompt_n}"


# spill files left by an abandoned branch share the conversation_id but not the
# token timeline - they must be rejected instead of restored into the new branch
def test_stale_timeline_checkpoint_rejected():
    server.checkpoint_min_step = 512
    # by default the rollback deletes the files of timeline A - keep them, this test is
    # about the merge-time binding check that runs when they are still on disk
    server.cache_multiverse = True
    server.start()
    log = LogReader(server.log_path)

    # timeline A: 11 turns
    assert _completion(_ask(_conversation(11), "qa1")).status_code == 200

    # timeline B: rewind to turn 8 and continue with different content past 6k tokens
    b1 = _ask(_conversation(8), "qb1")
    assert _completion(b1).status_code == 200

    b2 = _ask(b1 + _reply("rb1"), "qb2")
    assert _completion(b2).status_code == 200

    b3 = _ask(b2 + _reply("rb2"), "qb3")
    assert _completion(b3).status_code == 200

    assert _completion(_ask(b3 + _reply("rb3"), "qb4")).status_code == 200

    log.drain()

    # rollback on timeline B, into the range covered by timeline A's spill files
    res = _completion(_ask(b2 + _reply("rb2"), "qb9"))
    assert res.status_code == 200
    assert len(res.body["content"]) > 0

    out = log.drain()
    skipped = [int(n) for n in re.findall(r"checkpoint merge: .*tokens (\d+)", out)]
    assert skipped and max(skipped) > 0, "stale timeline files were not rejected"


# the common prefix runs a few tokens past the rollback prefix (the turn marker and the
# first words of the question are shared), so a file is only certainly on the abandoned
# branch when it covers this many tokens more
LCP_MARGIN = 64


def _rollback_to_prefix():
    # build a conversation past the SWA window, then roll back to its first 9 turns with
    # a different question. returns the spill files of the abandoned branch and of the
    # shared timeline, as they were before the rollback
    first = _ask(_conversation(11), "q1")
    assert _completion(first).status_code == 200

    second = _ask(first + _reply("r1"), "q2")
    assert _completion(second).status_code == 200

    prefix = _conversation(9)
    n_prefix = _n_tokens(prefix)
    assert n_prefix > 4096  # the rewind must land behind the SWA window

    before = _cp_headers()
    dead = {p: h for p, h in before.items() if h[1] > n_prefix + LCP_MARGIN}
    live = {p: h for p, h in before.items() if h[1] <= n_prefix}
    assert dead, f"no spill file past the divergence point: {sorted(before.values())}"
    assert live, f"no spill file below the divergence point: {sorted(before.values())}"

    res = _completion(_ask(prefix, "q3"))
    assert res.status_code == 200
    assert len(res.body["content"]) > 0

    return dead, live


def _still_there(files):
    # the subset still on disk with the same header (a later request can write a new
    # checkpoint under the same name - that is a different file)
    now = _cp_headers()
    return {p: h for p, h in files.items() if now.get(p) == h}


# by default a rollback deletes the spill files of the branch it abandons
def test_rollback_deletes_abandoned_branch_files():
    server.checkpoint_min_step = 512
    server.start()

    dead, live = _rollback_to_prefix()

    # the deletion happens while the request is served, the writer may still be busy
    left = _still_there(dead)
    for _ in range(20):
        if not left:
            break
        time.sleep(0.1)
        left = _still_there(dead)
    assert not left, f"abandoned branch files kept: {sorted(left.values())}"

    assert _still_there(live), "the shared timeline lost its checkpoints"


# --cache-multiverse keeps them
def test_multiverse_flag_keeps_branch_files():
    server.checkpoint_min_step = 512
    server.cache_multiverse = True
    server.start()

    dead, live = _rollback_to_prefix()

    assert _still_there(dead) == dead, "the abandoned branch was deleted despite --cache-multiverse"
    assert _still_there(live) == live
