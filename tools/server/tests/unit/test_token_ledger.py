import json
import pytest
from utils import *

server = ServerPreset.tinyllama2()

PROMPT = "I believe the meaning of life is to find your gift and give it away to the world"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1  # both requests land in the same slot, so the second one reuses the prefix
    server.server_metrics = True


def start_with_ledger(path):
    server.extra_args = ["--token-ledger", str(path)]
    server.start()


def complete(prompt, n_predict):
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": prompt,
        "max_tokens": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    u = res.body["usage"]
    return {
        "prompt": u["prompt_tokens"],
        "cached": u["prompt_tokens_details"]["cached_tokens"],
        "completion": u["completion_tokens"],
    }


def read_ledger(path):
    with open(path) as f:
        return json.load(f)


def check_totals(t, usages):
    assert t["requests"]          == len(usages)
    assert t["prompt_tokens"]     == sum(u["prompt"] for u in usages)
    assert t["cached_tokens"]     == sum(u["cached"] for u in usages)
    assert t["completion_tokens"] == sum(u["completion"] for u in usages)
    assert t["total_tokens"]      == t["prompt_tokens"] + t["completion_tokens"]
    assert t["processed_tokens"]  <= t["prompt_tokens"]
    assert t["reasoning_tokens"]  == 0  # no chat template with reasoning tags here


def test_token_ledger_persists(tmp_path):
    path = tmp_path / "ledger.json"

    start_with_ledger(path)
    u1 = complete(PROMPT, 8)
    u2 = complete(PROMPT + " and to be happy", 8)
    assert u1["cached"] == 0
    assert u2["cached"] > 0

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    check_totals(res.body["token_ledger"]["totals"], [u1, u2])

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert f"llamacpp:ledger_requests_total 2" in res.body
    assert f"llamacpp:ledger_prompt_tokens_cached_total {u2['cached']}" in res.body

    server.stop()

    j = read_ledger(path)
    assert j["version"] == 1
    check_totals(j["totals"], [u1, u2])
    assert len(j["days"]) == 1
    check_totals(next(iter(j["days"].values())), [u1, u2])
    since = j["since_unix"]

    # restart: totals carry over and grow
    start_with_ledger(path)
    u3 = complete(PROMPT, 4)
    server.stop()

    j = read_ledger(path)
    assert j["since_unix"] == since
    check_totals(j["totals"], [u1, u2, u3])


def test_token_ledger_corrupt_file(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{ not json")

    start_with_ledger(path)
    u1 = complete(PROMPT, 4)
    server.stop()

    check_totals(read_ledger(path)["totals"], [u1])
    assert len(list(tmp_path.glob("ledger.json.corrupt-*"))) == 1


@pytest.mark.parametrize("enable_thinking,all_reasoning", [(True, True), (False, False)])
def test_token_ledger_reasoning(tmp_path, enable_thinking, all_reasoning):
    # the template opens "<think>\n" in the generation prompt and the test model never closes it,
    # so with thinking on every generated token is reasoning; with thinking off the prompt closes the block
    path = tmp_path / "ledger.json"
    server.jinja = True
    server.chat_template_file = "../../../models/templates/Qwen3.5-4B.jinja"
    start_with_ledger(path)
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    })
    assert res.status_code == 200
    n_gen = res.body["usage"]["completion_tokens"]
    assert n_gen > 0
    server.stop()

    t = read_ledger(path)["totals"]
    assert t["completion_tokens"] == n_gen
    assert t["reasoning_tokens"] == (n_gen if all_reasoning else 0)
