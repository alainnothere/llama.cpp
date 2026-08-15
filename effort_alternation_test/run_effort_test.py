#!/usr/bin/env python3
"""Per-message reasoning_effort alternation test for llama.cpp.

Runs a multi-turn conversation against a llama.cpp server, sending a
reasoning_effort value inside each user message object. Requires:

  - a llama-server build with the reasoning_effort message-field passthrough
    (disk-cache-eviction branch, common/chat.{h,cpp})
  - --chat-template-file pointing at a template that renders
    message.reasoning_effort (models/templates/Qwen3.8-27B-per-msg-effort.jinja)

Usage:
  ./run_effort_test.py                                  # localhost:8091, default scenario
  ./run_effort_test.py --url http://127.0.0.1:8089
  ./run_effort_test.py --scenario questions_backup_saga.json --out results/myrun.txt

Prints one stats line per turn; writes the full transcript (questions,
complete reasoning, complete answers) to the output file. The pass criteria
are judged by a human reading the transcript: reasoning volume must track
the per-turn effort value, and each answer must follow from the earlier
turns.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def ask(url, messages, scenario):
    effort = messages[-1]["reasoning_effort"]
    payload = {
        "model": "effort-test",
        "messages": messages,
        "temperature": scenario.get("temperature", 0.6),
        "max_tokens": scenario["max_tokens"][effort],
        "chat_template_kwargs": scenario.get("chat_template_kwargs", {}),
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Talk to the server directly, ignoring http_proxy/https_proxy env vars
    # (this machine routes through a Squid proxy that cannot reach localhost).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=3600) as res:
        data = json.load(res)
    msg = data["choices"][0]["message"]
    return msg, data.get("usage", {}), data["choices"][0].get("finish_reason")


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8091",
                    help="llama-server base URL (no trailing slash)")
    ap.add_argument("--scenario", default=str(here / "questions_backup_saga.json"),
                    help="scenario JSON with turns/efforts")
    ap.add_argument("--out", default=None,
                    help="transcript output path (default: results/<stamp>.txt)")
    args = ap.parse_args()

    scenario = json.loads(Path(args.scenario).read_text())
    out = Path(args.out) if args.out else (
        here / "results" / time.strftime("run-%y%m%d%H%M.txt")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    messages = []
    report = []
    for i, turn in enumerate(scenario["turns"], 1):
        messages.append({
            "role": "user",
            "content": turn["question"],
            "reasoning_effort": turn["effort"],
        })
        try:
            msg, usage, finish = ask(args.url, messages, scenario)
        except Exception as e:
            print(f"TURN{i} {turn['effort']}: FAILED: {e}", file=sys.stderr)
            break
        r = msg.get("reasoning_content") or ""
        c = msg.get("content") or ""
        messages.append({"role": "assistant", "content": c,
                         "reasoning_content": r})
        line = (f"TURN{i} {turn['effort']:6s}: reasoning_chars={len(r):6d} "
                f"content_chars={len(c):6d} "
                f"completion_tokens={usage.get('completion_tokens')} "
                f"prompt_tokens={usage.get('prompt_tokens')} "
                f"finish={finish}")
        print(line, flush=True)
        report.append((line, turn["question"], r, c))

    with open(out, "w") as f:
        f.write(f"url={args.url}\nscenario={args.scenario}\n"
                f"date={time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for line, question, r, c in report:
            f.write(f"{'=' * 70}\n{line}\n\nUSER: {question}\n\n"
                    f"REASONING:\n{r}\n\nANSWER:\n{c}\n\n")
    print(f"transcript: {out}")


if __name__ == "__main__":
    main()
