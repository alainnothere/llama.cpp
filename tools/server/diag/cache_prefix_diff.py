#!/usr/bin/env python3
"""
cache_prefix_diff.py - given two POST-body dumps from --http-request-dump-path,
apply the model's chat template (offline, via Jinja2) and report the first byte
where the two rendered prompts diverge.

We render OFFLINE so this works while the server is busy answering live
privibe requests — no /apply-template HTTP call needed.

Use this to identify *why* the prompt cache hits with f_keep < 1.0 across two
consecutive requests for the same conversation. The dump filenames sort by
sequence number; pass them in chronological order.

Usage:
  cache_prefix_diff.py <dump1.json> <dump2.json> [--template PATH]
  cache_prefix_diff.py --all <dump_dir>                  # all consecutive pairs
  cache_prefix_diff.py --all <dump_dir> --by-system-len  # only same-terminal pairs

The --template flag points at the Jinja2 chat template saved earlier from
/props (we ran: curl ...:8089/props | jq .chat_template > /tmp/chat_template.j2)
Default: /tmp/chat_template.j2
"""
import argparse, json, os, glob, sys
from jinja2 import Environment, BaseLoader
from jinja2.exceptions import TemplateError


def raise_exception(msg):
    raise TemplateError(msg)


def make_env(template_text):
    env = Environment(
        loader=BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=False,
        keep_trailing_newline=False,
    )
    env.globals["raise_exception"] = raise_exception
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env.from_string(template_text)


def apply_template(tpl, messages, tools=None, add_generation_prompt=False):
    ctx = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "tools": tools or [],
        "add_vision_id": False,
        # Qwen3.5 template gates the <think> block on this:
        "enable_thinking": True,
    }
    return tpl.render(**ctx)


def load_dump(path):
    return json.load(open(path))


def diff_one(tpl, path_a, path_b):
    a = load_dump(path_a)
    b = load_dump(path_b)
    msgs_a = a.get("messages", [])
    msgs_b = b.get("messages", [])

    # A's prompt: full input + generation-prompt tail.
    # The slot's cached state after A also includes the model's generated
    # response tokens — we cannot reproduce those without the model.
    # So we compare the *input* prefix only; the divergence we are hunting
    # is always within the input prefix (otherwise f_keep would be ~1.0).
    try:
        p_a = apply_template(tpl, msgs_a, tools=a.get("tools"),
                              add_generation_prompt=True)
    except TemplateError as e:
        print(f"  template error rendering A: {e}")
        return

    # B's prefix: B's first len(msgs_a) messages, no gen-prompt — this is
    # what the cache will try to match against A's cached input prefix.
    n = len(msgs_a)
    if len(msgs_b) < n:
        print(f"--- {os.path.basename(path_a)} vs {os.path.basename(path_b)} ---")
        print(f"  SKIP: B has fewer messages than A ({len(msgs_b)} < {n}).")
        return
    try:
        p_b_prefix = apply_template(tpl, msgs_b[:n], tools=b.get("tools"),
                                     add_generation_prompt=False)
    except TemplateError as e:
        print(f"  template error rendering B prefix: {e}")
        return

    print(f"--- {os.path.basename(path_a)}  vs  {os.path.basename(path_b)} ---")
    print(f"  len(A input + genprompt)        = {len(p_a)}")
    print(f"  len(B's first {n} msgs)          = {len(p_b_prefix)}")

    mn = min(len(p_a), len(p_b_prefix))
    first = mn
    for i in range(mn):
        if p_a[i] != p_b_prefix[i]:
            first = i
            break

    if first == mn and len(p_a) == len(p_b_prefix):
        print(f"  PREFIX IDENTICAL ({mn} chars). cache should hit at full f_keep.")
        return
    if first == mn:
        print(f"  one is a prefix of the other (common len {mn}). diverges at end.")
    else:
        print(f"  first divergence at char {first}/{mn} ({100*first/max(1,mn):.2f}%)")
    pre, post = 80, 250
    lo = max(0, first - pre)
    hi_a = min(len(p_a), first + post)
    hi_b = min(len(p_b_prefix), first + post)
    print(f"  A [{lo}:{hi_a}]:")
    print(f"    {p_a[lo:hi_a]!r}")
    print(f"  B [{lo}:{hi_b}]:")
    print(f"    {p_b_prefix[lo:hi_b]!r}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="/tmp/chat_template.j2")
    ap.add_argument("--all", metavar="DIR",
                    help="run pairwise over all dumps in DIR (chronological)")
    ap.add_argument("--by-system-len", action="store_true",
                    help="group by system-prompt length so only same-terminal pairs are compared")
    ap.add_argument("files", nargs="*", help="two dump JSON files (older, newer)")
    args = ap.parse_args()

    tpl_text = open(args.template).read()
    tpl = make_env(tpl_text)

    if args.all:
        files = sorted(glob.glob(os.path.join(args.all, "*.json")))
        if args.by_system_len:
            groups = {}
            for f in files:
                d = json.load(open(f))
                msgs = d.get("messages", [])
                key = (len(msgs[0].get("content", ""))
                       if msgs and msgs[0].get("role") == "system" else -1)
                groups.setdefault(key, []).append(f)
            for key, fs in sorted(groups.items()):
                print(f"\n========== group sys_len={key} ({len(fs)} files) ==========")
                for a, b in zip(fs, fs[1:]):
                    diff_one(tpl, a, b)
        else:
            for a, b in zip(files, files[1:]):
                diff_one(tpl, a, b)
        return

    if len(args.files) != 2:
        ap.error("provide two files, or use --all DIR")
    diff_one(tpl, args.files[0], args.files[1])


if __name__ == "__main__":
    main()
