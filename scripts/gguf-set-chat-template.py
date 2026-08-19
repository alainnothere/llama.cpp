#!/usr/bin/env python3
"""Write a (modified) chat template back into a GGUF.

Usage:
    ggufSetTemplate.py MODEL.gguf TEMPLATE.jinja [OUT.gguf]

GGUF metadata is length-prefixed at the front of the file, so changing the
template means rewriting the whole model (this is a full copy, done via
gguf-py's gguf_new_metadata). OUT.gguf defaults to MODEL.newtemplate.gguf
next to the input; after verifying the result with llama-server, replace the
original yourself. Refuses to overwrite the input in place - a truncated
20GB model is not a lesson anyone needs twice.

NOTE: a new template = new rendered prompt bytes = every KV-cache prefix for
this model dies once. You know this. It is written here anyway.

Companion: gguf-get-chat-template.py extracts the template for editing.
"""

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "gguf-py/gguf/scripts/gguf_new_metadata.py"
)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2

    model = Path(sys.argv[1])
    template = Path(sys.argv[2])
    out = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else model.with_suffix(".newtemplate.gguf")
    )

    for f, what in ((model, "model"), (template, "template")):
        if not f.is_file():
            print(f"error: no such {what}: {f}", file=sys.stderr)
            return 1
    if out.resolve() == model.resolve():
        print("error: refusing to overwrite the input in place. Write to a new "
              "file, verify it loads, then replace the original.", file=sys.stderr)
        return 1
    if out.exists():
        print(f"error: {out} already exists; delete it first.", file=sys.stderr)
        return 1

    tmpl_text = template.read_text(encoding="utf-8")
    if not tmpl_text.strip():
        print(f"error: {template} is empty.", file=sys.stderr)
        return 1

    free = shutil.disk_usage(out.parent).free
    need = model.stat().st_size
    if free < need * 1.02:
        print(f"error: {out.parent} has {free / 1e9:.1f}GB free but the copy "
              f"needs ~{need / 1e9:.1f}GB.", file=sys.stderr)
        return 1

    print(f"{model.name} ({need / 1e9:.1f}GB) + {template.name} "
          f"({len(tmpl_text):,} chars) -> {out.name}")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            str(model), str(out),
            "--chat-template-file", str(template),
            "--force",
        ],
        check=False,
    )
    if result.returncode != 0:
        if out.exists():
            out.unlink()
            print(f"failed; removed partial {out}", file=sys.stderr)
        return result.returncode

    print(f"done. Verify with llama-server, then replace {model.name} yourself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
