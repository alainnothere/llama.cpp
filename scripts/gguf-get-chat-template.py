#!/usr/bin/env python3
"""Extract the chat template from a GGUF so it can be edited.

Usage:
    ggufGetTemplate.py MODEL.gguf [OUT.jinja]

Writes tokenizer.chat_template to OUT.jinja (default: MODEL.chat_template.jinja
next to the model). Also lists any variant templates
(tokenizer.chat_template.<name>, e.g. tool_use) and extracts them to
OUT.<name>.jinja. Read-only: never touches the GGUF.

Companion: gguf-set-chat-template.py writes a modified template back.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

from gguf import GGUFReader  # noqa: E402

KEY = "tokenizer.chat_template"


def field_str(reader: GGUFReader, key: str) -> str | None:
    field = reader.get_field(key)
    if field is None:
        return None
    return str(field.contents())


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2

    model = Path(sys.argv[1])
    if not model.is_file():
        print(f"error: no such file: {model}", file=sys.stderr)
        return 1

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else model.with_suffix(".chat_template.jinja")

    reader = GGUFReader(str(model), "r")

    main_tmpl = field_str(reader, KEY)
    if main_tmpl is None:
        print(f"error: {model.name} has no {KEY} metadata", file=sys.stderr)
        return 1
    out.write_text(main_tmpl, encoding="utf-8")
    print(f"{KEY}: {len(main_tmpl):,} chars -> {out}")

    # Variant templates are stored as chat_template.<name>, with an optional
    # tokenizer.chat_templates array listing the names.
    for key in sorted(reader.fields):
        if key.startswith(KEY + ".") :
            name = key[len(KEY) + 1:]
            variant = field_str(reader, key)
            if variant is None:
                continue
            vout = out.with_suffix(f".{name}.jinja")
            vout.write_text(variant, encoding="utf-8")
            print(f"{key}: {len(variant):,} chars -> {vout}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
