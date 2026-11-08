"""Bucket tagger for the MTP replay harness (examples/mtp-replay), 2026-11-07.

Reads raw int32 .tok streams (one conversation each, the disk-cache / tok_archive / mtp_dump format) and writes
<name>.buckets.json next to them (or into --out-dir): one label per token position for each of these dimensions

  role    system / user / assistant / tool / <other role word> / header / none
          header = <|im_start|> + the role word + its newline; none = the newline between <|im_end|> and the next
          <|im_start|>. assistant = exactly data.py assistant_mask(): <|im_start|>assistant\\n is header, the content
          and the closing <|im_end|> are assistant (eval_long's "asst" rows). --check-asst verifies the equality.
  think   think (between <think> and </think>, tags included) / answer (rest of an assistant turn) / - (non-assistant)
  kind    markup (control tokens: im_start/im_end/role header/think tags) / tool_call (<tool_call>..</tool_call>, tags
          included) / tool_response (<tool_response>..</tool_response>) / code (inside a ``` or ~~~ fence, fence lines
          included) / json (a line outside fences that starts with { [ } ] or "key":) / prose (everything else)
  lang    code language from the fence info string (lowercased, 'unknown' when empty) for code tokens, '-' otherwise
  script  cjk / latin / other (letters of another script) / neutral (no letters: digits, punctuation, whitespace,
          control tokens). Computed on the decoded UTF-8 text, so a CJK character split over two byte tokens tags both.
  asst    the assistant-only view production drafts on: (think_ if inside think) + kind, '-' for other roles.
          e.g. think_prose, think_code, prose, code, json, tool_call, markup.

Special ids (im_start/im_end/think/tool_call/tool_response) come from the model's gguf vocab by token text, not
hard-coded. Byte-level BPE pieces are decoded with the GPT-2 bytes_to_unicode table; control / user-defined tokens
are taken literally.

The replay attributes each scored chain step to the bucket of the token it predicts (the target t_{s+k}), the same
convention as eval_long.py (asst mask at r+1).

Sidecar format (runs keep a 288k-token conversation at a few hundred KB):
  {"version": 1, "source": ..., "n_tokens": N, "vocab": gguf path,
   "dims": {"role": {"labels": [...], "runs": [[start, end_excl, label_index], ...]}, ...}}
runs are sorted, contiguous and cover 0..N.

usage:
  ~/.venvs/mtp-train/bin/python tag_buckets.py [--out-dir DIR] [--check-asst] a.tok b.tok ...
  ~/.venvs/mtp-train/bin/python tag_buckets.py --show 200:400 a.tok      # print tokens with their tags
"""
import argparse, json, os, re, sys, time
import numpy as np

from gguf_import import gguf

MODEL = os.environ.get("MTP_MODEL", "/home/xlayn/ai/modelsOnOllamaToGguf/Qwen3.8-27B-UD-Q8_u.gguf")

TOKEN_TYPE_NORMAL = 1


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Vocab:
    def __init__(self, path=MODEL):
        t0 = time.time()
        r = gguf.GGUFReader(path)
        f = r.fields["tokenizer.ggml.tokens"]
        tt = r.fields["tokenizer.ggml.token_type"]
        self.path = path
        self.text = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        self.type = np.array([int(tt.parts[i][0]) for i in tt.data], dtype=np.int32)
        u2b = _bytes_to_unicode()
        self.bytes = []
        for s, ty in zip(self.text, self.type):
            if ty == TOKEN_TYPE_NORMAL:
                try:
                    self.bytes.append(bytes(u2b[c] for c in s))
                except KeyError:          # not a byte-level piece (should not happen for normal tokens)
                    self.bytes.append(s.encode("utf-8"))
            else:
                self.bytes.append(s.encode("utf-8"))
        ids = {s: i for i, s in enumerate(self.text) if self.type[i] != TOKEN_TYPE_NORMAL}
        need = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>"]
        missing = [s for s in need if s not in ids]
        if missing:
            raise SystemExit(f"vocab {path} lacks special tokens {missing}")
        self.sp = {s: ids[s] for s in need}
        self.nl = {i for i, b in enumerate(self.bytes) if self.type[i] == TOKEN_TYPE_NORMAL and b"\n" in b}
        print(f"[vocab] {len(self.text)} tokens from {path} in {time.time()-t0:.1f}s; special ids {self.sp}", file=sys.stderr)


# ---- script classes per character
def _script_of(cp):
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2FA1F or 0xF900 <= cp <= 0xFAFF or
            0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or
            0x3130 <= cp <= 0x318F or 0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF):
        return 3   # cjk (CJK punctuation and fullwidth forms included: they only occur in CJK text)
    if cp < 128:
        return 2 if (65 <= cp <= 90 or 97 <= cp <= 122) else 0
    if 0x00C0 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return 2 if cp not in (0xD7, 0xF7) else 0
    import unicodedata
    return 1 if unicodedata.category(chr(cp)).startswith("L") else 0   # other letters / neutral


_SCRIPT_CACHE = {}
SCRIPT_LABELS = ["neutral", "other", "latin", "cjk"]   # index = priority (higher wins inside one token)


def byte_scripts(buf: bytes):
    """per byte: script class of the character the byte belongs to (0 for invalid bytes)."""
    out = np.zeros(len(buf), dtype=np.int8)
    i, n = 0, len(buf)
    while i < n:
        b = buf[i]
        if b < 0x80:
            out[i] = 2 if (65 <= b <= 90 or 97 <= b <= 122) else 0
            i += 1
            continue
        ln = 2 if b >> 5 == 6 else 3 if b >> 4 == 14 else 4 if b >> 3 == 30 else 1
        try:
            cp = ord(buf[i:i + ln].decode("utf-8"))
        except (UnicodeDecodeError, TypeError):
            i += 1
            continue
        s = _SCRIPT_CACHE.get(cp)
        if s is None:
            s = _SCRIPT_CACHE[cp] = _script_of(cp)
        out[i:i + ln] = s
        i += ln
    return out


FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)")
JSONISH = re.compile(r'^[ \t]*([\[\]{}]|"[^"\n]{1,80}"[ \t]*:)')


def tag(tok, V: Vocab):
    N = len(tok)
    sp = V.sp
    is_norm = V.type[tok] == TOKEN_TYPE_NORMAL
    tb = [V.bytes[t] for t in tok]
    starts = np.zeros(N + 1, dtype=np.int64)
    np.cumsum([len(b) for b in tb], out=starts[1:])
    buf = b"".join(tb)
    bscript = byte_scripts(buf)

    role = ["none"] * N
    think = ["-"] * N
    kind = ["prose"] * N
    lang = ["-"] * N
    script = ["neutral"] * N

    # --- role + header: data.py assistant_mask semantics for assistant ('<|im_start|>' 'assistant' '\n' = header)
    i = 0
    cur_role = "none"
    while i < N:
        t = int(tok[i])
        if t == sp["<|im_start|>"]:
            j = i + 1
            while j < N and int(tok[j]) not in V.nl and is_norm[j]:
                j += 1
            hdr_end = min(j, N - 1)               # inclusive: the newline token
            name = b"".join(tb[i + 1:hdr_end]).decode("utf-8", "replace").strip() or "unknown"
            if hdr_end < N and int(tok[hdr_end]) in V.nl:
                nlb = tb[hdr_end]
                # a newline token that also carries content ("\n\n" or "\nfoo") is still the header's line end
                name = (b"".join(tb[i + 1:hdr_end]) + nlb.split(b"\n")[0]).decode("utf-8", "replace").strip() or name
            for k in range(i, hdr_end + 1):
                role[k] = "header"
                kind[k] = "markup"
            cur_role = name
            i = hdr_end + 1
            continue
        if t == sp["<|im_end|>"]:
            role[i] = cur_role
            kind[i] = "markup"
            cur_role = "none"
            i += 1
            continue
        role[i] = cur_role
        i += 1

    # --- think / tool_call / tool_response spans (tags included), reset at im boundaries
    in_think = in_call = in_resp = False
    for i in range(N):
        t = int(tok[i])
        if t in (sp["<|im_start|>"], sp["<|im_end|>"]):
            in_think = in_call = in_resp = False
            continue
        if t == sp["<think>"]:
            in_think = True
        if t == sp["<tool_call>"]:
            in_call = True
        if t == sp["<tool_response>"]:
            in_resp = True
        if role[i] == "assistant":
            think[i] = "think" if in_think else "answer"
        if in_call:
            kind[i] = "tool_call"
        elif in_resp:
            kind[i] = "tool_response"
        elif t in (sp["<think>"], sp["</think>"]) or not is_norm[i]:
            kind[i] = "markup"
        if t == sp["</think>"]:
            in_think = False
        if t == sp["</tool_call>"]:
            in_call = False
        if t == sp["</tool_response>"]:
            in_resp = False

    # --- code fences / json lines on the text of each run of normal tokens outside tool spans
    i = 0
    while i < N:
        if not is_norm[i] or kind[i] in ("tool_call", "tool_response", "markup"):
            i += 1
            continue
        j = i
        while j < N and is_norm[j] and kind[j] == "prose":
            j += 1
        seg = buf[starts[i]:starts[j]]
        base = int(starts[i])
        # per byte class: 0 prose, 1 json, 2 code; per byte lang id
        bcls = np.zeros(len(seg), dtype=np.int8)
        blang = {}
        fence = None        # (char, len, lang)
        pos = 0
        prev_cls = 0
        for line in seg.split(b"\n"):
            end = pos + len(line) + 1                     # include the newline
            text = line.decode("utf-8", "replace")
            if fence is not None:
                cls = 2
                m = FENCE.match(text)
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not m.group(2):
                    lg = fence[2]; fence = None           # closing fence line is code too
                else:
                    lg = fence[2]
            else:
                m = FENCE.match(text)
                if m:
                    fence = (m.group(1)[0], len(m.group(1)), (m.group(2) or "unknown").lower().strip(".{}"))
                    cls, lg = 2, fence[2]
                elif not text.strip():
                    cls, lg = prev_cls if prev_cls != 2 else 0, None
                elif JSONISH.match(text):
                    cls, lg = 1, None
                else:
                    cls, lg = 0, None
            bcls[pos:min(end, len(seg))] = cls
            if cls == 2:
                blang[(pos, min(end, len(seg)))] = lg
            prev_cls = cls
            pos = end
        lang_starts = sorted(blang)
        for k in range(i, j):
            a, b = int(starts[k]) - base, int(starts[k + 1]) - base
            if b <= a:
                continue
            seg_tok = seg[a:b]
            nz = len(seg_tok) - len(seg_tok.lstrip())     # classify by the first non-whitespace byte
            p = a + (nz if nz < len(seg_tok) else 0)
            c = int(bcls[p])
            kind[k] = ("prose", "json", "code")[c]
            if c == 2:
                # find the line range holding byte p
                lo, hi = 0, len(lang_starts) - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if lang_starts[mid][0] <= p:
                        lo = mid
                    else:
                        hi = mid - 1
                lang[k] = blang[lang_starts[lo]] or "unknown"
        i = j

    # --- script
    for k in range(N):
        if not is_norm[k]:
            continue
        a, b = int(starts[k]), int(starts[k + 1])
        if b > a:
            script[k] = SCRIPT_LABELS[int(bscript[a:b].max())]

    asst = ["-"] * N
    for k in range(N):
        if role[k] == "assistant":
            asst[k] = ("think_" if think[k] == "think" else "") + kind[k]

    return {"role": role, "think": think, "kind": kind, "lang": lang, "script": script, "asst": asst}


def to_runs(labels):
    lab, idx, runs = [], {}, []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            l = labels[s]
            if l not in idx:
                idx[l] = len(lab); lab.append(l)
            runs.append([s, i, idx[l]])
            s = i
    return {"labels": lab, "runs": runs}


def assistant_mask_eval_long(tok, V):
    """copy of data.py assistant_mask with the ids from the vocab (the check reference)."""
    IM_START, IM_END = V.sp["<|im_start|>"], V.sp["<|im_end|>"]
    ASSISTANT = V.text.index("assistant")
    m = np.zeros(len(tok), dtype=bool)
    st = np.nonzero((tok[:-1] == IM_START) & (tok[1:] == ASSISTANT))[0]
    ends = np.nonzero(tok == IM_END)[0]
    for a in st:
        b = ends[np.searchsorted(ends, a)] if np.searchsorted(ends, a) < len(ends) else len(tok) - 1
        m[a + 3:b + 1] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("toks", nargs="+")
    ap.add_argument("--out-dir", help="default: next to each .tok")
    ap.add_argument("--vocab", default=MODEL, help="gguf whose tokenizer the .tok ids belong to")
    ap.add_argument("--check-asst", action="store_true", help="assert role==assistant equals eval_long's mask")
    ap.add_argument("--show", help="A:B print tokens A..B-1 with their tags (first file only, writes nothing)")
    ap.add_argument("--force", action="store_true", help="rewrite existing sidecars")
    a = ap.parse_args()
    V = Vocab(a.vocab)
    if a.show:
        lo, hi = (int(x) for x in a.show.split(":"))
        tok = np.fromfile(a.toks[0], dtype=np.int32)
        T = tag(tok, V)
        for i in range(lo, min(hi, len(tok))):
            print(f"{i:7d} {tok[i]:7d} {V.bytes[tok[i]]!r:28s} " + " ".join(f"{d}={T[d][i]}" for d in T))
        return
    tot = {}
    for p in a.toks:
        name = os.path.basename(p)[:-4] if p.endswith(".tok") else os.path.basename(p)
        out = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(p)), name + ".buckets.json")
        if os.path.exists(out) and not a.force and not a.check_asst:
            print(f"skip {out} (exists)", file=sys.stderr)
            continue
        t0 = time.time()
        tok = np.fromfile(p, dtype=np.int32)
        T = tag(tok, V)
        if a.check_asst:
            ref = assistant_mask_eval_long(tok, V)
            mine = np.array([r == "assistant" for r in T["role"]])
            bad = np.nonzero(ref != mine)[0]
            print(f"check-asst {name}: {len(bad)} mismatches of {len(tok)} (asst {int(ref.sum())})"
                  + (f" first {bad[:10].tolist()}" if len(bad) else ""), file=sys.stderr)
        doc = {"version": 1, "source": os.path.abspath(p), "n_tokens": int(len(tok)), "vocab": V.path,
               "special_ids": V.sp, "dims": {d: to_runs(v) for d, v in T.items()}}
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, out)
        for d in ("kind", "asst", "script"):
            for l in T[d]:
                tot[(d, l)] = tot.get((d, l), 0) + 1
        print(f"wrote {out} ({len(tok)} tokens, {time.time()-t0:.1f}s)", file=sys.stderr)
    if tot:
        for (d, l), n in sorted(tot.items()):
            print(f"  {d:6s} {l:20s} {n:9d}", file=sys.stderr)


if __name__ == "__main__":
    main()
