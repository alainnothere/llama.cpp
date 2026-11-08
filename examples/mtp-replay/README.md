# llama-mtp-replay

Offline, head-only replay of the NextN/MTP draft chain (Qwen3.8 `blk.64`) against target hidden states recorded
from real conversations. No llama-server, no target context, no trunk layers: the target's `h` rows come from an
`llama-mtp-dump` dump, so a drafter variant (a `--circuit`, or a fine-tuned head via `--mtp-head`) is scored per
token bucket and per chain depth in minutes on the CPU.

## Pipeline

1. **Dump** the target hidden states of conversations (needs the GPUs or a lot of CPU time, the trunk runs once per
   token). The production disk cache sweep archives every conversation as a raw int32 `.tok` in `~/ai/tok_archive`.

   ```
   llama-mtp-dump -m Qwen3.8-27B-UD-Q8_u.gguf -c 290000 -b 8192 -ub 512 -ctk f16 -ctv f16 \
       --out-dir /home/xlayn/theOtherDocs/mtp_dump/dump [--truncate] ~/ai/tok_archive/NAME.tok ...
   ```

   Writes `NAME.h.f16` (`[n_tokens, n_embd]` f16, row i = h_i after `output_norm`), `NAME.tok`, `NAME.meta.json`.
   Use f16 KV: q8_0 KV makes the h differ from what an f16-KV target computes. The contract (pairing, positions,
   rope, chain h) is `HEAD_CONTRACT.md` in the dump dir.

2. **Tag** token buckets (Python, no torch needed):

   ```
   ~/.venvs/mtp-train/bin/python /home/xlayn/ai/claude_scratch/mtp_train/tag_buckets.py \
       --out-dir /home/xlayn/theOtherDocs/mtp_dump/buckets [--check-asst] /home/xlayn/theOtherDocs/mtp_dump/dump/*.tok
   ```

   One label per position for each dimension: `role` (system / user / assistant / header / none; assistant equals
   eval_long's assistant mask exactly, `--check-asst` asserts it), `think` (think / answer), `kind` (markup /
   tool_call / tool_response / code / json / prose), `lang` (code fence language), `script` (latin / cjk / other /
   neutral), `asst` (assistant-only view: `think_prose`, `think_code`, `prose`, `code`, `tool_call`, ...). Special ids
   come from the gguf vocab. `--show A:B file.tok` prints tokens with their tags. 47 conversations tag in ~15 s.

3. **Replay**:

   ```
   llama-mtp-replay -m Qwen3.8-27B-UD-Q8_u.gguf -t 10 \
       --circuit M0 [--circuit M0*2 ...] --depth 8 --stride 8 [--bucket-filter role:assistant] \
       [--vocab-subset ~/ai/mtp-vocab-top32k.txt [--vocab-subset-forced]] [--mtp-head head.gguf] \
       [--max-rows N] [-c N] --out res.json /home/xlayn/theOtherDocs/mtp_dump/dump/NAME ...
   ```

   Inputs are dump base paths (`DIR/NAME`, `.tok` / `.h.f16` suffixes are stripped). The sidecar is looked up in
   `--buckets-dir`, then `DIR/../buckets/`, then next to the `.tok`.

## Circuits

`--circuit` takes the pass-list grammar of `llama_mtp_set_circuit` (src/llama-ext.h, `M0`, `M0*2`, `M0,M0:b0`,
`T63,M0`, ...). Each circuit gets its own draft context, sized with `mtp_circuit_max_passes` = its own-K/V pass count
(parsed with `llama_mtp_circuit_parse`); the target head and the inputs are shared. CMake defines
`MTP_REPLAY_CIRCUIT_ENGINE` when `src/llama-mtp-circuit.h` exists; without the engine the tool still builds (weak
symbol) and runs `M0` only. Checked 2026-11-07: `M0` through the engine gives the same aggregates bit for bit as a
build without it.

## What it computes

Per conversation and circuit, one `LLAMA_CONTEXT_TYPE_MTP` context, built like draft-mtp's `ctx_dft` (unified KV,
`n_par + 1` sequences, KV = the whole conversation):

- catch-up prefill of MTP rows `p = 0..n` with token `t_p`, `h_{p-1}` (zeros at `p = 0`), exactly draft-mtp
  `process()`. A start row `s` is an output: its logits are depth 1 (predicts `t_{s+1}`), its MTP `h_nextn` seeds
  depth 2.
- per start `s`: seq `j` shares seq 0's cells `[0, s]`, then depths `2..D` decode one row per chain at `pos s+k-1`,
  token `t_{s+k-1}`, `h` = the head's own `h_nextn` of the previous chain row (the growing-KV branch of `draft()`).
  `--n-par` chains (default 64) share each decode.
- the chain is **teacher-forced** on the recorded tokens, like `eval_long.py`: depth `k` is conditional on depths
  `1..k-1` having been accepted.

Metrics, per (circuit, bucket of the **predicted** token `t_{s+k}`, depth k), means over n:

| field | meaning |
|---|---|
| `top1` | head argmax == recorded token |
| `top10` | recorded token in the head's top 10 |
| `p_rec` | q(recorded token), q = softmax(head top-10 / `--temp`) |
| `acc_prob` | sum over the head's top 10 of min(q, p), p = softmax(output.weight @ h_target / `--temp`) over the full vocab. eval_long.py's definition (temp 1): the expected acceptance of the probabilistic verifier |
| `acc2` | the same at production-like sampler settings (`--acc2 0.2,20,0.95` = temp, target top-k, target top-p; `off` disables) |
| `tgt_top1` | target argmax == recorded token (how predictable the position is for the target itself) |
| `alive` | top1 at every depth 1..k of this chain; sum over k = greedy accepted length (`Eg`) |
| `cum`, `cum2` | product of acc_prob (acc2) over 1..k; sum over k = probabilistic accepted length proxy (`Ep`, `Ep2`) |

The table prints rows = circuit x bucket (buckets with n >= `--min-n`, default 200), columns = depth: `top1 acc_prob`,
then `sumA` (sum of acc_prob over depths, the October overnight gate), `Eg`, `Ep`, `sumA2`, `Ep2`. The JSON (`--out`)
has every field, n per depth, the config and the timing split.

The target probabilities need `output.weight @ h` for every scored row: it runs as a plain ggml CPU graph on the
mmapped gguf (not through libllama) and is shared by all circuits of a run. `--no-target` skips it (top1 / alive only,
about 2x faster on the CPU).

## Correctness gate (2026-11-07)

On eval_long.py's held-out split (3 conversations, first 32768 rows each), circuit M0 with `--vocab-subset` (the plain
32k file = eval_long's `Ws`), `--stride 1 --depth 4` matches eval_long's `top1_sub` / `acc_prob` per depth: 98304 rows,
max |d top1| 0.0004 (all rows) / 0.0013 (6160 assistant rows), max |d acc_prob| 0.0002. Full table in
contextFiles/agent-reports/2026-11-07-mtp-replay-harness.md; JSONs in /home/xlayn/theOtherDocs/mtp_dump/replay/gate_*.
Two known, intended differences:

- eval_long also counts the depth >= 2 chain seeded at the zero-h row 0 (one item per conversation and depth);
  production never seeds there, the replay does not either.
- eval_long runs the head in bf16 with a dequantized output.weight; the replay runs the gguf's Q8_0 weights through
  ggml (Q8_0 activations on the CPU). Near-ties flip in both directions: ~1e-3 on top1.

`--vocab-subset` without `--vocab-subset-forced` is the eval_long subset; with it, the draft-mtp `vs_init` set
(file + control / EOG / digit / newline pieces). Production also adds each request's prompt ids; the replay does not
(that would leak ids of later turns).

## Memory and speed (CPU, 7900X, `-t 10`)

- The loader maps the whole 28 GB gguf (trunk included, never touched on the CPU unless a `T` circuit pass runs it).
  Resident: output.weight (2.6 GB) + the used token_embd rows + blk.64 (~0.45 GB) + per circuit the MTP KV
  (4 KiB per cell and own pass at f16) + the logits of the outputs of one prefill batch (1 MiB per output row).
- Weight repacking is off by default (`--repack` turns it on): a repacked CPU head refuses `llama_set_logits_subset`.
- Measured 2026-11-07: RSS 31 GB of which 29.7 GB is the file-backed gguf map (reclaimable page cache), 1.3 GB anon.
  1000 starts x depth 8 on a 21k-token conversation (stride 8): 34 s with the 32k subset head, 46 s full vocab,
  plus ~12 s model load from a warm page cache. Gate (98k starts x depth 4 at up to 32k context): 22 min, of which
  chain decode 825 s (2.8 ms per chain row at ~16k average context), prefill 171 s, target head 206 s.
- The prefill cost grows with the conversation (attention over all earlier rows): replay long conversations on the
  GPU, or cap them with `--max-rows` / `-c` on the CPU (a capped prefix is exact, h is causal).

## Reading the table: what a win looks like

- Compare circuits **on the same run** (same positions): the differences are paired, so a 0.003 shift on 5k items is
  real even though the per-bucket standard error is larger.
- The number that tracks production tokens/step is `Ep2` on `role:assistant` (the draft only runs while the model
  generates); `sumA` is the October gate (eval_long), keep reporting it for continuity.
- A win: `Ep2` (or `sumA`) up on `role:assistant` and on the big assistant buckets (`asst:think_prose`,
  `asst:tool_call`, `asst:prose`, `asst:think_code`, `asst:code`) with no bucket of n >= 2000 down by more than 0.005 at
  any depth. A circuit that only lifts depth 1 and costs depth 3+ loses at `--spec-draft-n-max 11`.
- `tgt_top1` caps what any drafter can do on a bucket: low `tgt_top1` (tool_response, user text) is the target's own
  uncertainty, not a drafter weakness.

## Feeding the --mtp-head fine-tune

The parked fine-tune pipeline (`/home/xlayn/ai/claude_scratch/mtp_train/`, design-mtp-drafter-finetune.md) writes a
head gguf with `write_head.py`. Replay it with `--mtp-head head.gguf` next to the shipped one (two runs on the same
inputs, or one run with two `--circuit` entries once circuits and heads mix) and compare per bucket and depth. The
per-bucket table says where to aim: weight the training loss by the buckets where the shipped head is weak at depth
2+ (the self-h chain), and check that a head that wins on the window eval also wins here at full context
(HEAD_CONTRACT s6: windowed training lost at full context every time in October).
