# Potential Bugs in the Idempotent synchronize() Change
> Status: fix implemented, second review corrected three false claims
> (2026-09-05), regression test added (`tests/test-context-sync.cpp`).
> The original research was done against branch `disk-cache-eviction` (commit
> 7a8dad907b); line numbers below refer to that commit unless marked
> otherwise. The change under scrutiny: commit 291ca57a62 "llama: make
> synchronize() idempotent to eliminate redundant GPU syncs" (also bundled in
> 7a8dad907b). Review of the implemented fix: see "Fix review - generation
> counter" below; corrections to that review: see "Second review - what the
> first review got wrong".

## Subject

Scrutiny of the `synced` flag added to `llama_context` to make
`synchronize()` idempotent. The flag is reset to `false` at the start of
`decode()` (`llama-context.cpp:1722`) and set to `true` after the first real
`ggml_backend_sched_synchronize()` (`llama-context.cpp:739`); any
`synchronize()` call that sees `true` returns immediately
(`llama-context.cpp:730-733`), skipping the GPU sync AND the perf-stats block
AND the 5s debug printout. The question: does that early return ever skip a
sync that is load-bearing, i.e. the only barrier between an async GPU->host
copy and the host read of the copied data?

## Current State (verified by reading source)

### How the flag actually behaves

- `synchronize()` structure: early-return on `synced` -> timed
  `ggml_backend_sched_synchronize()` -> `synced = true` -> perf stats
  (`t_eval_us`/`t_p_eval_us`/`n_eval`/`n_p_eval`, `t_load_us`) -> 5s debug
  breakdown print. The early return therefore skips three things, not just
  the sync.
- The flag is written in exactly three places: reset in `decode()`
  (`llama-context.cpp:1722`), set in `synchronize()` (`:739`), and declared
  (`llama-context.h:377`, plain `bool`, not atomic).
- There is NO reset in `encode()` (`llama-context.cpp:1475`), and the public
  `llama_encode` (`llama-context.cpp:4320`, `include/llama.h:999`) goes
  straight to `ctx->encode()` without touching the flag.

### There is no final sync inside decode() - the sync lives at the call sites

- Public `llama_decode` (`llama-context.cpp:4331`) is just
  `ctx->decode(batch)` + return. No sync.
- Private `llama_context::decode()` (`:1713`) contains exactly two live
  `synchronize()` calls: `:1805` (drain prior in-flight copies before
  `embd_seq.clear()`, only when `!embd_seq.empty()`) and `:2211` (inside
  `output_reserve()`, only when `buf_output` is allocated or grown). A third
  at `:2136` is commented out.
- The async output copies (raw logits `:1965`, embeddings `:1972`/`:1994-2020`,
  nextn, and the four sampling `copy_tensor_async_rows` calls) are enqueued
  in the ubatch loop AFTER `output_reserve()` is called at `:1871`.
- The drain that makes those copies visible to the host is the caller's
  `llama_synchronize()` / the `llama_get_*` getters
  (`llama-context.cpp:3927-4044`, each calls `ctx->synchronize()` first), or
  `common_sampler_sample` (`common/sampling.cpp:611`), or the server's
  explicit sync after decode (`tools/server/server-context.cpp:4243-4246`,
  `ret == 0 && has_output`).
- `encode()` mirrors the same shape: `:1508` embd_seq drain sync,
  `:1522` `output_reserve()`, then the async copy loop.

### output_reserve() growth semantics (`llama-context.cpp:2145-2236`)

- `n_outputs_max = max(n_outputs, n_seq_max())` (`:2149`); buffer size grows
  with the largest output count ever requested.
- Reallocation happens when `!buf_output || prev_size < new_size` (`:2205`);
  the pre-free `synchronize()` at `:2211` is unconditional on that path
  (the size-change log at `:2209` is `#ifndef NDEBUG` only, the sync is not).
- Context construction already calls `output_reserve(params.n_seq_max)`
  (`:387`), so growth starts from `n_seq_max` (1 for the server) and happens
  once per step-up of the max output count.

### Who else calls synchronize() (all become flag-dependent no-ops)

- Getters: `llama_get_logits*`, `llama_get_embeddings*`,
  `llama_get_sampled_*` (`llama-context.cpp:3927-4044`).
- State save/load: `llama_state_*` (`:4224-4310`).
- KV cache stream copy path (`llama-kv-cache.cpp:841` - sync BEFORE the
  copies, so it drains prior work; the copies it then enqueues rely on a
  later real sync).
- Warmup (`common/common.cpp:1542`), the one-shot seq_rm capability probe
  `common_context_can_seq_rm` (`common/common.cpp:1620`, called once at server
  init - NOT a defrag thread, see the second review), server metrics flush
  (`tools/server/server-context.cpp:4821`), the instrumentation guard
  (`tools/server/server-context.cpp:3306-3314`, syncs ctx_tgt when
  `!has_output`; now runtime-gated by `--performance-instrumentation`).

### The documented invariant that relies on getter syncs being real

`tools/server/server-context.cpp:4470-4482` ([TAG_SPEC_OVERLAP] comment):
the draft decode is submission-only and "stays in flight across subsequent
steps"; "the window closes at the first read of draft output or draft state"
- explicitly naming `llama_get_logits*/llama_get_embeddings*` on ctx_dft and
`llama_synchronize(ctx_dft)` as the things that close it. That logic
assumes those syncs are real barriers.

## The bugs (ranked)

### Bug 1 - REALLOCATION ITERTION: output copies enqueued after the flag-set sync, read without a drain

Sequence on any decode/encode whose `output_reserve()` reallocates
`buf_output`:
1. `synced = false` (decode start, `:1722`).
2. `output_reserve()` -> reallocation -> `synchronize()` at `:2211` runs for
   real (drains PREVIOUS work, needed before freeing the old buffer) and
   sets `synced = true`.
3. Ubatch loop enqueues THIS decode's async D2H copies (`:1965` etc.).
4. Caller's drain sync (server `:4245`, or the first `llama_get_*` getter,
   or `common_sampler_sample`) sees `synced == true` -> NO-OP.
5. Host samples/reads logits (or embeddings, or nextn) from `buf_output`
   while the copy is still in flight.

Trigger frequency is not rare for a spec-decode server: `n_outputs_max`
starts at `n_seq_max` (1) and the draft auto-tuner probes draft lengths 1..9,
so the verify batch (1 + draft_len outputs) steps the buffer up ~9 times per
server start, on BOTH ctx_tgt and ctx_dft. After the max settles, it is
stable. Every step-up iteration is one raced read. (The observed run did
probe up to 9 and settled n=5, so growth did happen in that capture - the
reallocation itself is invisible at default verbosity because the log is
`LLAMA_LOG_DEBUG`.)

### Bug 2 - EMBEDDINGS CONTEXTS: the embd_seq drain sync fires every iteration and eats the iteration's own drain

For any context with `cparams.embeddings` (CLI embedding/encode paths):
after the first call, `embd_seq` is non-empty at the next decode/encode
start, so the drain sync at `:1805` (decode) / `:1508` (encode) runs for real
every iteration and sets the flag BEFORE this iteration's embedding copies
are enqueued. The `llama_get_embeddings*` getter sync that is supposed to
drain THIS iteration's copy is then a no-op. Raced read on EVERY
embedding iteration, not just growth steps.

### Bug 3 - llama_encode never resets the flag: encoder-only contexts are permanently poisoned

`encode()` has no `synced = false`. Once any real sync happens on an
encoder path (first-call reallocation `:1522`->`:2211`, or embd drain
`:1508`), the flag is true forever for that context unless a `decode()`
happens to intervene. Encoder-only usage (CLIP-style: `llama_encode` +
`llama_get_embeddings`, no decoder) then races on every single read,
indefinitely.

### Bug 4 - The [TAG_SPEC_OVERLAP] draft-window invariant is silently broken

The server comment (`server-context.cpp:4470-4482`) says the draft's
in-flight window closes at the first draft-output read because those reads
synchronize. If the draft's own `decode()` set the flag (Bug 1 growth step on
ctx_dft, which MTP drafts also hit via their own `output_reserve`, or the
embd path), the "closing" sync is a no-op and the nextn/logits read races.
The comment is now a trap: it documents a guarantee the flag can void.

### Bug 5 - Cross-thread: plain bool, shared with the defrag thread (WITHDRAWN)

Original claim: `synced` is a non-atomic `bool` touched by the update_slots
thread and "the defrag thread" (`common/common.cpp:1620`), so a foreign
sync landing mid-decode could poison the drain.

Withdrawn on the second review: `common/common.cpp:1620` is inside
`common_context_can_seq_rm`, a one-shot probe the server runs at init
(`server-context.cpp:1496`, `:1527`). There is no defrag code in `src/`
(only a dead `defrag_thold` default) and no thread in `tools/server` touches
a `llama_context` besides the worker loop. `llama_context` is
single-threaded by contract. The bug never existed; the atomics work item
is moot.

What survives of it is the general shape: ANY real sync that lands between
the last graph submission and the caller's drain would poison a
"first-sync-wins" scheme. The only in-process path that can do that is a
`llama_synchronize()` from inside `cb_eval` - see the second review.

### Benign-but-confusing side effects (instrumentation, not correctness)

- On a reallocation iteration, the REAL sync counted in `t_sync`/`n_sync`
  is the `:2211` pre-allocation barrier (draining prior work), and the
  actual end-of-decode drain is the skipped no-op. The 5s `sync` metric
  therefore measures the wrong event on those iterations.
- The perf stats block is skipped by the early return, so on a reallocation
  iteration `t_eval_us`/`t_p_eval_us` get the prep-only span measured at
  `:2211` (before compute) and the real compute span is never counted -
  slot "eval time" / "prompt eval time" prints undercount.
- The 5s breakdown printout lives inside `synchronize()` after the early
  return, so flagged (skipped) syncs never trigger a print. Print cadence is
  now tied to real syncs.
- The `:2211`/`:1805`/`:1508` syncs are legitimately real work (pre-free
  barriers) but are indistinguishable in the counters from a decode drain.

## Fix review - generation counter (working tree, uncommitted)

The fix implements the "Cleaner" option from Fix directions: a generation
pair instead of the per-decode bool. Verified against the working tree (post
7a8dad907b, uncommitted):

- `uint64_t async_gen` bumps once per `graph_compute()` submission, immediately
  before `ggml_backend_sched_graph_compute_async` (`llama-context.cpp:2621`).
  `graph_compute()` (`:1467`) is its only caller and `:2622` the only submit
  site in `src/`, so every graph submission is counted.
- `uint64_t synced_gen` records `async_gen` after every real
  `ggml_backend_sched_synchronize()`: in `synchronize()` (`:741`) and in the
  pipeline-parallel reuse barrier (`:1421-1422`, a necessary add - without
  the recording, the next `synchronize()` would re-drain what the barrier
  just drained). These are the only two raw sync sites in the file; both
  recorded. No unrecorded raw syncs remain (checked `llama_context::reset`
  too - no such member exists in this file).
- `synchronize()` early-returns only when `synced_gen == async_gen`
  (`:733`). The reset at decode start is gone (`:1727` comment: a batch left
  in flight by a mid-prompt chunk is still drained by the next real sync).
- No leftover `synced` references; no new includes needed.

Verdict per documented bug:

- Bug 1 (reallocation iteration): FIXED. The `:2211` reallocation sync
  records the pre-bump gen; the ubatch's `graph_compute()` bump happens
  after it, so the caller's drain sync is real and drains this decode's
  copies.
- Bug 2 (embd_seq drain every iteration): FIXED, same mechanism. Side
  effect: when the previous iteration's getter already synced, the `:1805`
  drain is itself a no-op - the embeddings path no longer double-drains.
- Bug 3 (llama_encode never resets): FIXED by deleting the reset; the
  counter is cumulative per-context, so no entry point owns a reset anymore.
- Bug 4 ([TAG_SPEC_OVERLAP] draft window): FIXED. The draft decode bumps
  ctx_dft's `async_gen` per ubatch and never syncs; the first draft-output
  read sees the mismatch and does a real sync, closing the window exactly
  as the server comment promises.
- Bug 5 (cross-thread non-atomic): FIXED SEMANTICALLY, NOT MEMORY-MODEL. A
  mid-decode defrag-thread sync records the current gen and the next bump
  makes the main thread's drain real again - the poisoning hazard is gone.
  But the counters are plain `uint64_t`, not `std::atomic`: still UB, benign
  on x86-64 (aligned 64-bit stores), and a torn read on a 32-bit platform is
  worse than the torn bool it replaced.
- Original motivation (kill the ~1622 redundant sampler/getter syncs):
  PRESERVED. After the per-iteration drain the gens are equal, so the
  per-draft-token `common_sampler_sample` syncs (`common/sampling.cpp:611`)
  and repeated getter calls still no-op.

Instrumentation side effects (from the bug list): the end-of-decode drain is
now always a real sync, so perf stats get the full span and the 5s printout
fires every iteration - both resolved. Residual: on reallocation iterations
`t_sync`/`n_sync` count two real syncs (pre-free barrier + drain). At least
it is the truth, unlike the bool era where the drain was silently skipped.

Coverage checks that passed: `mctx->apply()` KV setup enqueues before the
bump (covered); a failed `graph_compute_async` leaves the gen bumped,
costing one redundant real drain later (conservative, fine); initial state
(0,0) makes a pre-decode `synchronize()` a no-op, which is correct because
nothing is submitted yet.

Coverage check that was WRONG in the first review: "output/sampling copies
enqueue after the bump (covered)". They are covered only if no real sync
lands between the bump and the enqueue. See the second review for the fix
(a second bump after the copies).

Remaining work as listed by the first review (all four items superseded by
the second review below):

1. ~~`std::atomic<uint64_t>` for both counters.~~ Moot: no second thread.
2. ~~Soften "equal means nothing in flight" because KV stream / defrag
   copies never bump `async_gen`.~~ False: the KV stream copy loop uses
   `ggml_backend_tensor_copy`, which is synchronous (`ggml-backend.cpp:488-509`:
   tensor_get + tensor_set or a blocking buffer copy). Defrag does not exist.
   Nothing is left in flight. The comment is accurate.
3. ~~Re-run with `-lv 4` and grep "reallocating output buffer".~~ Cannot
   work: that log is inside `#ifndef NDEBUG` (`llama-context.cpp:2213-2216`)
   and both build trees are Release. Needs a Debug build, or the log moved
   out of the guard.
4. Nothing in `tools/server` needs changing for correctness: the
   [TAG_SPEC_OVERLAP] comment at `server-context.cpp:4470-4482` is again
   accurate as written. (The instrumentation there was made runtime-gated
   separately, see below.)

## Second review - what the first review got wrong (2026-09-05)

Verified against the working tree; every claim below was checked in source.

- **No defrag thread** (Bug 5 withdrawn, see above).
- **KV stream copies are synchronous** (work item 2 withdrawn, see above).
- **Open question 3 answered - yes.** `ggml_backend_sched_synchronize` loops
  `ggml_backend_synchronize` over every backend (`ggml-backend.cpp:2036`);
  the Vulkan one (`ggml-vulkan.cpp:16822` -> `ggml_vk_synchronize`) submits
  the pending transfer context and waits on a fence, so raw
  `ggml_backend_tensor_get_async` copies are drained by the sched sync.
- **Open question 4 answered - no.** Nothing runs concurrently with
  `update_slots` on the same context.
- **Open question 1 is unanswerable in a Release build** (NDEBUG guard).
- **The one real residual hole.** The window between the submission bump
  (`graph_compute`) and the output-copy enqueues in the ubatch loop is
  exactly where a real sync must NOT land: it records the current gen, the
  copies are enqueued after it, and the caller's drain then no-ops. On the
  main thread nothing syncs in that window. `cb_eval` runs inside
  `ggml_backend_sched_graph_compute_async`, i.e. inside the window, and the
  old comment at the bump said a `llama_synchronize()` from there "drains
  this graph too" - true, and then it poisons the drain for that ubatch's
  copies. No in-tree callback does that (eval-callback and imatrix use the
  synchronous `ggml_backend_tensor_get`), but the comment promised the
  opposite of what happened.

  **Fix (implemented):** `async_gen` now also bumps at the END of every
  ubatch iteration in `decode()` (after the D2H copies and
  `extract_layer_inputs`) and after the copies in `encode()`. So
  `async_gen` counts async submission points, not graphs: a graph
  submission and a batch of output copies each bump. A sync that lands
  between them records a gen the trailing bump immediately invalidates.
  The submission-time bump stays so a mid-compute sync is still real.
  Cost: one increment per ubatch. The header comment and the `graph_compute`
  comment were rewritten to say this.

- **Regression test:** `tests/test-context-sync.cpp` (ctest label `model`,
  uses the tinyllamas fixture). It reads the counters through test-only
  accessors on `llama_context` (`sync_async_gen()`, `sync_synced_gen()`,
  `sync_n_real()` - the last is a lifetime count of real
  `ggml_backend_sched_synchronize` calls, independent of the timing
  instrumentation). Cases:
  - fresh context is drained and a sync on it is a no-op;
  - decode -> drain pending -> getter performs exactly one real sync ->
    repeated getters/syncs are no-ops (the original motivation);
  - Bug 1: previous batch in flight + a batch with more outputs than
    `n_seq_max` (forces `output_reserve` reallocation and its real
    mid-decode sync) -> drain still pending after decode -> getter real;
  - Bug 2: embeddings context, second decode has the real embd_seq drain
    at the top -> drain still pending after decode -> getter real; then the
    steady state (read, decode, read);
  - no per-decode reset: two decodes, one real sync drains both;
  - cb_eval hole: a callback that calls `llama_synchronize()` mid-compute
    -> drain still pending after decode -> getter real. This case FAILS if
    the trailing bump is removed (mutation-checked on 2026-09-05);
  - encode path, skipped on decoder-only models.
  Passes on CPU and on Vulkan (`-ngl 0` / `-ngl 99`).

- **Instrumentation made runtime-gated:** `--performance-instrumentation`
  (`LLAMA_ARG_PERF_INSTRUMENTATION`, off by default) plumbs
  `common_params.perf_instrumentation` -> `llama_context_params.perf_instrumentation`
  -> `cparams.perf_instrumentation`. In libllama every `llama_debug_timer`,
  the raw `ggml_time_us()` reads in `decode()`/`synchronize()`, the
  "decode instrumentation active" banner and the 5s breakdown print are
  gated. In the server the compile-time `DEBUG_TIMINGS` define is gone; the
  `scoped_timer`s, the 5s window print and the extra `llama_synchronize` on
  mid-prompt chunks (which trades the target/draft prefill overlap for an
  honest `t_decode`) are gated on the same flag. `test-arg-parser` covers
  the flag.

## Why the observed run looked fine

On this Vulkan setup `graph_compute` is effectively blocking, and the
x_logits copy after the per-device pinned-buffer fix is a microseconds-scale
`copyBuffer` enqueued immediately after compute. The host then spends
milliseconds in post_decode/pre_decode before sampling, so the queue is
almost always drained by read time. The race is LATENT here: the broken
contract (synchronize-before-read) is honored in practice by timing luck,
not by the flag. It stops being luck on a genuinely async backend (ROCm/
CUDA with a deep queue), when the pinned fast path is missed (the old
fallback), or in the multi-subbatch in-flight draft scenario the
[ TAG_SPEC_OVERLAP ] comment itself describes.

## Open questions

1. Did the captured run actually hit Bug 1 (reallocations during draft-length
   probing)? Reallocation logging is `LLAMA_LOG_DEBUG`, invisible at the
   default `-lv 3`. Re-run with `-lv 4` and grep for "reallocating output
   buffer" to confirm how many growth steps occurred and when.
2. Which sampler reads hit a raced buffer first on a reallocation iteration:
   the server's `common_sampler_sample` (sampling.cpp:611, syncs first -
   no-op) reading `logits.data`, or the `llama_get_*` getters in
   post_decode? Order matters for the exact corruption window, not for
   existence.
3. Does `ggml_backend_sched_synchronize` drain raw
   `ggml_backend_tensor_get_async` copies (not sched-tracked copies)? Believed
   yes (Vulkan backend sync = queue wait idle covers the whole queue) but
   not verified per-backend - on a backend where backend sync is narrower
   than the sched sync, even the non-idempotent code would have a latent
   gap.
4. Does the server's defrag actually run concurrently with decode in this
   workload (Bug 5 precondition)? The disk-cache-eviction branch does
   lazy checkpoint re-hydration; whether that overlaps update_slots needs a
   check of the defrag trigger path.
5. Is `buf_output` growth bounded by the draft auto-tuner's max probe length,
   or can long prompts with many outputs (multi-slot batching) also step it
   up later? A/B: the capacity formula is `max(n_outputs, n_seq_max())`, so
   ANY larger batch grows it - multi-slot verify batches count.

## Gotchas

- "Idempotent" here means "first sync since decode start wins", NOT "second
  call is redundant". Those are only equivalent if no async work is
  enqueued between the first sync and the second. The code enqueues plenty
  of async work (all output D2H copies) in between.
- The flag reset is in `decode()` only. Every other entry point that can
  enqueue async work and then expect a sync (`encode()`, getters, state
  save/load, defrag, KV stream copy) inherits whatever flag state the last
  decode left or the last sync set.
- The pre-existing invariant "llama_decode does not synchronize; the getter
  or the caller does" means the drain sync and the flag-setting sync are
  DIFFERENT calls in the normal flow - the idempotency window always spans
  the entire output-copy phase.
- The [TAG_SPEC_OVERLAP] comment in server-context.cpp is the most detailed
  sync-semantics documentation in the codebase and it predates the flag;
  nobody has re-checked it against the flag.
- The perf-stats FIXME at `llama-context.cpp:741-743` (multiple single-token
  evals without sync land in prompt stats) gets WORSE, not better, with the
  flag: now the stats are added by whichever sync happened to be the first,
  wherever that was.
- The first review claimed the header comment on the counters ("equal means
  nothing in flight") was an overclaim because of KV/defrag copies. It was
  not: those copies are synchronous and defrag does not exist. After the
  trailing-bump fix the comment is exactly right. The gotcha is the
  opposite one: do not add an async enqueue site anywhere in
  `llama_context` without either a bump after it or a sync before the next
  host read. `tests/test-context-sync.cpp` will not catch a brand-new site.
- "Reallocating output buffer" is logged under `#ifndef NDEBUG` only. In a
  Release build no verbosity level shows it.
## Fix directions (the generation-counter option was chosen and implemented - see Fix review)

- Minimal: clear the flag at every async-copy enqueue site (the
  `tensor_get_async`/`copy_tensor_async_rows` calls in the decode/encode
  ubatch loop, ~6 sites) instead of only at decode start. Cheap, targeted,
  keeps the redundant-sync elimination for the spec-verify sampler-spam case
  that motivated the change.
- Cleaner: generation counter. Bump `enqueue_gen` on every async enqueue,
  record `synced_gen` on real sync; skip only when equal. Immune to missed
  sites only if enqueues are centralized.
- Or: keep the flag but make it mean exactly what the server comment says -
  "context fully drained" - by setting it only in a designated final-sync
  path, not in the pre-allocation/pre-free barriers at `:2211`/`:1805`/`:1508`
  (those would use a raw `ggml_backend_sched_synchronize` call instead of
  `synchronize()`).
- Whichever is chosen: make the variable `std::atomic<bool>` (or fold it
  into the generation counter) for the defrag thread, and re-verify the
  [TAG_SPEC_OVERLAP] window-closing guarantee in server-context.cpp.

## What Changed

- **Initial creation.** First pass over the idempotent `synchronize()` change
  (commit 291ca57a62). Mapped every write site of the `synced` flag (three:
  reset in decode, set in synchronize, declaration), every caller of
  synchronize() (getters, state save/load, KV stream copy, warmup, defrag,
  server syncs, sampler), and the two in-decode syncs that can set the flag
  before output copies are enqueued (`:2211` output_reserve reallocation,
  `:1805`/`:1508` embd_seq drain). Identified five potential bugs
  (reallocation-iteration raced read, embeddings every-iteration raced read,
  encoder-only permanent poisoning, broken [TAG_SPEC_OVERLAP] draft-window
  invariant, cross-thread non-atomic bool) plus instrumentation side
  effects. Verified all line references against the working tree at
    7a8dad907b. No code changes made.
- **Fix review (handover).** The fix was implemented in the working tree
  (uncommitted) using the "Cleaner" generation-counter option. Added a
  "Fix review - generation counter" section: mechanism mapping with verified
  line refs (bump at `llama-context.cpp:2621`, records at `:741` and
  `:1421-1422`, gate at `:733`, reset removed at `:1727`), per-bug verdicts
  (Bugs 1-4 fixed; Bug 5 fixed semantically but counters still non-atomic;
  the original redundant-sync elimination preserved), coverage checks
  (single submission path, mctx_apply ordering, failed-submit
  conservatism), and a four-item remaining-work list. Updated the status
  header (no longer RESEARCH ONLY), annotated the Fix directions heading
  (option chosen and implemented), and added a gotcha on the overclaiming
  header comment at `llama-context.h:379`. Review only - no code changes in
  this revision.
- **Second review + fix + test + flag (2026-09-05).** Verified the first
  review against the tree and withdrew three of its claims: no defrag
  thread exists (Bug 5 never happened, atomics moot); KV stream copies are
  synchronous (header comment was right); the "reallocating output buffer"
  log is NDEBUG-only so the `-lv 4` grep cannot work in Release. Answered
  open questions 3 (Vulkan sched sync drains async gets) and 4 (nothing
  concurrent). Found the one real residual hole (a sync from `cb_eval`
  lands between the submission bump and the output copies) and closed it
  with a trailing bump per ubatch in `decode()` and after the copies in
  `encode()`. Added `tests/test-context-sync.cpp` (7 cases, mutation-checked)
  and the `--performance-instrumentation` runtime flag replacing the
  server's compile-time `DEBUG_TIMINGS` and gating all libllama timers.

## Open work (handover checklist)

1. ~~atomics~~ withdrawn (no second thread).
2. ~~reword header comment~~ withdrawn (it was right).
3. ~~grep "reallocating output buffer" at `-lv 4`~~ impossible in Release;
   only worth doing if someone wants to date the old exposure window.
4. Committed 2026-09-05 together with the trailing bump, the test and the
   flag.
5. Re-run the server with `--performance-instrumentation` and compare
   `sync`/`n_sync` against the last capture: expectation is one real drain
   sync per decode iteration, plus one extra (the pre-free barrier) on
   reallocation iterations only. Throughput should be unchanged to slightly
   better; correctness is the point, not the clock.
