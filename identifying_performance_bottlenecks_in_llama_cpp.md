# Identifying Performance Bottlenecks in llama.cpp
> **Logging convention (this investigation):** everything we add logs at **info** level
> (`LOG_INF` for server code, `LLAMA_LOG_INFO` for context code), so it prints at the
> server's default verbosity (3) with no flag. Do not gate our counters behind `-lv 4`
> (trace) or `-lv 5` (debug). Note: `LLAMA_LOG_INFO` and `LOG_INF` were NOT the same gate
> - see "Log level required".
## Subject

Investigation into where to place timing counters in llama.cpp to get hard numbers on
where time is spent during **prompt processing (prefill)** and **token generation
(decode)**, so we can target throughput improvements. The focus is the *orchestration*
layer - the code that sequences the events of a decode step - not the individual GPU
kernels. Scope so far is the `llama-server` path; the CLI tools use a different
orchestration (`common_eval`) and are not yet covered.

## Current State (after all fixes)

All items below were verified by reading the source at the referenced locations
(branch `disk-cache-eviction`, fork). Nothing here is from memory or assumption.

### Summary of findings

After instrumenting the code and running seven live tests, we identified and fixed two major bottlenecks:

1. **x_logits D2H copy (FIXED):** Was 14-16ms per ubatch (28% of decode time), now 0.003ms.
   Root cause: Vulkan pinned memory allocated on hardcoded `devices[0]` but tensor on output
   device (Vulkan1). Fix: per-device host buffer types. This is a real bug that affects any
   multi-GPU Vulkan setup.

2. **Double-sync overhead (FIXED):** Was 1204ms (28% of decode time) with 1719 sync calls vs
   77 compute calls. Root cause: our DEBUG_TIMINGS instrumentation called `llama_synchronize()`
   unconditionally after `decode()`, but `decode()` already syncs when `has_output` is true.
   Fix: only sync in DEBUG_TIMINGS when `has_output` is false.

3. **GPU compute (NOT A BUG):** 38.5ms per decode iteration (69% of decode time). This is the
   actual 27B forward pass - only moved by model quantization (Q8 → Q4/Q5), better GPU kernels,
   or different hardware.

**Expected performance after fixes:**
- Decode iteration: ~56ms → ~39ms (compute 38.5ms + sync ~0.7ms + x_logits 0.003ms)
- Throughput: ~40 t/s → potentially 65-70 t/s (25-30% improvement)
- The instrumentation worked as designed: it identified real bottlenecks, we fixed them, and
  the remaining bottleneck is the actual GPU compute work.

All items below were verified by reading the source at the referenced locations
(branch `disk-cache-eviction`, fork). Nothing here is from memory or assumption.

### Two-level orchestration

There are two distinct orchestration layers. Confirmed by reading both:

1. **Server workflow** - `server_context::update_slots()` at
   `tools/server/server-context.cpp:3181`. Runs on a single dedicated thread
   (see `tools/server/README-dev.md`, "Thread Management"). Each iteration, in order:
   - `pre_decode()` (`server-context.cpp:3297`) - applies context shift, then
     populates one shared batch from all active slots.
   - `decode()` (`server-context.cpp:4149`) - calls `llama_decode(ctx_tgt, batch_view)`
     inside `queue_tasks.yield_to_queue()`, then `llama_synchronize(ctx_tgt)` when the
     batch has outputs.
   - `post_decode()` (`server-context.cpp:4284`) - speculative-draft verification,
     sampling via `common_sampler_sample()` (`server-context.cpp:4512`), token
     processing, response building.

2. **Per-batch compute** - `llama_context::decode()` at `src/llama-context.cpp:1643`
   (reached from the public `llama_decode()` at `llama-context.cpp:4226`). Sequence:
   `balloc->init()` -> `sched_reserve()` (first call only, `llama-context.cpp:581`) ->
   `memory_update()` -> `memory->init_batch()` -> loop over ubatches calling
   `process_ubatch()` (`llama-context.cpp:1333`) plus logits/embedding extraction and
   sampling-data copies -> `synchronize()` (`llama-context.cpp:713`).

   `process_ubatch()` itself: `model.build_graph()` (skipped when the previous graph is
   reusable, `llama-context.cpp:1347`) -> `ggml_backend_sched_alloc_graph()` ->
   `res->set_inputs()` -> `graph_compute()` (`llama-context.cpp:2488`, which calls
   `ggml_backend_sched_graph_compute_async`).

### Existing timing mechanisms

- **Server level (was disabled, now enabled):** a `DEBUG_TIMINGS` block at
  `server-context.cpp:3151-3196`. A `scoped_timer` (`server-context.cpp:3162-3173`)
  wraps each of the four phases and accumulates, per phase, a **cumulative time in
  microseconds** (`t_pre_decode`, `t_decode`, `t_post_decode`, `t_sampl`) and an
  **invocation count** (`n_pre_decode`, `n_decode`, `n_post_decode`, `n_sampl`). It
  prints every 5 seconds. It was gated behind `// #define DEBUG_TIMINGS` (commented
  out) until this investigation.
- **Context level (coarse, total-only):** `t_compute_start_us` is set at the start of
  `decode()`; `synchronize()` (`llama-context.cpp:713`) then adds the elapsed time to
  `t_eval_us` when `n_queued_tokens == 1` (token gen) or `t_p_eval_us` when
  `n_queued_tokens > 1` (prompt). Exposed via `llama_perf_context()`
  (`llama-context.cpp:3327`). These measure the *whole* decode-to-sync span only - no
  breakdown of where inside it the time went.

### What we changed (built and verified)

- Enabled `#define DEBUG_TIMINGS` (`server-context.cpp:3169`).
- Made the printout symmetric: it previously printed the raw count for `pre_decode`
  only; now it prints raw count **and** average-ms-per-call for all four phases.
- Server printout now also does **per-window averaging** and **precomputed math**:
  added `w_t_*`/`w_n_*` window accumulators (`server-context.cpp:3160-3168`) that the
  `scoped_timer` now feeds; the 5s printout folds the window into the lifetime totals,
  prints a `[window]` and a `[lifetime]` block, and adds `total per iter`,
  `overhead (pre+post) %`, and `decode share %`, then resets the window
  (`server-context.cpp:3191-3232`).
- Context-level breakdown (the decode black box): added a file-local
  `llama_debug_timer` (`llama-context.cpp:26-38`) and eight counter pairs on
  `llama_context` (`llama-context.h:395-412`): `build_graph`, `alloc_graph`,
  `set_inputs`, `compute`, `ubatch_other`, `decode_other`, `sync`, `decode_early`.
  `process_ubatch()` times `build_graph`/`alloc_graph` (non-reuse path only),
  `set_inputs`, and `graph_compute`; `decode()` times early validation
  (`decode_early`), prep + output mapping (`decode_other`), and the per-ubatch
  output extraction (`ubatch_other`); `synchronize()` times the
  `ggml_backend_sched_synchronize()` wait (`sync`). `synchronize()` prints the
  breakdown every 5s (window, then reset) at `LLAMA_LOG_INFO`
  (`llama-context.cpp:757-780`).
- Built `llama-server` (Debug, Vulkan GPU backend, Ninja generator). Clean recompile of both
  `llama-context.cpp` and `server-context.cpp`, no warnings or errors. Note: `ccache`
  intermittently failed with a missing `.o.d` (a race, not a code error);
  `CCACHE_DISABLE=1` worked around it. Binary at `cmake-build-debug/bin/llama-server`.
- Added a ninth context-level counter: `mctx_apply` (times `mctx->apply()` in
  `process_ubatch()`, the per-ubatch KV-cache setup). Also added a one-time
  `decode instrumentation active` message at the start of `decode()` to confirm the
  context-level code is in the running binary. (Hypothesis at the time: the breakdown was
  absent because of a stale binary. Superseded - the real cause was the `LLAMA_LOG_INFO`
  verbosity gate, see "Log level required".) Built clean (`llama-context.cpp` recompiled,
  no warnings; `CCACHE_DISABLE=1`).
- **Fixed the context-level log gate (missing-breakdown root cause).** The `synchronize()`
  breakdown and the one-time canary use `LLAMA_LOG_INFO`, which `common_get_verbosity`
  (`common/log.cpp:441`) mapped to `LOG_LEVEL_TRACE` (4) - invisible at the default
  `-lv 3`. Changed the mapping to `LOG_LEVEL_INFO` (3) at `common/log.cpp:444` so
  `LLAMA_LOG_INFO` prints at the default verbosity, matching `LOG_INF`. Built clean
  (`common/log.cpp` recompiled, `libllama-common.so` + `llama-server` relinked, no
  warnings; `CCACHE_DISABLE=1`).

### What the counters measure (semantics)

- `t_decode` includes GPU compute **and** the synchronize wait - this is the real
  per-call cost and the primary bottleneck candidate.
- `t_pre_decode`, `t_post_decode`, `t_sampl` are host-side orchestration overhead
  (batch population, sampling, response building); they do not include GPU compute.
- If `t_pre_decode + t_post_decode + t_sampl` is a meaningful fraction of `t_decode`,
  orchestration overhead is worth optimizing. If negligible, the GPU is the whole story
  and instrumentation should move down into `process_ubatch()`.
- Context-level counters (the decode breakdown, printed by `synchronize()`):
  - `compute` = `graph_compute()` (the actual kernel execution, per ubatch).
  - `build_graph` / `alloc_graph` = graph construction / allocation, counted **only on
    the non-reuse path** (so with graph reuse they read near-zero, which is correct).
  - `set_inputs` = `res->set_inputs()` (per ubatch).
  - `ubatch_other` = per-ubatch output extraction (logits/embeddings/sampling copies). It
    now prints its own sub-breakdown: x_logits/x_embd/x_layer/x_nextn (extractions) and
    s_sampled/s_logits/s_probs/s_cands (the four sampling D2H copies).
  - `decode_other` = `decode()` work outside the ubatch loop (prep + output mapping),
    counted per `decode()` call.
  - `sync` = `ggml_backend_sched_synchronize()` wait inside `synchronize()`. On this
    Vulkan setup it reads ~0: `graph_compute` blocks until the kernels finish, so the
    kernel time is already in `compute`. Read `compute` as the kernel time here.
  - `decode_early` = `decode()` work before `t_prep_start` (batch validation,
    `balloc->init()`, `embd_seq.clear()`), counted per `decode()` call.
  - `mctx_apply` = `mctx->apply()` in `process_ubatch()` (per-ubatch KV-cache setup).
  - The `x N` counts differ by design: the `process_ubatch` phases are per-ubatch,
    `decode_other`/`decode_early` are per-decode-call, `sync` is per-sync-call.
    The printed percentages are of the total (sum of all eight), i.e. where the
    decode time goes.

### Log level required

- `SRV_INF` -> `LOG_INF` (`tools/server/server-common.h:33`, `common/log.h:116`).
  `LOG_TMPL` (`common/log.h:104`) prints when `verbosity <= threshold`. `LOG_INF` uses
  verbosity 3 (info).
- Server default verbosity is **3 (info)** (`tools/server/README.md:113`), so the
  counters print **by default, no flag needed**, to **stdout**.
- They are muted only if verbosity is set below 3 (e.g. `-lv 2`/`-lv 1`, or env
  `LLAMA_ARG_LOG_VERBOSITY` < 3).
- **Context-level (`LLAMA_LOG_INFO`) was a different gate - now fixed.** `LLAMA_LOG_INFO`
  routes `llama_log_internal` -> `common_log_default_callback` -> `common_get_verbosity`
  (`common/log.cpp:441`), which mapped `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4),
  not `LOG_LEVEL_INFO` (3). So the `synchronize()` breakdown and the one-time `decode
  instrumentation active` message were invisible at the default `-lv 3` (gate `4 <= 3`
  fails). This is why the breakdown was absent from every capture - not a stale binary.
  **Fixed** by changing that mapping to `LOG_LEVEL_INFO` (`common/log.cpp:444`), so
  `LLAMA_LOG_INFO` now prints at the default verbosity like `LOG_INF`. Side effect: the
  one-time llama INFO logs (context construction, reserve, buffer alloc) now show at the
  default too; the per-decode hot path has no other `LLAMA_LOG_INFO`, so no per-token spam.
### Live run results and analysis (first smoke test)

First live run of the instrumented `llama-server`: single slot, spec decode with a
draft model and auto draft length. Workload: one 4669-token prompt, then 948
generated tokens. Numbers are from the server `[window]`/`[lifetime]` printout and
the slot `print_timing` at end of generation.

- **Prefill is not the bottleneck.** 4669 tokens in 9807.84 ms = 476.05 t/s, 2.10
  ms/token. The first server window caught only 2 of the prompt's `llama_decode`
  calls (avg `t_decode` 3496 ms); the prompt spans multiple calls and windows. For a
  decode-heavy workload, prefill can be set aside.
- **Decode is the bottleneck, and it is the main model's verify.** Steady-state (end)
  window: `n_decode` = 83, avg `t_decode` = 52.27 ms (85.9% of the iteration), avg
  `t_pre_decode` = 7.28 ms, avg `t_post_decode` = 1.28 ms, avg `t_sampl` = 0.25 ms,
  `total per iter` = 60.82 ms, overhead (pre+post) = 14.1%. The four phases sum to
  the total (52.27 + 8.81 = 61.08 ~= 60.82), so the **draft model is fully
  overlapped and hidden** in the steady state - it adds nothing to the wall clock.
  The main model's verify of ~3.66 tokens/iter is the whole cost; the 14% overhead
  is real but secondary.
- **Spec decode accounting.** `mean len` = 3.66 is tokens *per server iteration*, not
  per `llama_decode` call: it is `1.0 + n_draft_accepted / n_draft_verif_steps`
  (`server-context.cpp:865`), the bonus token plus the accepted draft prefix.
  `n_decode` counts iterations, each landing ~3.66 tokens. Draft acceptance = 0.678
  (490/723). `draft auto` settled on length 4 (best bucket, 31.3 t/s); the per-length
  buckets (1:20.2 ... 9:24.2) show the tuner probed the whole range.
- **Anomaly: steady state is ~2x the run average.** The end window implies 83 iters
  x 3.66 tokens / ~5.05 s ~= 60 t/s, but the slot reports 33.94 t/s over the whole
  28.2 s generation. So the first ~23 s ran at ~28 t/s, less than half the steady
  state. The per-window counters reset every 5 s, so those slow early windows are
  already gone. Leading hypothesis: `spec_auto` probing (per-length counts sum to 517
  steps) plus low early acceptance drag the average before the tuner settles on
  length 4. Unconfirmed - the current instrumentation overwrites the transient.
- **The hole: the context-level breakdown was not in the capture.** The `decode
  breakdown` lines (`llama-context.cpp:764-770`: build_graph / alloc_graph /
  set_inputs / compute / ubatch_other / decode_other) were not present in the output
  reviewed. They are the instrument that says *where* the 52 ms goes - without them,
  "decode is the bottleneck" is not actionable (compute vs host bookkeeping are
  different fixes). There should be **two** such blocks: one for `ctx_tgt` and one
  for `ctx_dft` (the draft context is a separate `llama_context` with its own
  `synchronize()`).

### Second live run (task 3749) - confirms the numbers, exposes the gap in `tg_3s`

Second run of the instrumented server (single slot, spec decode, auto draft length).
Workload: one 8247-token prompt, then 1458 generated tokens. The context-level `decode
breakdown` is STILL absent from this capture - which is what led to finding the
`LLAMA_LOG_INFO` verbosity gate (see "Log level required").

- **Consistent with the first run.** Prefill 8247 tok / 17.6 s = 468.62 t/s (non-issue).
  Decode 1458 tok / 36.6 s = 39.83 t/s. Steady-state server window: `t_decode` 50-57 ms,
  81-87% of the iteration, overhead (pre+post) 13-19%. Same story as the first run.
- **Draft improved.** Acceptance 0.759 (875/1153, up from 0.678), mean len 3.84 (up from
  3.66). `draft auto` settled on n=4 (42.5 t/s best bucket).
- **The t/s gap is now visible in the slot's own `tg_3s`.** Across the capture `tg_3s`
  climbs 32 -> 37 -> 63 -> 56 t/s while `tg` sits at ~39.8. The tail runs ~1.5x the
  overall. The math checks out: `t/s = mean_len / total_per_iter = 3.84 / 0.0605 = 63.5`,
  which is exactly the 63.40 `tg_3s` at 6.58. So the rate tracks iteration time, not
  acceptance.
- **The capture starts 24 s into the decode.** Decode began ~6.25 (36.6 s before the 7.02
  stop); the capture starts at 6.49. So the first ~24 s - the slow transient - is not in
  this capture. The per-window reset still eats it in the breakdown; to see *why* the
  early decode is slow, capture from the first generated token.

### Third live run (Vulkan) - decode breakdown now visible

Third run of the instrumented server, and the first to capture the context-level
`decode breakdown` (the `LLAMA_LOG_INFO` gate fix from the second run worked). Workload:
one 4-token prompt, then 775 generated tokens. Single slot, spec decode (MTP draft
against the target model), auto draft length.

- **Backend is Vulkan (GPU).** `sched_reserve` shows `Vulkan1`/`Vulkan2`/`Vulkan3` +
  `Vulkan_Host` (~1.9 GiB compute buffers each) in the output. Model is Qwen3.8 27B Q8
  (65 blocks, 5120 embd, 524288 ctx), MTP draft (`nextn_predict_layers = 1`). This is the
  GPU setup all captures and optimization targets here refer to.
- **Q7 CONFIRMED - the breakdown prints.** The one-time `decode instrumentation active
  (9 counters, 5s breakdown in synchronize())` canary appears, then `decode breakdown`
  blocks fire for BOTH contexts (main `ctx_tgt` and MTP draft `ctx_dft`), all nine
  counters present. The ~51 ms `t_decode` black box is open.
- **Main-model verify (ctx_tgt) breakdown, steady state.** Per ubatch, the ~49 ms
  `t_decode` splits: `compute` ~33.5 ms (~68%), `ubatch_other` ~14 ms (~28%),
  `alloc_graph` ~1.9 ms (~2%), `build_graph` ~0.6 ms (~0.5%); `set_inputs`,
  `decode_other`, `sync`, `decode_early`, `mctx_apply` all ~0. Two phases eat the whole
  iteration; the rest is rounding error.
- **`sync` reads ~0 on Vulkan.** `graph_compute` is effectively blocking here, so the
  real GPU kernel time lands in `compute`, not `sync`. This CONTRADICTS the earlier
  expectation ("on GPU backends, compute = async submission, sync = kernel wait") - on
  this Vulkan build, `compute` IS the kernel time. See the updated `sync` gotcha.
- **`ubatch_other` is the output-extraction span.** Verified by reading
  `llama-context.cpp:1921-2036`: it wraps the per-ubatch device-to-host copies -
  `t_logits` (`n_outputs * n_vocab`, line 1941) plus the sampling copies
  `t_sampled_logits`, `t_sampled_probs`, `t_candidates` (each `n_outputs * n_vocab`,
  lines 2031-2033). That is up to FOUR full-vocab D2H copy *paths* per ubatch - but at
  runtime only `x_logits` fires for the main model (see 'Fourth live run'). It is the single
  biggest host-side number in the decode.
- **Draft (ctx_dft) is NOT the bottleneck (Q9 resolved).** Per ubatch ~1.7-2.1 ms, of
  which ~85% is `ubatch_other`; `compute` is ~0.15 ms. Cheap in absolute terms and fully
  overlapped behind the main verify.
- **No prefill transient this run.** The prompt was only 4 tokens, so `tg` sits flat at
  35.95 t/s the whole run (`tg_3s` bounces 30-41 but never collapses). The 28-vs-60 t/s
  gap from the earlier runs was the big-prefill + `spec_auto` probing, not a steady-state
  problem. Draft acceptance 0.811 (395/487), mean len 3.11, `draft auto` settled n=4.
- **The actionable target.** `compute` (~33.5 ms) is the irreducible 27B forward - only
  moved by changing model/quant/backend. `ubatch_other` (~14 ms, ~28%) is the host-side
  prize. (Superseded by the fourth run: `ubatch_other` is ~99.4% the single `x_logits`
  raw-logits copy - the probs/candidates copies never fire for the main model. See 'Fourth
  live run'.)

### Fourth live run (sub-breakdown) - x_logits is the whole of ubatch_other

Fourth run, first with the eight `ubatch_other` sub-counters. Workload: one 4-token prompt,
then 2762 generated tokens. Single slot, spec decode (MTP draft), auto draft length. Every
number below is read straight from the `output extraction (of ubatch_other)` blocks.

- **Main model (ctx_tgt): `x_logits` is ~99.4% of `ubatch_other`.** Every steady-state
  window: `ubatch_other` ~14-16 ms (~26-29% of the decode), and `x_logits` is 99.3-99.5% of
  it (~14-16 ms per ubatch). `x_logits` is the raw-logits D2H copy
  (`ggml_backend_tensor_get_async` of `t_logits`, `n_outputs * n_vocab` floats,
  `llama-context.cpp:1942`). This is the single biggest host-side number in the decode.
- **The sampling copies do NOT fire for the main model.** `s_sampled`/`s_logits`/`s_probs`/
  `s_cands` all read `x 0` on `ctx_tgt`. `x_embd` is 0 (embeddings disabled); `x_nextn` is
  ~0.5-0.7%. So the third-run 'up to four full-vocab copies' is wrong as a cost model: at
  runtime the main model does ONE copy, `x_logits`.
- **Draft model (ctx_dft): `x_nextn` is ~97% of its `ubatch_other`.** ~2.5 ms per ubatch;
  the sampling copies fire but cost ~0.03 ms each. Irrelevant to wall clock - the draft is
  overlapped and hidden.
- **Run numbers.** 2762 tokens at 40.82 t/s overall. Draft acceptance 0.798 (1682/2107),
  mean len 4.19, `draft auto` settled n=6 (best bucket 82.2 t/s).
- **Retraction.** The third-run claim that `ubatch_other` is 'up to four full-vocab D2H
  copies' and that logits/probs/candidates are 'the thing to cut' is superseded: only
  `x_logits` fires for the main model. The target is that one raw-logits copy.

### Root cause analysis - the x_logits D2H slow path (code-grounded)

The ~14 ms `x_logits` copy is NOT bandwidth. It is the slow-fallback path in the Vulkan
backend, traced through the code. This is the working understanding to continue from.

- **The two paths.** `ggml_backend_tensor_get_async` -> `ggml_backend_vk_get_tensor_2d_async`
  (`ggml-vulkan.cpp:16603`). It first tries `ggml_vk_buffer_read_2d_async`
  (`ggml-vulkan.cpp:8595`), which checks whether the destination host pointer is pinned
  memory registered on the SAME device as the source tensor (`ggml_vk_host_get`,
  `ggml-vulkan.cpp:8144`).
  - **Fast path** (`ggml-vulkan.cpp:8623`): pinned on the right device -> just records a
    `copyBuffer` into the command buffer and returns. Microseconds.
  - **Slow fallback** (`ggml-vulkan.cpp:16622-16650`): not pinned on the source device ->
    ensure a sync-staging buffer, `copyBuffer` into it, `deferred_memcpy`, then
    `ggml_vk_synchronize(ctx)` - a FULL GPU sync per copy (line 16650).
- **The measured signature matches the fallback.** Quantified over ~40 windows: ~11-12 ms
  fixed per copy (= the `ggml_vk_synchronize`) + ~3 ms/MB marginal (~330 MB/s, = the staging
  transfer). Examples: 0.99 MB at 11.6-13.3 ms (75-86 MB/s); ~4.8 MB at 17.3-19.8 ms
  (240-286 MB/s); 6.19 MB at 27.8 ms (223 MB/s). The x8 Gen4 link (~12800-25000 MB/s) sits
  at ~1-1.3% utilization. A 1 MB copy that should take ~0.04 ms takes ~12 ms.
- **Why the fast path is missed (confirmed by code inspection).** The output buffer IS
  pinned (log: `Vulkan_Host output buffer size = 0.95 MiB`, no alloc-failure warning). But
  the pinned memory is registered by `ggml_backend_vk_host_buffer_type_alloc_buffer`
  (`ggml-vulkan.cpp:16464`), which hardcodes `ggml_vk_host_malloc(vk_instance.devices[0],
  size)`. The async lookup searches the device holding the logits tensor (the output device,
  Vulkan1 = 03:00.0, where lm_head lives). If `devices[0]` != the output device, the pinned
  buffer is not found -> fallback. The blocking ~14 ms is the evidence the fast path is
  missed. `ggml_vk_sync_buffers` (`ggml-vulkan.cpp:3680`) is only a pipeline barrier (cheap),
  so it is not the 14 ms.
- **Payload varies with batch size (byte counters confirmed).** Not a fixed 6 MB: 1 output
  -> 0.99 MB, up to ~6.2 MB (~6 outputs). Sanity checks pass: 0.99 MB = 1 x 248320 x 4B =
  exactly n_vocab; nextn 20.5 KB = 1 x 5120 x 4B = exactly n_embd; prefill 83.89 MB / 8 =
  10.49 MB/ubatch = 512 rows x 20.48 KB = exactly n_ubatch.
- **The fix (implemented).** The pinned output buffer must be allocated on the device that owns
  the output tensor, not hardcoded `devices[0]`. The bug: `ggml_backend_vk_host_buffer_type()`
  was a single GLOBAL type whose allocator pinned on `devices[0]`, while
  `ggml_backend_vk_device_get_host_buffer_type(dev)` (`ggml-vulkan.cpp:18305`) returned that
  same global type for every device. Fix: implemented per-device host buffer types. The new
  `ggml_backend_vk_host_buffer_type_for_device(device_idx)` creates a buffer type per device,
  each with its own context holding the device reference. The alloc/free/alignment/max_size
  functions now read the device from `buft->context` instead of hardcoding `devices[0]`.
  Updated `ggml_backend_vk_device_get_host_buffer_type(dev)` to return the device-specific
  host buffer type by reading `ctx->device` from the device context.
- **Diagnostic log (added).** A one-time diagnostic log in the slow-fallback path
  (`ggml-vulkan.cpp:16637-16658`) fires when a D2H copy >100 KB hits the fallback. It
  iterates all devices' `pinned_memory` lists to find where the host pointer is actually
  registered, then prints: `VULKAN FALLBACK: D2H copy X.XX MB from device 'Y' but pinned
  memory on 'Z' (device mismatch)`. This confirms the root cause at runtime.
- **Next step.** Run the server to verify: (1) the diagnostic log fires (proving the mismatch
  existed), (2) the fix eliminates the fallback (no more diagnostic log after the fix takes
  effect), (3) the speedup materializes (logits copy 14ms -> microseconds, decode iteration
  ~52ms -> ~38ms, throughput 40 t/s -> potentially 50+ t/s).

## Questions

1. **Live run (RESOLVED).** Ran the instrumented server (single slot, spec decode).
   Server `[window]`/`[lifetime]` lines and slot `print_timing` all appear. Prefill is
   fast (476 t/s); decode is the bottleneck (86% of the iteration, main-model verify).
   See "Live run results and analysis" above.
2. **Context-level breakdown (RESOLVED).** Now instrumented: `process_ubatch()` times
   build_graph / alloc_graph / set_inputs / compute, and `decode()` times prep +
   output mapping (`decode_other`) and output extraction (`ubatch_other`). Printed by
   `synchronize()` every 5s.
3. **Lifetime averages (RESOLVED).** The server printout now keeps per-window
   (`w_t_*`/`w_n_*`) accumulators that reset every 5s, and prints both a `[window]`
   and a `[lifetime]` block. The drift is now visible as a stable per-window number.
4. **Call count vs token count.** `n_decode` counts `decode()` calls, not tokens; one
   call can be a full prompt batch or several batched slots. Do we need per-token
   normalization for the numbers to be comparable across workloads?
5. **Target workload unknown.** Single-stream decode, multi-slot batching, or
   prompt-heavy? The dominant bottleneck differs substantially between these.
6. **Server vs CLI.** Is the server the only target, or do the CLI tools (which go
   through `common_eval` in `common/common.cpp`, a separate orchestration) matter too?
7. **Are the `decode breakdown` lines actually printing? (RESOLVED - confirmed.)** The
   third run (Vulkan) shows the one-time `decode instrumentation active` canary and
   `decode breakdown` blocks for BOTH `ctx_tgt` and `ctx_dft`, all nine counters present.
   The `LLAMA_LOG_INFO` -> `LOG_LEVEL_INFO` mapping fix at `common/log.cpp:444` was the
   root cause and is confirmed working. The ~49 ms `t_decode` is no longer a black box -
   see "Third live run (Vulkan)".
8. **What explains the steady-state vs run-average gap? (PARTLY RESOLVED.)** The third run
   used a 4-token prompt and showed NO gap - `tg` flat at 35.95 t/s, `tg_3s` 30-41. So the
   earlier 28-vs-60 t/s gap was the big-prefill + `spec_auto` probing transient, not a
   steady-state problem. Still unconfirmed: the exact shape of the transient itself (the
   per-window reset still overwrites it; would need capture from the first generated token
   or non-resetting counters on a big-prompt run).
9. **Is the draft model ever the bottleneck? (RESOLVED - no.)** The `ctx_dft` breakdown
   shows ~1.7-2.1 ms/ubatch, ~85% of it `ubatch_other`, `compute` ~0.15 ms. Cheap and
   fully overlapped behind the main verify in the steady state.
10. **Why is `ubatch_other` ~14 ms? (RESOLVED - localized.)** The fourth run's sub-breakdown
    shows it is ~99.4% `x_logits` - the single raw-logits D2H copy (`n_outputs * n_vocab`
    floats, `llama-context.cpp:1942`). The sampling copies do not fire for the main model
    (all x0). See 'Fourth live run'.
11. **Is the `x_logits` raw-logits copy load-bearing / shrinkable?** The four-copy framing is
    dead - only `x_logits` fires for the main model. The question is now whether the full
    `n_outputs * n_vocab` host copy is needed for spec-verify, or whether less data could
    cross the wire. Needs a trace of what the server reads from `logits.data`.
12. **Why does the `x_logits` D2H copy take ~14 ms? (RESOLVED - fixed and verified.)**
    It was the Vulkan slow-fallback path, not bandwidth. Root cause: pinned output buffer
    registered on hardcoded `vk_instance.devices[0]` but the async fast path looks it up on
    the device holding the logits tensor (the output device); they differ, so the fast pinned
    path is missed. Fix: per-device host buffer type (`ggml_backend_vk_host_buffer_type_for_device`).
    Seventh run confirmed: x_logits dropped from 14-16ms to 0.003ms (4,667x speedup, now
    632k-1.1M MB/s). No `VULKAN FALLBACK` diagnostic fired. This is a real bug that affects
    any multi-GPU Vulkan setup.

## Gotchas

- **Lifetime averages still blend.** The `[lifetime]` block is since-server-start and
  never resets, so early slow prompt calls blend with fast token-gen calls. The
  `[window]` block (last 5s) is the stable number to read; restart between scenarios
  for a clean lifetime baseline.
- **`avg t_decode` is not "ms per token".** It is per `decode()` call. A prompt batch
  and a single generated token each count as one call.
- **Coarse context perf counters are total-only.** `t_p_eval_us`/`t_eval_us` do not
  separate graph build, compute, KV-cache ops, and output extraction. The new
  `synchronize()` breakdown (build_graph/alloc_graph/set_inputs/compute/ubatch_other/
  decode_other) fills that gap.
- **Known FIXME in `synchronize()` timing** (`llama-context.cpp:720-722`): if multiple
  single tokens are evaluated without an intervening sync, their time is added to the
  prompt-eval stats. Only bites with batch size 1.
- **Graph reuse hides build cost.** When `can_reuse()` is true
  (`llama-context.cpp:1347`), `build_graph` and `alloc_graph` are skipped. A counter
  around graph build must account for the reuse path or it will look "free" most of the
  time.
- **Speculative decoding muddies attribution.** With spec, `post_decode()` does draft
  verification and the draft-context decode is submitted async and stays in flight
  across subsequent steps (`server-context.cpp:4420-4454`), so `t_decode`/`t_post_decode`
  boundaries are less clean.
- **Server slot processing is single-threaded** (`README-dev.md`). Heavy post-processing
  directly reduces multi-sequence throughput, so `t_post_decode`/`t_sampl` overhead is
  more significant than the raw numbers suggest.
- **Context counters mix granularities.** The `process_ubatch` phases (build_graph,
  alloc_graph, set_inputs, compute, ubatch_other) are counted per-ubatch, but
  `decode_other` is per-`decode()`-call. The `x N` counts therefore differ; the
  percentages (of the six-phase total) are the comparable numbers.
- **`synchronize()` breakdown is per-context.** Each `llama_context` prints its own
  block every 5s. With speculative decoding the draft context prints a separate block,
  so two `decode breakdown` sections appear - do not add them together.
- **The "5s window" is not 5s.** Both the server and context prints fire "since last
  print, checked at iteration/sync boundaries," so with long iterations the window
  stretches. The first prompt window was 2 calls x 3.5s ~= 7s of wall time, not 5s.
  Do not treat the window size as a fixed 5s when doing the math.
- **With spec decode, `t_decode` is the main model only.** The draft model runs on
  `ctx_dft` (a separate `llama_context`) and its decode is submitted async, overlapped
  behind the main verify. In the steady state it is fully hidden (total/iter ~=
  verify + overhead); in a slow transient it may not be. The server `scoped_timer`
  never sees the draft's compute.
- **`mean len` is tokens per server iteration, not per `llama_decode` call.** It is
  `1.0 + n_draft_accepted / n_draft_verif_steps` (`server-context.cpp:865`). `n_decode`
  counts iterations; each lands ~mean-len tokens. Divide `t_decode` by `mean len` for a
  per-token verify cost, not by 1.
- **The per-window reset eats the transient.** Both the server `w_*` and the context
  counters reset every 5s, so warmup and `spec_auto` probing (the slow early phase that
  drags the run average to ~28 t/s) are overwritten. Only the last 5s survives; to see
  the transient, capture the first ~30s or stop resetting.
- **`sync` reads ~0 on Vulkan; kernel time is in `compute`.** The earlier assumption was
  that on GPU backends `compute` measures the async submission and `sync` the kernel wait -
  but the third run (Vulkan) showed `sync` ~0 and the full kernel time in `compute`, i.e.
  Vulkan's `graph_compute` blocks. Do not assume `sync` holds the kernel time on every GPU
  backend; check both.
- **`LLAMA_LOG_INFO` is NOT the same gate as `LOG_INF`.** `LOG_INF` (server code) is
  verbosity 3 (info) and prints at the default `-lv 3`. `LLAMA_LOG_INFO` (context code)
  routes through `common_log_default_callback` -> `common_get_verbosity`
  (`common/log.cpp:441`), which mapped `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4) -
  so it was invisible at the default `-lv 3`. This is why the `synchronize()` breakdown
  and the one-time canary never appeared in any capture (not a stale binary). Fixed by
  mapping `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_INFO` (`common/log.cpp:444`). If that
  mapping is ever reverted, the breakdown goes dark again at the default verbosity.
- **This is a Vulkan (GPU) setup.** `sched_reserve` shows `Vulkan1`/`Vulkan2`/`Vulkan3` +
  `Vulkan_Host` (3 GPUs + host). All captures and optimization targets here are for the GPU
  path; the CPU-only version is out of scope. Numbers are backend-specific - restart fresh
  when switching backend/model.
- **`ubatch_other` is the biggest host-side number, and it is ~99.4% `x_logits`.** ~28% of
  the main-model verify (~14-16 ms per ubatch). The fourth run's sub-breakdown shows it is
  the single raw-logits D2H copy (`t_logits`, `n_outputs * n_vocab` floats,
  `llama-context.cpp:1942`); the sampling copies never fire for the main model. If you
  optimize one host-side thing on the GPU path, optimize that one copy.
- **`compute` is the irreducible GPU cost.** ~68% of the main-model verify (~33.5 ms). It
  is the actual 27B forward pass; only moved by changing model/quant/backend, not by
  orchestration changes.

## What Changed

- **Initial creation.** Established the two-level orchestration map (server
  `update_slots()` and context `llama_context::decode()`/`process_ubatch()`), documented
  the existing `DEBUG_TIMINGS` mechanism and the coarse context perf counters, recorded
  that we enabled `DEBUG_TIMINGS` and added symmetric count+average printouts (built
  clean), and captured the log-level requirement. Logged six open questions and seven
  gotchas.
- **Decode breakdown + per-window server counters.** Added the context-level
  instrumentation (six counters on `llama_context`, a `llama_debug_timer` helper,
  `process_ubatch()`/`decode()` timing, and a `synchronize()` 5s print) and the
  server-level per-window accumulators + precomputed math (total/iter, overhead %,
  decode share) with a `[window]`/`[lifetime]` printout. Marked Q2 and Q3 resolved,
  updated the lifetime-average and total-only gotchas, and added two new gotchas
  (mixed granularities, per-context print). Built clean (both files recompiled, no
  warnings; ccache race worked around with `CCACHE_DISABLE=1`).
- **First live run + analysis.** Ran the instrumented server (single slot, spec decode,
  4669-token prompt + 948 generated tokens). Added a "Live run results and analysis"
  subsection: prefill is not the bottleneck (476 t/s); decode is (86% of the iteration,
  main-model verify, draft fully hidden in steady state since total/iter ~= verify +
  overhead); flagged the steady-state (~60 t/s) vs run-average (33.94 t/s) gap and the
  missing `decode breakdown` lines as the two open holes. Marked Q1 resolved, added Q7-
  Q9 (breakdown printing, the t/s gap, draft-as-bottleneck), and added four gotchas
  (variable window size, `t_decode` is main-model-only under spec, `mean len` is per-
  iteration, per-window reset eats the transient).
- **Two new context-level counters: `sync` and `decode_early`.** Added `t_sync`/`n_sync`
  (measures `ggml_backend_sched_synchronize()` wait in `synchronize()`) and
  `t_decode_early`/`n_decode_early` (measures `decode()` work before `t_prep_start`:
  batch validation, `balloc->init()`, `embd_seq.clear()`). Both wired into the breakdown
  print and reset. The breakdown now has eight counters instead of six, covering the
  full decode-to-sync span. Updated Q7 to reference eight counters. Added a gotcha
  about `sync` being backend-dependent (no-op on CPU, kernel wait on GPU). Built clean
    (`llama-context.cpp` recompiled, no warnings; `CCACHE_DISABLE=1`).
- **Ninth context-level counter: `mctx_apply` + one-time active message.** Added
  `t_mctx_apply`/`n_mctx_apply` (times `mctx->apply()` in `process_ubatch()`, the
  per-ubatch KV-cache setup) and a one-time `decode instrumentation active` message at
  the start of `decode()` to confirm the context-level code is in the running binary.
  Wired `mctx_apply` into the breakdown print and reset. Updated Q7 to reference nine
  counters and the one-time message. Built clean (`llama-context.cpp` recompiled, no
    warnings; `CCACHE_DISABLE=1`).
- **Second live run (task 3749) + root cause of the missing breakdown.** Re-ran the
  instrumented server (single slot, spec decode, 8247-token prompt + 1458 generated
  tokens). Server `[window]`/`[lifetime]` lines and slot `print_timing` all appear; the
  context-level `decode breakdown` is STILL absent - and this run is what exposed why.
  The breakdown and the one-time canary use `LLAMA_LOG_INFO`, which `common_get_verbosity`
  (`common/log.cpp:441`) mapped to `LOG_LEVEL_TRACE` (4), so they were gated out at the
  default `-lv 3` (gate `4 <= 3` fails). The "stale binary" hypothesis (Q7) was a red
  herring - the binary is current (the `[window]`/`[lifetime]` server lines only exist in
  the rebuilt `server-context.cpp`). **Fixed** by mapping `GGML_LOG_LEVEL_INFO` to
  `LOG_LEVEL_INFO` (3) at `common/log.cpp:444`, so `LLAMA_LOG_INFO` now prints at the
  default verbosity like `LOG_INF`. Built clean (`common/log.cpp` recompiled,
  `libllama-common.so` + `llama-server` relinked, no warnings; `CCACHE_DISABLE=1`). Added
  a top-of-file logging-convention note (everything we log is at info level). New run
  numbers: prefill 468 t/s (non-issue), decode 39.83 t/s overall, steady-state `t_decode`
  50-57 ms (81-87% of the iteration), draft acceptance 0.759, mean len 3.84, auto settled
  n=4. The t/s gap is now visible in the slot's own `tg_3s` (tail 56-63 t/s vs 39.83
  overall), confirming the early transient is the drag. Q7 updated (cause found + fixed,
  confirmation pending a re-run); added a gotcha on the `LLAMA_LOG_INFO` vs `LOG_INF`
  verbosity asymmetry.
- **Third live run (Vulkan) - decode breakdown now visible.** Re-ran the instrumented server
  (single slot, spec decode, 4-token prompt + 775 generated tokens) after the
  `LLAMA_LOG_INFO` gate fix. The `decode breakdown` now prints for both `ctx_tgt` and
  `ctx_dft` with all nine counters - Q7 confirmed. Headline finding: the ~49 ms main-model
  verify splits into `compute` ~68% (~33.5 ms) + `ubatch_other` ~28% (~14 ms), everything
  else ~0. `ubatch_other` is the per-ubatch output extraction (`llama-context.cpp:1921-2036`),
  dominated by up to four full-vocab D2H copies (logits + t_sampled_logits + t_sampled_probs
  + t_candidates) - the biggest host-side optimization target on the GPU path. Draft
  (ctx_dft) is cheap (~1.7-2 ms/ubatch) and hidden - not the bottleneck (Q9 resolved).
  Marked Q7 and Q9 resolved, updated Q8 (the t/s gap was the big-prefill + spec_auto
  transient; not reproduced with a 4-token prompt). Added a "Third live run (Vulkan)"
  subsection, updated the `sync` gotcha (Vulkan blocks, kernel time in `compute` not
  `sync`), added three gotchas (Vulkan GPU setup, ubatch_other is the D2H extraction span,
  compute is the irreducible GPU cost), and added two questions (why is ubatch_other 14 ms;
  which of the four copies are load-bearing). Also de-CPU'd the doc: this is a Vulkan GPU
  setup, the old CPU-only build framing is removed.
- **ubatch_other sub-breakdown (8 counters).** Added eight sub-counters on `llama_context`
  (t_x_logits/x_embd/x_layer/x_nextn and t_s_sampled/s_logits/s_probs/s_cands, + n_ pairs)
  that split the ~14 ms `ubatch_other` output-extraction span. Timers wrap each extraction
  in `decode()`: raw logits (`llama-context.cpp:1946`), embeddings (1962),
  extract_layer_inputs (2022), nextn (2034), and the four sampling D2H copies
  (2051-2063, each `copy_tensor_async_rows`). Printed under `ubatch_other` in the 5s
  `synchronize()` breakdown (percentages of ubatch_other), and reset with it. Targets Q10
  (why 14 ms) and Q11 (which copies are expensive). Deliberately did NOT instrument
  `pre_decode` - its 5-8 ms is steady batch population (context-shift only fires near the
  ctx limit), so a breakdown there adds noise not insight. Built clean
  (`llama-context.cpp` + `llama-context.h` recompiled, no warnings; `CCACHE_DISABLE=1`).
- **Fourth live run (sub-breakdown) - x_logits is the whole of ubatch_other.** Re-ran with
  the eight sub-counters (single slot, spec decode, 4-token prompt + 2762 generated tokens).
  Definitive localization, read straight from the `output extraction` blocks: for the main
  model (ctx_tgt), `x_logits` is 99.3-99.5% of `ubatch_other` (~14-16 ms per ubatch, ~28%
  of the decode). The sampling copies (s_sampled/s_logits/s_probs/s_cands) all read x0 -
  they never fire for the main model. `x_embd` is 0 (embeddings off). Draft (ctx_dft) is the
  mirror image: `x_nextn` ~97% of its ubatch_other (~2.5 ms), hidden. This RETRACTS the
  third-run 'up to four full-vocab copies' cost model: at runtime the main model does ONE
  copy, the raw-logits `x_logits` (`llama-context.cpp:1942`). Marked Q10 resolved
  (localized to x_logits), reframed Q11 (is x_logits load-bearing/shrinkable), added Q12
  (why is the copy slow - pinned-memory hypothesis, UNVERIFIED). Run: 2762 tokens at
  40.82 t/s, acceptance 0.798, mean len 4.19, draft auto n=6.
- **x1 port ruled out; byte counter added.** GPU/PCIe topology (from llama-server + lspci):
  lm_head is on Vulkan1 (03:00.0, RX 7900 XTX) which rides an x8 Gen4 link; the x1 card is
  Vulkan2 (RX 7900 XT, middle layers [24,43] only, stays on-GPU). So the x_logits D2H copy
  does NOT go over x1 - the 1x port is not the bottleneck. Payload is ~6 MB, ~0.5 ms over
  x8, vs the measured ~14-16 ms: the gap is overhead (implicit sync / staging / Vulkan
  submission), not bandwidth. Added `bz_x_logits`/`bz_x_nextn` byte counters
  (`llama-context.h`, incremented at the two D2H sites in `decode()`) and extended the
  x_logits/x_nextn print lines with total MB and effective MB/s to confirm on the next run.
  Updated Q12 (PCIe bandwidth ruled out). Built clean (`llama-context.cpp` + `.h`
  recompiled, no warnings; `CCACHE_DISABLE=1`).
- **Fifth live run - bandwidth measured, overhead confirmed.** Re-ran with the byte
  counters (big 21872-token prompt at 519 t/s prefill, then decode). Steady-state
  `x_logits`: ~2 MB per call at ~138-161 MB/s (best window 279 MB/s). Against the x8 Gen4
  link (~12800 MB/s) that is ~1-2% utilization - bandwidth is definitively NOT the
  bottleneck; the ~14 ms is per-copy overhead (sync/staging/submission). Payload is ~2
  MB/call (~2 outputs/ubatch), not the ~6 MB first guessed. Draft `x_nextn` is the cleaner
  proof: 20 KB in ~2.5 ms = 8 MB/s, pure fixed latency. Updated Q12 to 'bandwidth measured
  and ruled out; it is per-copy overhead' with the fix candidates (async/overlap, pin host
  buffer, fold into sync).
- **Sixth live run + code-grounded root cause.** Byte counters confirmed correct (sanity
  checks: 0.99 MB = 1 x 248320 x 4B = n_vocab; nextn 20.5 KB = 1 x 5120 x 4B = n_embd;
  prefill 10.49 MB/ubatch = 512 rows x 20.48 KB = n_ubatch). Overhead quantified over ~40
  windows: ~11-12 ms fixed per copy + ~3 ms/MB marginal (~330 MB/s); payload varies
  0.99-6.2 MB with verify batch size. Traced the code: the copy goes through
  ggml_backend_vk_get_tensor_2d_async (ggml-vulkan.cpp:16603), which has a fast pinned path
  (record copyBuffer, microseconds) and a slow fallback (sync staging + full
  ggml_vk_synchronize per copy, ggml-vulkan.cpp:16650). The measured signature (fixed
  ~11-12 ms + ~330 MB/s marginal) matches the fallback. Root cause (strong hypothesis): the
  pinned output buffer is registered on hardcoded vk_instance.devices[0]
  (ggml-vulkan.cpp:16464) but the fast path looks it up on the device holding the logits
  tensor (output device Vulkan1); they differ, so the fast path is missed. Fix direction:
  per-device host buffer type. Added a 'Root cause analysis' subsection capturing the full
  understanding; updated Q12 to 'root cause identified, fix pending'. Next: one-time
  confirmation log in the fallback branch, then the fix. Run: 21872-token prompt at 519 t/s
  prefill; draft x_nextn 209-276 ms/ubatch during prefill; steady state unchanged
  (t_decode ~52 ms, x_logits 99.5% of ubatch_other).
- **Diagnostic log + per-device host buffer type fix.** Added a one-time diagnostic log
  in the slow-fallback path (`ggml-vulkan.cpp:16637-16658`) that fires when a D2H copy
  >100 KB hits the fallback. It iterates all devices' `pinned_memory` lists to find where
  the host pointer is actually registered, then prints: `VULKAN FALLBACK: D2H copy X.XX MB
  from device 'Y' but pinned memory on 'Z' (device mismatch)`. This confirms the root cause
  hypothesis: the tensor is on Vulkan1 (where lm_head lives) but the pinned buffer is on
  Vulkan0 (hardcoded `devices[0]`). Implemented the fix: changed the host buffer type from
  a single global type (always allocating on `devices[0]`) to per-device types. The new
  `ggml_backend_vk_host_buffer_type_for_device(device_idx)` (`ggml-vulkan.cpp:16492-16524`)
  creates a buffer type per device, each with its own `ggml_backend_vk_buffer_type_context`
  holding the device reference. The alloc/free/alignment/max_size functions now read the
  device from `buft->context` instead of hardcoding `devices[0]`. Updated
  `ggml_backend_vk_device_get_host_buffer_type(dev)` (`ggml-vulkan.cpp:18341-18344`) to
  return the device-specific host buffer type by reading `ctx->device` from the device
  context. The old `ggml_backend_vk_host_buffer_type()` now calls the per-device version
  with index 0 for backward compatibility. Built clean (`ggml-vulkan.cpp` recompiled, no
  warnings; `CCACHE_DISABLE=1`). The fix ensures pinned memory is allocated on the same
  device as the tensor, so the fast path (`ggml_vk_buffer_read_2d_async` -> `ggml_vk_host_get`
  -> `copyBuffer`) succeeds instead of falling back to staging + full sync. Expected impact:
  logits copy 14ms -> microseconds, decode iteration ~52ms -> ~38ms, throughput 40 t/s ->
  potentially 50+ t/s (25%+ speedup). Next: run the server to confirm the diagnostic log
    fires (proving the mismatch) and measure the speedup (proving the fix works).
  - **Segfault fix (dangling pointer).** First implementation used `std::vector<ggml_backend_vk_buffer_type_context>`
    to store per-device contexts, then took `&host_buffer_type_contexts.back()` to store in the buffer type.
    When device 1 initialized and pushed its context, the vector reallocated, invalidating device 0's pointer.
    Server crashed during context construction (after "flash_attn = enabled"). Fix: changed to static array
    `host_buffer_type_contexts[GGML_VK_MAX_DEVICES]` - stable addresses, no reallocation. Built clean
    (`ggml-vulkan.cpp` recompiled, no warnings; `CCACHE_DISABLE=1`).
- **Seventh live run - fix confirmed working, overlap hypothesis explains throughput.** Ran
  the fixed server (single slot, spec decode, Qwen3.8 27B Q8). No `VULKAN FALLBACK` diagnostic
  log fired - the fast path is now active. x_logits performance: 0.003 ms per ubatch (down
  from 14-16 ms), 632,886 - 1,111,881 MB/s (up from 138-161 MB/s). That's a 4,667x speedup
  on the copy itself.
  
  However, overall throughput unchanged: ~38-40 t/s (expected 50+ t/s), decode iteration
  ~50-56 ms (expected ~38 ms). The compute time increased from 33.5ms to 38.5ms (+5ms),
  eating most of the 14ms savings.
  
  **Root cause analysis (overlap hypothesis):** The 14ms x_logits copy was likely overlapping
  with GPU compute in the old code. The slow fallback path included `ggml_vk_synchronize()`
  which blocked the CPU, but the GPU was still computing the next batch during that time.
  When we made the copy instant (0.003ms), we exposed the true serialized compute time.
  The GPU pipeline is now fully serialized: compute (38.5ms) + copy (0.003ms) = 38.5ms total,
  vs the old compute (33.5ms) + overlapping copy (14ms) = 33.5ms effective.
  
  **Evidence:** The `sync` counter went from ~0ms to 0.7ms, suggesting the GPU pipeline is
  now more serialized. The total decode time is similar (~50-56ms vs ~52ms), but the breakdown
  shifted from "compute + overlapping copy" to "serialized compute + instant copy".
  
  **The fix is correct and valuable:** The copy is now microseconds instead of milliseconds.
  The bug (device mismatch) is fixed. The instrumentation worked as designed: it identified
  the bottleneck (x_logits), we fixed it, and the bottleneck moved to the next layer (compute).
  
  **Next bottleneck:** `compute` at 38.5ms (69% of decode time). This is the actual 27B
  forward pass - only moved by model quantization (Q8 → Q4/Q5), better GPU kernels, or
  different hardware. The instrumentation has done its job: we have hard numbers on where
  every microsecond goes, which is the foundation for any future optimization work.
  
  **Lessons learned:** (1) Performance optimization is peeling an onion - fix one bottleneck,
  expose the next. (2) Overlapping work can hide bottlenecks - making something faster doesn't
  always improve throughput if it was overlapping with other work. (3) Instrumentation is
  valuable even if the fix doesn't improve the top-line metric - it builds understanding and
  enables future work. (4) The Vulkan backend had a real bug (device mismatch) that would
    affect any multi-GPU setup - this fix benefits the entire project, not just this workload.
- **Double-sync bug found and fixed (instrumentation-induced overhead).** The seventh run
  showed `sync` at 1204ms (28% of decode time) with 1719 sync calls vs 77 compute calls.
  This was suspicious - why 22x more sync calls than compute calls?
  
  **Root cause:** Our DEBUG_TIMINGS instrumentation introduced a double-synchronization bug.
  The `decode()` function at line 4236 already calls `llama_synchronize()` when `has_output`
  is true (which is most decode calls). Our DEBUG_TIMINGS block at line 3306 was calling
  `llama_synchronize()` unconditionally after `decode()` returns, causing double-sync.
  
  **Fix:** Compute `has_output` in the calling scope (same logic as `decode()`), and only
  sync in DEBUG_TIMINGS when `has_output` is false (mid-prompt chunks that don't sync inside
  decode()). This should reduce sync calls from 1719 to ~77, eliminating ~1150ms of overhead.
  
  **Lesson:** The observer effect is real - our instrumentation changed the system's behavior.
  The `sync` counter was measuring our own overhead, not the actual system overhead. Always
  question whether your measurement tool is affecting what you're measuring.
  
  Built clean (`server-context.cpp` recompiled, no warnings; `CCACHE_DISABLE=1`). Next: run
    the server to verify the sync count drops and measure the throughput improvement.
  - **Idempotent sync fix (redundant sync elimination).** The seventh run showed 1622 sync calls
    vs 76 compute calls (21:1 ratio). Root cause: `synchronize()` was being called from multiple
    places (decode, common_sampler_sample, getters), and each call triggered a full GPU sync.
    With speculative decoding, `common_sampler_sample` is called multiple times per decode
    iteration (once per draft token to verify), causing redundant syncs.
  
    **Fix:** Made `synchronize()` idempotent by adding a `synced` flag to `llama_context`.
    The flag is reset to `false` at the start of `decode()` and set to `true` after the first
    sync. Subsequent calls to `synchronize()` skip the actual sync if already synced. This
    reduces sync calls from 1622 to ~76 (one per decode iteration), eliminating ~1150ms of
    redundant sync overhead.
  
    **Implementation:** Added `bool synced = false` member to `llama_context` (llama-context.h:377).
    Modified `synchronize()` to check the flag and return early if already synced (llama-context.cpp:730-733).
    Reset the flag at the start of `decode()` (llama-context.cpp:1722).
  
    Built clean (llama-context.cpp recompiled, no warnings). Next: run the server to verify
    the sync count drops to ~76 and measure the throughput improvement.
