# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### EAGLE-3 (`draft-eagle3`)

EAGLE-3 uses a small draft model that reads the target model's hidden states to predict the next tokens, so it
reaches higher acceptance than a standalone draft model of the same size. The draft is a one-layer transformer
trained for a specific target model; it shares the target model's tokenizer and, optionally, uses a reduced draft
vocabulary with its own `lm_head`, which is mapped back using a `d2t` table.

Convert the EAGLE-3 checkpoint with `--target-model-dir` so it inherits the target's tokenizer and the layer
indices to read. Both the SpecForge `LlamaForCausalLMEagle3` and the vLLM/AngelSlim `Eagle3LlamaForCausalLM`
checkpoint formats are supported (for example [`AngelSlim/Qwen3-4B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
for `Qwen/Qwen3-4B`):

```bash
python convert_hf_to_gguf.py AngelSlim/Qwen3-4B_eagle3 \
    --target-model-dir Qwen/Qwen3-4B --outtype bf16 --outfile Qwen3-4B-eagle3.gguf

llama-server -m Qwen3-4B.gguf -md Qwen3-4B-eagle3.gguf --spec-type draft-eagle3
```

Supported EAGLE-3 draft models include:

- [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
- [yuhuili/EAGLE3-LLaMA3.3-Instruct-70B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B)
- [RedHatAI/gemma-4-31B-it-speculator.eagle3](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3)
- [RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3)
- [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3)
- [Tengyunw/qwen3_30b_moe_eagle3](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3)
- [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)
- [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
- [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)
- [AngelSlim/Qwen3-14B_eagle3](https://huggingface.co/AngelSlim/Qwen3-14B_eagle3)
- [AngelSlim/Qwen3-32B_eagle3](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)
- [AngelSlim/Qwen3-a3B_eagle3](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3)
- [RedHatAI/gpt-oss-20b-speculator.eagle3](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3)
- [lmsys/EAGLE3-gpt-oss-120b-bf16](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16)
- [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)

For the full and up-to-date list of supported models, see #18039.

### DFlash (`draft-dflash`)

DFlash produces an entire block of draft tokens in a single forward pass (block diffusion) and
injects the target model's hidden states into the draft model's attention, instead of drafting one
token at a time. This keeps the draft model small while making drafting GPU-friendly. Unlike EAGLE-3
(a single-layer autoregressive draft), the DFlash draft uses several transformer layers but emits a
whole block per draft step.

The draft is a small block-diffusion model trained for a specific target (for example
`z-lab/Qwen3-4B-DFlash` for `Qwen/Qwen3-4B`). Convert it with `--target-model-dir` so it inherits the
target's tokenizer and token embeddings:

```bash
python convert_hf_to_gguf.py z-lab/Qwen3-4B-DFlash \
    --target-model-dir Qwen/Qwen3-4B --outtype bf16 --outfile Qwen3-4B-DFlash.gguf

llama-server -m Qwen3-4B.gguf -md Qwen3-4B-DFlash.gguf \
    --spec-type draft-dflash --spec-draft-n-max 15 -fa on --jinja
```

`--spec-draft-n-max` is clamped to the draft model's trained block size.

See:

- #22105

### DSpark (`draft-dspark`)

DSpark extends DFlash with a semi-autoregressive _Markov head_: the draft still emits a whole
block per forward pass, but each block position's logits are biased by a low-rank term keyed on
the previous token, chained in-graph across the block. This keeps drafting at one decode per
block while recovering some of the left-to-right signal that pure block diffusion loses.

The draft is a small DeepSpec checkpoint trained for a specific target (for example
[`deepseek-ai/dspark_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)
for `Qwen/Qwen3-4B`). Convert it with `--target-model-dir` so it inherits the target's tokenizer
and token embeddings:

```bash
python convert_hf_to_gguf.py deepseek-ai/dspark_qwen3_4b_block7 \
    --target-model-dir Qwen/Qwen3-4B --outtype bf16 --outfile Qwen3-4B-DSpark.gguf

llama-server -m Qwen3-4B.gguf -md Qwen3-4B-DSpark.gguf \
    --spec-type draft-dspark --spec-draft-n-max 7 -fa on --jinja
```

`--spec-draft-n-max` is clamped to the draft model's trained block size.

`--spec-draft-conf-min P` truncates each drafted block at the first position whose predicted
acceptance (from the draft's confidence head, if present) falls below `P` (default 0 = disabled).

Currently only drafts with a Qwen3 backbone are supported; support for other backbones
(e.g. Gemma4) is planned.

DSpark drafts exported in the [speculators](https://github.com/vllm-project/speculators) format
(for example [`RedHatAI/gemma-4-31B-it-speculator.dspark`](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark))
convert the same way.

See:

- #25173

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

Two such files can be supplied:

- `-lcs`, `--lookup-cache-static FNAME`: a prebuilt corpus, used read-only for validation. It is never written back, and a missing file is a hard error.
- `-lcd`, `--lookup-cache-dynamic FNAME`: statistics accumulated from generation. It is written back when the server shuts down gracefully (SIGINT/SIGTERM, and when the server goes to sleep), so lookup statistics survive a restart. Pointing it at a file that does not exist yet is the normal way to start a fresh cache - the server warns and begins with an empty one.

The write-back merges what was generated since startup into the statistics that were loaded, and replaces the file atomically (write to `FNAME.tmp`, then rename), so an interrupted write cannot destroy the cache already on disk. A process killed outright (`SIGKILL`) never reaches the write-back and loses the statistics of that run.

```
llama-server [...] --spec-type ngram-cache -lcd /var/lib/llama/lookup.bin
```

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

The pool is never wiped by default. `--spec-ngram-mod-reset-occupancy F` restores the old behaviour of wiping it at the start of a request once more than the fraction `F` of its slots is used, and `--spec-ngram-mod-size N` sets the number of slots (4 bytes each). A slot that keeps getting its drafts rejected does not wipe the shared pool either: it stops drafting from it for a while (16 draft calls, doubling up to 1024 on repeated streaks) and lets the next speculative type take the turn.

With `--spec-ngram-mod-cache FNAME` the pool is loaded at startup and written back on graceful shutdown, so it survives server restarts. A missing file starts an empty pool; a file built with a different `--spec-ngram-mod-n-match` or `--spec-ngram-mod-size` is ignored and replaced on shutdown.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

When an n-gram type is combined with a draft model (for example `--spec-type draft-mtp,ngram-mod`), the n-gram
draft wins the step whenever it has one, and by default it is cut to the draft model's cap
(`--spec-draft-n-max`, `--spec-draft-ctx-step`, `--spec-draft-auto`). n-gram drafts cost nothing to build, so
`--spec-ngram-draft-n-max N` (server only) gives them their own cap, bounded only by the context room and the
remaining `n_predict`. The auto tuner ignores the steps an n-gram type drafted, so long n-gram verify batches do not
end up in the draft model's buckets.

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

If a draft model is combined with a draftless decoding the draftless decoding has higher precedence.

### Backend Sampling

Use `--backend-sampling` to run supported target-model samplers on the model backend. Draft-model sampling uses the backend by default and can be controlled with `--spec-draft-backend-sampling` and `--no-spec-draft-backend-sampling`.

Unsupported samplers and device layouts fall back to CPU sampling. Tensor split mode does not support backend sampling. A fixed seed produces repeatable random draws, but stochastic CPU and backend sampling can still select different tokens because floating-point operations can differ between implementations and devices. Use greedy sampling when exact output matching is required.

### Synthetic Acceptance

`llama-server` and `llama-cli` can replace normal speculative verification with synthetic decisions for benchmarking. The generated output is not valid model output because accepted draft tokens do not have to match the target model.

Use exactly one of these options:

- `--spec-synth-rates P0,P1,...` sets unconditional per-position acceptance probabilities. Entry `i` is the probability that the first `i+1` draft tokens are all accepted. The number of entries must match the effective maximum draft length. Values must be finite, within `[0, 1]`, and monotonically non-increasing.
- `--spec-synth-len L` sets the target mean acceptance length, including the target token. For `K` maximum draft tokens, `L` must be within `[1, K+1]`. The server finds a constant conditional probability `p` such that `p + p^2 + ... + p^K = L - 1`, then uses unconditional rates `[p, p^2, ..., p^K]`.

### General Speculative Parameters

```
--spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        comma-separated list of types of speculative decoding to use
                                        (default: none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
--spec-default                          use default speculative decoding config
                                        (enables ngram-mod)
```

### Draft Model Parameters

```
--spec-draft-model, -md, --model-draft  FNAME
                                        draft model for speculative decoding (default: unused)
                                        (env: LLAMA_ARG_SPEC_DRAFT_MODEL)
--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft  <user>/<model>[:quant]
                                        HuggingFace repository for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_HF_REPO)
--spec-draft-n-max                      N
                                        number of tokens to draft for speculative decoding (default: 3)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min                      N
                                        minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-ctx-step                   N
                                        reduce the draft length by one for every N tokens of context, never below 1 (default: 0, 0 = disabled)
                                        useful with a large --spec-draft-n-max: long drafts pay off at short context and stop paying off once the verify batch is dominated by KV reads
                                        (env: LLAMA_ARG_SPEC_DRAFT_CTX_STEP)
--spec-draft-auto
                                        automatically tune the draft length per slot, using --spec-draft-n-max as the ceiling: keeps a running estimate of tokens landed per unit of wall time for every draft length tried, uses the best one and periodically probes its neighbours (default: disabled)
                                        (env: LLAMA_ARG_SPEC_DRAFT_AUTO)
--spec-draft-p-split, --draft-p-split   P
                                        speculative decoding split probability (default: 0.10)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_SPLIT)
--spec-draft-p-min, --draft-p-min       P
                                        minimum speculative decoding probability (greedy) (default: 0.00)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
                                        max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS_DRAFT)
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
                                        comma-separated list of devices to use for offloading the draft model
                                        (use --list-devices to see available devices)
```

`--spec-draft-n-max` is a fixed cap for the whole run, but the best draft length is not: the target's
verify batch reads the KV cache for every drafted token, so a draft that wins at 1k tokens of context
can cost more than it saves at 32k. `--spec-draft-ctx-step N` makes the cap follow the context - the
effective per-step limit is `max(1, n_max - n_tokens_in_context/N)`, evaluated per slot on every
generation step. For example, `--spec-draft-n-max 16 --spec-draft-ctx-step 2048` drafts 16 tokens on a
short prompt, 8 tokens at 16k of context, and 1 token past 30k. The default `0` disables the schedule
and keeps `n_max` constant.

`--spec-draft-ctx-step` needs you to know the shape of the curve up front. `--spec-draft-auto` measures
it instead: for every draft length it tries, each slot keeps a running estimate of how many tokens the
step landed and how long the step took, then runs on the length with the best tokens per second and
probes a neighbour every 8 steps to follow the curve as the context grows or the acceptance rate
changes. `--spec-draft-n-max` stays the ceiling (and `--spec-draft-ctx-step`, if set, still applies on
top of it). The estimates live per slot and carry across requests. Server logs print the table as
`draft auto = n=6 | 4:98.1 t/s 5:104.3 t/s ...` at the end of each request.

### Draft Model CPU Scheduling Parameters

```
--spec-draft-threads, -td, --threads-draft  N
                                        number of CPU threads to use during generation
--spec-draft-threads-batch, -tbd, --threads-batch-draft  N
                                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
--spec-draft-cpu-mask, -Cd, --cpu-mask-draft  M
                                        Draft model CPU affinity mask. Complements cpu-range-draft
--spec-draft-cpu-range, -Crd, --cpu-range-draft  lo-hi
                                        Ranges of CPUs for affinity. Complements --cpu-mask-draft
--spec-draft-cpu-strict, --cpu-strict-draft  <0|1>
                                        Use strict CPU placement for draft model (default: same as --cpu-strict)
--spec-draft-prio, --prio-draft  N
                                        set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime
--spec-draft-poll, --poll-draft  <0|1>
                                        Use polling to wait for draft model work (default: same as --poll)
--spec-draft-cpu-mask-batch, -Cbd, --cpu-mask-batch-draft  M
                                        Draft model CPU affinity mask for batch. Complements cpu-range-batch-draft
--spec-draft-cpu-range-batch, -Crbd, --cpu-range-batch-draft  lo-hi
                                        Ranges of CPUs for affinity for batch. Complements --cpu-mask-batch-draft
--spec-draft-cpu-strict-batch, --cpu-strict-batch-draft  <0|1>
                                        Use strict CPU placement for draft model batch (default: --cpu-strict-draft)
--spec-draft-prio-batch, --prio-batch-draft  N
                                        set draft process/thread priority for batch : 0-normal, 1-medium, 2-high, 3-realtime
--spec-draft-poll-batch, --poll-batch-draft  <0|1>
                                        Use polling to wait for draft model work for batch (default: --poll-draft)
```

### Draft Model KV Cache and Tensor Override Parameters

```
--spec-draft-type-k, -ctkd, --cache-type-k-draft  TYPE
                                        KV cache data type for K for the draft model
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (env: LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K)
--spec-draft-type-v, -ctvd, --cache-type-v-draft  TYPE
                                        KV cache data type for V for the draft model
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (env: LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V)
--spec-draft-override-tensor, -otd, --override-tensor-draft  <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type for draft model
--spec-draft-cpu-moe, -cmoed, --cpu-moe-draft
                                        keep all Mixture of Experts (MoE) weights in the CPU for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_CPU_MOE)
--spec-draft-n-cpu-moe, --spec-draft-ncmoe, -ncmoed, --n-cpu-moe-draft  N
                                        keep the MoE weights of the first N layers in the CPU for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE)
```

### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
--spec-ngram-mod-size                   N
                                        ngram-mod hash table size in slots, 4 bytes each (default: 4194304 = 16 MB)
                                        (env: LLAMA_ARG_SPEC_NGRAM_MOD_SIZE)
--spec-ngram-mod-reset-occupancy        F
                                        wipe the ngram-mod table at the start of a request when its occupancy exceeds this fraction, 0 = never (default: 0.00)
                                        (env: LLAMA_ARG_SPEC_NGRAM_MOD_RESET_OCCUPANCY)
--spec-ngram-mod-cache                  FNAME
                                        path to an ngram-mod table file, loaded at startup and written back on graceful shutdown (default: none)
                                        (env: LLAMA_ARG_SPEC_NGRAM_MOD_CACHE)
--spec-ngram-mod-shadow-n               N
                                        log only: lookup length of a second ngram-mod table that never drafts, its would-land stats are printed per request, persisted next to --spec-ngram-mod-cache as FNAME.shadow-nN when that is set, 0 = off (default: 0)
                                        (env: LLAMA_ARG_SPEC_NGRAM_MOD_SHADOW_N)
--spec-ngram-draft-n-max                N
                                        cap for n-gram drafts (all n-gram types), independent of the draft-model cap and the
                                        auto tuner; -1 = same cap as the draft model (default: -1)
                                        (env: LLAMA_ARG_SPEC_NGRAM_DRAFT_N_MAX)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
```

### `--spec-type TYPE`

Specifies a comma-separated list of speculative decoding types to use.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding (default) |
| `draft-simple` | Use a simple draft model for speculation |
| `draft-eagle3` | Use an EAGLE-3 draft model that reads the target's hidden states |
| `draft-dflash` | Use a DFlash block-diffusion draft model that emits a block per step |
| `draft-dspark` | Use a DSpark draft model (DFlash backbone + semi-autoregressive Markov head) |
| `draft-mtp` | Use Multi Token Prediction (MTP) heads from the main model |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram-keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram-keys and up to four m-gram values (experimental) |
| `ngram-mod` | Use basic ngram hasher for speculative decoding with shared pool |

**Example:** Server-instance used to refactor source code.
```bash
./llama-server [...] --spec-type ngram-simple
```

**Example:** Multiple speculative implementations.
```bash
./llama-server [...] --spec-type ngram-mod,ngram-map-k4v
```

### `--spec-ngram-*-size-n N`

Sets the size N of the lookup n-gram for n-gram map based speculative decoding.
The n-gram size N determines how many tokens in a row to look back when searching for matching patterns.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-n` for `ngram-simple`
- `--spec-ngram-map-k-size-n` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-n` for `ngram-map-k4v`
- `--spec-ngram-mod-n-match` for `ngram-mod`

### `--spec-ngram-*-size-m M`

Sets the size M of the draft m-gram for n-gram map based speculative decoding.
The m-gram size determines how many tokens to draft when a match is found.
Larger values can provide more speedup but may reduce acceptance rate.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-m` for `ngram-simple`
- `--spec-ngram-map-k-size-m` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-m` for `ngram-map-k4v`

### `--spec-ngram-*-min-hits H`

This option defines how often a key has to appear in the token history to be used as a draft (default is 1).

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-min-hits` for `ngram-simple`
- `--spec-ngram-map-k-min-hits` for `ngram-map-k`
- `--spec-ngram-map-k4v-min-hits` for `ngram-map-k4v`

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).

The server also prints, at the default log level and after the `draft acceptance` line of each request, one line per
implementation that won at least one step:

```
draft by impl: ngram-mod | steps 13 | built 7.8 | offered 7.4 | accepted 1.7 (0.23) | full 1 | step 0.8 ms | 3564.5 t/s | acc hist 0:4 1-4:8 5-8:1 9-16:0 17-32:0 33-48:0 49+:0 | built hist 0:0 1-4:0 5-8:13 9-16:0 17-32:0 33-48:0 49+:0
draft by impl: none | steps 164 | step 0.2 ms
```

- `built`, `offered`, `accepted`: mean draft length per step as built by the implementation, after the cap (what the
  target verified), and accepted; the ratio is accepted / offered
- `full`: steps where every offered token was accepted - many of them mean the cap is censoring the drafter
- `step`, `t/s`: mean wall time of a draft + verify step, and landed tokens (accepted + the sampled one) per second
- `acc hist`, `built hist`: steps per length bucket
- `none`: steps where drafting was attempted but no implementation produced a draft

Each `draft by impl` line ends with `ms by rows: R:step/draft(count) ...`: for every verify batch size R (offered
tokens + 1, 65+ pooled) the mean step wall time and the mean time spent in the drafters' `draft()` calls, in ms, and
the number of steps. This is the measured cost curve of the target: the Vulkan mat-vec path ends at 12 columns, so
expect a jump at R = 13.

Shadow lines (log only, nothing changes what is drafted or verified) follow when they have data:

```
chain truth: ngram | chains 32 | censored 1 | would land mean 5.4 | full 16 | hist 0:3 1-4:8 5-8:21 9-16:0 17-32:0 33-48:0 49-64:0 65+:0
ext shadow: mtp+ngram | attempts 718 | hits 40 | tested 12 | landed mean 9.5 (over tested) | hist ... | built hist ...
alt table: n=8 | attempts 165 | hits 12 | would land mean 2.0 | hist ... | built hist ... | censored 0
mtp confidence: stop cap:600(acc 3.9) pmin:118(acc 2.1) other:0 | p bins landed/tested: <0.5 a/b 0.5 c/d 0.6 e/f 0.7 g/h 0.8 i/j 0.9 k/l
```

- `chain truth`: every n-gram chain is kept in full (before the cap) and compared with the tokens the target later
  committed; `would land` is how many of the chain would have been accepted with no cap, `censored` chains ended with
  the request before they resolved. Input for choosing `--spec-ngram-draft-n-max`.
- `ext shadow`: on steps a draft model won, the ngram-mod table is looked up with the context plus the draft; `hits`
  built a chain, `tested` are hits where the whole draft landed so the extension was really exercised, `landed` is how
  many extension tokens would have been accepted. Input for an "n-gram extends the draft-model draft" feature.
- `alt table` (`--spec-ngram-mod-shadow-n N`): a second ngram-mod table with N-token keys, looked up only when the main
  table did not draft; `would land` as above. With `--spec-ngram-mod-cache FNAME` it is persisted as `FNAME.shadow-nN`
  (loaded at startup, written on shutdown like the main table), otherwise it is RAM only and starts cold.
- `mtp confidence`: why draft-mtp stopped (`cap`, `pmin` or `other`, with the mean accepted length of those steps) and,
  per probability bin of the offered token, how many landed of how many the target tested. The gate `--spec-draft-p-min`
  checks the top candidate; in probabilistic mode the offered token can have a lower p than the gate saw.

With `-v` every step is also logged as `spec step: impl=... built=... offered=... accepted=... ext=... t=... ms`
(`ext` = length of the shadow extension built that step). `spec_log_summary.py` in the bench folder sums these lines
over a log file.

## Benchmarking

To measure the end-to-end effect of speculative decoding (throughput, latency, and draft acceptance) across diverse prompts, see the SPEED-Bench client in [tools/server/bench/speed-bench](../tools/server/bench/speed-bench/README.md).
It runs against a running `llama-server` and can compare a baseline run against a speculative-decoding run.
