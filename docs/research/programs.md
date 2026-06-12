# Autoresearch Programs

This file collects the protocols that guided the autonomous optimization loops.

## Contents

- [QwenASR Autoresearch Program](#qwenasr-autoresearch-program)
- [QwenASR Codex Autoresearch Program](#qwenasr-codex-autoresearch-program)

## QwenASR Autoresearch Program

## QwenASR Autoresearch Program

> Autonomous optimization of the QwenASR Rust inference engine using the autoresearch pattern.
> Human writes this file. Agent executes the loop.

### Goal

Maximize inference performance (lower latency, higher realtime factor) and/or improve transcription accuracy (lower WER/CER) of the QwenASR Rust inference engine on the current hardware, without breaking correctness.

### Setup Phase (one-time, confirm with human)

1. Create the branch: `git checkout -b autoresearch/<tag>` from current `main`.
2. Read the in-scope files. The repo is a Rust workspace. Read these for full context:
   - `README.md` — repository context, build instructions, feature flags
   - `Cargo.toml` — workspace config, dependencies, features
   - `crates/` — the core library crate(s)
   - `src/` — CLI entry point and all core modules:
     - `kernels/` — SIMD kernels (generic, NEON, AVX), BLAS bindings, thread pool ← **primary optimization target**
     - `encoder.rs` — Conv2D stem + windowed transformer
     - `decoder.rs` — GQA decoder + KV cache
     - `audio.rs` — WAV decode, resample, mel spectrogram
     - `transcribe.rs` — offline / segmented / streaming orchestration
     - `config.rs`, `context.rs`, `safetensors.rs`, `tokenizer.rs`
   - `bench/` — benchmark scripts and comparison tools
   - `tests/` — kernel, audio, tokenizer, regression tests
3. Verify model exists: Check that `qwen3-asr-0.6b/` (or `qwen3-asr-1.7b/`) contains model files. If not, tell the human to download them per README.
4. Verify benchmark audio exists: Check `bench/samples/` for WAV files and `audio.wav` in root. If missing, tell human.
5. Establish baseline:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | tail -5
   bench/run.sh --label baseline --runs 3
   ```
6. Initialize `results.tsv` with header:
   ```
   experiment	description	build_ok	test_ok	offline_time_ms	offline_rtf	segmented_time_ms	segmented_rtf	streaming_time_ms	streaming_rtf	status
   ```
   Record baseline results as experiment 0.
7. Confirm setup looks good. Once you get human confirmation, kick off experimentation.

### Experiment Loop

Repeat indefinitely:

#### 1. Pick an idea

Choose ONE focused change per experiment. Ideas to explore (not exhaustive):

**Kernel / SIMD optimizations:**
- Vectorize hot loops that are still using generic.rs fallbacks
- Improve NEON/AVX kernel implementations (fused ops, reduce branching)
- Optimize matmul tiling, cache blocking, prefetch hints
- Try different BLAS call patterns or batch sizes
- Reduce unnecessary memory allocations in hot paths

**Decoder / Encoder optimizations:**
- KV cache memory layout (contiguous vs strided, pre-allocation)
- Attention computation optimizations (fused softmax, flash-attention-style chunking)
- Layer fusion opportunities (combine adjacent operations)
- Reduce redundant computation in streaming mode rollback

**Audio pipeline:**
- Mel spectrogram computation optimization
- FFT/windowing optimizations
- Silence detection efficiency

**Memory & allocation:**
- Reduce heap allocations in the inference hot path
- Use arena allocators or pre-allocated buffers
- Optimize tensor layout for cache locality

**Architecture-level:**
- Parallelism tuning (thread count, work distribution)
- Batch processing of encoder windows
- Async I/O for audio loading

#### 2. Implement the change

Edit the relevant Rust source file(s). Keep changes minimal and focused. Write a clear git commit message describing the hypothesis.

```bash
git add -A && git commit -m "experiment: <brief description of change>"
```

#### 3. Build

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 > build.log
```

If build fails, read `tail -30 build.log`, attempt a fix. If you can't fix after 2 attempts, revert and move on.

#### 4. Run tests (quick sanity check)

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release --test kernels --test audio 2>&1 > test.log
tail -5 test.log
```

If tests fail, the change broke correctness. Revert immediately.

#### 5. Benchmark

```bash
bench/run.sh --label "exp-$(date +%s)" --runs 3 > bench.log 2>&1
```

If bench/run.sh is not available or doesn't work, fall back to direct timing:

```bash
## Offline mode benchmark
time ./target/release/qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent > /dev/null 2>&1

## If segmented mode test audio exists:
time ./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav -S 30 --silent > /dev/null 2>&1
```

Run each 3 times, take the best (lowest) time.

#### 6. Evaluate

Extract timing results. Compare against the current best baseline.

**Keep criteria (ALL must hold):**
- Build succeeds
- Tests pass
- Inference time improved (even by a small margin) OR accuracy improved without significant speed regression
- No correctness regression (spot-check transcript output hasn't degraded)

#### 7. Keep or revert

**If improved:**
```bash
## Record in results.tsv
echo "<exp>\t<description>\tyes\tyes\t<offline_ms>\t<offline_rtf>\t<seg_ms>\t<seg_rtf>\t<stream_ms>\t<stream_rtf>\tkept" >> results.tsv
## This commit stays. It becomes the new baseline.
```

**If not improved or regressed:**
```bash
echo "<exp>\t<description>\tyes\tyes\t<offline_ms>\t<offline_rtf>\t<seg_ms>\t<seg_rtf>\t<stream_ms>\t<stream_rtf>\treverted" >> results.tsv
git reset --hard HEAD~1
```

#### 8. Repeat

Go back to step 1. Try a different idea. Learn from what worked and what didn't.

### Rules

- **Do NOT modify test files** to make tests pass. If tests fail, the code change is wrong.
- **Do NOT modify bench/ scripts** unless they are genuinely broken.
- **One idea per experiment.** Compound changes make it impossible to attribute results.
- **Always build in release mode** with `--release` and `target-cpu=native`.
- **Redirect output.** Never let build/test/bench output flood your context. Use `> file.log 2>&1`.
- **Be bold but reversible.** Try architectural changes, not just parameter tweaks. Git makes everything reversible.
- **Track everything** in results.tsv. Do not commit results.tsv (keep it untracked).
- **If stuck after 3 failed experiments in a row**, step back and re-read the source to find a new angle.
- **Unsafe Rust:** You may use `unsafe` blocks for performance-critical SIMD code, but be extra careful. Run tests after every unsafe change.

### Context for the Agent

This is a pure Rust, CPU-only ASR inference engine. There is no GPU, no CUDA, no Python in the hot path. Performance gains come from:
1. Better utilization of CPU SIMD (NEON on ARM, AVX2+FMA on x86)
2. Better memory access patterns (cache-friendly layouts)
3. Reducing allocations and copies
4. Algorithmic improvements in the attention/matmul/FFT paths
5. Better thread utilization

The model weights are fixed (loaded from safetensors). You are optimizing the inference code, not training a model. The "loss function" here is wall-clock inference time for a given audio input.

---

## QwenASR Codex Autoresearch Program

## QwenASR Codex Autoresearch Program

> Autonomous optimization of the QwenASR Rust inference engine using a code-grounded experiment loop.
> Human writes this file. Agent executes the loop indefinitely.
> This version is allowed to implement, benchmark, keep, or revert changes.

### Goal

Run an infinite code-grounded optimization loop for the Rust CPU inference engine, with emphasis on:

- reducing repeated weight-preparation cost in decoder prefill and large static projections
- reducing data movement and transient allocations in encoder convolution and transcription orchestration
- filling missing high-value platform coverage, especially x86_64 INT8 decode kernels

Hard constraints for this document and the experiment loop:

- CPU-only, Rust-only inference engine
- correctness first; do not trade away transcription quality casually
- no public API changes unless they are clearly required by an internal optimization
- no model architecture changes, no training changes, no GPU work
- one focused idea per experiment
- every kept improvement must stay as a git commit
- every reverted experiment must leave the branch clean except for this document and intentionally preserved notes

### Setup Phase

1. Create the branch from current `main`:
   ```bash
   git checkout -b codex-auto-research
   ```
2. Read the repository context and benchmark protocol:
   - [README.md](../README.md)
   - [comparison.md](../benchmarks/comparison.md)
   - [ledger.md](./ledger.md)
   - [bench/run.sh](../../bench/run.sh)
3. Read the current implementation hotspots and state holders:
   - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
   - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs)
   - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs)
   - [crates/qwen-asr/src/audio.rs](crates/qwen-asr/src/audio.rs)
   - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs)
   - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs)
   - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
   - [crates/qwen-asr/src/kernels/avx.rs](crates/qwen-asr/src/kernels/avx.rs)
   - [crates/qwen-asr/src/kernels/neon.rs](crates/qwen-asr/src/kernels/neon.rs)
4. Verify local assets needed for non-mutating research runs:
   - model directories such as `qwen3-asr-0.6b/` and `qwen3-aligner-0.6b/`
   - benchmark inputs under `bench/samples/`
   - optional larger regression assets if present
5. Establish a baseline before the first optimization experiment:
   ```bash
   git status --short --branch
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   cargo test --release --test regression -- --nocapture
   bench/run.sh --label codex-research-baseline --runs 3
   ```
   If assets are missing, record the blocked measurement and continue with the best available checks.
6. Use this file as the living experiment protocol and ledger. Keep the backlog, baseline, and kept-experiment history current.

#### Baseline Snapshot

- Branch: `codex-auto-research`
- Build: `RUSTFLAGS="-C target-cpu=native" cargo build --release` succeeded
- Regression tests: `cargo test --release --test regression -- --nocapture` passed, but several cases self-skipped because the test harness expects specific local model/reference asset filenames
- Benchmark working baseline from `bench/run.sh --label codex-exp-encoder-x-reuse --runs 3`:
  - offline: `1318ms`, `21.37x` realtime, `WER=0.0270`
  - segmented: `1326ms`, `21.24x` realtime, `WER=0.0270`
  - streaming: `5037ms`, `5.59x` realtime, `WER=0.0270`
- Offline `--profile` snapshot on `bench/samples/audio.wav` after `exp-03`:
  - total inference: `1472ms`
  - encode: `387ms`
  - decode: `1085ms`
  - hottest counters:
    - `sgemm`: `476.0ms`
    - `bf16_matvec`: `399.9ms`
    - `attention_causal`: `252.3ms`
    - `conv2d_op`: `126.6ms`
- Initial read of this baseline:
  - decode remains the larger half of end-to-end latency
  - encoder forward reuse of `x` and `window_starts` improved all three benchmark modes, even though the single `--profile` spot check was noisier than the benchmark sweep
  - the next experiment should likely stay in encoder/transcription memory reuse, especially reducing the remaining `enc_output` allocation or window metadata rebuilds

#### Kept Experiment Ledger

Append one entry for each kept optimization. Keep entries compact and chronological.

Template:

```md
- `exp-NN`
  - date:
  - hypothesis:
  - touched files:
  - benchmark delta:
  - correctness check:
  - kept commit:
  - notes:
```

- `exp-01`
  - date: `2026-04-17`
  - hypothesis: preconverting decoder prefill weights from BF16 to reusable F32 matrices at load time will remove repeated weight-preparation cost from multi-token prefill
  - touched files: `crates/qwen-asr/src/decoder.rs`
  - benchmark delta: offline `1484ms -> 1467ms`, segmented `1493ms -> 1364ms`, streaming `5706ms -> 5311ms`, WER unchanged at `0.0270`
  - correctness check: `RUSTFLAGS="-C target-cpu=native" cargo build --release` succeeded; `cargo test --release --test regression -- --nocapture` passed
  - kept commit: `experiment: prepack decoder prefill weights`
  - notes: direct `--profile` dropped `bf16_matvec` from the earlier `756.1ms` snapshot to `426.2ms`; some of that work moved into `sgemm`, but the end-to-end result stayed positive in offline, segmented, and streaming modes

- `exp-02`
  - date: `2026-04-17`
  - hypothesis: reusing encoder stem temporaries and the `conv2d` im2col workspace across calls will cut allocator churn enough to improve encoder latency
  - touched files: `crates/qwen-asr/src/encoder.rs`, `crates/qwen-asr/src/kernels/mod.rs`
  - benchmark delta: offline `1467ms -> 1387ms`, segmented `1364ms -> 1343ms`, streaming `5311ms -> 5233ms`, WER unchanged at `0.0270`
  - correctness check: `RUSTFLAGS="-C target-cpu=native" cargo build --release` succeeded; `cargo test --release --test regression -- --nocapture` passed
  - kept commit: `experiment: reuse encoder stem workspace`
  - notes: direct `--profile` reduced encode time from `392ms` to `378ms` and nudged `conv2d_op` from `128.8ms` to `125.1ms`; the larger end-to-end gain likely comes from removing repeated stem allocations across chunk processing rather than changing convolution math itself

- `exp-03`
  - date: `2026-04-17`
  - hypothesis: reusing the encoder forward `x` activation buffer and attention `window_starts` metadata across calls will reduce per-call allocation overhead enough to improve end-to-end latency
  - touched files: `crates/qwen-asr/src/encoder.rs`
  - benchmark delta: offline `1387ms -> 1318ms`, segmented `1343ms -> 1326ms`, streaming `5233ms -> 5037ms`, WER unchanged at `0.0270`
  - correctness check: `RUSTFLAGS="-C target-cpu=native" cargo build --release` succeeded; `cargo test --release --test regression -- --nocapture` passed
  - kept commit: `experiment: reuse encoder forward buffers`
  - notes: benchmark movement was clearly positive in all three modes, but the single direct `--profile` run did not improve proportionally; keep decision is based on the repeated benchmark sweep rather than that one noisier spot profile

### Current State Map

This section must keep a strict distinction between `already optimized`, `partially optimized / still leaking cost`, and `missing opportunity`.
It is the map the agent should use to choose the next experiment.

#### Already optimized

- `decoder` single-token decode on `aarch64` already has INT8 coverage for `QKV`, `O-proj addto`, `SwiGLU`, and `argmax`.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) quantizes decoder weights and `lm_head` at load time.
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) wires `linear_nobias_int8_qkv`, `linear_nobias_int8_addto`, `linear_nobias_int8_swiglu`, and `argmax_matvec_int8`.
  - [ledger.md](ledger.md) records these as prior kept wins.
- KV cache layout is already head-contiguous and tuned for causal attention scans.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) stores cache as `[layer][head][pos][head_dim]`.
  - [ledger.md](ledger.md) shows this was a major kept optimization.
- Encoder transformer path already includes several fused or threaded improvements.
  Evidence:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) uses `linear_accumulate()` for residual branches.
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) has threaded `im2col`, threaded `gelu`, threaded `swiglu_multiply`, persistent thread pool, and BLAS-backed dense paths.
- Decoder prefill now reuses load-time F32 copies of its large projection weights instead of reconverting those BF16 weights on every multi-token prefill.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) stores `wq_weight_f32_prefill`, `wk_weight_f32_prefill`, `wv_weight_f32_prefill`, `wo_weight_f32_prefill`, `gate_up_fused_f32_prefill`, and `down_weight_f32_prefill` per layer.
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) uses `linear_nobias()` with those reusable F32 matrices throughout `decoder_prefill()`.
- Encoder stem now reuses chunk buffers and the `conv2d` im2col workspace across encoder calls.
  Evidence:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) stores reusable `chunk_mel`, `c1`, `c2`, `c3`, `reshaped`, `pe`, and `conv_cols` inside `EncoderBuffers`.
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) exposes `conv2d_with_cols()` so encoder forward can avoid per-call `cols` allocation.
- Encoder forward now reuses its main `x` activation buffer and `window_starts` metadata across calls.
  Evidence:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) stores reusable `x` and `window_starts` inside `EncoderBuffers`.
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) writes chunk projections and transformer updates directly into the reusable `x` slice.

#### Partially optimized / still leaking cost

- Decoder prefill layer projections no longer reconvert weights, but logits-oriented prefill still allocates fresh `x_norm` and `logits` and still converts the `lm_head` through BF16 scratch.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) allocates fresh `x_norm` and `logits` in `decoder_prefill_logits()`.
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) still calls `linear_nobias_bf16_scratch()` for `lm_head` projection in `decoder_prefill_logits()`.
- Encoder convolution uses BLAS and threaded `im2col`, but per-chunk transient buffers are still rebuilt each call.
- Encoder stem and main forward activations are now reused, but encoder forward still allocates a fresh final `enc_output` return buffer on every call.
  Evidence:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) still allocates `enc_output` for each forward pass before returning.
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) still rebuilds `chunk_sizes` per call.
- Streaming and segmented transcription reuse some state, but still copy large embedding buffers aggressively.
  Evidence:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) allocates `input_embeds`, `enc_output`, and `tmp_embed` on hot paths.
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) persists `prev_prefill_embeds` with `to_vec()` and compares prior rows by float slice equality.
- Aligner and logits-oriented paths still materialize full intermediate tensors even after decoder prefill reuse exists elsewhere.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) allocates fresh `x_norm` and `logits` in `decoder_prefill_logits()`.

#### Missing opportunity

- x86_64 INT8 decode kernels are still absent.
  Evidence:
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) returns `unimplemented!()` for non-`aarch64` INT8 matvec, fused QKV, fused SwiGLU, and INT8 argmax.
- There is still no backend-specific packed-B abstraction for repeatedly used matrices beyond plain row-major F32 preconversion.
  Evidence:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) now stores reusable row-major F32 prefill weights, but not a backend-tuned packed representation.
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) eagerly converts many weights to F32, but keeps them in plain row-major vectors.
- Tokenizer loading is repeated across entry points.
  Evidence:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) calls `load_tokenizer()` per top-level transcription flow.
  - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs) independently re-encodes prompt text against the tokenizer path.
- Audio front-end still lacks a reusable workspace and a narrower FFT-specific specialization path.
  Evidence:
  - [crates/qwen-asr/src/audio.rs](crates/qwen-asr/src/audio.rs) allocates `padded`, `windowed_all`, `re`, `im`, `power`, `mel`, plus silence-compaction helper buffers.

### Experiment Loop

Repeat indefinitely. Each pass is one experiment.

1. Pick one focused idea from `Initial Prioritization` or `Opportunity Backlog`.
   Prefer the highest-priority item that still matches the current hotspot profile.
2. Reconfirm the current implementation before editing.
   Read the relevant call chain and check [ledger.md](ledger.md) so the experiment does not duplicate a kept optimization that already landed.
3. State the hypothesis in one sentence.
   Examples:
   - prepacking decoder prefill weights will cut repeated BF16 to F32 conversion cost
   - moving encoder conv temporaries into persistent workspace will reduce allocator noise and conv tail latency
4. Implement exactly one focused change.
   Keep the diff minimal enough that benchmark movement can still be attributed to the change.
5. Build in release mode.
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```
6. Run correctness checks.
   At minimum:
   ```bash
   cargo test --release --test regression -- --nocapture
   ```
   If the changed area has a more targeted test, run that too.
7. Benchmark the result.
   Always run:
   ```bash
   bench/run.sh --label codex-exp-<tag> --runs 3
   ./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav --profile
   ```
   Add segmented, streaming, or aligner spot checks when the experiment targets those paths.
8. Evaluate keep versus revert.
   Keep criteria:
   - build succeeds
   - correctness checks pass
   - benchmark improves in the intended area without unacceptable regression elsewhere
   - profile movement matches the hypothesis, or the end-to-end gain is still clearly positive
9. If the experiment is a win:
   - commit it with a message like `experiment: <brief hypothesis>`
   - append an entry to `Kept Experiment Ledger`
   - update `Baseline Snapshot` only when this becomes the new working baseline
   - continue to the next experiment from the new HEAD
10. If the experiment is not a win:
   - revert the code change and return to the previous HEAD
   - optionally record a short rejected note in local scratch, but do not clutter the ledger with failures
   - continue to the next experiment

Rules for this loop:

- do not batch multiple unrelated optimizations into one experiment
- do not modify tests only to justify an optimization
- do not modify benchmark scripts unless they are actually broken
- do not leave broken or unbenchmarked code committed
- always preserve kept improvements as explicit git commits
- prefer benchmark evidence over intuition when choosing whether to keep a change

### Opportunity Backlog

Every entry below must keep the same fields, so an agent can convert backlog items directly into experiments:

- `Priority`
- `Why`
- `Current evidence`
- `Likely touch points`
- `Expected payoff`
- `Validation metrics`
- `Risks / unknowns`

#### 1. Static weight prepack for decoder prefill and large heads

- `Priority`: `P0`
- `Why`: decoder prefill repeatedly touches the same static matrices and still pays BF16 to F32 conversion plus row-major-to-kernel-unfriendly traversal on every invocation.
- `Current evidence`:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) prefill path repeatedly calls `linear_nobias_bf16_scratch()`.
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) converts the full right-hand weight matrix into scratch for `seq_len > 1`.
  - `lm_head` and aligner head are also static, repeatedly reused projections.
- `Likely touch points`:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
- `Expected payoff`:
  - lower decoder prefill latency
  - smaller gap between first-call and later-call behavior
  - less scratch write traffic
  - more stable cache behavior in prefill and classification head paths
- `Validation metrics`:
  - `--profile` time attributed to `bf16_matvec` and `bf16_to_f32_conv`
  - offline and streaming prefill timings
  - load-time increase versus decode-time decrease
- `Risks / unknowns`:
  - pack format should not overfit one backend too early
  - load-time memory growth may be large on 1.7B
  - `blas` and `no-default-features` builds may want different pack layouts

#### 2. Encoder conv workspace reuse and stem specialization

- `Priority`: `P0`
- `Why`: encoder stem is still a front-end ingestion hotspot with repeated buffer allocation and full `im2col` materialization.
- `Current evidence`:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) allocates `chunk_mel`, `c1`, `c2`, `c3`, `reshaped`, and `pe` per chunk.
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) allocates `cols` in each `conv2d()`.
  - [ledger.md](ledger.md) shows threaded `im2col` helped, implying the path is still relevant.
- `Likely touch points`:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs)
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
- `Expected payoff`:
  - lower encoder latency
  - reduced allocator noise
  - reduced peak temporary memory
  - cleaner path to later specialized `3x3s2 pad1` conv kernels
- `Validation metrics`:
  - `conv2d_op` time from `--profile`
  - encoder-only time in offline and streaming modes
  - peak RSS or coarse process memory if measurable
- `Risks / unknowns`:
  - workspace sizing depends on chunk width and model size
  - specialized stem path must preserve current padding semantics exactly

#### 3. x86_64 INT8 decode coverage

- `Priority`: `P0`
- `Why`: the highest-value single-token decode kernels exist on `aarch64` but are explicitly absent on `x86_64`.
- `Current evidence`:
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) gates INT8 kernels behind `#[cfg(target_arch = "aarch64")]` and uses `unimplemented!()` otherwise.
  - [crates/qwen-asr/src/kernels/avx.rs](crates/qwen-asr/src/kernels/avx.rs) currently contains BF16 and float SIMD work, but no matching INT8 decode kernels.
- `Likely touch points`:
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/avx.rs](crates/qwen-asr/src/kernels/avx.rs)
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs)
- `Expected payoff`:
  - major single-token latency reduction on desktop and server CPUs
  - parity of algorithmic strategy across architectures
  - better value from existing load-time quantization
- `Validation metrics`:
  - token latency in streaming decode
  - `argmax`, `QKV`, `O-proj`, and `FFN` portion timing via `--profile`
  - architecture-specific benchmark comparison on x86 hosts
- `Risks / unknowns`:
  - AVX2 is the minimum viable target; VNNI/AVX512VNNI should remain optional
  - epilogue fusion choices may constrain shared abstractions with `aarch64`

#### 4. Eliminate `conv3 -> reshaped -> conv_out` full reorder

- `Priority`: `P1`
- `Why`: even after conv outputs are computed, the encoder currently performs a full tensor re-layout before the projection into model width.
- `Current evidence`:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs) builds `reshaped` by walking `[channel][freq][time]` into `[time][conv_proj_dim]`.
- `Likely touch points`:
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
- `Expected payoff`:
  - reduced memory traffic after conv stem
  - lower encoder chunk tail latency
  - possible path to packed projection weights
- `Validation metrics`:
  - encoder total time
  - CPU profile or `--profile` if a new counter is added later
  - differential measurement on short versus long chunks
- `Risks / unknowns`:
  - BLAS expects dense row-major inputs; a direct projection path may need a custom kernel or tile packing

#### 5. Direct KV-cache write path for prefill K/V

- `Priority`: `P1`
- `Why`: prefill currently materializes interleaved `pref_k` and `pref_v` buffers and then scatters them into the head-contiguous KV cache.
- `Current evidence`:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) writes K/V into `pref_k` and `pref_v`, then loops over sequence positions to call `k_write_pos()` and `v_write_pos()`.
- `Likely touch points`:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
- `Expected payoff`:
  - less scatter overhead in prefill
  - fewer intermediate writes
  - better alignment with head-contiguous attention consumption
- `Validation metrics`:
  - prefill latency
  - cache-write time from targeted microbenchmarks if later added
  - streaming chunk prefill time
- `Risks / unknowns`:
  - rope and per-head RMSNorm currently operate on the interleaved layout
  - fused direct write may complicate code reuse with single-token decode

#### 6. Reusable transcription embedding workspace

- `Priority`: `P1`
- `Why`: top-level transcription and streaming flows repeatedly allocate and copy large prompt plus encoder embedding buffers even though the sequence shape is predictable enough to reuse storage.
- `Current evidence`:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) allocates `input_embeds`, `enc_output`, and `tmp_embed` in multiple flows.
  - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs) performs similar embedding assembly work.
- `Likely touch points`:
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs)
  - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs)
- `Expected payoff`:
  - reduced allocator churn for offline, segmented, streaming, and aligner flows
  - better reuse of prompt and special-token embeddings
- `Validation metrics`:
  - offline and streaming total time
  - counts of transient allocations from external profilers if available
  - first-run versus repeated-run stability
- `Risks / unknowns`:
  - sequence lengths vary between modes
  - ownership between `QwenCtx`, streaming state, and aligner helpers must stay clear

#### 7. Streaming LCP reuse without float-row snapshot cloning

- `Priority`: `P1`
- `Why`: current reuse logic persists the previous prefix as raw float embeddings and compares row-by-row, which is expensive and fragile versus token-level reuse metadata.
- `Current evidence`:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) stores `prev_prefill_embeds = input_embeds[..prefill_len * dim].to_vec()`.
  - The reuse loop compares float rows to find the common prefix.
- `Likely touch points`:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs)
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
- `Expected payoff`:
  - lower streaming chunk overhead
  - smaller memory copies during rolling prefill reuse
  - more direct mapping between token history and KV reuse
- `Validation metrics`:
  - streaming chunk latency
  - bytes copied into `prev_prefill_embeds` or successor state
  - stale/degen reset behavior stability
- `Risks / unknowns`:
  - prefix equality in embedding space currently also covers encoder output positions, not just tokens
  - cache key design must distinguish prompt, encoder, suffix, and carryover text regions

#### 8. Tokenizer and prompt cache lifetime cleanup

- `Priority`: `P1`
- `Why`: tokenizer reload is not the hottest inner-loop issue, but repeated top-level invocations still pay repeated JSON loading and prompt re-encoding costs.
- `Current evidence`:
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs) loads the tokenizer per transcription call.
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs) already caches prompt tokenization readiness, indicating a natural home for longer-lived tokenizer state.
- `Likely touch points`:
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/transcribe.rs](crates/qwen-asr/src/transcribe.rs)
  - [crates/qwen-asr/src/align.rs](crates/qwen-asr/src/align.rs)
- `Expected payoff`:
  - better repeated-call latency
  - less duplicated setup work between offline, streaming, and aligner flows
- `Validation metrics`:
  - cold versus warm top-level invocation latency
  - prompt-preparation time if profiled separately
- `Risks / unknowns`:
  - keeping tokenizer in `QwenCtx` affects load semantics and memory lifetime

#### 9. Audio front-end workspace and FFT-specific specialization

- `Priority`: `P2`
- `Why`: the mel front-end still materializes several whole-frame matrices and may be leaving performance on the table versus reusable workspace or real-FFT-specific paths.
- `Current evidence`:
  - [crates/qwen-asr/src/audio.rs](crates/qwen-asr/src/audio.rs) allocates `padded`, `windowed_all`, `re`, `im`, `power`, `mel`.
  - Silence compaction also allocates `rms_vals`, `smooth_vals`, `sorted`, and output buffers.
- `Likely touch points`:
  - [crates/qwen-asr/src/audio.rs](crates/qwen-asr/src/audio.rs)
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
- `Expected payoff`:
  - lower encoder-front latency
  - lower peak temporary memory
  - possible library-assisted gains on Apple via vDSP or real-FFT specialization
- `Validation metrics`:
  - isolated mel spectrogram timing
  - offline end-to-end timing on short clips where front-end cost is more visible
  - silence-skip workloads
- `Risks / unknowns`:
  - existing BLAS batching may already be close to optimal on some machines
  - real FFT path must match current spectrogram numerics closely enough

#### 10. Thread-local scratch buffers for fused kernels

- `Priority`: `P2`
- `Why`: several fused kernels still allocate per-thread temporary `Vec`s inside hot closures.
- `Current evidence`:
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs) allocates `gate_up_local` or `gate_buf` inside BF16 and INT8 fused SwiGLU paths.
- `Likely touch points`:
  - [crates/qwen-asr/src/kernels/mod.rs](crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](crates/qwen-asr/src/kernels/neon.rs)
  - [crates/qwen-asr/src/kernels/avx.rs](crates/qwen-asr/src/kernels/avx.rs)
- `Expected payoff`:
  - less allocator overhead in highly threaded decode
  - smoother scaling with worker count
- `Validation metrics`:
  - single-token decode timing under varied thread counts
  - allocator traces if available
- `Risks / unknowns`:
  - thread-local scratch adds complexity to the custom thread pool
  - benefits may be small after larger P0 work lands

#### 11. Parallel load-time pack and quantize

- `Priority`: `P2`
- `Why`: once more load-time prepack work is added, model startup can become the new bottleneck; research should include how to parallelize that work safely.
- `Current evidence`:
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs) already quantizes all decoder and `lm_head` weights during load.
  - There is no explicit parallelization across layers or heads yet.
- `Likely touch points`:
  - [crates/qwen-asr/src/context.rs](crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/decoder.rs](crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/encoder.rs](crates/qwen-asr/src/encoder.rs)
- `Expected payoff`:
  - bounded startup growth after adding packed representations
  - cleaner separation between one-time preprocessing cost and runtime benefit
- `Validation metrics`:
  - wall-clock `QwenCtx::load()` time
  - memory peak during load
  - amortized benefit over repeated transcriptions
- `Risks / unknowns`:
  - loading already relies on mmap lifetimes and owned fused buffers
  - parallelization strategy should not over-contend with BLAS or OS page faults

#### Candidate Internal Design Sketches

These are internal implementation sketches only. They are not public API proposals.

- `PackedLinearWeight`
  - owning scope: `Decoder`, `Encoder`, possibly aligner-specific head holder
  - main fields: original source pointer or owned source buffer, packed data, logical `in_dim`, logical `out_dim`, pack kind, backend affinity
  - phase mapping:
    - stage 1: F32 conversion plus simple blocked layout
    - stage 2: backend-specific packed layout
  - platform affinity:
    - generic stage should work with `blas` and `no-default-features`
    - later variants may diverge for `aarch64` and `x86_64`
- `PackedInt8Weight`
  - owning scope: decoder layer and `lm_head`
  - main fields: packed int8 payload, per-row or per-block scales, optional zero-points if ever needed, epilogue metadata
  - phase mapping:
    - initial x86_64 work can keep per-row scales
    - later versions may add VNNI-friendly block packing
  - platform affinity:
    - `x86_64` first
    - keep compatibility with existing `aarch64` quantized data if possible
- `EncoderConvWorkspace`
  - owning scope: `EncoderBuffers`
  - main fields: `chunk_mel`, `conv1`, `conv2`, `conv3`, `im2col`, `reshaped`, `pe`, capacity markers
  - phase mapping:
    - stage 1: persistent reusable buffers
    - stage 2: tile-local stem workspace for specialized conv
  - platform affinity:
    - architecture-neutral
    - should work in both `blas` and fallback matmul builds
- `TranscribeScratch`
  - owning scope: `QwenCtx` or streaming state
  - main fields: prompt embedding cache, `input_embeds`, `enc_output`, `tmp_embed`, previous-prefix reuse metadata
  - phase mapping:
    - stage 1: reuse exact current shapes and copies
    - stage 2: replace float-row LCP snapshots with structured metadata
  - platform affinity:
    - architecture-neutral

Public API note: this research track assumes no public API changes. All candidate types above are internal ownership and layout aids only.

### Measurement Plan

All measurements in this phase support keep/revert decisions. Always benchmark after implementing a change, and prefer the smallest benchmark set that still validates the hypothesis.

#### Core commands

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
cargo test --release --test regression -- --nocapture
bench/run.sh --runs 3 --label codex-research
./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav --profile
./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav -S 30 --profile
./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav --stream --profile
./target/release/qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English --profile
```

#### Observation buckets

- `offline`
  - total latency
  - encoder time
  - prefill time
  - decode time
- `segmented`
  - per-segment stability
  - prefix / prompt reuse behavior
  - allocator sensitivity across segments
- `streaming`
  - chunk prefill time
  - token latency
  - rollback and degeneracy reset behavior
- `aligner`
  - per-position logits overhead
  - effect of `lm_head` or classify-head related opportunities

#### Topic-specific checks

- static weight prepack:
  - compare first prefill versus repeated prefill
  - inspect `bf16_matvec` profile share
- conv workspace and stem specialization:
  - inspect `conv2d_op` share
  - compare short and long audio sensitivity
- x86 INT8 decode:
  - inspect token latency and `argmax` share
  - compare decode-heavy workloads versus encoder-heavy workloads
- transcription workspace:
  - watch repeated-call stability and transient allocation patterns
- audio front-end:
  - isolate mel spectrogram cost on short clips and silence-skip inputs

If a measurement is blocked by missing local assets or hardware, note the blocker explicitly and use the next-best measurement rather than guessing.

### Rules

- Allowed actions:
  - static reading and code search
  - source edits under `crates/qwen-asr/src/`
  - release builds
  - tests
  - benchmarks
  - profiling
  - document edits to this file
  - git commits for kept improvements
- Avoid editing:
  - tests, unless a real behavior change requires new or updated coverage
  - benchmark scripts, unless they are genuinely broken
  - `results.tsv`, unless the experiment explicitly adopts it as an external ledger
- Keep conclusions tied to actual repository evidence, not generic ML systems advice.
- When a topic is already optimized, say so and move on.
- When a topic is only partially optimized, identify exactly which cost was removed and which cost still remains.
- When an opportunity is missing, specify where the code path stops today and what future work would need to replace it.
- When a kept experiment lands, ensure the working tree is clean except for intentional documentation updates.
- When an experiment loses, revert it completely before starting the next one.

### Initial Prioritization

#### First wave

1. Static weight prepack for decoder prefill and large heads
   - highest expected payoff because it attacks repeated BF16 to F32 conversion and scratch traffic on static weights
2. Encoder conv workspace reuse plus fixed stem specialization study
   - highest front-end data-movement payoff and likely allocator win
3. x86_64 INT8 decode coverage
   - highest platform-coverage payoff and most obvious missing hot path

#### Second wave

4. Eliminate `conv3 -> reshaped -> conv_out` reorder
5. Direct KV-cache write path for prefill K/V
6. Reusable transcription embedding workspace
7. Streaming LCP reuse without float-row snapshot cloning
8. Tokenizer and prompt cache lifetime cleanup

#### Third wave

9. Audio front-end workspace and FFT specialization
10. Thread-local scratch buffers for fused kernels
11. Parallel load-time pack and quantize

#### Default assumptions for follow-up implementation

- No public API changes.
- `aarch64` and `x86_64` may diverge in internal packed layout.
- `blas` builds and `no-default-features` builds must both remain valid.
- Prefer stage-wise delivery:
  - first remove repeated preparation work
  - then reduce copy / reorder overhead
  - then add architecture-specific microkernels
- Do not start from more SIMD micro-functions unless the data path above is already cleaned up.
- If a smaller precursor change can de-risk a larger `P0` idea, it is acceptable as long as it is benchmarked and committed separately.

---

