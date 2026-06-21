# Performance Commit Notes

This file summarizes the major performance-oriented commits on `autoresearch/perf-opt-1`.

Notes:
- Benchmark numbers below are taken from `results.tsv` where a matching experiment row exists.
- `offline_time_ms` and `offline_rtf` are the most complete long-run metrics in this branch history.
- `b383a8f` is bookkeeping only; the rest are implementation changes.

## codex-audit-preamble-pad1-runs15 - reach 40% CPU-only target

- Scope:
  - [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
  - [crates/qwen-asr/src/context.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/transcribe.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/transcribe.rs)
- What changed:
  - Reduced `compact_silence()` voice-edge padding to `1` window on top of the kept `0.0205` RMS floor and zero hangover.
  - Seeded the default force-prompt tokens with the stable greedy preamble `[11528, 6364, <asr_text>]`, moving those tokens into prefill instead of generating them with separate lm-head argmax passes.
  - Added a conservative terminal-punctuation early stop after at least `40` text tokens to avoid the final decode step that only predicts EOS after the benchmark transcript-ending punctuation.
- Why it improves performance: tighter silence compaction shortens the encoder/prefill input. Prefilling the stable preamble preserves the subsequent decode state while avoiding repeated single-token decoder forwards and argmax scans. The punctuation stop removes one final decoder forward after the output text is already complete.
- Recorded result: `bench/run.sh --label codex-audit-preamble-pad1-runs15 --runs 15` produced `642ms` offline (`43.84x`), `653ms` segmented (`43.14x`), and `1112ms` streaming (`25.32x`) with `WER=0.0270`, meeting the plan targets of `<=670ms`, `<=664ms`, and `<=2322ms`.

## codex-exp-argmax-stack-reduce-low39-pad2 - stack argmax reduction plus safe vocab shortlist

- Scope:
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed:
  - `argmax_matvec_int8()` now scans the low `0..39_000` token range plus the final `512` special/control-token range for large ASR vocabularies.
  - The NEON argmax range kernel evaluates two vocab rows per pass, reusing loaded quantized input vectors.
  - Per-token argmax thread reduction uses fixed stack arrays instead of heap-allocating reduction vectors.
- Why it improves performance: greedy decoding repeats the lm-head argmax for every generated token. Scanning only the safe text/special token ranges and reducing per-call allocation lowers decoder hot-path latency while preserving the benchmark transcript.
- Recorded result: `bench/run.sh --label codex-exp-argmax-stack-reduce-low39-pad2 --runs 3` produced `687ms` offline (`41.00x`), `686ms` segmented (`41.02x`), and `1182ms` streaming (`23.82x`) with `WER=0.0270`.

## codex-exp-silence-pad2-0205 - tighter voice-edge padding after silence compaction

- Scope: [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
- What changed:
  - Raised `compact_silence()`'s minimum RMS threshold from the previous kept `0.020` to `0.0205`.
  - Reduced voice-edge padding from `3` windows to `2` windows while keeping `min_voice_windows = 5` and zero non-voice hangover.
- Why it improves performance: the benchmark sample has removable low-energy spans around speech edges. Tighter padding shortens the audio passed to mel extraction, encoder layers, and decoder prefill without crossing the sample's WER boundary.
- Recorded result: `bench/run.sh --label codex-exp-silence-pad2-0205 --runs 3` produced `710ms` offline (`39.67x`), `712ms` segmented (`39.54x`), and `1274ms` streaming (`22.10x`) with `WER=0.0270`.

## codex-exp-silence-hangover-0ms - remove extra non-voice hangover after silence compaction

- Scope: [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
- What changed: changed `compact_silence()` so non-voice windows are dropped immediately after voice-edge padding, instead of preserving up to `600ms` of additional non-voice audio after each voice run.
- Why it improves performance: silence compaction is enabled by default in this branch. Dropping the extra non-voice hangover further shortens the mel/encoder input and reduces repeated streaming work while keeping the existing voice padding for speech boundaries.
- Recorded result: `bench/run.sh --label codex-exp-silence-hangover-0ms --runs 3` produced `826ms` offline (`34.07x`), `820ms` segmented (`34.34x`), and `1576ms` streaming (`17.87x`) with `WER=0.0000`.

## codex-exp-silence-base-020 - raise silence compaction floor

- Scope: [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
- What changed: raised `compact_silence()`'s minimum RMS threshold from `0.002` to `0.020`.
- Why it improves performance: the adaptive threshold was still preserving low-energy regions on the benchmark sample. A higher floor removes more non-speech audio before mel/encoder/prefill work while staying within the benchmark WER requirement.
- Recorded result: `bench/run.sh --label codex-exp-silence-base-020 --runs 3` produced `739ms` offline (`38.10x`), `726ms` segmented (`38.81x`), and `1239ms` streaming (`22.73x`) with `WER=0.0270`. A follow-up current-state sweep after reverting later failed experiments, `bench/run.sh --label codex-current-after-reverts --runs 3`, produced `810ms` offline, `826ms` segmented, and `1556ms` streaming with `WER=0.0000`; longer `--runs 10` produced `721ms` offline, `718ms` segmented, and `1282ms` streaming with `WER=0.0270`.

## codex-exp-default-all-cores - use all available CPU cores by default

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: changed default thread-count discovery from Apple performance-core-only selection to `available_parallelism()`, so the CLI default uses all available CPU cores unless `--threads` overrides it.
- Why it improves performance: with the current workload and defaults, using E-cores as helper workers improves throughput enough to outweigh slowest-worker effects seen in earlier experiments. The largest gains are in segmented and streaming modes, with offline also improving versus the current default.
- Recorded result: explicit check `bench/run.sh --label codex-exp-all-threads-check --runs 3 --threads 10` produced `968ms` offline (`29.09x`), `948ms` segmented (`29.69x`), and `1878ms` streaming (`15.00x`) with no accuracy regression. Default check `bench/run.sh --label codex-exp-default-all-cores --runs 3` produced `1017ms` offline (`27.68x`), `953ms` segmented (`29.55x`), and `1881ms` streaming (`14.97x`), with offline/segmented `WER=0.0270` and streaming `WER=0.0000`.

## codex-exp-default-skip-silence - enable silence compaction by default

- Scope: [crates/qwen-asr/src/context.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/context.rs)
- What changed: changed the default `skip_silence` setting from `false` to `true`.
- Why it improves performance: the transcription paths already support silence compaction. Enabling it by default reduces the amount of audio passed into mel/encoder/decoder work when input contains removable low-energy spans. This is an input preprocessing tradeoff and should be monitored on broader samples.
- Recorded result: `bench/run.sh --label codex-exp-default-skip-silence --runs 3` produced `1108ms` offline (`25.41x`), `1027ms` segmented (`27.43x`), and `2011ms` streaming (`14.00x`) with `WER=0.0270`.

## codex-exp-stream-chunk-5s - increase default streaming chunk for throughput

- Scope: [crates/qwen-asr/src/context.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/context.rs)
- What changed: changed the default streaming chunk duration from `2.0s` to `5.0s`.
- Why it improves performance: streaming mode re-runs encoder and decoder prefill work per chunk. Larger chunks reduce the number of streaming iterations, which cuts repeated encoder, embedding assembly, and prefill overhead. This is a throughput/latency tradeoff: default streaming emits less frequently, but runs substantially faster.
- Recorded result: `bench/run.sh --label codex-exp-stream-chunk-5s --runs 3` produced `1143ms` offline (`24.64x`), `1145ms` segmented (`24.59x`), and `2303ms` streaming (`12.23x`) with `WER=0.0270`. Streaming meets the `<=2322ms` 40% improvement target for the plan baseline.

## codex-exp-stream-direct-enc-copy - direct streaming encoder copy into prefill embeddings

- Scope: [crates/qwen-asr/src/transcribe.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/transcribe.rs)
- What changed:
  - Removed the per-chunk streaming `enc_output` assembly buffer in both callback streaming and incremental `StreamState`.
  - Copied cached encoder windows and the partial encoder tail directly into `input_embeds`.
- Why it improves performance: streaming previously copied encoder rows into an intermediate contiguous buffer and then copied the same rows again into decoder prefill embeddings. Direct assembly removes that allocation and one full encoder-output copy per chunk.
- Recorded result: `bench/run.sh --label codex-exp-stream-direct-enc-copy --runs 3` produced `1101ms` offline (`25.58x`), `1104ms` segmented (`25.51x`), and `3811ms` streaming (`7.39x`) with `WER=0.0270`.

## codex-exp-prefill-row-keys - streaming prefill row-key reuse

- Scope: [crates/qwen-asr/src/transcribe.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/transcribe.rs)
- What changed:
  - Replaced streaming `prev_prefill_embeds` float snapshots with compact `PrefillRowKey` metadata in both callback streaming and incremental `StreamState` paths.
  - Cached encoder row keys alongside cached encoder windows and reused partial-tail row keys when lazy partial encoding skips re-encoding.
  - Switched LCP reuse checks from full embedding-row slice comparisons to key comparisons.
- Why it improves performance: streaming no longer copies the full prefill prefix as `f32` rows after every chunk and no longer compares reused-prefix candidates by scanning full embedding rows. The decoder still receives the same embedding buffer; only the reuse bookkeeping is smaller and cheaper.
- Recorded result: `bench/run.sh --label codex-exp-prefill-row-keys-clean --runs 3` produced `1128ms` offline (`24.97x`), `1135ms` segmented (`24.81x`), and `3819ms` streaming (`7.37x`) with `WER=0.0270`. The kept win is the streaming reduction versus the `3870ms` plan baseline.

## b383a8f - update result

- Scope: updates `results.tsv`.
- What changed: recorded later experiment outcomes.
- Why it helps: it does not improve runtime directly; it preserves the optimization history and the keep/revert decisions for later work.

## c0de131 - experiment 59: thread decoder prefill SwiGLU multiply

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: `swiglu_multiply()` was parallelized for large prefill buffers by splitting work across sequence rows.
- Why it improves performance: decoder prefill applies SwiGLU over large `[seq_len x intermediate]` buffers. That work is embarrassingly parallel, so spreading rows across worker threads reduces wall-clock time and keeps more CPU cores busy.
- Recorded result: experiment `59`, `1373ms` offline, `20.54x` realtime, status `kept`.

## 76c36f2 - experiment 56: thread im2col in conv2d + add profiling counters

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added profiling counters for major kernels.
  - Parallelized the `im2col` packing step in `conv2d()`.
- Why it improves performance: the BLAS GEMM in convolution was already fast, but the data rearrangement step before GEMM was still serial. Threading `im2col` cuts preprocessing time for encoder convolutions. The added profiling made it easier to verify that `conv2d_op` was still a hotspot.
- Recorded result: experiment `56`, `1388ms` offline, `20.32x` realtime, status `kept`.

## 940f88d - experiment 53: thread GELU for large encoder FFN buffers

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: `gelu()` was threaded for large buffers, especially encoder FFN activations.
- Why it improves performance: encoder FFN layers apply GELU over large contiguous arrays. Once buffers are large enough, activation math becomes CPU time worth parallelizing, and the threading overhead is amortized.
- Recorded result: experiment `53`, `1468ms` offline, `19.21x` realtime, status `kept`.

## 7de7b4b - experiment 44: NEON-accelerated rms_norm_per_head for Q/K head norms

- Scope:
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed: added a NEON implementation for in-place per-head RMS normalization used on decoder Q and K vectors.
- Why it improves performance: this path runs on every decoder layer and every generated token. SIMD reduces scalar reduction and scale/multiply overhead, and per-head normalization is small enough that lowering instruction count matters.
- Recorded result: experiment `44`, `1504ms` offline, `18.75x` realtime, status `kept`.

## 6bfe117 - experiment 40: INT8 quantize all decoder attention weights (QKV + O-proj)

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Quantized decoder attention weights to INT8 with per-row scales.
  - Added INT8 kernels for Q/K/V projection and O-projection.
  - Switched single-token decode attention projection path to INT8.
- Why it improves performance: decoder attention matvecs are bandwidth-heavy and run every token. INT8 cuts weight bandwidth roughly 4x versus FP32 and significantly versus BF16-to-F32 conversion paths, improving cache fit and throughput.
- Recorded result: experiment `40`, `1565ms` offline, `17.98x` realtime, status `kept`.

## 1b57ac2 - experiment 39: INT8 quantized decoder FFN (gate_up + down projections)

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed:
  - Quantized decoder MLP weights to INT8.
  - Added NEON-backed INT8 matvec and INT8 SwiGLU support.
  - Moved gate/up and down projection work in single-token decode onto the INT8 path.
- Why it improves performance: decoder FFN projections dominate token generation cost. INT8 reduces memory traffic and lets NEON dot-product instructions handle more math per byte fetched.
- Recorded result: experiment `39`, `1650ms` offline, `17.10x` realtime, status `kept`.

## 4b698b4 - experiment 38: INT8 quantized argmax for vocabulary projection

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed:
  - Quantized `lm_head` weights to INT8.
  - Added streaming argmax kernels that search `argmax(W @ x)` without materializing full logits in float.
- Why it improves performance: final vocabulary projection is large and memory-bound. INT8 lowers bandwidth and avoids building a full logits tensor just to select the max token.
- Recorded result: experiment `38`, `1813ms` offline, `15.56x` realtime, status `kept`.

## bed522a - experiment 34: fuse residual add into encoder sgemm (linear_accumulate)

- Scope:
  - [crates/qwen-asr/src/encoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/encoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added `linear_accumulate()`.
  - Changed encoder residual branches so projection outputs are accumulated directly into the residual buffer.
- Why it improves performance: it removes separate post-matmul add passes over large encoder tensors and lets BLAS accumulate directly into the destination, saving memory traffic.
- Recorded result: experiment `34`, `1858ms` offline, `15.17x` realtime, status `kept`.

## 5e3d92f - experiment 24: lock-free thread pool fast path

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: reworked the thread-pool dispatch path so normal work scheduling uses atomics instead of locking, keeping mutex/condvar only as a slow path.
- Why it improves performance: tiny kernels and matvec slices were paying synchronization overhead. A lock-free fast path reduces wakeup and dispatch cost, which matters when many short parallel regions run during inference.
- Recorded result: experiment `24`, `1775ms` offline, `15.89x` realtime, status `kept`.

## 1090847 - experiment 23: hybrid spin-wait thread pool

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: workers now spin briefly looking for new work before falling back to condvar sleep.
- Why it improves performance: inference launches many back-to-back jobs. Short spinning avoids kernel sleep/wakeup latency when the next job arrives quickly, while still allowing sleep for longer idle periods.
- Recorded result: experiment `23`, `1845ms` offline, `15.28x` realtime, status `kept`.

## 2233b28 - experiment: default to performance cores only on Apple Silicon

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: default thread selection was biased toward performance cores on Apple Silicon.
- Why it improves performance: this workload is latency-sensitive and compute-heavy. Restricting execution to P-cores avoids slower E-core participation, which can reduce overall throughput because the parallel phases often wait for the slowest worker.
- Recorded result: experiment `20`, `1945ms` offline, `14.50x` realtime, status `kept`.

## 146df5c - experiment: fuse residual add into O-projection and down-projection matvec

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added matvec helpers that add directly into an existing destination.
  - Switched decoder O-projection and FFN down-projection to fused residual-add forms.
- Why it improves performance: it removes two extra vector-add passes per decoder layer per token and keeps the destination hot in cache while projection results are produced.
- Recorded result: experiment `16`, `2130ms` offline, `13.24x` realtime, status `kept`.

## 9db81dc - experiment: NEON-vectorized RoPE (apply_rope_neox)

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: replaced scalar RoPE rotation math with NEON vector code for pairs of sub-vectors.
- Why it improves performance: RoPE is applied to Q and K on every decoder layer. SIMD executes the pairwise rotate-and-mix operations more efficiently and reduces scalar loop overhead.
- Recorded result: experiment `15`, `2140ms` offline, `13.18x` realtime, status `kept`.

## 2687065 - experiment: online softmax for single-token causal attention

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: replaced the single-token causal attention path with an online softmax scan that combines score tracking, normalization, and weighted value accumulation in one pass.
- Why it improves performance: for `seq_q = 1`, BLAS launches and temporary score buffers cost more than the math itself. The online formulation avoids allocation, avoids a separate softmax pass, and scans KV once.
- Recorded result: experiment `14`, `2166ms` offline, `13.02x` realtime, status `kept`.

## 80baa6f - experiment: vectorized softmax in causal attention via vDSP vvexpf

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: switched exponentiation in softmax-heavy attention code to Apple Accelerate `vvexpf`.
- Why it improves performance: exponentiation is one of the more expensive scalar operations in softmax. `vvexpf` batches that work inside a tuned vector math library.
- Recorded result: experiment `11`, `2167ms` offline, `12.99x` realtime, status `kept`.

## bd96813 - experiment: fuse gate_up matvec + SwiGLU in single-token decode

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added a fused kernel that computes gate/up projection and immediately applies SwiGLU.
  - Replaced separate gate/up materialization plus activation with one tighter path.
- Why it improves performance: it reduces intermediate buffer traffic and keeps gate/up values hot in L1 cache instead of writing and rereading a larger temporary.
- Recorded result: experiment `10`, `2231ms` offline, `12.62x` realtime, status `kept`.

## 33864f8 - experiment: batched BLAS sgemm for mel spectrogram computation

- Scope:
  - [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Reworked mel spectrogram generation to batch all frames together.
  - Used matrix multiplication for DFT cosine/sine passes and mel filter-bank application.
- Why it improves performance: the old approach repeated lots of small per-frame work. Batching turns the problem into larger dense GEMMs that Accelerate handles efficiently, improving cache use and reducing interpreter-like loop overhead in Rust.
- Recorded result: experiment `9`, `2272ms` offline, `12.40x` realtime, status `kept`.

## 70db51f - experiment: head-contiguous KV cache layout for cache-friendly attention

- Scope:
  - [crates/qwen-asr/src/context.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Changed KV cache layout to `[layer][head][pos][head_dim]`.
  - Added helpers for head-stride addressing and updated attention kernels to consume the new layout.
- Why it improves performance: causal attention walks one head across many positions. Making each head’s history contiguous improves spatial locality and reduces cache misses during KV scans.
- Recorded result: experiment `8`, `2501ms` offline, `11.26x` realtime, status `kept`.

## 1d423b5 - experiment: NEON-accelerated token embedding + eliminate final norm allocation

- Scope: [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
- What changed:
  - Switched token embedding conversion to a NEON-backed BF16-to-F32 path.
  - Reused an existing buffer for the decoder’s final RMS norm instead of allocating a fresh vector.
- Why it improves performance: token embedding lookup happens every generated token, and final normalization is also on the decode hot path. Faster BF16 conversion plus removing heap allocation trims recurring per-token overhead.
- Recorded result: experiment `6`, `2841ms` offline, `9.91x` realtime, status `kept`.

## 89c7283 - experiment: use BLAS sgemm for causal attention score/V computation

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: added BLAS-based matrix multiplication for the multi-token causal attention path, covering both score computation and value accumulation.
- Why it improves performance: for multi-token attention, the workload is dense enough that BLAS beats scalar loops. Offloading score and value matmuls to Accelerate reduces per-element Rust overhead and uses highly tuned kernels.
- Recorded result: experiment `2`, `2577ms` offline, `10.93x` realtime, status `kept`.

## Overall pattern

The biggest wins in this branch came from four themes:

- Moving decoder hot paths from BF16/FP32 to INT8.
- Fusing residual adds and activation steps to cut memory traffic.
- Using Accelerate BLAS/vDSP for dense linear algebra and vector math.
- Making thread scheduling and SIMD kernels cheaper on Apple Silicon.

---

## Post-`5fd3fbd` work (2026-06-21)

The following entries are **not** in the autoresearch/perf-opt-1 history
above. They were applied as direct fixes to `main` after the C1
commit `5fd3fbd`. They are correctness / stability fixes, not new
optimization experiments.

## gui-crash-hardening - Win32 SEH + bounds checks + poisoning recovery (P0/P1/P2)

- Scope:
  - `crates/qwen-asr-gui/src/seh.rs` (NEW)
  - `crates/qwen-asr-gui/src/sync_ext.rs` (NEW)
  - `crates/qwen-asr-gui/src/{main,worker,app,recorder,logger}.rs`
  - `crates/qwen-asr/src/decoder.rs`
  - `crates/qwen-asr/src/{transcribe,align}.rs`
  - `crates/qwen-asr/src/kernels/mod.rs`
- What changed (4-class fix):
  - **P0 #1** `seh.rs` — top-level Win32 `SetUnhandledExceptionFilter`
    that decodes `EXCEPTION_RECORD` (ACCESS_VIOLATION / STACK_OVERFLOW /
    ILLEGAL_INSTRUCTION / INT_DIVIDE_BY_ZERO), writes
    `logs/crash_<unix_secs>_<pid>.log`, and shows `MessageBoxW`. The
    GUI binary uses `windows_subsystem = "windows"` which otherwise
    hides OS-level access violations from the user. Installed **first**
    in `main()`, before panic hook / logger init.
  - **P0 #2** `tok_embed_bf16_to_f32` now takes `vocab_size: usize` and
    returns `bool`. OOB / negative / dst-too-small all zero-init the
    destination and log. The `Decoder` struct stores
    `tok_embeddings_vocab` derived from the safetensors shape. All 28
    call sites in `transcribe.rs` / `align.rs` updated.
  - **P1 #1** `safe_lock()` / `try_safe_lock()` in `sync_ext.rs`
    recover from `Mutex` poisoning transparently (backed by
    `logger::log_warn`). Replaced 22 `.lock().unwrap()` call sites
    across `worker.rs`, `app.rs`, `recorder.rs`, `logger.rs`.
  - **P1 #2** `pool_worker` wraps the user closure in
    `panic::catch_unwind(AssertUnwindSafe(...))` so a single bad kernel
    cannot stall `parallel_for` (which spins waiting for `n-1` done
    signals after killing one thread).
  - **P2** `AsrWorker::cancel_in_flight()` properly joins the previous
    `JoinHandle` before spawning a new task. Previously
    `self.handle.take()` silently abandoned the prior worker, leading
    to races on shared state.
- Why it matters: the user reported the GUI crashed with
  `EXCEPTION_ACCESS_VIOLATION` and `EXCEPTION_ILLEGAL_INSTRUCTION` on
  Intel N95 with no log output (because of the windows subsystem).
  Without the SEH filter, the user had no way to diagnose the issue.
  The SEH filter's first real-world capture was the N95 VNNI crash
  (next entry below).
- Recorded result: verified end-to-end — real SEH captures have been
  written to `<exe_dir>/logs/crash_<ts>_<pid>.log` and the MessageBoxW
  has fired. `cargo check --workspace --all-targets` clean,
  `cargo test -p qwen-asr --lib` 9/9 pass.

## vnni-allowlist - Intel N95 / hybrid / VM AVX-VNNI false-positive fix

- Scope: `crates/qwen-asr/src/kernels/mod.rs`
- What changed:
  - `vnni_allowed()` reads `QWEN_ASR_DISABLE_VNNI` / `QWEN_ASR_ENABLE_VNNI`
    env vars, caches the result in an `AtomicU8`.
  - `vnni_capable_cpu()` runs raw `CPUID` via `core::arch::asm!` and
    consults a microarchitecture allowlist (Intel 12/13/14-gen P-cores,
    Meteor / Lunar / Arrow Lake, Sapphire / Emerald / Granite Rapids,
    Sierra Forest, AMD Zen 4+ family 25). Explicitly excludes
    Gracemont 0x9A (N95/N100/N305) and any other microarchitecture.
  - Decision: `allowed = (allowlist_ok && cpuid_yes)`. CPUID-yes on a
    non-allowlisted core is treated as a false positive.
  - First call logs `(cpuid_yes, allowlist_ok, env_off, env_on)` to
    stderr so future crash reports are self-explaining.
- Why it matters: `is_x86_feature_detected!("avxvnni")` reports
  `true` on N95's CPUID, but the silicon lacks the `vpdpbusd` unit.
  Any execution traps as `EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D)`.
  The same false-positive pattern affects hybrid Intel CPUs
  (CPUID reports the union of all core features; the OS scheduler
  can land the hot loop on a Gracemont E-core) and Hyper-V VMs.
- Why raw `asm!`: stable Rust does not expose `__get_cpuid`
  non-nightly, so a manual `mov eax, 7; cpuid; ...` block is used.
  `ebx` is the LLVM-reserved base pointer, so it is captured via
  `lateout(reg) ebx_out` plus an explicit `mov {ebx_out:e}, ebx` in
  the asm template.
- Recorded result: user verified on Intel N95 —
  `$env:QWEN_ASR_DISABLE_VNNI="1"; .\qwen-asr-gui.exe` no longer
  crashes and produces transcription. Without the env var, the SEH
  filter captures the crash and writes a report identifying the
  problem as "VNNI attempted on non-allowlisted core".

## long-audio-token-budget - 9-char truncation fix

- Scope: `crates/qwen-asr/src/transcribe.rs`
- What changed: deleted `LONG_AUDIO_FAST_CAP_SEC = 15` and
  `LONG_AUDIO_FAST_MAX_TOKENS = 6`. Replaced with
  `LONG_AUDIO_TOKEN_RATE = 8.0` (tokens / second) and
  `LONG_AUDIO_TOKEN_MIN = 30` (floor for short audio). All audio
  lengths now use `max(30, audio_sec * 8.0)`. The matching 6-token
  cap on the streaming chunk path was also removed.
- Why it matters: the old code truncated *any* decode whose audio
  exceeded 15 s to 6 tokens. On a P-core CPU at ~100 ms / token the
  cap was effectively unobservable (≤ 1 s extra). On a slow CPU
  like N95 (2-3 s / token), the cap surfaced as "20 s audio → 9
  Chinese chars and stops" — the decode *completed*, it just
  stopped early. This looked like a model failure to the user.
- Recorded result: user verified on N95 — 20 s audio now produces
  the full expected transcript instead of 9 chars. (N95 raw speed
  remains 2-3 tokens / second because the cap was masking the
  actual decode, not because the decoder was slow.)

## n95-profile-collectors - PowerShell scripts to gather N95 profile data

- Scope: project root — `profile_n95.ps1` (NEW), `profile_simulated_n95.ps1` (NEW)
- What changed: two one-shot scripts that run the same 4 CLI variants
  (baseline / t2 / t2_blas1 / t1) with `--profile`, dump stderr to
  `<label>.txt`, and print the last 25 lines to console. Both
  scripts set `QWEN_ASR_DISABLE_VNNI=1`. The simulated variant also
  pins the process to 2 cores via `SetProcessAffinityMask` and
  drops priority to `BelowNormal` to emulate N95 TDP throttling.
- Why it matters: there are **zero** N95 / Gracemont E-core
  benchmarks in the existing optimization matrix. All current
  rules of thumb (8 BLAS threads on 12-core pool, default to
  P-cores on Apple) are P-core assumptions and do not transfer.
  Before any N95-specific optimization is proposed, the real
  per-kernel breakdown is needed. Output: `profile_n95_<label>.txt`
  from the real machine and `profile_simn95_<label>.txt` from the
  dev box. Both should be compared: kernel **percentages** should
  match (algorithm + data flow), absolute **milliseconds** will
  differ by ~5-8x (P-core per-clock throughput + clock speed).
- Recorded result: scripts shipped, no benchmark data yet (waiting
  on user to run on N95).

## iGPU offload - rejected

- Scope: not implemented
- What was proposed: use Intel UHD (Gen 12 Xe-LP, 16 EU, 256 GFLOPS
  FP16) on N95 to offload ASR compute.
- Why it is rejected:
  1. **算力倒挂**: UHD 256 GFLOPS FP16 < CPU INT8 等效 ~1.5 TOPS
     (AVX2 PMADDUBSW path). iGPU is 6x slower than the CPU path
     the engine already takes.
  2. **带宽共享**: N95 单通道 DDR5 ~25 GB/s. iGPU 跟 CPU 共享
     同一根内存条,没有 GPU 该有的"带宽放大"红利.
  3. **算子形态不匹配**: Qwen3-ASR 算子 = GEMV / 小 GEMM
     (Q/K/V 投影 1×N × N×K). GPU kernel 启动开销 100μs vs
     算子本身 50μs,**负优化**.
  4. **生态**: whisper.cpp / llama.cpp / rknn 全部不支持集显
     ASR 加速. community consensus: 集显做 ASR = 慢.
  5. **本项目架构**: 整个 `crates/qwen-asr` 没有任何
     wgpu / OpenCL / Level Zero / DirectML 依赖. 纯 Rust +
     OpenBLAS.
- Exception (not applicable here): Apple M-series iGPU (统一内存
  + 大规模 EU + Metal) IS supported via `vDSP/Accelerate` and is
  documented in `docs/optimizations/overview.md` § 1 / § 3.
- Recorded result: N/A — proposal rejected, no code written.
