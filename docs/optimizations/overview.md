# QwenASR Performance Optimizations

This document catalogs the performance optimizations implemented in the pure-Rust QwenASR CPU inference engine. Current HEAD reaches 64+× realtime on Apple M5 for offline transcription of the 28.2 s benchmark sample.

## 1. Memory Traffic & Allocation Reduction

- **INT8 Quantization for Decoder**: Decoder weights (`QKV`, `O`-projection, `FFN` gate/up/down, `lm_head`) are quantized to INT8 with per-row scales at load time. This cuts memory traffic by ~4x compared to FP32 and avoids BF16-to-F32 conversion overhead. Implemented via NEON INT8 matvec/argmax kernels.
- **Reusable Workspaces**: Eliminated transient heap allocations in hot paths.
  - **Encoder**: `EncoderBuffers` persists scratch spaces for `chunk_mel`, convolution variables, and `im2col`. The main activation buffer (`x`) and `window_starts` metadata are reused across calls.
  - **Decoder**: `DecoderBuffers` provides pre-allocated scratch for BF16-to-F32 conversions, removing ~140 allocations per prefill pass.
  - **Transcription**: Embedding assembly buffers are reused instead of being reallocated per chunk.
- **Static Weight Prepacking**: Multi-token decoder prefill weights are preconverted from BF16 to F32 at load time and stored in a reusable matrix. This skips repetitive conversions across streaming chunks or segmented prefills.

## 2. Kernel Fusion & Cache Locality

- **Fused Residual Adds**: Replaced separate `y = y + x` loops with `linear_accumulate` and `linear_nobias_bf16_addto`. Matvec/GEMM operations accumulate directly into the destination residual buffer, saving read/write passes.
- **Fused Matvec + SwiGLU**: A fused kernel computes the `gate_up` projection and applies the `SwiGLU` activation in one pass, keeping intermediate values in L1 cache.
- **Head-Contiguous KV Cache**: Cache layout is `[layer][head][pos][head_dim]`. Storing heads contiguously improves spatial locality and reduces cache misses during causal attention scans.

## 3. SIMD & Platform Acceleration

- **Explicit SIMD Intrinsics**: 
  - Vectorized `rms_norm`, `gelu`, and `swiglu` using fast polynomial exponential approximations.
  - RoPE uses NEON vector code for pairwise sub-vector rotations.
  - Bulk BF16 conversions use `vshll_n_u16` (NEON) and `_mm256_cvtepu16_epi32` (AVX2).
- **Apple Accelerate & vDSP**: Dense linear algebra (causal attention scores, mel spectrogram generation) is offloaded to Accelerate (BLAS). Uses `vvexpf` for batched softmax exponentiation and `vDSP_dotpr` for AMX coprocessor utilization.

## 4. Threading & Concurrency

- **Lock-Free Thread Pool Fast Path**: Work scheduling uses atomics and spin-waiting before falling back to mutex/condvar sleep, reducing OS context-switch latency for micro-jobs.
- **Threaded Non-Matmul Operations**: Parallelized operations beyond GEMMs:
  - `im2col` packing for encoder convolutions.
  - `gelu` and `swiglu` activations over large FFN buffers.
  - Bidirectional attention across attention heads.

## 5. Algorithmic Improvements

- **Silence Compaction**: Energy-based VAD preprocesses audio to strip non-speech segments. Edge padding is reduced to 2 windows and extra non-voice hangover is eliminated, minimizing data sent to the encoder.
- **Lazy Encoder Re-encoding**: In streaming mode, the partial encoder tail is only re-encoded every other chunk. This provides near-perfect Longest Common Prefix (LCP) reuse and reduces decoder prefill cost by ~50% on skipped chunks.
- **Online Softmax**: Single-token causal attention uses an online softmax scan, combining score tracking, normalization, and value accumulation into a single loop. This avoids temporary score buffer allocations and separate exponentiation passes for `seq_len = 1` queries.

## 6. Windows x86 (AVX2) Optimizations

- **AVX2 INT8 GEMM Kernel**: PMADDUBSW (u8×s8→i16) + PMADDWD (i16×i16→i32) + XOR 0x80 sign-flip compensation. 4-row batched kernel for cache efficiency.
- **OpenBLAS Thread Tuning**: 8 OpenBLAS threads optimal with 12-thread custom pool (2/3 ratio). Environment variable `OPENBLAS_NUM_THREADS` set in `main()` before any BLAS call (DLL only exports `cblas_sgemm`).
- **Tiled INT8 GEMM**: Outer loop = o_block (4 output rows), inner loop = tokens. Weights loaded once per o_block and reused across all tokens, reducing weight memory traffic from `seq_len × out_dim × in_dim` to `out_dim × in_dim`.
- **Fused QKV Quantization**: Quantize input activation once, reuse for Q/K/V projections. Saves 2/3 of quantization cost (memory alloc + compute). Both encoder (same out_dim, with bias) and decoder (GQA: q_dim/kv_dim, no bias) variants.
- **Result**: sgemm -91.8%, realtime 1.03x → 3.56x on 1.7B model (28.2s audio).

## 6a. AVX-VNNI microarchitecture allowlist (2026-06-21)

Some Intel / VM platforms report `avxvnni=1` in CPUID but do **not** have the
`vpdpbusd` execution unit. Hitting that path traps as
`EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D)`. The most common false
positive is **Intel N95/N100/N305 (Alder Lake-N Gracemont, family 6,
model 0x9A)**: a pure E-core chip with no AVX-VNNI silicon. Hybrid Intel
chips and Hyper-V VMs can land the hot loop on a similar false-positive
core under scheduler pressure.

Two-layer mitigation in `crates/qwen-asr/src/kernels/mod.rs`:

- **Environment gate** (manual override for any user):
  - `QWEN_ASR_DISABLE_VNNI=1` — force AVX2 path.
  - `QWEN_ASR_ENABLE_VNNI=1` — force VNNI (use only if you know your CPU
    has it; bypasses the allowlist).
- **Raw CPUID allowlist** in `vnni_capable_cpu()`:
  - Implementation uses `core::arch::asm!` to read CPUID directly, since
    stable Rust does not expose `__get_cpuid` non-nightly.
  - Allowlist includes Intel 12/13/14th-gen P-cores, Meteor Lake, Lunar
    Lake, Arrow Lake, Sapphire / Emerald / Granite Rapids, Sierra
    Forest, and AMD Zen 4+ (family 25).
  - **Excludes** Gracemont (N95/N100/N305 0x9A) and any
    microarchitecture not explicitly in the allowlist.
  - Decision logic: `allowed = (allowlist_ok && cpuid_yes)` — both must
    agree. A CPUID-yes on a non-allowlisted core is treated as a false
    positive. First call logs the actual
    `(cpuid_yes, allowlist_ok, env_off, env_on)` tuple to stderr so
    crash reports are self-explaining.

The decision is cached in an `AtomicU8` after the first call. Cost:
one CPUID instruction, paid once per process.

## 7. Experience Knowledge Base

See [experience-ledger.md](experience-ledger.md) for the complete optimization experience knowledge base, including:
- Success stories and failure lessons across all platforms
- Performance evolution data (Apple M5 + Windows x86)
- Platform-specific pitfalls and best practices
- Continuous update mechanism for future optimizations

## 8. Long-Audio Token Budget (2026-06-21)

Long-audio token-budget fix in `crates/qwen-asr/src/transcribe.rs`:

- **Before**: separate `LONG_AUDIO_FAST_CAP_SEC = 15` and
  `LONG_AUDIO_FAST_MAX_TOKENS = 6` constants hard-capped **any** decode
  whose audio exceeded 15 s to 6 tokens. On a P-core CPU at ~100 ms /
  token, the cap was effectively unobservable (≤ 1 s extra). On a slow
  CPU like N95 (2-3 s / token) the cap surfaced as "20 s audio → 9
  Chinese chars and stops" — the decode *completed*, it just stopped
  early. This looked like a model failure to the user.
- **After**: removed both constants. The token budget is now
  `max(30, audio_sec * 8.0)` applied uniformly to all audio lengths,
  with `LONG_AUDIO_TOKEN_RATE = 8.0` (tokens / second) and
  `LONG_AUDIO_TOKEN_MIN = 30` (floor for very short audio).
- **Streaming path**: the matching 6-token cap on stream chunks was
  removed too. Long audio in streaming is handled by the chunked /
  LCP-reuse mechanism (`stream_push_audio`), not by a global token cap.

This is a P1-class fix (correctness, not speed): the user gets the full
text on long audio instead of a hard 6-9 token truncation.
