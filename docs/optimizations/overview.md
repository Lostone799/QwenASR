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

## 7. Experience Knowledge Base

See [experience-ledger.md](experience-ledger.md) for the complete optimization experience knowledge base, including:
- Success stories and failure lessons across all platforms
- Performance evolution data (Apple M5 + Windows x86)
- Platform-specific pitfalls and best practices
- Continuous update mechanism for future optimizations
