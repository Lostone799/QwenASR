# Speed Improvement Experiments — Round 2 (profiling-driven, structural)

Goal: improve speed without WER regression. Gate: 100-file LibriSpeech offline WER must stay `<= 0.04` (baseline `0.0387`). These experiments are **structural / engineering** optimizations identified by profiling (load overhead, GEMM fusion, threading) and deliberately avoid the quality knobs already exhausted in `speed-improvement-expirement.md` (token caps, vocab shortlist, encoder window size, silence thresholds).

Machine: Apple M5 Pro. Model: `qwen3-asr-0.6b`. Speed via `bench/run.sh --runs 5` (median wall = load+infer, median inference = `total_ms`). WER via `librispeech_wer.py --limit 100 --mode offline`.

## Baseline (HEAD, `base-e0`)

| Mode | Wall (ms) | Inference (ms) |
|------|-----------|----------------|
| offline | 1071 | 487 |
| segmented -S30 | 964 | 372 |
| streaming | 969 | 382 |

- 100-file offline WER: **0.0387**
- Fixed startup floor (0.5s clip): ~0.58s, single-threaded
- Peak RSS: 5.1 GB
- Kernel breakdown (offline 28s): sgemm 289ms, conv2d 67, attn_causal 45, bf16_matvec 42, attn_bidir 11, gelu 9

Profiling notes: load is single-threaded; inference does not scale past ~4 threads (shared AMX coprocessor). Decoder holds 3 weight copies (bf16 mmap + 1.76GB f32 prefill + 0.44GB int8).

## E1: default threads = P-core count (5 instead of 15)

Change: default thread count uses `hw.perflevel0.physicalcpu` (P-cores=5) instead of all CPUs (15).

Clean A/B on same binary, offline 28s, runs=5 (median inference ms): `t15=489, t8=486, t6=496, t5=502`.

Decision: **Rejected.** Capping to P-cores slightly *regresses* the encoder-heavy offline path. The earlier hypothesis was wrong: the parallelized non-matmul ops (im2col, gelu, bidirectional attention) and Accelerate's own threading do benefit from the efficiency cores. More threads (8–15) is marginally better, not worse. Reverted.

## E2: parallelize model load conversions ✅

Change: load encoder/decoder layers in parallel via `std::thread::scope` (each layer's bf16→f32 prefill conversion + INT8 quantization is independent). Also switched the encoder's `load_bf16_as_f32` from a scalar loop to the SIMD `kernels::bf16_to_f32_buf`.

Measured load (tiny clip, instrumented): encoder 73→25ms, decoder 272→94ms; total load ~345→~130ms.

| Mode | Wall before | Wall after | Inference |
|------|-------------|-----------|-----------|
| offline | 1071 | **859** (−20%) | 488 (unchanged) |
| segmented | 964 | **743** (−23%) | 373 (unchanged) |
| streaming | 969 | **756** (−22%) | 384 (unchanged) |

- 100-file offline WER: **0.0387** (unchanged — load produces identical weights)
- Library tests: pass

Decision: **Accepted.** Large wall-clock win, zero inference/WER impact, zero quality risk. Note: profiling showed the decoder f32-prefill conversion is 164ms of the decoder load; E2 parallelizes it rather than removing it (see E3).

## E3: lazy / on-demand f32 prefill weights ❌

Idea: stop building the 1.76GB f32 prefill weight copies at load; convert bf16→f32 on the fly (or lazily) so load is cheaper and RAM drops.

Analysis (settled from measured numbers rather than full implementation, which is invasive): every benchmark mode performs ~1 decoder prefill (offline 1; segmented -S30 on 28s = 1 segment; streaming skips non-final prefills per S27–S30, so ~1). The f32 conversion is 164ms serial, already parallelized into the 94ms decoder load by E2. Making it lazy/on-the-fly therefore *relocates* the same conversion out of (parallel) load into (per-prefill, single-threaded) inference: net ≈ −35ms load, +164ms inference per run = **wall-clock regression**. Wall = load+infer is conserved; only RAM (~1.76GB) improves, which is not the speed gate.

Decision: **Rejected** on the speed criterion. The genuinely beneficial removal of the f32 copies is to make prefill use INT8 weights so the conversion never has to happen at all — that is E11, not a lazy rebuild.

## E4: fused Q/K/V GEMM in encoder ❌

Change: concatenate per-layer wq/wk/wv into one `[3*d_model, d_model]` weight at load, run one BLAS GEMM into `qkv[T, 3*d_model]`, then split each token row into contiguous q/k/v buffers.

| Mode | Wall (E2) | Wall (E4) | Inference (E2→E4) |
|------|-----------|-----------|-------------------|
| offline | 859 | 859 | 488 → 489 |
| segmented | 743 | 742 | 373 → 371 |
| streaming | 756 | 754 | 384 → 383 |

Decision: **Rejected.** No measurable change (all within noise). Apple Accelerate already schedules the 3 separate QKV GEMMs efficiently on AMX, and the extra split-copy of `qkv[T,3d]` into contiguous q/k/v offsets any fusion benefit. Reverted. (Verified correctness is unaffected: the empty output on the local `short.wav` sample is a pre-existing edge case present on the committed E2 binary too, not introduced here.)

## E5: fused Q/K/V GEMM in decoder prefill ❌

Change: same fusion as E4 applied to the decoder prefill (concat wq/wk/wv f32 prefill weights → one GEMM into `pref_qkv` → split into q/k/v).

| Mode | Wall (E2) | Wall (E5) | Inference |
|------|-----------|-----------|-----------|
| offline | 859 | 871 | 489 (=) |
| segmented | 743 | 753 | 371 (=) |
| streaming | 756 | 768 | 384 (=) |

Decision: **Rejected.** Inference unchanged (same AMX behavior as E4); wall slightly *worse* because the fused weight is an extra ~470MB copy that lengthens load. Reverted.

## E6: batch conv / reuse im2col across chunks ❌ (unsafe)

The encoder conv front-end processes the mel in ~19 chunks (`enc_chunk_size`≈147), each convolved with its own zero-padding at the chunk edges — this matches the reference model and is baked into the WER. Merging chunks into one full-width conv would change the boundary padding and therefore the output (WER divergence), so it is not a safe speedup. im2col buffers can't be reused across chunks (different data), and parallelizing the chunk loop would oversubscribe the conv internals (im2col is already threaded and the GEMM is Accelerate-threaded).

Decision: **Rejected** — no safe lever that preserves output.

## E7: conv1 single-channel kernel + gelu fusion ❌

conv1 has only 1 input channel, so its im2col+GEMM has K=9 (tiny, latency-bound). But conv1 is a small fraction of total conv FLOPs — conv2/conv3 have c_in=480 (K=4320) and dominate, and they already run on optimal Accelerate BLAS. A naive direct conv1 would be cache-unfriendly and likely slower than the current im2col+AMX path; a competitive hand-vectorized direct-conv kernel is high-effort/high-risk for a sub-1% gain.

Decision: **Rejected** on cost/benefit — conv2/conv3 (the bulk) are already optimal; conv1's ceiling is negligible.

## E8: batched (flash-style) prefill causal attention ✅

Change: the multi-token causal-attention path did two N=1 BLAS calls per (head, query) — for prefill with seq_q≈350 × 16 heads × 28 layers that is a huge number of tiny matvec calls. Replaced with two real GEMMs per head: `S = scale·Q_h·K_hᵀ`, causal-masked row softmax (masked keys zeroed), then `O = S·V_h`. Single-token decode path unchanged.

- `attention_causal` profile: 45.0ms → **24.9ms** (−44%)

| Mode | Wall (E2) | Wall (E8) | Inference (E2→E8) |
|------|-----------|-----------|-------------------|
| offline | 859 | **836** | 488 → 468 |
| segmented | 743 | **731** | 373 → 360 |
| streaming | 756 | **739** | 384 → 373 |

- 100-file offline WER: **0.0387** (unchanged; CER 0.0164→0.0162)
- Library tests: pass

Decision: **Accepted.** Halves prefill attention time; ~3-4% inference / ~2-3% wall improvement with zero WER impact. (Computes a few masked-out scores in the upper triangle, but real GEMMs vastly outweigh the eliminated per-call overhead.)

## E9: parallel_for end backoff ❌  &  E10: pin workers to P-cores ❌

Both are thread-placement / spin tweaks. The benchmark runs on an otherwise-idle 15-core machine (5 P + 10 E):

- **E10** (restrict/pin workers to the 5 performance cores) is functionally identical to E1, which was measured and *regressed* the encoder-heavy offline path (the parallelized im2col/gelu/attention and Accelerate's own threading benefit from the efficiency cores). Rejected for the same reason.
- **E9** (add `sched_yield`/backoff to the completion spin) cannot improve wall-time when cores are idle — the spinning thread occupies its own otherwise-free core, and yielding only adds wakeup latency. Its benefit (lower energy/contention) does not register on an isolated speed benchmark and risks a small latency regression.

Decision: **Rejected** — no isolated-benchmark speed benefit; E10≡E1 (already shown to regress).

## E1-revisited: default thread count = performance cores ✅ (after E8)

While investigating decode threading, profiling a *real* (uncapped, 11.7s) clip showed decode dominates inference (decoding 382ms vs encoding 108ms) and is highly thread-count sensitive: the small, bandwidth-bound single-token matvecs slow down badly when spread across efficiency cores. Crucially, **after E8** (batched-GEMM attention changed the threading profile), fewer threads now wins on *every* mode — the opposite of E1's pre-E8 result.

Stable medians (perf-core default = 5 vs old default = 15):

| Metric | t15 (old) | t5 (perf cores) |
|--------|-----------|-----------------|
| offline wall / infer | 847 / 469 | **822 / 450** |
| segmented wall / infer | 731 / 357 | **711 / 340** |
| streaming wall / infer | 742 / 368 | **722 / 351** |
| decode (real 11.7s clip) | 381ms | **286ms** (−25%) |

Change: default thread count uses `hw.perflevel0.physicalcpu` (P-cores) instead of all CPUs.

- 100-file offline WER: **0.0379** (≤0.04, marginally better than 0.0387 — FP accumulation order differs slightly with thread count)
- Library tests: pass

Decision: **Accepted.** Improves every benchmark mode and cuts real-world decode latency ~25%, with WER within the gate. (Note: a finer-grained attempt to cap *only* the decode matvecs to 4 threads while keeping the encoder at full width required thread-pool surgery that introduced a race; the global perf-core default is the safe form and captures essentially the same benefit since the encoder also prefers P-cores post-E8.)

## E11: INT8 GEMM for decoder prefill ❌

Idea: replace the f32 prefill GEMMs (Accelerate sgemm) with an INT8 GEMM reusing the already-quantized weights, eliminating the f32 prefill copies (load + 1.76GB RAM).

Analysis: prefill is compute-bound and runs on Apple's AMX coprocessor via Accelerate f32 sgemm (~2 TFLOP/s). A hand-written CPU/NEON INT8 GEMM cannot access AMX's INT8 path through `cblas` and will not beat AMX f32 for these sizes; a per-token looped INT8 matvec would be far worse (tens of thousands of tiny dispatches per prefill). The only upside is load/RAM, which E2 already parallelized. Net compute would regress.

Decision: **Rejected** — CPU INT8 GEMM cannot beat AMX f32 here; load benefit is secondary and already addressed.

## E12: INT4 decoder weights ❌ (WER)

Decode is bandwidth-bound (reads ~500MB of INT8 weights per token), so INT4 would cut decode bandwidth ~2x. Probed the WER impact cheaply by coarsening the INT8 decode weights to INT4 precision (15 levels, per-row symmetric) while keeping the existing kernel:

- output visibly degraded; 100-file **macro WER 0.2514, CER 0.1735** (gate 0.04)

Decision: **Rejected.** Naive per-row symmetric INT4 destroys accuracy (~6x over the WER gate). Only group-wise GPTQ/AWQ-style INT4 could preserve quality — a research-grade effort, not a kernel tweak. The cheap probe avoided building the full NEON INT4 kernel for a change that fails the gate.

## E13: speculative decoding ❌ (infeasible)

Speculative decoding needs a separate small draft model to propose tokens for the main model to verify in parallel. No draft model exists for Qwen3-ASR, and self-speculative / n-gram (prompt-lookup) variants rely on repetitive output that ASR transcripts don't have. Not implementable in this codebase without training/shipping a draft model.

Decision: **Deferred** — no draft model available; out of scope for a local kernel/threading optimization pass.

---

## Summary

| Exp | Change | Result |
|-----|--------|--------|
| E2 | Parallelize model load conversions | ✅ wall −20-23%, WER 0.0387 |
| E8 | Batched-GEMM prefill causal attention | ✅ attn_causal −44%, infer −3-4%, WER 0.0387 |
| E1-rev | Default threads = performance cores (post-E8) | ✅ all modes faster, decode −25%, WER 0.0379 |
| E1 | Threads = P-cores (pre-E8) | ❌ regressed offline (superseded by E1-rev) |
| E3 | Lazy f32 prefill | ❌ wall-neutral (cost relocated), RAM-only |
| E4/E5 | Fused Q/K/V GEMM (encoder/prefill) | ❌ no AMX benefit |
| E6 | Merge conv chunks | ❌ unsafe (changes padding/WER) |
| E7 | conv1 specialization | ❌ negligible (conv2/3 dominate, already optimal) |
| E9/E10 | parallel_for backoff / P-core pin | ❌ no isolated-bench benefit |
| E11 | INT8 prefill GEMM | ❌ can't beat AMX f32 |
| E12 | INT4 decode weights | ❌ WER 0.25 (naive int4) |
| E13 | Speculative decoding | ❌ no draft model |

**Net accepted gains (vs baseline `base-e0`):**

| Mode | Wall before | Wall after | Δ |
|------|-------------|-----------|---|
| offline | 1071 | ~822 | −23% |
| segmented | 964 | ~711 | −26% |
| streaming | 969 | ~722 | −25% |
| real-clip decode (11.7s) | 381ms | 286ms | −25% |

100-file offline WER: 0.0387 → 0.0379 (within gate). Three commits on branch `perf-round2`.

