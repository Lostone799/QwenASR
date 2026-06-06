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

