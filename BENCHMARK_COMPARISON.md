# Benchmark Comparison — perf-round2 vs previous impl

Apples-to-apples comparison of the optimization round (`perf-round2`) against the
previous implementation (`main` @ `9e8205f`). Both binaries built with
`RUSTFLAGS="-C target-cpu=native"` and run through the **same** current harness
(`bench/run.sh`, median of 10 runs) on the same machine, back-to-back.

- Machine: Apple M5 Pro (5 performance + 10 efficiency cores)
- Model: `qwen3-asr-0.6b`
- Speed sample: `bench/samples/audio.wav` (28 s)
- Decode-heavy sample: a LibriSpeech `dev-clean` clip (11.7 s, uncapped)
- WER: `librispeech_wer.py --limit 100 --mode offline`
- "Previous" default threads = all CPUs (15); "latest" default = performance cores (5)

## Speed (median of 10) — wall = load + inference

| Mode | Metric | Prev (9e8205f) | Latest (perf-round2) | Δ |
|------|--------|---------------:|---------------------:|----:|
| offline    | wall      | 1106 ms | **860 ms** | **−22.2%** |
| offline    | inference |  495 ms | **470 ms** | −5.1% |
| segmented  | wall      |  987 ms | **740 ms** | **−25.0%** |
| segmented  | inference |  378 ms | **356 ms** | −5.8% |
| streaming  | wall      | 1003 ms | **753 ms** | **−24.9%** |
| streaming  | inference |  390 ms | **365 ms** | −6.4% |

Inference realtime factor: offline 56.9× → **59.9×**, segmented 74.4× → **79.2×**,
streaming 72.3× → **77.1×**.

## Real-world decode-heavy clip (11.7 s, no long-audio cap)

The 28 s speed sample triggers the long-audio token cap, so its decode is tiny
and it under-represents normal usage. On a real uncapped clip decode dominates:

| Phase | Prev | Latest | Δ |
|-------|-----:|-------:|----:|
| decoding | 398 ms | **302 ms** | **−24.1%** |
| encoding | 109 ms | 111 ms | ~0 |

## Startup / memory

| Metric | Prev | Latest | Δ |
|--------|-----:|-------:|----:|
| load floor (0.5 s clip, wall) | 0.39 s | **0.17 s** | **−56%** |
| peak RSS | 5.04 GB | 5.04 GB | 0 |

(RSS is unchanged: the load *conversions* were parallelized, not removed —
the RAM-reducing experiments E3/E11 were rejected on the speed/quality gate.)

## Accuracy (100-file LibriSpeech offline)

| Metric | Prev | Latest | Δ |
|--------|-----:|-------:|----:|
| Corpus WER | 0.0387 | **0.0379** | −0.0008 (better) |
| Macro WER  | 0.0428 | **0.0418** | better |
| Corpus CER | 0.0164 | **0.0152** | better |

## What changed (accepted optimizations)

1. **E2 — parallel model-load conversions** (`thread::scope` over encoder/decoder
   layers + SIMD encoder bf16→f32). Load floor 0.39 → 0.17 s. This is the bulk of
   the wall-clock win.
2. **E8 — batched-GEMM prefill causal attention** (two real GEMMs per head instead
   of `2·seq_q` tiny N=1 BLAS calls). `attention_causal` −44%; inference −5-6%.
3. **Default threads = performance cores** (became a win only after E8 changed the
   threading profile). All modes faster; real-clip decode −24%.

Nine other ideas were tried and rejected/deferred with evidence — see
`speed-improvement-experiment-2.md`.

## Bottom line

~22-25% faster end-to-end on the standard sample, ~24% faster decode on real
clips, 56% faster cold start, **with slightly better WER**.
