# Benchmark Results

Latest results for the standard speed sample and LibriSpeech WER.

## Contents

- [Speed Benchmark](#speed-benchmark)
- [WER Benchmark](#wer-benchmark)

## Speed Benchmark

## Speed Benchmark

Speed benchmark for the standard 28.2 s mono WAV sample (`bench/samples/audio.wav`).

### Methodology

- Machine: Apple M5 Pro (5 performance + 10 efficiency cores), 32 GB RAM
- Model: `qwen3-asr-0.6b`
- Audio: `bench/samples/audio.wav` (28.2 s)
- Binary: `target/release/qwen-asr` built with `RUSTFLAGS="-C target-cpu=native"`
- Modes:
  - `offline` — full-file transcription
  - `segmented` — `-S 30`
  - `streaming` — `--stream`
- Metric: median inference time across 10 standalone runs
- Reference transcript: `bench/samples/audio.txt`

### Reproduce

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./bench/run.sh --label current --runs 10
```

Results are written to `bench/results/current/`.

### Latest Results

> Generated on: 2026-06-12
> Commit: `9ecde04`
> Hardware: Apple M5 Pro, 32 GB RAM
> Threads: performance cores (5) — `bench/run.sh` reports the system CPU count (15) for metadata

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor | WER (sample) |
|---|---:|---:|---:|---:|---:|
| offline | 447 | 460.7 | 444 | 63.09× | 0.9189 |
| segmented | 336 | 337.0 | 333 | 83.93× | 0.9189 |
| streaming | 345 | 344.5 | 342 | 81.74× | 0.9189 |

#### Wall-clock timing

| Mode | Median wall ms | Mean ms | Best ms | Wall realtime factor |
|---|---:|---:|---:|---:|
| offline | 816.9 | 917.7 | 811.4 | 34.55× |
| segmented | 706.1 | 708.6 | 700.7 | 39.94× |
| streaming | 716.8 | 716.5 | 712.7 | 39.34× |

#### Note on sample WER

The speed sample WER is high (0.9189) because the current default configuration applies a long-audio token cap to the 28.2 s benchmark clip. This is an explicit speed/quality tradeoff: short utterances (including the 100-file LibriSpeech gate) use the quality path, while long files favor latency. See [`experiments.md`](../research/experiments.md) (Round 1, S37) for the rationale.

#### Kernel profile (offline)

When run with `--profile`, the offline run reports per-kernel timings. The latest profile will be inserted here after regeneration.

### Historical context

- Initial Rust port (`bf52daf`): 1,612 ms offline / 17.49× RTF (cross-implementation run, `--threads 15`)
- Current implementation (`9ecde04`): 447 ms offline / 63.09× RTF (dedicated speed benchmark, performance cores)

See [`comparison.md`](./comparison.md) for the latest cross-implementation numbers and [`experiments.md`](../research/experiments.md) for the full optimization diaries.

---

## WER Benchmark

## WER Benchmark

Word-error-rate benchmark on LibriSpeech `dev-clean`.

### Methodology

- Dataset: LibriSpeech `dev-clean` (cached locally as `librispeech-wer-bench/dev-clean-2/`)
- Model: `qwen3-asr-0.6b`
- Binary: `target/release/qwen-asr`
- Mode: `offline` (default for the 100-file gate)
- Metric: corpus WER/CER and macro WER/CER
- Preprocessing: lowercasing and punctuation stripping before Levenshtein distance

### Reproduce

#### 100-file offline gate

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline
```

#### Full 1,089-utterance dataset

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-full \
  --mode offline
```

#### Auto-download dataset

If `dev-clean-2/` is not present:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --download-dataset \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline
```

### Latest Results

> Generated on: 2026-06-12
> Commit: `9ecde04`
> Dataset: LibriSpeech `dev-clean-2`
> Items evaluated: 100
> Mode: offline

| Metric | Value |
|---|---:|
| Corpus WER | 0.0379 |
| Macro WER | 0.0418 |
| Corpus CER | 0.0152 |
| Macro CER | 0.0155 |
| Failed utterances | 0 / 100 |

### Historical context

- Early baseline (`step0-current`, `12663c5`): corpus WER 0.1101
- After WER recovery tuning: corpus WER 0.0387
- Latest target: keep corpus WER ≤ 0.04 while improving speed

See [`experiments.md`](../research/experiments.md) for the full tuning diary.

---

