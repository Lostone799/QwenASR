# qwen-asr

[![OctoCounts](https://api.octocounts.com/badge/huanglizhuo/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2Fhuanglizhuo%2FQwenASR&ref=main)

A **blazing fast**, pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. Zero heavy runtime dependencies (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports 0.6B and 1.7B models with offline, segmented, streaming, live capture, VAD live, and forced alignment modes.

## Performance

On an Apple M5 Pro, qwen-asr transcribes a 28-second audio clip in **576 ms** — about **49× faster than realtime** in the dedicated local benchmark. It's faster than the upstream C implementation and the measured MLX GPU baselines.

| Implementation | Median inference | Realtime factor |
|---|---:|---:|
| qwen-asr (latest, dedicated) | 576 ms | 48.92× |
| mlx-audio Python MLX | 688 ms | 40.92× |
| second-state/qwen3_asr_rs MLX GPU | 1,401 ms | 20.10× |
| pure C upstream | 1,650 ms | 17.06× |
| qwen-asr (first Rust port) | 1,669 ms | 16.90× |

<p float="left">
  <img src="docs/benchmarks/charts/benchmark-unified-latency.png" width="48%" alt="Latency comparison" />
  <img src="docs/benchmarks/charts/benchmark-unified-rtf.png" width="48%" alt="Realtime factor comparison" />
</p>

> Benchmarked on the same 28.2 s sample with 10 runs each. The qwen-asr latest row is the current dedicated benchmark on the working tree (`f28145c` + retained fixes); external baseline rows come from the latest full cross-implementation run (`20260702T053724Z`). See [`docs/benchmarks/comparison.md`](docs/benchmarks/comparison.md) for full details and reproduction steps.

## Documentation

- [`docs/benchmarks/`](docs/benchmarks/) — benchmark methodology, latest results, and reproduction instructions
- [`docs/optimizations/overview.md`](docs/optimizations/overview.md) — catalog of implemented performance optimizations
- [`docs/optimizations/optimization-matrix-evaluation.md`](docs/optimizations/optimization-matrix-evaluation.md) — 8-combination A/B matrix on i5-10400
- [`docs/research/`](docs/research/) — historical autoresearch experiment logs and protocols

## GUI (Windows)

A standalone Windows GUI binary is provided at `crates/qwen-asr-gui/`.
It is built with eframe / egui and uses a `windows_subsystem = "windows"`
release binary, so it launches without a console window. A top-level Win32
SEH filter is installed first in `main()`; if the engine ever crashes
(`EXCEPTION_ACCESS_VIOLATION` / `STACK_OVERFLOW` / `ILLEGAL_INSTRUCTION` /
`INT_DIVIDE_BY_ZERO`), a `MessageBoxW` is shown and a detailed report is
written to `<exe_dir>/logs/crash_<unix_secs>_<pid>.log`.

### AVX-VNNI false-positive (Intel N95 / hybrid CPUs / VMs)

Intel N95 / N100 / N305 (Alder Lake-N Gracemont) reports `avxvnni=1` in
CPUID but the silicon lacks the `vpdpbusd` unit. Executing the AVX-VNNI
INT8 path on it traps as `EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D)`.
A microarchitecture allowlist in `crates/qwen-asr/src/kernels/mod.rs`
already excludes Gracemont, but if you hit the crash on another hybrid
Intel CPU or a Hyper-V VM, set the env var before launching:

```powershell
$env:QWEN_ASR_DISABLE_VNNI = "1"
.\qwen-asr-gui.exe
```

`QWEN_ASR_ENABLE_VNNI=1` does the opposite (forces VNNI, bypasses the
allowlist — use only if you know your CPU has the unit).

## Quick Start

```bash
# Install
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

Or download a pre-built binary from [GitHub Releases](https://github.com/huanglizhuo/QwenASR/releases).

## Usage

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav              # basic
qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent      # transcript only
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin     # pipe from stdin
qwen-asr -d qwen3-asr-0.6b -i long.wav -S 30           # segmented
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream       # streaming
qwen-asr -d qwen3-asr-0.6b --live --device "BlackHole 2ch"         # live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"   # VAD live
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English  # alignment
```

<details>
<summary>All options</summary>

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz) | off |
| `--live` | Live capture from audio device (macOS) | off |
| `--device <name>` | Input device for live capture | system default |
| `--list-devices` | List audio input devices | — |
| `--vad` | VAD live mode | off |
| `-t <n>` | Thread count | performance cores |
| `-S <secs>` | Segment target seconds | 0 (full) |
| `--stream` | Streaming mode | off |
| `--stream-chunk-sec <s>` | Chunk size for streaming | 2.0 |
| `--language <lang>` | Force output language (`en`, `zh`, `ja`, ...) | auto |
| `--silent` | Transcript only, no status output | off |
| `--profile` | Print timing breakdown | off |

</details>

## Build

**Always use release mode.** Debug builds are 10–50× slower.

```bash
# macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Linux
sudo apt install libopenblas-dev   # Debian/Ubuntu
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Without BLAS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

# iOS (static library + C-FFI)
cargo build --release --target aarch64-apple-ios --features ios

# Android (shared library + JNI)
cargo ndk -t arm64-v8a build --release --features android
```

| Feature | Description |
|---------|-------------|
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API |
| `android` | JNI API |

## Reproducing Benchmarks

```bash
# Speed benchmark
./bench/run.sh --label current --runs 10

# WER benchmark (100-file LibriSpeech offline)
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline

# Cross-implementation comparison (30–60 min)
./bench/benchmark-all.sh --runs 10
```

See [`docs/benchmarks/`](docs/benchmarks/) for full details.

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Acknowledgments

Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a pure C implementation of Qwen3-ASR inference by antirez.

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
