# qwen-asr (Lostone799 fork)

> **Fork 说明**: 这是 [huanglizhuo/QwenASR](https://github.com/huanglizhuo/QwenASR) 的个人优化 fork,聚焦**低端 CPU 主机**(无独立 GPU)上的推理延迟与内存优化,**精度优先,禁止量化**。

[![OctoCounts](https://api.octocounts.com/badge/Lostone799/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2FLostone799%2FQwenASR&ref=main)

A **blazing fast**, pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. Zero heavy runtime dependencies (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports 0.6B and 1.7B models with offline, segmented, streaming, live capture, VAD live, and forced alignment modes.

---

## Fork 独有改动

### P0 关键修复

| Commit | 修复 | 影响 |
|--------|------|------|
| `9981be5` | **oneDNN crash + 中文音频被误识别为英文** | 修复 oneDNN FFI 崩溃;默认语言 token 从 `6364` ("English") 改为 `8453` ("Chinese"),修复中文音频识别为英文的 bug |
| `2e0ec6a` | **移除 LONG_AUDIO_FAST token 上限** | 修复 #31,长音频识别被截断的问题 |
| `210e9f1` | **移除剩余 transcript 截断原因** | 修复 #31,与上一 commit 配合彻底解决截断 |
| `a6aaeff` | **工作区清理 + flutter 目录移除** | 清理无关文件,减少仓库体积 |

### 性能优化

| Commit | 优化 | 状态 |
|--------|------|------|
| `5fd3fbd` | **C1: P2 heap alloc 消除 + BLAS 8 线程** | 当前最佳配置 |
| `5c5a4d5` | **Encoder Conv2D + bias-add + GELU 融合** | P1 路径减少 3 次内存往返 |
| `4d3fe44` | **Phase 5 AVX2 4-row kernel 优化 (FAIL)** | 6.4% 劣化,保留为备用路径,附重启用条件矩阵 |

### GUI 与稳定性

| Commit | 改进 | 说明 |
|--------|------|------|
| `bc815b7` (v0.1) | **GUI crash hardening + AVX-VNNI allowlist + 长音频修复 + N95 profile 脚本** | Win32 SEH 过滤器;Intel N95/Gracemont CPU AVX-VNNI 误报检测;`QWEN_ASR_DISABLE_VNNI` 环境变量 |

### 设计约束 (Hard Constraints)

- **GUI 按钮(清空/复制)在录音/识别时禁用**,防止用户误操作
- 所有 panic 错误必须通过 `setup_panic_hook()` 捕获崩溃详情
- 音频回调使用安全错误处理(`if let Ok(...)` 替代 `unwrap()`),避免 Mutex 中毒崩溃
- 所有优化必须提供禁用开关(如 `QWEN_ASR_DISABLE_P1`),便于 A/B 测试
- oneDNN 必须通过 `QWEN_ASR_ENABLE_ONEDNN` 选择性启用(默认禁用),因 FFI 崩溃和性能劣化
- 当未指定 `--language` 时,ASR prompt 使用中文 token (8453) 启用正确的中文音频识别
- `QwenCtx.stream_unfixed_chunks=0` 确保 token 从第一个 chunk 开始发射(防止 99 秒不出字)
- `QwenCtx.stream_rollback=1` (从 2 降低) 防止早期 chunk token 丢失
- `StreamState` 在音频 backlog 超过 20 秒时重置,避免推理时间恶性增长
- `refine_enabled` 标志必须检查后再执行 refine 逻辑,防止不必要的 358s 阻塞
- 段式 KV cache 复用(`reuse_prefix_kv`)默认启用,减少 decoder prefill
- 段结束时重置 `StreamState` 和 KV cache,防止 O(n²) attention 增长
- `conv2d_3x3_s2_p1_parallel()` 仅对 Intel Gracemont/Tremont/Atom 低功耗 E-core 启用;Skylake+ P-core 必须用 im2col+sgemm
- 缓冲区管理使用预分配和复用(reserve + set_len)消除运行时 resize/heap allocation

## 运行时依赖

本 fork 在 `runtime/` 目录提供 Windows 运行时所需的 DLL:

| 文件 | 大小 | 说明 |
|------|------|------|
| `runtime/libopenblas.dll` | ~50 MB | OpenBLAS BLAS 后端,矩阵运算 |
| `runtime/qwen_asr.dll` | ~111 KB | 项目自身编译产物(cdylib) |

**Windows 运行方式**:
1. 从 [Releases](https://github.com/Lostone799/QwenASR/releases) 下载 `qwen-asr.exe` / `qwen-asr-gui.exe`
2. 复制 `runtime/*.dll` 到 exe 同目录(或加入 PATH)
3. 运行 `qwen-asr.exe -d <model_dir> -i audio.wav`

**Linux/macOS**: 通过包管理器安装 OpenBLAS(见下方 Build 章节),无需 DLL。

## Performance

On an Apple M5 Pro, qwen-asr transcribes a 28-second audio clip in **576 ms** — about **49× faster than realtime** in the dedicated local benchmark. It's faster than the upstream C implementation and the measured MLX GPU baselines.

| Implementation | Median inference | Realtime factor |
|---|---:|---:|
| qwen-asr (latest, dedicated) | 576 ms | 48.92× |
| mlx-audio Python MLX | 688 ms | 40.92× |
| second-state/qwen3_asr_rs MLX GPU | 1,401 ms | 20.10× |
| pure C upstream | 1,650 ms | 17.06× |
| qwen-asr (first Rust port) | 1,669 ms | 16.90× |

> Benchmarked on the same 28.2 s sample with 10 runs each. See [`docs/benchmarks/comparison.md`](docs/benchmarks/comparison.md) for full details and reproduction steps.

## Documentation

- [`docs/plans/2026-06-23-low-end-optimization.md`](docs/plans/2026-06-23-low-end-optimization.md) — **低端 CPU 优化计划**(本 fork 主线工作,9 任务 3 阶段)
- [`docs/optimizations/overview.md`](docs/optimizations/overview.md) — 已实现性能优化目录
- [`docs/optimizations/optimization-matrix-evaluation.md`](docs/optimizations/optimization-matrix-evaluation.md) — i5-10400 上 8 组合 A/B 矩阵
- [`docs/research/failed-optimizations-backup-paths.md`](docs/research/failed-optimizations-backup-paths.md) — 失败优化保留为备用路径(oneDNN + 4-row kernel)
- [`docs/optimizations/2026-07-02-avx2-kernel-methodology.md`](docs/optimizations/2026-07-02-avx2-kernel-methodology.md) — AVX2 kernel 优化方法论与 DRAM 带宽墙判定法
- [`docs/benchmarks/`](docs/benchmarks/) — benchmark 方法论、最新结果、复现步骤
- [`docs/research/`](docs/research/) — 历史实验日志

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

### P1 Encoder Kernel (CPU 微架构 allowlist)

P1 hand-written 3×3 stride=2 AVX2 kernel 仅在 Intel Gracemont/Tremont/Atom 低功耗 E-core 上启用;Skylake+ P-core 因数值精度问题(1 token "。" vs P0 的 44 tokens)禁用。

- 自动检测:`crates/qwen-asr/src/kernels/mod.rs::p1_target_cpu()`
- 强制禁用:`$env:QWEN_ASR_DISABLE_P1 = "1"`
- 强制启用(仅测试用):`$env:QWEN_ASR_ENABLE_P1 = "1"`

## Quick Start

```bash
# Install
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

Or download a pre-built binary from [GitHub Releases](https://github.com/Lostone799/QwenASR/releases).

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
# Windows (with OpenBLAS)
$env:OPENBLAS_DIR = "C:\path\to\openblas"
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Windows (without BLAS)
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

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
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux/Windows) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API |
| `android` | JNI API |

### Windows OpenBLAS 配置

Windows 构建需要 OpenBLAS。可通过以下方式获取:

1. **使用本 fork 提供的 DLL**: 从 `runtime/libopenblas.dll` 复制到 exe 同目录
2. **从源码编译**: 参考 [OpenBLAS 官方](https://github.com/OpenBLAS/OpenBLAS)
3. **设置 `OPENBLAS_DIR` 环境变量** 指向包含 `openblas.lib` 的目录

`crates/qwen-asr/build.rs` 会读取 `OPENBLAS_DIR`,默认值为 `C:\Users\Administrator\clawd\openblas`。

## Environment Variables

本 fork 新增的环境变量:

| 变量 | 默认 | 说明 |
|------|------|------|
| `QWEN_ASR_DISABLE_P1` | 0 | 禁用 P1 hand-written conv2d kernel,回退到 P0 im2col+sgemm |
| `QWEN_ASR_ENABLE_P1` | 0 | 强制启用 P1 kernel(仅测试用,绕过 CPU allowlist) |
| `QWEN_ASR_DISABLE_VNNI` | 0 | 禁用 AVX-VNNI INT8 路径(N95/Gracemont 误报修复) |
| `QWEN_ASR_ENABLE_VNNI` | 0 | 强制启用 AVX-VNNI(仅测试用) |
| `QWEN_ASR_ENABLE_ONEDNN` | 0 | 启用 oneDNN(默认禁用,因 FFI 崩溃和性能劣化) |
| `QWEN_ASR_AUTO_STOP_SEC` | 0 | 自动测试模式下的录音停止秒数(0 = 无限制) |
| `QWEN_ASR_SEGMENT_SEC` | 60 | 段式处理的段长度(秒) |

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

**A/B 测试强制规则**: 跨会话对比无效(系统状态/热状态/后台进程不同)。必须在同一会话内交替运行 baseline/optimized 各 ≥4 次,取 trimmed mean。

See [`docs/benchmarks/`](docs/benchmarks/) for full details.

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Acknowledgments

- Original Rust port: [huanglizhuo/QwenASR](https://github.com/huanglizhuo/QwenASR)
- Original C implementation: [antirez/qwen-asr](https://github.com/antirez/qwen-asr)

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
