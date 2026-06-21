# Changelog

## [Unreleased] - 2026-06-21

### Fixed

- **GUI crash on Windows release builds** (P0/P1/P2 hardening)
  - **P0** Top-level Win32 `SetUnhandledExceptionFilter` in
    `crates/qwen-asr-gui/src/seh.rs` captures `EXCEPTION_ACCESS_VIOLATION`
    / `STACK_OVERFLOW` / `ILLEGAL_INSTRUCTION` / `INT_DIVIDE_BY_ZERO`,
    writes `logs/crash_<unix_secs>_<pid>.log`, and shows a MessageBoxW.
    The GUI binary uses `windows_subsystem = "windows"` which otherwise
    hides OS-level access violations from the user. SEH filter is
    installed first in `main()`, before panic hook / logger init.
  - **P0** `tok_embed_bf16_to_f32` now takes `vocab_size: usize` and
    returns `bool`. Out-of-bounds / negative / short-destination all
    zero-init the destination and log. The `Decoder` struct stores
    `tok_embeddings_vocab` derived from the safetensors shape. All 28
    call sites in `transcribe.rs` / `align.rs` updated.
  - **P1** `safe_lock()` / `try_safe_lock()` in
    `crates/qwen-asr-gui/src/sync_ext.rs` recover from `Mutex`
    poisoning transparently (backed by `logger::log_warn`). Replaced
    22 `.lock().unwrap()` call sites across `worker.rs`, `app.rs`,
    `recorder.rs`, `logger.rs`.
  - **P1** `pool_worker` in `crates/qwen-asr/src/kernels/mod.rs` wraps
    the user closure in `panic::catch_unwind(AssertUnwindSafe(...))`
    so a single bad kernel cannot stall `parallel_for`.
  - **P2** `AsrWorker::cancel_in_flight()` properly joins the previous
    `JoinHandle` before spawning a new task (the old code did
    `self.handle.take()` and silently abandoned the prior worker).

- **AVX-VNNI illegal-instruction on Intel N95 / hybrid CPUs / VMs**
  - Two-layer mitigation in `crates/qwen-asr/src/kernels/mod.rs`:
    1. Environment gate: `QWEN_ASR_DISABLE_VNNI=1` (force AVX2 path) or
       `QWEN_ASR_ENABLE_VNNI=1` (force VNNI, bypasses allowlist).
    2. Raw CPUID microarchitecture allowlist via `core::arch::asm!`
       (stable Rust does not expose `__get_cpuid` non-nightly).
       Allows VNNI only for verified cores (Intel 12/13/14-gen
       P-cores, Meteor / Lunar / Arrow Lake, Sapphire / Emerald /
       Granite Rapids, Sierra Forest, AMD Zen 4+ family 25).
       Explicitly excludes Gracemont (N95/N100/N305 0x9A) and any
       other microarchitecture. Decision:
       `allowed = (allowlist_ok && cpuid_yes)`. CPUID-yes on a
       non-allowlisted core is treated as a false positive. First
       call logs `(cpuid_yes, allowlist_ok, env_off, env_on)` to
       stderr for self-explaining SEH reports.
  - Verified end-to-end on Intel N95: previously crashed with
    `EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D)` after 4 s; with
    `QWEN_ASR_DISABLE_VNNI=1` runs to completion and produces
    correct transcription.

- **Long-audio truncation at 9 characters**
  - `crates/qwen-asr/src/transcribe.rs`: deleted
    `LONG_AUDIO_FAST_CAP_SEC = 15` and `LONG_AUDIO_FAST_MAX_TOKENS = 6`.
    These hard-capped any decode whose audio exceeded 15 s to 6 tokens.
    On a P-core CPU at ~100 ms / token the cap was effectively
    unobservable (≤ 1 s extra). On a slow CPU like N95 (2-3 s / token)
    the cap surfaced as "20 s audio → 9 Chinese chars and stops" —
    the decode *completed*, it just stopped early. Replaced with
    `LONG_AUDIO_TOKEN_RATE = 8.0` (tokens / second) and
    `LONG_AUDIO_TOKEN_MIN = 30` (floor for short audio). Token budget
    is now `max(30, audio_sec * 8.0)`, applied uniformly to all
    audio lengths. The matching 6-token cap on the streaming chunk
    path was also removed.

### Added

- **`crates/qwen-asr-gui/`** — new GUI crate
  - `Cargo.toml` / `src/main.rs` / `src/{app,worker,recorder,logger,
    params,seh,sync_ext}.rs`.
  - Plain Windows eframe / egui binary. Subsystem set to `windows`
    for release builds (no console window).
  - All five fixes above are exercised end-to-end by this GUI.

- **`profile_n95.ps1`** — N95 real-machine profile collector
  - Runs 4 CLI variants with `--profile`: `baseline` / `t2` /
    `t2_blas1` / `t1`.
  - Dumps stderr to `profile_n95_<label>.txt` and prints last 25
    lines to console.
  - Auto-sets `QWEN_ASR_DISABLE_VNNI=1`.

- **`profile_simulated_n95.ps1`** — dev-box N95 simulator
  - Same 4 variants + `SetProcessAffinityMask` (2 cores) +
    `BelowNormal` priority to emulate E-core L2 cluster sharing
    and TDP throttling.
  - Cannot simulate AVX2 microarchitecture latency, PMADDUBSW
    throughput, or memory subsystem, but kernel **percentages**
    match the real N95.

### Documentation

- `docs/optimizations/overview.md` — added § 6a (AVX-VNNI allowlist)
  and § 8 (long-audio token budget).
- `docs/optimizations/overview.zh.md` — same in Chinese.
- `docs/optimizations/optimization-matrix-evaluation.md` — added § 7
  (platform coverage + N95 collection guidance + rejected optimization
  candidates under the precision-first rule).
- `docs/research/ledger.md` — added `gui-crash-hardening`,
  `vnni-allowlist`, `long-audio-token-budget`, `n95-profile-collectors`,
  and `iGPU offload rejected` entries after the C1 commit `5fd3fbd`.

## [0.2.3] - 2026-02-23

### Features

- **Live audio capture** (macOS) — `--live` flag captures from audio input devices (microphone, BlackHole) in real time with segmented, streaming, or VAD modes
- **VAD live mode** — `--vad` flag uses energy-based Voice Activity Detection to detect speech segments and transcribe each independently, with cross-segment prompt conditioning for improved accuracy
- **Model download subcommand** — `qwen-asr download --list` and `qwen-asr download <model>` for built-in model management
- **Forced alignment** — `--align` flag produces word-level timestamps for a known transcript using the ForcedAligner model variant

### Performance

- **Lazy encoder re-encoding** — Partial encoder tail is only re-encoded every other chunk in streaming mode, giving near-perfect LCP (Longest Common Prefix) reuse and cutting decoder prefill cost by ~50% on skip chunks
- **Streaming robustness** — Degeneracy detection resets decoder state when stale or repetitive output is detected; periodic re-anchoring prevents unbounded sequence growth

### Changed

- Debug messages (`[stream degen]`, `[stream reanchor]`) now only appear in `--debug` mode
- Project restructured into workspace: `crates/qwen-asr` (library), `crates/qwen-asr-cli` (CLI binary)
- Removed WIP banner from all README files

## [0.2.0] - 2026-02-15

### Performance

- **Reusable BF16→F32 scratch buffer** — Pre-allocated scratch in `DecoderBuffers` eliminates ~140 heap allocations per prefill pass
- **SIMD BF16→F32 bulk conversion** — NEON (`vshll_n_u16`) and AVX2 (`_mm256_cvtepu16_epi32`) paths for 4-8x faster weight conversion
- **Threaded encoder attention** — `bidirectional_attention` parallelized across heads via thread pool for near-linear scaling
- **Cached mel filter bank** — `OnceLock`-based lazy initialization avoids redundant computation in streaming mode
- **SIMD activation functions** — Vectorized `rms_norm`, `gelu`, and `swiglu` with fast polynomial exp approximation (NEON + AVX2)
- **Encoder buffer reuse** — New `EncoderBuffers` struct with persistent scratch avoids per-call allocations in encoder forward pass
- **vDSP integration** (macOS, `--features vdsp`) — `vDSP_dotpr`, `vDSP_vsmul`, `vDSP_vsma`, `vvexpf` leverage Apple AMX coprocessor

### Features

- **Built-in profiling** — `--profile` flag prints per-operation timing breakdown (call count, total/avg time)
- **iOS support** — Static library target with C-FFI API (`src/c_api.rs`): `qwen_asr_load_model`, `qwen_asr_transcribe_file`, `qwen_asr_transcribe_pcm`, `qwen_asr_free`
- **Android support** — Shared library target with JNI API (`src/jni_api.rs`) for `com.qwenasr.QAsrEngine` Java class
- **Feature flags** — `blas` (default), `vdsp`, `ios`, `android` for platform-specific builds
- **Cross-compilation config** — `.cargo/config.toml` with tuned CPU targets for iOS (`apple-a14`) and Android (`cortex-a76`)

### Changed

- Library crate renamed to `qwen_asr` (was `q-asr`) for valid Rust identifier in imports
- Library target now produces `lib`, `staticlib`, and `cdylib` outputs
- Thread pool workers recover from poisoned mutex instead of panicking
- Regression tests serialized via `Mutex` to prevent thread pool race conditions
- README updated with per-platform build instructions (macOS, Linux, iOS, Android)

## [0.1.0] - 2026-02-15

Initial release — pure Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- CPU-only Qwen3-ASR inference (0.6B and 1.7B models)
- Three runtime modes: offline, segmented (`-S`), streaming (`--stream`)
- NEON SIMD (aarch64) and AVX2+FMA (x86_64) acceleration
- BLAS via Accelerate (macOS) / OpenBLAS (Linux)
- Zero runtime Rust crate dependencies (only `libc`)
- 22 tests (kernels, audio, tokenizer, regression)
