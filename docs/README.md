# QwenASR Documentation

This directory contains the authoritative documentation for QwenASR.

## Quick Links

- [Project README](../README.md) — overview, install, and quick-start
- [Benchmarks](./benchmarks/) — how to reproduce and interpret all benchmarks
- [Optimizations](./optimizations/) — catalog of performance techniques
- [Research Logs](./research/) — historical experiment diaries and research protocols

## Directory Guide

### [`benchmarks/`](./benchmarks/)

- [`README.md`](./benchmarks/README.md) — one-page guide to every benchmark script
- [`results.md`](./benchmarks/results.md) — latest speed and WER results
- [`comparison.md`](./benchmarks/comparison.md) — cross-implementation comparison

### [`optimizations/`](./optimizations/)

- [`overview.md`](./optimizations/overview.md) — consolidated catalog of the implemented performance optimizations
- [`overview.zh.md`](./optimizations/overview.zh.md) — 中文优化清单

### [`research/`](./research/)

- [`README.md`](./research/README.md) — index of experiment logs and protocols
- [`experiments.md`](./research/experiments.md) — merged optimization experiment diaries (speed round 1, speed round 2, WER recovery, perf-round2 comparison)
- [`ledger.md`](./research/ledger.md) — ledger of major autoresearch commits and measured gains
- [`programs.md`](./research/programs.md) — merged autoresearch protocols

## Updating These Docs

Benchmark pages are updated by running the commands in [`benchmarks/README.md`](./benchmarks/README.md) and copying the fresh numbers into [`benchmarks/results.md`](./benchmarks/results.md). If you add a new optimization or experiment, append it to the relevant page under `optimizations/` or `research/` and update the index above.
