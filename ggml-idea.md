# GGML Optimization Ideas for qwen-asr

Source pass: DeepWiki MCP over `ggml-org/whisper.cpp` and `ggml-org/llama.cpp`, filtered against `docs/research/experiments.md`.

This file now keeps only methods that are not already checked by the documented S/E/A/B/C/D/F experiments. Ideas that were already accepted, rejected, deferred, or confirmed already implemented were removed.

## Quantization and Weight Layout

## CPU Kernels

## Attention and KV Cache

## Scheduling, Batching, and Threading

## Audio and ASR Pipeline


## Model Loading and Residency


## Backend and Accelerator Options

## Profiling and Benchmarking

- Add kernel-shape benchmark tooling similar to llama-bench for matvec, GEMM, attention, conv, quantize, dequantize, lm_head argmax, and mel.
- Add automated sweeps for chunk size, prefill batch size, quantization format, KV cache type, VAD aggressiveness, and backend choice.

## Conditional Ideas
