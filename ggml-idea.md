# GGML Optimization Ideas for qwen-asr

Source pass: DeepWiki MCP over `ggml-org/whisper.cpp` and `ggml-org/llama.cpp`, filtered against `docs/research/experiments.md`.

This file now keeps only methods that are not already checked by the documented S/E/A/B/C/D/F experiments. Ideas that were already accepted, rejected, deferred, or confirmed already implemented were removed.

## Quantization and Weight Layout

- Add group-wise low-bit decoder quantization, such as GPTQ/AWQ-style INT4 or ggml K-quant/IQ-style formats with small groups and zero-points. The documented E12 probe rejected naive per-row symmetric INT4, but not calibrated group-wise formats.
- Quantize the encoder transformer and projection weights. Current experiments focus mainly on decoder INT8, decoder INT4, and prefill copies; encoder weight quantization remains unchecked.
- Use mixed quantization by tensor role: keep sensitive tensors in f16/bf16/f32 and use lower-bit formats only for memory-bound FFN, projection, lm_head, or selected encoder matrices.
- Repack quantized weights into SIMD-native interleaved layouts for formats beyond the current INT8 SDOT path. The I8MM/SMMLA experiment was checked and rejected, but block-quant layout work for Q4/Q5/K-quant-style kernels remains unchecked.
- Consider per-layer or per-block activation quantization scales backed by offline calibration. The global static scale experiment failed; calibrated local scales remain a different method.

## CPU Kernels

- Add tiny-shape specialized kernels for common qwen-asr dimensions where BLAS/custom-kernel crossover points are known from benchmarks.
- Add fused dequantize-dot-accumulate kernels for future low-bit formats so dequantized f32 blocks are not materialized.

## Attention and KV Cache

- Evaluate true tiled flash-attention-style prefill for larger contexts. E8 accepted batched-GEMM prefill attention, but a memory-efficient tiled implementation remains a separate idea.

## Scheduling, Batching, and Threading

- Add graph/stage-level scheduling boundaries similar to whisper.cpp's separate conv, encoder, and cross/decoder schedulers. Existing code has explicit stages and profiling, but not a scheduler abstraction.
- Add micro-batching for repeated decoder prefill work across independent utterances or streams.
- Add batched decode across independent segments in segmented mode so each token step reads weights once for multiple segment states.
- Add segment-level pipelining: encode segment N+1 while decoding segment N, using AMX-heavy encoder work and NEON-heavy decode work concurrently.
- Add adaptive work partitioning thresholds based on measured shapes, choosing single-thread, thread-pool, BLAS, or custom kernels per operation.
- Add CPU/AMX overlap pipelining inside the encoder/prefill path, running CPU-side ops such as im2col, softmax, norms, or activations while AMX GEMMs are in flight.

## Audio and ASR Pipeline


## Model Loading and Residency


## Backend and Accelerator Options

- Evaluate f16 or bf16 GEMM through BNNS/AMX for encoder and decoder prefill. This is distinct from rejected hand-written INT8 prefill GEMM and could remove or shrink f32 prefill copies.

## Profiling and Benchmarking

- Add kernel-shape benchmark tooling similar to llama-bench for matvec, GEMM, attention, conv, quantize, dequantize, lm_head argmax, and mel.
- Add automated sweeps for chunk size, prefill batch size, quantization format, KV cache type, VAD aggressiveness, and backend choice.

## Conditional Ideas
