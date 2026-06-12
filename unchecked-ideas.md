# Unchecked Performance Ideas

Candidate optimizations that have **not** been tried yet (not covered by round-1 S1–S37, round-2 E1–E13, or the WER-recovery steps in `docs/research/experiments.md`). Each idea is grounded in the current profile:

- Offline 28 s sample: ~447 ms inference / ~817 ms wall → **~370 ms fixed startup** is the largest single chunk.
- Encoder + decoder prefill are **AMX-bound** (Accelerate sgemm, doesn't scale with threads).
- Single-token decode is **bandwidth-bound** (reads ~500 MB INT8 weights per token; SDOT NEON matvec).
- Peak RSS 5.1 GB (bf16 mmap + 1.76 GB f32 prefill + 0.44 GB INT8).
- KV cache is f32 (`decoder.rs::KvCache`).

Gate for all ideas: 100-file LibriSpeech offline WER ≤ 0.04.

---

## A. Startup / wall-clock (~370 ms fixed floor)

### A1. Pre-quantized weight cache on disk
Serialize INT8 weights + scales (and optionally the f32 prefill matrices) to a sidecar file on first run; mmap it directly afterwards. Removes the per-run bf16→f32/INT8 conversion entirely instead of parallelizing it (E2) or relocating it (rejected E3). Could also let `download.rs` fetch pre-quantized artifacts.
*Impact: high (most of remaining load). Effort: medium. Risk: low (bit-identical weights).*

### A2. Overlap model load with the audio front-end
WAV decode, resample, silence compaction, and mel extraction need no weights. Run them on a separate thread concurrently with weight loading so wall ≈ max(load, mel) instead of load + mel.
*Status: ✅ accepted. −9% to −12% wall time; small inference-time increase from contention, WER unchanged. Currently overlaps WAV decode/resample/compaction; mel still inside inference.*
*Impact: medium. Effort: low–medium. Risk: low.*

### A3. Tokenizer binary cache / lazy build
If vocab.json parse + trie construction is a measurable slice of the ~370 ms floor, cache the parsed tokenizer in a binary format, or defer building it until the first decoded token.
*Status: ❌ rejected. Lazy merge-map build gave mixed wall results and inference regressions; binary cache not pursued.*
*Impact: small–medium (measure first). Effort: low. Risk: low.*

### A4. Daemon / server mode
`--serve` (socket or stdin protocol) keeping weights resident; load cost amortizes to zero across requests. Doesn't speed a single run but changes practical latency for repeated use (and helps the Flutter/JNI embedding story).
*Impact: high for repeated use. Effort: medium. Risk: low.*

### A5. Page-fault prefaulting of the mmap'd model
`madvise(MADV_WILLNEED)` / parallel page-touch on the 1.87 GB mmap before the conversion pass so first-touch faults don't serialize inside load loops.
*Status: ✅ accepted. −8% to −11% inference / wall on top of E1, WER unchanged.*
*Impact: small–medium. Effort: low. Risk: none.*

### A6. Per-phase wall breakdown in bench
Not an optimization itself: add a startup-phase breakdown (mmap, convert, quantize, tokenizer, mel) to `--profile` so A1–A5 can be sized and tracked for regressions.
*Status: ✅ accepted as tooling. Added model/encoder/decoder/tokenizer/audio/mel profile counters; no speed impact.*
*Impact: enabler. Effort: low.*

---

## B. Decode (bandwidth-bound; dominates real uncapped clips)

### B1. NEON i8mm (SMMLA) matvec kernels ❌ rejected
Current kernels use SDOT. On cores with i8mm, SMMLA computes a 2×2×4 int8 block per instruction. A runtime-detected SMMLA variant was implemented for `matvec_int8` and `argmax_int8_range` (interleaving two rows of W and broadcasting x into the B matrix).
*Status: ❌ rejected. Regressed −5% to −9% inference across modes vs the existing well-unrolled SDOT kernels. Memory bandwidth is the bottleneck; the extra load/shuffle overhead to form SMMLA inputs outweighs any instruction-throughput advantage on this workload.*
*Impact: medium (compute side of matvec; bandwidth still caps it). Effort: medium. Risk: low.*

### B2. Group-wise INT4 (GPTQ/AWQ-style)
E12 only probed naive per-row symmetric INT4 (WER 0.25 — rejected). Group size 32–64 with zero-points is the known-good recipe; halves decode weight bandwidth, the only remaining ~2× lever on decode.
*Impact: high. Effort: high (research-grade quantizer + NEON kernel). Risk: WER gate.*

### B3. Batched decode across segments (lockstep segment decoding)
In segmented mode, segments are independent. Decode B segments in lockstep so each step is a `[B, dim]` skinny GEMM instead of B separate matvecs — weights are read **once per step for all B tokens**, amortizing the dominant bandwidth cost by ~B×. Probably the single biggest throughput idea for long files.
*Impact: high (long files). Effort: high (batched decoder state, per-segment KV). Risk: low for WER (same math per segment).*

### B4. Self-speculative streaming decode (draft = previous chunk's transcript)
E13 rejected speculative decoding for lack of a draft model — but streaming re-decodes overlapping audio every chunk, and chunk N+1's output largely reproduces chunk N's text. Use the previous chunk's tokens as draft, verify them in one batched prefill pass, and fall back to token-by-token decode only at the divergence point. This is the decode-side analogue of the existing LCP prefill reuse.
*Impact: high (streaming). Effort: medium–high. Risk: low (verification preserves greedy output exactly).*

### B5. Fused QKV INT8 matvec (single-token path)
One pass over the input activation feeding three weight streams (one x load instead of three). E4/E5 rejected the *GEMM* fusion on AMX; the NEON single-token matvec path was never tested.
*Status: ✅ already implemented. `linear_nobias_int8_qkv` shares the quantized activation across Q/K/V matvecs.*
*Impact: small. Effort: low. Risk: low.*

### B6. Software prefetch (`prfm`) in INT8 matvec/argmax inner loops
Weight streams are perfectly sequential; explicit prefetch ahead of the SDOT loop is sometimes worth 5–10% on bandwidth-bound aarch64 loops.
*Status: ❌ rejected. Added 3–4% overhead on Apple M5 Pro; hardware prefetcher already covers the sequential streams.*
*Impact: small–medium. Effort: low. Risk: none.*

### B7. f16 (or INT8) KV cache
KV cache is f32 today. f16 halves attention-scan bandwidth per token; matters more as context grows (long offline clips, streaming carry-over). INT8 KV is the aggressive variant.
*Impact: small–medium (grows with context length). Effort: medium. Risk: small WER risk; gate it.*

### B8. Two-stage lm_head argmax
First pass over a coarser representation (e.g., INT4 or even per-row scale bounds) to shortlist top-k candidates, then exact INT8 rescoring of only the shortlist. Complements the existing 0–39k range shortlist.
*Impact: small–medium (argmax is the per-token tail). Effort: medium. Risk: must guarantee exact argmax (bound-based pruning can be exact).*

### B9. Overlap lm_head argmax with next-token start
Let argmax shards return early and allow one thread to begin next-token layer-0 work (embedding fetch can be deferred but norms/buffers can be staged). Marginal pipelining of the per-token tail.
*Impact: small. Effort: medium. Risk: complexity.*

### B10. Static activation quantization scales ❌ rejected
If the INT8 matvec currently re-quantizes activations dynamically per call, calibrated static scales would skip the quantize pass per token per layer.
*Status: ❌ rejected. A global static scale either clips (observed max_abs up to 421.7 on one file) or, if enlarged to cover the range, maps typical activations to int8 values near 0 and destroys precision. The speed-sample WER jumped to 1.0000 and the 100-file run timed out. Per-layer calibrated scales might work but need substantial offline calibration for a small compute win.*
*Impact: small. Effort: low–medium. Risk: WER gate exceeded with naive global scale.*

---

## C. Encoder / prefill (AMX-bound)

### C1. CPU/AMX overlap pipelining
sgemm runs on the shared AMX coprocessor while P-cores mostly wait. Restructure so CPU-side ops (softmax, norms, activations, im2col for the next chunk) run concurrently with the current AMX GEMM instead of strictly alternating.
*Impact: medium. Effort: high (restructuring + careful sync). Risk: medium (complexity).*

### C2. Segment-level pipelining for `-S` mode
Encode segment N+1 while decoding segment N (encoder uses AMX, decode uses NEON cores — naturally complementary resources). On long files this hides nearly all encoder time behind decode.
*Impact: high (long files). Effort: medium–high. Risk: low.*

### C3. f16 GEMM for encoder/prefill via BNNS
AMX f16 has ~2× f32 throughput and halves weight bandwidth; also halves the 1.76 GB f32 prefill copy. f16 (unlike naive INT4) usually survives a WER gate.
*Impact: medium–high. Effort: medium (BNNS integration). Risk: WER gate; BNNS API surface.*

### C4. bf16 GEMM via BNNS/AMX directly from the mmap
Same as C3 but consuming weights as-is: removes the f32 prefill copy, its load-time conversion, **and** 1.76 GB RAM — the win E3/E11 aimed at, via a route not yet evaluated (E11 only ruled out hand-written CPU INT8 GEMM).
*Impact: medium–high (wall + RAM). Effort: medium. Risk: depends on BNNS bf16 support/perf on target macOS.*

### C5. Winograd transform for encoder 3×3 convs
Reduces conv FLOPs ~2.25× vs im2col+GEMM. conv2/conv3 dominate conv time (67 ms profile bucket). AMX GEMM is very fast, so the realized gain may be far below theoretical.
*Impact: small–medium. Effort: high. Risk: numeric drift at chunk boundaries (E6 showed boundary sensitivity).*

### C6. Batch encoder windows into larger GEMMs
If encoder attention/FFN GEMMs are issued per window, batching multiple windows raises M and improves AMX utilization per call. (Verify current shapes first — may already be effectively batched.)
*Impact: unknown until measured. Effort: low to probe. Risk: low.*

---

## D. Threading / scheduling

### D1. Per-phase thread counts (safe re-do)
E1-revisited found decode wants ~4–5 P-cores while pre-E8 encoder liked more threads; a finer-grained attempt raced and was abandoned. Safe version: `parallel_for` takes an explicit `max_workers` per call site (no pool resize), so decode matvecs cap at 4–5 while encoder-side parallel ops can use more.
*Impact: small–medium. Effort: low–medium. Risk: low if no pool surgery.*

### D2. macOS QoS hints instead of pinning
Set worker threads to `QOS_CLASS_USER_INTERACTIVE` so the scheduler keeps them on P-cores under load (pinning isn't available; E10's approach was rejected on an idle machine, but QoS matters on a busy one).
*Status: ❌ rejected. Slight regression on idle benchmark (+2–6%); no contention gate in harness.*
*Impact: small on idle bench; real under contention. Effort: low. Risk: none.*

### D3. Superpages for hot weight allocations ✅ accepted
Allocate the INT8/f32 weight buffers with 2 MB superpages (`posix_memalign` to 2 MB, with fallback to normal `Vec`) to cut TLB misses during the ~500 MB/token streaming weight reads.
*Status: ✅ accepted. −1% to −5% inference/wall across modes; WER unchanged.*
*Impact: small–medium. Effort: low–medium. Risk: allocation may fail/fragment — fallback handles it.*

---

## E. Build-level / system

### E1. Fat LTO + `codegen-units=1` + PGO
Profile is `lto = "thin"` today. Fat LTO, `codegen-units = 1`, and PGO on the benchmark workload are cheap one-time experiments, occasionally 3–8% on hot scalar/glue code.
*Status: ✅ fat LTO + codegen-units=1 accepted; PGO not yet tested. Measured −28% to −36% inference speedup across modes, WER unchanged.*
*Impact: small. Effort: low. Risk: none (measurable, revertible).*

### E2. vDSP FFT for mel
Mel DFT is currently a dense GEMM (O(n²) per frame batch); `vDSP_fft_zrip` is O(n log n). Small slice today but scales with audio length.
*Impact: small (grows with file length). Effort: medium. Risk: numeric drift vs reference mel — gate it.*

### E3. Metal GPU offload for encoder GEMMs (optional feature)
Against the CPU-only ethos, but as a cargo feature: MPS matmul for encoder/prefill frees AMX/CPU entirely and likely several-× on the encoder. Listed for completeness.
*Impact: high. Effort: high. Risk: scope/philosophy; feature-gated.*

### E4. Core ML / ANE encoder offload (optional feature)
Export the encoder to a Core ML model to run on the idle Neural Engine while Rust keeps the decoder. Same scope caveat as E3.
*Impact: high. Effort: very high. Risk: scope; conversion fidelity.*

---

## F. Memory (secondary to speed, but affects small machines)

### F1. Release f32 prefill copies after last prefill
Offline mode performs exactly one prefill; `madvise(MADV_FREE)`/drop the 1.76 GB f32 copies afterwards. Peak RSS 5.1 → ~3.3 GB. No bench speedup on a 32 GB machine, but avoids swap-driven slowdowns on 8–16 GB targets (and matters for the mobile/JNI builds).
*Status: ❌ rejected on speed gate. Slight wall-time regression (+2–5%) from deallocation on benchmark machine; no WER impact.*
*Impact: none on bench; real on small RAM. Effort: low. Risk: none.*

### F2. Skip building unused weight copies per mode
Audit which modes actually touch which copies (e.g., a pure-streaming session that always skips non-final prefills may not need all f32 prefill matrices); build only what the selected mode uses.
*Impact: small wall + RAM. Effort: medium (audit). Risk: low.*

---

## Suggested priority

| Rank | Idea | Why |
|------|------|-----|
| 1 | A1 disk weight cache | Largest remaining wall chunk, low risk |
| 2 | B3 batched segment decode | Only ~B× lever on bandwidth-bound decode for long files |
| 3 | B4 self-speculative streaming | Big streaming win, exact-output-preserving |
| 4 | A2 load/mel overlap | Cheap, stacks with A1 |
| 5 | C2 segment pipelining | Hides encoder behind decode on long files |
| 6 | B1 i8mm kernels | Straightforward kernel upgrade |
| 7 | C4 bf16 BNNS prefill | Kills the 1.76 GB copy + load cost if BNNS delivers |
| 8 | B2 group-wise INT4 | Highest decode ceiling, highest effort |
