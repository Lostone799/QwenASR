# AVX2 INT8 Kernel 4-Row + Prefetch Optimization

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

> **结论 (2026-07-02)**: 本计划全部 3 个任务已执行完毕，**整体 FAIL**。
> - T-AVX2-001 (测试): PASS — 正确性测试保留作为 2-row kernel 回归守卫
> - T-AVX2-002 (实现): 完成 + 单测 PASS，但 A/B 显示 4-row 慢 6.4%，**代码已 REVERTED**
> - T-AVX2-003 (benchmark): FAIL — 4-row 2909ms vs 2-row 2734ms，DRAM 带宽墙
>
> 根因: INT8 matvec 权重 1.7GB 远超 L3 12MB，每 token 全量 DRAM 读取。2-row/4-row 总流量相同，kernel 重构无法突破带宽墙。
> 方法论总结: `docs/optimizations/2026-07-02-avx2-kernel-methodology.md`
> 备用路径 (硬件升级后重启用): `docs/research/failed-optimizations-backup-paths.md`

**Goal:** Upgrade `matvec_int8_avx2` (decode-path INT8 matvec, #1 bottleneck at 39.3%) from 2-row to 4-row processing with weight prefetch, targeting 15-30% reduction in `bf16_matvec` time.

**Architecture:** The decode path (single-token, 3024 calls) currently uses `matvec_int8_avx2` which processes 2 output rows per iteration. The prefill path already uses `int8_gemm_4rows_avx2` (4-row). We inline the 4-row pattern into `matvec_int8_avx2` (with bias support + 2-row/1-row tail), and add `_mm_prefetch` to preload the next weight cache line while computing the current one.

**Tech Stack:** Rust, x86 AVX2 intrinsics (`core::arch::x86_64`), `_mm_prefetch` (PREFETCHT0).

**Baseline:** Inference 6400ms, `bf16_matvec` 2516.6ms (3024 calls, 0.83ms avg), 39.3% of total.

---

## Context: Why 4-row helps

INT8 matvec is **memory-bound**. Each weight byte is used once (no reuse across tokens in decode). The bottleneck is cache-miss latency on weight streams. By processing 4 rows simultaneously, we:
1. Share each `x_int8` load (32 bytes) across 4 weight rows → 4x arithmetic intensity per x load
2. Amortize loop overhead / branch pressure
3. With prefetch: hide weight-cache-line miss latency by issuing prefetch 1 iteration ahead

The 4-row pattern already exists in `int8_gemm_4rows_avx2` (avx.rs:695) but lacks bias support and only handles exactly 4 rows. We generalize it inside `matvec_int8_avx2`.

## Reference: existing 4-row kernel pattern (int8_gemm_4rows_avx2, avx.rs:695-767)

```rust
let mut acc0..acc3 = _mm256_setzero_si256();
while k + 32 <= in_dim {
    let x = _mm256_loadu_si256(x_int8.add(k) as *const __m256i);
    let xu = _mm256_xor_si256(x, sf256);
    acc0 = dot_i8_avx2_acc_256v2(acc0, xu, _mm256_loadu_si256(w0.add(k) ...), ones256);
    acc1 = dot_i8_avx2_acc_256v2(acc1, xu, _mm256_loadu_si256(w1.add(k) ...), ones256);
    acc2 = dot_i8_avx2_acc_256v2(acc2, xu, _mm256_loadu_si256(w2.add(k) ...), ones256);
    acc3 = dot_i8_avx2_acc_256v2(acc3, xu, _mm256_loadu_si256(w3.add(k) ...), ones256);
    k += 32;
}
// 16-byte tail, scalar tail, then finalize 4 rows
```

---

### Task 1: Write failing correctness test for 4-row matvec (RED)

**Files:**
- Modify: `c:\Users\Administrator\clawd\QwenASR\crates\qwen-asr\src\kernels\mod.rs` (append to `#[cfg(test)]` module at line 3391)

**Step 1: Write the failing test**

Add a test that builds a known INT8 matvec problem and checks `matvec_int8_avx2` output matches a scalar reference, covering edge cases where `out_dim` is NOT a multiple of 4 (forces 4-row + 2-row + 1-row paths):

```rust
#[test]
fn test_matvec_int8_avx2_4row_correctness() {
    // Safety: AVX2 required. Skip on non-x86_64 / no AVX2.
    if !is_x86_feature_detected!("avx2") {
        eprintln!("skipping: AVX2 not available");
        return;
    }
    let in_dim = 256usize;
    // Test several out_dim values to exercise 4-row, 2-row, 1-row tails
    for &out_dim in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17] {
        let mut w_int8: Vec<i8> = Vec::with_capacity(out_dim * in_dim);
        let mut w_scales: Vec<f32> = Vec::with_capacity(out_dim);
        let mut w_sums: Vec<i32> = Vec::with_capacity(out_dim);
        for o in 0..out_dim {
            let scale = 0.01 + (o as f32) * 0.001;
            w_scales.push(scale);
            let mut row_sum: i32 = 0;
            for i in 0..in_dim {
                let v = (((o * 7 + i * 3) % 201) - 100) as i8; // -100..100
                w_int8.push(v);
                row_sum += v as i32;
            }
            w_sums.push(row_sum);
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 50) as f32 - 25.0) * 0.1).collect();
        // Quantize x to int8 (mirror quantize_f32_to_int8)
        let xmax = x.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        let x_scale = xmax / 127.0;
        let x_int8: Vec<i8> = x.iter().map(|&v| (v / x_scale).round() as i8).collect();

        let bias: Vec<f32> = (0..out_dim).map(|o| o as f32 * 0.5).collect();

        // Scalar reference: y[o] = (sum_i x_int8[i]*w[o,i] - 128*w_sum[o]) * x_scale * w_scale[o] + bias[o]
        let mut y_ref = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut s: i32 = 0;
            for i in 0..in_dim {
                s += x_int8[i] as i32 * w_int8[o * in_dim + i] as i32;
            }
            let corrected = s - 128 * w_sums[o];
            y_ref[o] = corrected as f32 * x_scale * w_scales[o] + bias[o];
        }

        // AVX2 kernel
        let mut y_simd = vec![0.0f32; out_dim];
        unsafe {
            avx::matvec_int8_avx2(
                &mut y_simd, x_int8.as_ptr(), x_scale,
                w_int8.as_ptr(), &w_scales, &w_sums, Some(&bias),
                in_dim, out_dim,
            );
        }

        for o in 0..out_dim {
            let diff = (y_simd[o] - y_ref[o]).abs();
            let tol = (y_ref[o].abs() * 1e-4 + 1e-3).max(1e-2);
            assert!(diff < tol, "out_dim={} o={}: simd={} ref={} diff={}", out_dim, o, y_simd[o], y_ref[o], diff);
        }
    }
}
```

**Step 2: Run test to verify it PASSES on current 2-row kernel**

Run: `cargo test -p qwen-asr test_matvec_int8_avx2_4row_correctness --release -- --nocapture`
Expected: PASS (the current 2-row kernel is already correct; this test guards against regressions when we change to 4-row)

**Step 3: Commit**

```bash
git add crates/qwen-asr/src/kernels/mod.rs
git commit -m "test: add 4-row matvec_int8_avx2 correctness guard"
```

---

### Task 2: Implement G — 4-row main loop in matvec_int8_avx2 (GREEN)

**Files:**
- Modify: `c:\Users\Administrator\clawd\QwenASR\crates\qwen-asr\src\kernels\avx.rs:840-916`

**Step 1: Rewrite matvec_int8_avx2 with 4-row main loop + 2-row/1-row tail**

Replace the body of `matvec_int8_avx2`. The structure:

```rust
pub unsafe fn matvec_int8_avx2(
    y: &mut [f32], x_int8: *const i8, x_scale: f32,
    w_int8: *const i8, w_scales: &[f32], w_sums: &[i32],
    bias: Option<&[f32]>,
    in_dim: usize, out_dim: usize,
) {
    let sf256 = sign_flip_256();
    let sf128 = sign_flip_128();
    let ones256 = _mm256_set1_epi16(1);
    let mut o = 0usize;

    // === 4-row main loop: share x_int8 across 4 weight rows ===
    while o + 4 <= out_dim {
        let w0 = w_int8.add(o * in_dim);
        let w1 = w_int8.add((o + 1) * in_dim);
        let w2 = w_int8.add((o + 2) * in_dim);
        let w3 = w_int8.add((o + 3) * in_dim);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut k = 0usize;

        while k + 32 <= in_dim {
            let x = _mm256_loadu_si256(x_int8.add(k) as *const __m256i);
            let xu = _mm256_xor_si256(x, sf256);
            // Prefetch next-iteration weight cache lines (hide miss latency)
            if k + 64 <= in_dim {
                _mm_prefetch::<_MM_HINT_T0>(w0.add(k + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w1.add(k + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w2.add(k + 64) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w3.add(k + 64) as *const i8);
            }
            acc0 = dot_i8_avx2_acc_256v2(acc0, xu,
                _mm256_loadu_si256(w0.add(k) as *const __m256i), ones256);
            acc1 = dot_i8_avx2_acc_256v2(acc1, xu,
                _mm256_loadu_si256(w1.add(k) as *const __m256i), ones256);
            acc2 = dot_i8_avx2_acc_256v2(acc2, xu,
                _mm256_loadu_si256(w2.add(k) as *const __m256i), ones256);
            acc3 = dot_i8_avx2_acc_256v2(acc3, xu,
                _mm256_loadu_si256(w3.add(k) as *const __m256i), ones256);
            k += 32;
        }
        // 16-byte tail
        while k + 16 <= in_dim {
            let xu = _mm_xor_si128(_mm_loadu_si128(x_int8.add(k) as *const __m128i), sf128);
            acc0 = dot_i8_avx2_acc_256(acc0, xu, _mm_setzero_si128(),
                _mm_loadu_si128(w0.add(k) as *const __m128i), _mm_setzero_si128(), ones256);
            acc1 = dot_i8_avx2_acc_256(acc1, xu, _mm_setzero_si128(),
                _mm_loadu_si128(w1.add(k) as *const __m128i), _mm_setzero_si128(), ones256);
            acc2 = dot_i8_avx2_acc_256(acc2, xu, _mm_setzero_si128(),
                _mm_loadu_si128(w2.add(k) as *const __m128i), _mm_setzero_si128(), ones256);
            acc3 = dot_i8_avx2_acc_256(acc3, xu, _mm_setzero_si128(),
                _mm_loadu_si128(w3.add(k) as *const __m128i), _mm_setzero_si128(), ones256);
            k += 16;
        }
        let tail0 = dot_i8_tail(x_int8, w0, k, in_dim);
        let tail1 = dot_i8_tail(x_int8, w1, k, in_dim);
        let tail2 = dot_i8_tail(x_int8, w2, k, in_dim);
        let tail3 = dot_i8_tail(x_int8, w3, k, in_dim);

        let mut v0 = finalize_int8(hsum_epi32(acc0), w_sums[o], tail0, x_scale, w_scales[o]);
        let mut v1 = finalize_int8(hsum_epi32(acc1), w_sums[o+1], tail1, x_scale, w_scales[o+1]);
        let mut v2 = finalize_int8(hsum_epi32(acc2), w_sums[o+2], tail2, x_scale, w_scales[o+2]);
        let mut v3 = finalize_int8(hsum_epi32(acc3), w_sums[o+3], tail3, x_scale, w_scales[o+3]);
        if let Some(b) = bias {
            v0 += b[o]; v1 += b[o+1]; v2 += b[o+2]; v3 += b[o+3];
        }
        y[o] = v0; y[o+1] = v1; y[o+2] = v2; y[o+3] = v3;
        o += 4;
    }

    // === 2-row tail (unchanged from original) ===
    while o + 1 < out_dim {
        // ... (keep original 2-row block verbatim)
        o += 2;
    }

    // === 1-row tail (unchanged from original) ===
    while o < out_dim {
        // ... (keep original 1-row block verbatim)
        o += 1;
    }
}
```

**Step 2: Run test to verify PASS**

Run: `cargo test -p qwen-asr test_matvec_int8_avx2_4row_correctness --release -- --nocapture`
Expected: PASS

**Step 3: Build the CLI**

Run: `cargo build --release -p qwen-asr-cli --bin qwen-asr`
Expected: Compiles clean

**Step 4: Commit**

```bash
git add crates/qwen-asr/src/kernels/avx.rs
git commit -m "perf(avx2): 4-row main loop + prefetch in matvec_int8_avx2"
```

---

### Task 3: A/B benchmark to verify performance (TP-AVX2-001)

**Files:** none (benchmark only)

**Step 1: Run profile benchmark**

```
$vcrt = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT"
$env:Path = "$vcrt;c:\Users\Administrator\clawd\onednn_build\src\Release;c:\Users\Administrator\clawd\openblas\bin;" + $env:Path
cd c:\Users\Administrator\clawd\QwenASR
.\target\release\qwen-asr.exe -d models\qwen3-asr-rust-1.7b -i audio.wav -S 0 --profile
```

**Step 2: Compare against baseline**

- Baseline: `bf16_matvec` 2516.6ms, inference 6400ms
- Target: `bf16_matvec` ≤ 2139ms (≥15% reduction), inference ≤ 6000ms

**Step 3: Record result in 05-test-plan.md**

---

## Verification

- Correctness: `test_matvec_int8_avx2_4row_correctness` PASS (edge cases out_dim 1..17)
- Performance: `bf16_matvec` ≥ 15% faster than baseline 2516.6ms
- No regression: end-to-end output unchanged (same transcription)
