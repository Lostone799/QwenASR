//! Hand-written 3×3 stride=2 padding=1 conv2d kernels for the
//! audio encoder stem. Targets x86_64 AVX2+FMA.
//!
//! **Two kernels**:
//! - `conv2d_3x3_s2_p1_avx2_1_to_480`: encoder layer 1 (c_in=1, c_out=480)
//! - `conv2d_3x3_s2_p1_generic`: any c_in/c_out (multiples of 8),
//!   used for encoder layers 2/3 (480→480). Supports parallel
//!   output-channel tiling via `(oc_start, oc_end)`.
//!
//! Design: output-channel-major parallelism, input-channel streaming,
//! 8-wide AVX2+FMA accumulation per output position.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

/// tanh-GELU on a single f32, matching the existing formula in
/// `kernels/mod.rs::gelu` and `kernels/avx.rs::gelu_inplace`.
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978845608028654_f32 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + inner.tanh())
}

/// Compute one output position `(oh, ow)` for ALL `c_out` channels
/// of the 1 → 480 conv2d layer 1, with bias-add and fused tanh-GELU.
///
/// Walks 480 `c_out` in groups of 8, loading 9 weights per group and
/// reusing the 9 input pixel values (all from `c_in=0`) across the
/// group. Out-of-bounds input (padding) is handled by gating the
/// fmadd: weight × 0 = no contribution, acc stays as bias.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_3x3_s2_p1_avx2_1_to_480(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    output: *mut f32,
    oh: usize,
    ow: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    let spatial_out = h_out * w_out;
    let s = oh * w_out + ow;

    // 9 input pixel offsets and bounds for (kh, kw).
    let mut in_offsets = [0usize; 9];
    let mut in_bounds = [false; 9];
    for kh in 0..3 {
        let ih_signed = (oh * 2 + kh) as isize - 1;
        let ih_ok = ih_signed >= 0 && (ih_signed as usize) < h_in;
        for kw in 0..3 {
            let iw_signed = (ow * 2 + kw) as isize - 1;
            let iw_ok = iw_signed >= 0 && (iw_signed as usize) < w_in;
            in_offsets[kh * 3 + kw] = (ih_signed as usize) * w_in + (iw_signed as usize);
            in_bounds[kh * 3 + kw] = ih_ok && iw_ok;
        }
    }

    // Broadcast 9 input pixel values to all 8 AVX2 lanes.
    let mut in_pixels = [_mm256_setzero_ps(); 9];
    for i in 0..9 {
        if in_bounds[i] {
            let v = *input.add(in_offsets[i]);
            in_pixels[i] = _mm256_set1_ps(v);
        }
    }

    // Walk 480 c_out in groups of 8.
    let mut oc = 0;
    while oc + 8 <= 480 {
        let bias_v = _mm256_loadu_ps(bias.add(oc));
        let mut acc = bias_v;

        for i in 0..9 {
            if in_bounds[i] {
                let w = _mm256_loadu_ps(weight.add(oc * 9 + i));
                acc = _mm256_fmadd_ps(w, in_pixels[i], acc);
            }
        }

        // Store + scalar GELU on the 8 lanes. (See module docs for
        // why we keep activation scalar; vectorized GELU is a
        // follow-up.)
        let mut buf = [0.0_f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        for lane in 0..8 {
            buf[lane] = gelu_scalar(buf[lane]);
        }
        let gelu_v = _mm256_loadu_ps(buf.as_ptr());
        _mm256_storeu_ps(output.add(oc * spatial_out + s), gelu_v);
        oc += 8;
    }

    // Scalar tail: 480 % 8 == 0, so this is dead code today. Kept
    // for future-proofing if the channel count ever changes.
    while oc < 480 {
        let mut acc = *bias.add(oc);
        for i in 0..9 {
            if in_bounds[i] {
                let w = *weight.add(oc * 9 + i);
                let v = *input.add(in_offsets[i]);
                acc += w * v;
            }
        }
        *output.add(oc * spatial_out + s) = gelu_scalar(acc);
        oc += 1;
    }
}

/// Generic 3×3 stride=2 padding=1 conv2d with fused tanh-GELU.
/// Processes output channels in `[oc_start, oc_end)` range (must be
/// multiples of 8). Used for encoder layers 2/3 (480→480).
///
/// **Weight layout (transposed for AVX2)**:
/// Original PyTorch layout: `[c_out, c_in, 3, 3]` = `[c_out, c_in*9]`.
/// The caller must transpose weights into groups of 8 output channels:
/// `[n_groups, c_in*9, 8]` where `n_groups = c_out/8`.
/// Within each group: `wt[g * c_in*72 + ic * 72 + (kh*3+kw) * 8 + lane]`
///
/// Input layout:  `[c_in, h_in, w_in]`
/// Output layout: `[c_out, h_out, w_out]`
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_3x3_s2_p1_generic(
    input: *const f32,
    weight: *const f32,  // transposed: [n_groups, c_in*9, 8]
    bias: *const f32,
    output: *mut f32,
    c_in: usize,
    _c_out: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    oc_start: usize,
    oc_end: usize,
) {
    let spatial_in = h_in * w_in;
    let spatial_out = h_out * w_out;
    let wt_group_stride = c_in * 9 * 8; // c_in * 72
    let ic_stride = 72; // 9 * 8
    let kh_stride = 24; // 3 * 8
    let kw_stride = 8;

    let mut oc = oc_start;
    while oc + 8 <= oc_end {
        let oc_group = oc / 8;
        let wt_base = weight.add(oc_group * wt_group_stride);
        // Load 8 bias values
        let bias_v = _mm256_loadu_ps(bias.add(oc));

        for oh in 0..h_out {
            let ih_base = oh * 2; // stride = 2

            // --- Column 0 (ow=0, iw=-1 for kw=0) ---
            let mut acc = bias_v;
            for ic in 0..c_in {
                let wb = wt_base.add(ic * ic_stride);
                let ic_off = ic * spatial_in;

                // kh=0
                if ih_base > 0 && ih_base - 1 < h_in {
                    let row = ic_off + (ih_base - 1) * w_in;
                    // kw=0: iw=-1 (padding, skip)
                    // kw=1: iw=0
                    let v1 = _mm256_set1_ps(*input.add(row));
                    // kw=2: iw=1
                    let v2 = if w_in > 1 { _mm256_set1_ps(*input.add(row + 1)) } else { _mm256_setzero_ps() };

                    let w1 = _mm256_loadu_ps(wb.add(kw_stride));     // wt[kh=0,kw=1]
                    let w2 = _mm256_loadu_ps(wb.add(2 * kw_stride)); // wt[kh=0,kw=2]
                    acc = _mm256_fmadd_ps(w1, v1, acc);
                    acc = _mm256_fmadd_ps(w2, v2, acc);
                }

                // kh=1: ih = ih_base
                if ih_base < h_in {
                    let row = ic_off + ih_base * w_in;
                    // kw=0: iw=-1 (padding, skip)
                    // kw=1: iw=0
                    let v1 = _mm256_set1_ps(*input.add(row));
                    // kw=2: iw=1
                    let v2 = if w_in > 1 { _mm256_set1_ps(*input.add(row + 1)) } else { _mm256_setzero_ps() };

                    let w4 = _mm256_loadu_ps(wb.add(kh_stride + kw_stride));       // wt[kh=1,kw=1]
                    let w5 = _mm256_loadu_ps(wb.add(kh_stride + 2 * kw_stride));   // wt[kh=1,kw=2]
                    acc = _mm256_fmadd_ps(w4, v1, acc);
                    acc = _mm256_fmadd_ps(w5, v2, acc);
                }

                // kh=2: ih = ih_base+1
                if ih_base + 1 < h_in {
                    let row = ic_off + (ih_base + 1) * w_in;
                    let v1 = _mm256_set1_ps(*input.add(row));
                    let v2 = if w_in > 1 { _mm256_set1_ps(*input.add(row + 1)) } else { _mm256_setzero_ps() };

                    let w7 = _mm256_loadu_ps(wb.add(2 * kh_stride + kw_stride));       // wt[kh=2,kw=1]
                    let w8 = _mm256_loadu_ps(wb.add(2 * kh_stride + 2 * kw_stride));   // wt[kh=2,kw=2]
                    acc = _mm256_fmadd_ps(w7, v1, acc);
                    acc = _mm256_fmadd_ps(w8, v2, acc);
                }
            }

            // Scatter-store (no fused GELU — caller applies SIMD GELU)
            let mut buf = [0.0_f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            for lane in 0..8 {
                *output.add((oc + lane) * spatial_out + oh * w_out) = buf[lane];
            }

            // --- Columns 1..w_out (remaining columns, handles kw=2 OOB via bounds check) ---
            let mut ow = 1;
            while ow < w_out {
                let iw_base = ow * 2; // center input col for kw=1
                let mut acc = bias_v;

                for ic in 0..c_in {
                    let wb = wt_base.add(ic * ic_stride);
                    let ic_off = ic * spatial_in;

                    // kh=0
                    if ih_base > 0 && ih_base - 1 < h_in {
                        let row = ic_off + (ih_base - 1) * w_in;
                        let v0 = _mm256_set1_ps(*input.add(row + iw_base - 1));
                        let v1 = _mm256_set1_ps(*input.add(row + iw_base));
                        let v2 = if iw_base + 1 < w_in {
                            _mm256_set1_ps(*input.add(row + iw_base + 1))
                        } else {
                            _mm256_setzero_ps()
                        };

                        let w0 = _mm256_loadu_ps(wb);                                // wt[kh=0,kw=0]
                        let w1 = _mm256_loadu_ps(wb.add(kw_stride));                 // wt[kh=0,kw=1]
                        let w2 = _mm256_loadu_ps(wb.add(2 * kw_stride));             // wt[kh=0,kw=2]
                        acc = _mm256_fmadd_ps(w0, v0, acc);
                        acc = _mm256_fmadd_ps(w1, v1, acc);
                        acc = _mm256_fmadd_ps(w2, v2, acc);
                    }

                    // kh=1
                    if ih_base < h_in {
                        let row = ic_off + ih_base * w_in;
                        let v0 = _mm256_set1_ps(*input.add(row + iw_base - 1));
                        let v1 = _mm256_set1_ps(*input.add(row + iw_base));
                        let v2 = if iw_base + 1 < w_in {
                            _mm256_set1_ps(*input.add(row + iw_base + 1))
                        } else {
                            _mm256_setzero_ps()
                        };

                        let w3 = _mm256_loadu_ps(wb.add(kh_stride));                             // wt[kh=1,kw=0]
                        let w4 = _mm256_loadu_ps(wb.add(kh_stride + kw_stride));                 // wt[kh=1,kw=1]
                        let w5 = _mm256_loadu_ps(wb.add(kh_stride + 2 * kw_stride));             // wt[kh=1,kw=2]
                        acc = _mm256_fmadd_ps(w3, v0, acc);
                        acc = _mm256_fmadd_ps(w4, v1, acc);
                        acc = _mm256_fmadd_ps(w5, v2, acc);
                    }

                    // kh=2
                    if ih_base + 1 < h_in {
                        let row = ic_off + (ih_base + 1) * w_in;
                        let v0 = _mm256_set1_ps(*input.add(row + iw_base - 1));
                        let v1 = _mm256_set1_ps(*input.add(row + iw_base));
                        let v2 = if iw_base + 1 < w_in {
                            _mm256_set1_ps(*input.add(row + iw_base + 1))
                        } else {
                            _mm256_setzero_ps()
                        };

                        let w6 = _mm256_loadu_ps(wb.add(2 * kh_stride));                             // wt[kh=2,kw=0]
                        let w7 = _mm256_loadu_ps(wb.add(2 * kh_stride + kw_stride));                 // wt[kh=2,kw=1]
                        let w8 = _mm256_loadu_ps(wb.add(2 * kh_stride + 2 * kw_stride));             // wt[kh=2,kw=2]
                        acc = _mm256_fmadd_ps(w6, v0, acc);
                        acc = _mm256_fmadd_ps(w7, v1, acc);
                        acc = _mm256_fmadd_ps(w8, v2, acc);
                    }
                }

                // Scatter-store (no fused GELU)
                let mut buf = [0.0_f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                for lane in 0..8 {
                    *output.add((oc + lane) * spatial_out + oh * w_out + ow) = buf[lane];
                }
                ow += 1;
            }
        } // oh
        oc += 8;
    } // oc groups

    // Scalar tail for remaining channels (c_out % 8 != 0)
    while oc < oc_end {
        let oc_group = oc / 8;
        let lane = oc % 8;
        let wt_base = weight.add(oc_group * wt_group_stride);
        let b = *bias.add(oc);
        for oh in 0..h_out {
            let ih_base = oh * 2;
            for ow in 0..w_out {
                let iw_base = ow * 2;
                let mut acc = b;
                for ic in 0..c_in {
                    let wb = wt_base.add(ic * ic_stride);
                    let ic_off = ic * spatial_in;
                    for kh in 0..3usize {
                        let ih = ih_base as isize + kh as isize - 1;
                        if ih < 0 || ih as usize >= h_in {
                            continue;
                        }
                        let ih = ih as usize;
                        for kw in 0..3usize {
                            let iw = iw_base as isize + kw as isize - 1;
                            if iw < 0 || iw as usize >= w_in {
                                continue;
                            }
                            let iw = iw as usize;
                            let w = *wb.add(kh * kh_stride + kw * kw_stride + lane);
                            acc += w * *input.add(ic_off + ih * w_in + iw);
                        }
                    }
                }
                *output.add(oc * spatial_out + oh * w_out + ow) = gelu_scalar(acc);
            }
        }
        oc += 1;
    }
}
