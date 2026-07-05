//! Audio encoder: Conv2D stem + windowed transformer + projection cascade.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

pub struct EncLayer {
    pub wq_weight: Vec<f32>,
    pub wq_bias: Vec<f32>,
    pub wk_weight: Vec<f32>,
    pub wk_bias: Vec<f32>,
    pub wv_weight: Vec<f32>,
    pub wv_bias: Vec<f32>,
    pub wo_weight: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
    /// INT8 quantized weights + per-row scales + per-row w_sums
    pub wq_int8: Vec<i8>,
    pub wq_int8_scales: Vec<f32>,
    pub wq_int8_sums: Vec<i32>,
    pub wk_int8: Vec<i8>,
    pub wk_int8_scales: Vec<f32>,
    pub wk_int8_sums: Vec<i32>,
    pub wv_int8: Vec<i8>,
    pub wv_int8_scales: Vec<f32>,
    pub wv_int8_sums: Vec<i32>,
    pub wo_int8: Vec<i8>,
    pub wo_int8_scales: Vec<f32>,
    pub wo_int8_sums: Vec<i32>,
    pub fc1_int8: Vec<i8>,
    pub fc1_int8_scales: Vec<f32>,
    pub fc1_int8_sums: Vec<i32>,
    pub fc2_int8: Vec<i8>,
    pub fc2_int8_scales: Vec<f32>,
    pub fc2_int8_sums: Vec<i32>,
}

pub struct EncoderBuffers {
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub chunk_mel: Vec<f32>,
    pub c1: Vec<f32>,
    pub c2: Vec<f32>,
    pub c3: Vec<f32>,
    pub reshaped: Vec<f32>,
    pub pe: Vec<f32>,
    pub conv_cols: Vec<f32>,
    pub window_starts: Vec<i32>,
    pub cap_tokens: usize,
}

impl Default for EncoderBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderBuffers {
    pub fn new() -> Self {
        EncoderBuffers {
            x: Vec::new(),
            x_norm: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            proj_out: Vec::new(),
            ffn_mid: Vec::new(),
            ffn_out: Vec::new(),
            chunk_mel: Vec::new(),
            c1: Vec::new(),
            c2: Vec::new(),
            c3: Vec::new(),
            reshaped: Vec::new(),
            pe: Vec::new(),
            conv_cols: Vec::new(),
            window_starts: Vec::new(),
            cap_tokens: 0,
        }
    }

    pub fn ensure(&mut self, total_tokens: usize, d_model: usize, ffn_dim: usize) {
        if total_tokens <= self.cap_tokens {
            return;
        }
        let mut new_cap = if self.cap_tokens > 0 {
            self.cap_tokens
        } else {
            256
        };
        while new_cap < total_tokens {
            new_cap *= 2;
        }
        self.x.resize(new_cap * d_model, 0.0);
        self.x_norm.resize(new_cap * d_model, 0.0);
        self.q.resize(new_cap * d_model, 0.0);
        self.k.resize(new_cap * d_model, 0.0);
        self.v.resize(new_cap * d_model, 0.0);
        self.attn_out.resize(new_cap * d_model, 0.0);
        self.proj_out.resize(new_cap * d_model, 0.0);
        self.ffn_mid.resize(new_cap * ffn_dim, 0.0);
        self.ffn_out.resize(new_cap * d_model, 0.0);
        self.cap_tokens = new_cap;
    }

    pub fn ensure_stem(&mut self, chunk_w: usize, d_model: usize) {
        let h1 = (128 + 2 - 3) / 2 + 1;
        let w1 = (chunk_w + 2 - 3) / 2 + 1;
        let h2 = (h1 + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let h3 = (h2 + 2 - 3) / 2 + 1;
        let w3 = (w2 + 2 - 3) / 2 + 1;
        let conv_proj_dim = CONV_HIDDEN * h3;

        self.chunk_mel.resize(128 * chunk_w, 0.0);
        self.c1.resize(CONV_HIDDEN * h1 * w1, 0.0);
        self.c2.resize(CONV_HIDDEN * h2 * w2, 0.0);
        self.c3.resize(CONV_HIDDEN * h3 * w3, 0.0);
        self.reshaped.resize(w3 * conv_proj_dim, 0.0);
        self.pe.resize(w3 * d_model, 0.0);

        // Pre-size conv_cols for the largest im2col buffer used by the stem.
        // Layer 2/3 use CONV_HIDDEN input channels and dominate the size.
        let max_patch = CONV_HIDDEN * 3 * 3;
        let spatial_out_l2 = h2 * w2;
        let required_cols = max_patch * spatial_out_l2;
        if self.conv_cols.len() < required_cols {
            self.conv_cols.resize(required_cols, 0.0f32);
        }
    }
}

pub struct Encoder {
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv3_weight: Vec<f32>,
    pub conv3_bias: Vec<f32>,
    pub conv_out_weight: Vec<f32>,
    pub layers: Vec<EncLayer>,
    pub ln_post_weight: Vec<f32>,
    pub ln_post_bias: Vec<f32>,
    pub proj1_weight: Vec<f32>,
    pub proj1_bias: Vec<f32>,
    pub proj2_weight: Vec<f32>,
    pub proj2_bias: Vec<f32>,
    /// INT8 quantized projection weights
    pub conv_out_int8: Vec<i8>,
    pub conv_out_int8_scales: Vec<f32>,
    pub conv_out_int8_sums: Vec<i32>,
    pub proj1_int8: Vec<i8>,
    pub proj1_int8_scales: Vec<f32>,
    pub proj1_int8_sums: Vec<i32>,
    pub proj2_int8: Vec<i8>,
    pub proj2_int8_scales: Vec<f32>,
    pub proj2_int8_sums: Vec<i32>,
    /// INT8 quantized conv2d weights (conv2/conv3 only; conv1 keeps F32 due to small patch_size=9)
    pub conv2_int8: Vec<i8>,
    pub conv2_int8_scales: Vec<f32>,
    pub conv2_int8_sums: Vec<i32>,
    pub conv3_int8: Vec<i8>,
    pub conv3_int8_scales: Vec<f32>,
    pub conv3_int8_sums: Vec<i32>,
}

const ENC_PREFIX: &str = "thinker.audio_tower.";

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("encoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_as_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let (si, t) = ms.find(name).or_else(|| {
        eprintln!("encoder: weight not found: {}", name);
        None
    })?;

    let bf16_ptr = ms.shards[si].get_bf16_direct(t)?;
    let n = t.numel();
    let mut f32_data = vec![0.0f32; n];
    let src = unsafe { std::slice::from_raw_parts(bf16_ptr, n) };
    kernels::bf16_to_f32_buf(&mut f32_data, src);
    Some(f32_data)
}

/// Load one encoder transformer layer (bf16->f32 weight conversion). Independent
/// per layer, so callable in parallel. Also quantizes weights to INT8.
fn load_enc_layer(ms: &MultiSafetensors, i: usize, d_model: usize, ffn_dim: usize) -> Option<EncLayer> {
    let lp = format!("{}layers.{}", ENC_PREFIX, i);
    let wq_weight = load_bf16_as_f32(ms, &format!("{}.self_attn.q_proj.weight", lp))?;
    let wq_bias = load_f32(ms, &format!("{}.self_attn.q_proj.bias", lp))?;
    let wk_weight = load_bf16_as_f32(ms, &format!("{}.self_attn.k_proj.weight", lp))?;
    let wk_bias = load_f32(ms, &format!("{}.self_attn.k_proj.bias", lp))?;
    let wv_weight = load_bf16_as_f32(ms, &format!("{}.self_attn.v_proj.weight", lp))?;
    let wv_bias = load_f32(ms, &format!("{}.self_attn.v_proj.bias", lp))?;
    let wo_weight = load_bf16_as_f32(ms, &format!("{}.self_attn.out_proj.weight", lp))?;
    let wo_bias = load_f32(ms, &format!("{}.self_attn.out_proj.bias", lp))?;
    let attn_norm_weight = load_f32(ms, &format!("{}.self_attn_layer_norm.weight", lp))?;
    let attn_norm_bias = load_f32(ms, &format!("{}.self_attn_layer_norm.bias", lp))?;
    let fc1_weight = load_bf16_as_f32(ms, &format!("{}.fc1.weight", lp))?;
    let fc1_bias = load_f32(ms, &format!("{}.fc1.bias", lp))?;
    let fc2_weight = load_bf16_as_f32(ms, &format!("{}.fc2.weight", lp))?;
    let fc2_bias = load_f32(ms, &format!("{}.fc2.bias", lp))?;
    let ffn_norm_weight = load_f32(ms, &format!("{}.final_layer_norm.weight", lp))?;
    let ffn_norm_bias = load_f32(ms, &format!("{}.final_layer_norm.bias", lp))?;

    // INT8 quantize all weights (per-row symmetric quantization)
    let (wq_int8, wq_int8_scales, wq_int8_sums) = kernels::quantize_f32_weights_to_int8(&wq_weight, d_model, d_model);
    let (wk_int8, wk_int8_scales, wk_int8_sums) = kernels::quantize_f32_weights_to_int8(&wk_weight, d_model, d_model);
    let (wv_int8, wv_int8_scales, wv_int8_sums) = kernels::quantize_f32_weights_to_int8(&wv_weight, d_model, d_model);
    let (wo_int8, wo_int8_scales, wo_int8_sums) = kernels::quantize_f32_weights_to_int8(&wo_weight, d_model, d_model);
    let (fc1_int8, fc1_int8_scales, fc1_int8_sums) = kernels::quantize_f32_weights_to_int8(&fc1_weight, ffn_dim, d_model);
    let (fc2_int8, fc2_int8_scales, fc2_int8_sums) = kernels::quantize_f32_weights_to_int8(&fc2_weight, d_model, ffn_dim);

    Some(EncLayer {
        wq_weight,
        wq_bias,
        wk_weight,
        wk_bias,
        wv_weight,
        wv_bias,
        wo_weight,
        wo_bias,
        attn_norm_weight,
        attn_norm_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        ffn_norm_weight,
        ffn_norm_bias,
        wq_int8, wq_int8_scales, wq_int8_sums,
        wk_int8, wk_int8_scales, wk_int8_sums,
        wv_int8, wv_int8_scales, wv_int8_sums,
        wo_int8, wo_int8_scales, wo_int8_sums,
        fc1_int8, fc1_int8_scales, fc1_int8_sums,
        fc2_int8, fc2_int8_scales, fc2_int8_sums,
    })
}

impl Encoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let p = ENC_PREFIX;

        let conv1_weight = load_f32(ms, &format!("{}conv2d1.weight", p))?;
        let conv1_bias = load_f32(ms, &format!("{}conv2d1.bias", p))?;
        let conv2_weight = load_f32(ms, &format!("{}conv2d2.weight", p))?;
        let conv2_bias = load_f32(ms, &format!("{}conv2d2.bias", p))?;
        let conv3_weight = load_f32(ms, &format!("{}conv2d3.weight", p))?;
        let conv3_bias = load_f32(ms, &format!("{}conv2d3.bias", p))?;
        let conv_out_weight = load_bf16_as_f32(ms, &format!("{}conv_out.weight", p))?;

        // Per-layer weights are independent and conversion-heavy (bf16->f32),
        // so load encoder layers in parallel.
        let nlayers = cfg.enc_layers;
        let nthreads = kernels::get_num_cpus().min(nlayers).max(1);
        let chunk = nlayers.div_ceil(nthreads);
        let mut indexed: Vec<(usize, EncLayer)> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(nlayers);
                if start >= end {
                    break;
                }
                handles.push(s.spawn(move || {
                    let mut out = Vec::with_capacity(end - start);
                    for i in start..end {
                        out.push((i, load_enc_layer(ms, i, cfg.enc_d_model, cfg.enc_ffn_dim)?));
                    }
                    Some(out)
                }));
            }
            let mut all: Vec<(usize, EncLayer)> = Vec::with_capacity(nlayers);
            for h in handles {
                all.extend(h.join().ok()??);
            }
            Some(all)
        })?;
        indexed.sort_by_key(|(i, _)| *i);
        let layers: Vec<EncLayer> = indexed.into_iter().map(|(_, l)| l).collect();

        let ln_post_weight = load_f32(ms, &format!("{}ln_post.weight", p))?;
        let ln_post_bias = load_f32(ms, &format!("{}ln_post.bias", p))?;
        let proj1_weight = load_bf16_as_f32(ms, &format!("{}proj1.weight", p))?;
        let proj1_bias = load_f32(ms, &format!("{}proj1.bias", p))?;
        let proj2_weight = load_bf16_as_f32(ms, &format!("{}proj2.weight", p))?;
        let proj2_bias = load_f32(ms, &format!("{}proj2.bias", p))?;

        // INT8 quantize projection weights
        let d_model = cfg.enc_d_model;
        let conv_proj_dim = conv_out_weight.len() / d_model;
        let (conv_out_int8, conv_out_int8_scales, conv_out_int8_sums) =
            kernels::quantize_f32_weights_to_int8(&conv_out_weight, d_model, conv_proj_dim);
        let (proj1_int8, proj1_int8_scales, proj1_int8_sums) =
            kernels::quantize_f32_weights_to_int8(&proj1_weight, d_model, d_model);
        let proj2_out_dim = proj2_weight.len() / d_model;
        let (proj2_int8, proj2_int8_scales, proj2_int8_sums) =
            kernels::quantize_f32_weights_to_int8(&proj2_weight, proj2_out_dim, d_model);

        // INT8 quantize conv2/conv3 weights (conv2/conv3 have patch_size=4320, benefit from INT8)
        // Weight layout: [c_out, c_in, kh, kw] -> treat as [c_out, c_in*kh*kw] for per-row quantization
        let conv2_patch = conv2_weight.len() / CONV_HIDDEN; // c_out = CONV_HIDDEN
        let (conv2_int8, conv2_int8_scales, conv2_int8_sums) =
            kernels::quantize_f32_weights_to_int8(&conv2_weight, CONV_HIDDEN, conv2_patch);
        let conv3_patch = conv3_weight.len() / CONV_HIDDEN;
        let (conv3_int8, conv3_int8_scales, conv3_int8_sums) =
            kernels::quantize_f32_weights_to_int8(&conv3_weight, CONV_HIDDEN, conv3_patch);

        Some(Encoder {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            conv3_weight,
            conv3_bias,
            conv_out_weight,
            layers,
            ln_post_weight,
            ln_post_bias,
            proj1_weight,
            proj1_bias,
            proj2_weight,
            proj2_bias,
            conv_out_int8, conv_out_int8_scales, conv_out_int8_sums,
            proj1_int8, proj1_int8_scales, proj1_int8_sums,
            proj2_int8, proj2_int8_scales, proj2_int8_sums,
            conv2_int8, conv2_int8_scales, conv2_int8_sums,
            conv3_int8, conv3_int8_scales, conv3_int8_sums,
        })
    }

    /// Run encoder forward pass on mel spectrogram.
    /// mel: [128, mel_frames], returns [total_tokens, output_dim].
    pub fn forward(
        &self,
        cfg: &QwenConfig,
        mel: &[f32],
        mel_frames: usize,
        enc_bufs: Option<&mut EncoderBuffers>,
    ) -> Option<(Vec<f32>, usize)> {
        let _pg_total = kernels::ProfileGuard::new(&kernels::PROF.enc_forward_total);
        let d_model = cfg.enc_d_model;
        let n_heads = cfg.enc_heads;
        let head_dim = cfg.enc_head_dim;
        let ffn_dim = cfg.enc_ffn_dim;
        let output_dim = cfg.enc_output_dim;
        let chunk_size = cfg.enc_chunk_size;
        let n_window_infer = cfg.enc_n_window_infer;

        // Determine tokens per full chunk
        let tokens_per_chunk = {
            let w = chunk_size;
            let w1 = (w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            (w2 + 2 - 3) / 2 + 1
        };

        let n_chunks = mel_frames.div_ceil(chunk_size);

        // Pre-calculate total tokens
        let mut total_tokens = 0;
        let mut chunk_sizes = Vec::new();
        for c in 0..n_chunks {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(mel_frames);
            let chunk_w = end - start;
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let w3 = (w2 + 2 - 3) / 2 + 1;
            total_tokens += w3;
            chunk_sizes.push((start, end, w3));
        }

        // Transformer + stem scratch buffers (reusable or fresh)
        let mut _owned_bufs;
        let bufs: &mut EncoderBuffers = match enc_bufs {
            Some(b) => {
                b.ensure(total_tokens, d_model, ffn_dim);
                b
            }
            None => {
                _owned_bufs = EncoderBuffers::new();
                _owned_bufs.ensure(total_tokens, d_model, ffn_dim);
                &mut _owned_bufs
            }
        };

        // Main sequence buffer: [total_tokens, d_model]
        let td = total_tokens * d_model;
        let mut token_offset = 0;

        // Process each chunk through Conv2D + reshape + project + sinusoidal PE
        for &(start, end, w3) in &chunk_sizes {
            let chunk_w = end - start;
            bufs.ensure_stem(chunk_w, d_model);

            // Extract chunk mel: [128, chunk_w]
            let chunk_mel = &mut bufs.chunk_mel[..128 * chunk_w];
            for m in 0..128 {
                chunk_mel[m * chunk_w..(m + 1) * chunk_w]
                    .copy_from_slice(&mel[m * mel_frames + start..m * mel_frames + end]);
            }

            // Conv2D layer 1: [1, 128, chunk_w] -> [480, h1, w1]
            let h1 = (128 + 2 - 3) / 2 + 1; // 64
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let c1 = &mut bufs.c1[..CONV_HIDDEN * h1 * w1];
            // P1: try hand-written 3×3 stride=2 AVX2 kernel (no fused GELU)
            let ok = kernels::conv2d_3x3_s2_p1_parallel(
                c1, chunk_mel, &self.conv1_weight, &self.conv1_bias,
                1, CONV_HIDDEN, 128, chunk_w,
            );
            if !ok {
                // P0 fallback: im2col + sgemm + bias-add + GELU
                kernels::conv2d_with_cols(
                    c1, chunk_mel, &self.conv1_weight, Some(&self.conv1_bias),
                    &mut bufs.conv_cols, 1, CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1,
                );
                kernels::gelu(c1, CONV_HIDDEN * h1 * w1);
            } else {
                // P1 path: apply SIMD GELU separately
                kernels::gelu(c1, CONV_HIDDEN * h1 * w1);
            }

            // Conv2D layer 2: [480, h1, w1] -> [480, h2, w2]
            let h2 = (h1 + 2 - 3) / 2 + 1; // 32
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let c2 = &mut bufs.c2[..CONV_HIDDEN * h2 * w2];
            let ok = kernels::conv2d_3x3_s2_p1_parallel(
                c2, c1, &self.conv2_weight, &self.conv2_bias,
                CONV_HIDDEN, CONV_HIDDEN, h1, w1,
            );
            if !ok {
                kernels::conv2d_with_cols(
                    c2, c1, &self.conv2_weight, Some(&self.conv2_bias),
                    &mut bufs.conv_cols, CONV_HIDDEN, CONV_HIDDEN, h1, w1, 3, 3, 2, 1,
                );
                kernels::gelu(c2, CONV_HIDDEN * h2 * w2);
            } else {
                kernels::gelu(c2, CONV_HIDDEN * h2 * w2);
            }

            // Conv2D layer 3: [480, h2, w2] -> [480, h3, w3]
            let h3 = (h2 + 2 - 3) / 2 + 1; // 16
            let _w3_calc = (w2 + 2 - 3) / 2 + 1;
            debug_assert_eq!(_w3_calc, w3);
            let c3 = &mut bufs.c3[..CONV_HIDDEN * h3 * w3];
            let ok = kernels::conv2d_3x3_s2_p1_parallel(
                c3, c2, &self.conv3_weight, &self.conv3_bias,
                CONV_HIDDEN, CONV_HIDDEN, h2, w2,
            );
            if !ok {
                kernels::conv2d_with_cols(
                    c3, c2, &self.conv3_weight, Some(&self.conv3_bias),
                    &mut bufs.conv_cols, CONV_HIDDEN, CONV_HIDDEN, h2, w2, 3, 3, 2, 1,
                );
                kernels::gelu(c3, CONV_HIDDEN * h3 * w3);
            } else {
                kernels::gelu(c3, CONV_HIDDEN * h3 * w3);
            }

            // Reshape [480, h3, w3] -> [w3, 480*h3]
            // Loop order: ch → f → t for sequential reads from c3
            let conv_proj_dim = CONV_HIDDEN * h3;
            let reshaped = &mut bufs.reshaped[..w3 * conv_proj_dim];
            for ch in 0..CONV_HIDDEN {
                for f in 0..h3 {
                    let src_off = ch * h3 * w3 + f * w3;
                    let dst_col = ch * h3 + f;
                    for t in 0..w3 {
                        reshaped[t * conv_proj_dim + dst_col] = c3[src_off + t];
                    }
                }
            }

            // Project: [w3, 7680] -> [w3, d_model]
            let projected = &mut bufs.x[token_offset * d_model..(token_offset + w3) * d_model];
            kernels::linear_nobias(projected, reshaped, &self.conv_out_weight, w3, conv_proj_dim, d_model);

            // Add per-chunk sinusoidal PE
            let pe = &mut bufs.pe[..w3 * d_model];
            kernels::sinusoidal_pe(pe, w3, d_model);
            kernels::add_inplace(projected, pe, w3 * d_model);

            token_offset += w3;
        }

        // Build attention window boundaries
        let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
        let n_windows = total_tokens.div_ceil(window_token_size);
        bufs.window_starts.resize(n_windows + 1, 0);
        let window_starts = &mut bufs.window_starts[..n_windows + 1];
        for (w, ws) in window_starts.iter_mut().enumerate().take(n_windows) {
            *ws = (w * window_token_size) as i32;
        }
        window_starts[n_windows] = total_tokens as i32;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let tf = total_tokens * ffn_dim;

        for layer in &self.layers {
            // Self-attention
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.attn_norm_weight,
                &layer.attn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            kernels::linear_int8_qkv_prefill_fused(
                &mut bufs.q[..td], &mut bufs.k[..td], &mut bufs.v[..td],
                &bufs.x_norm[..td],
                &layer.wq_int8, &layer.wq_int8_scales, &layer.wq_int8_sums, &layer.wq_bias,
                &layer.wk_int8, &layer.wk_int8_scales, &layer.wk_int8_sums, &layer.wk_bias,
                &layer.wv_int8, &layer.wv_int8_scales, &layer.wv_int8_sums, &layer.wv_bias,
                total_tokens, d_model, d_model,
            );

            kernels::bidirectional_attention(
                &mut bufs.attn_out[..td],
                &bufs.q[..td],
                &bufs.k[..td],
                &bufs.v[..td],
                total_tokens,
                n_heads,
                head_dim,
                scale,
                &window_starts,
                n_windows,
            );

            // Fused: x += wo_bias + attn_out @ wo_weight.T
            kernels::linear_accumulate_int8(&mut bufs.x[..td], &bufs.attn_out[..td], &layer.wo_int8, &layer.wo_int8_scales, &layer.wo_int8_sums, Some(&layer.wo_bias), total_tokens, d_model, d_model, &mut bufs.ffn_out[..td]);

            // FFN
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.ffn_norm_weight,
                &layer.ffn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            kernels::linear_int8(&mut bufs.ffn_mid[..tf], &bufs.x_norm[..td], &layer.fc1_int8, &layer.fc1_int8_scales, &layer.fc1_int8_sums, Some(&layer.fc1_bias), total_tokens, d_model, ffn_dim);
            kernels::gelu(&mut bufs.ffn_mid[..tf], tf);
            // Fused: x += fc2_bias + ffn_mid @ fc2_weight.T
            kernels::linear_accumulate_int8(&mut bufs.x[..td], &bufs.ffn_mid[..tf], &layer.fc2_int8, &layer.fc2_int8_scales, &layer.fc2_int8_sums, Some(&layer.fc2_bias), total_tokens, ffn_dim, d_model, &mut bufs.proj_out[..td]);
        }

        // Final LayerNorm: use x_norm as temp, then swap into x
        kernels::layer_norm(
            &mut bufs.x_norm[..td],
            &bufs.x[..td],
            &self.ln_post_weight,
            &self.ln_post_bias,
            total_tokens,
            d_model,
            1e-5,
        );
        bufs.x[..td].copy_from_slice(&bufs.x_norm[..td]);

        // Projection: proj1 (GELU) -> proj2 (reuse scratch buffers)
        kernels::linear_int8(&mut bufs.q[..td], &bufs.x[..td], &self.proj1_int8, &self.proj1_int8_scales, &self.proj1_int8_sums, Some(&self.proj1_bias), total_tokens, d_model, d_model);
        kernels::gelu(&mut bufs.q[..td], td);

        let mut enc_output = vec![0.0f32; total_tokens * output_dim];
        kernels::linear_int8(&mut enc_output, &bufs.q[..td], &self.proj2_int8, &self.proj2_int8_scales, &self.proj2_int8_sums, Some(&self.proj2_bias), total_tokens, d_model, output_dim);

        Some((enc_output, total_tokens))
    }
}
