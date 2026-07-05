//! Parameter definitions with recommended defaults

use qwen_asr::kernels;

/// All tunable ASR parameters with recommended defaults
#[derive(Clone, Debug)]
pub struct AsrParams {
    /// Thread pool size (0 = auto-detect)
    pub n_threads: i32,
    /// OpenBLAS thread count
    pub blas_threads: i32,
    /// Segment length in seconds (-1 = auto)
    pub segment_sec: f32,
    /// Search window in seconds (-1 = auto)
    pub search_sec: f32,
    /// Encoder window in seconds (-1 = auto)
    pub enc_window_sec: f32,
    /// Max new tokens for streaming (-1 = auto)
    pub stream_max_new_tokens: i32,
    /// Stream chunk size in seconds (-1 = auto)
    pub stream_chunk_sec: f32,
    /// Enable two-stage refinement: streaming draft + periodic whole-audio re-transcription
    pub refine_enabled: bool,
    /// Interval (seconds) between refinement passes during streaming
    pub refine_interval_sec: f32,
    /// Skip silence in audio
    pub skip_silence: bool,
    /// Past text conditioning mode (0=off, 1=on)
    pub past_text_mode: i32,
    /// Enable profiling
    pub profile: bool,
    /// Verbosity level
    pub verbosity: i32,
}

/// Default recommended parameters (benchmark-optimized)
impl Default for AsrParams {
    fn default() -> Self {
        let n_threads = kernels::get_num_perf_cpus() as i32;
        Self {
            n_threads,
            // Benchmark matrix shows BLAS 4 slightly outperforms 8 on 6C/12T
            blas_threads: 4,
            segment_sec: -1.0,       // auto
            search_sec: -1.0,        // auto
            enc_window_sec: -1.0,    // auto
            stream_max_new_tokens: -1, // auto
            stream_chunk_sec: 4.0,   // 4s chunks: 2s 仍导致 chunk#1 空转(0 tokens)
            refine_enabled: false,  // 禁用整块修正：358s 推理太慢，用户无反馈
            refine_interval_sec: 30.0, // refine every 30s of accumulated audio
            skip_silence: false,
            past_text_mode: 1,        // on (recommended for streaming)
            profile: false,
            verbosity: 1,
        }
    }
}

impl AsrParams {
    /// Apply parameters to a QwenCtx instance
    pub fn apply_to_ctx(&self, ctx: &mut qwen_asr::context::QwenCtx) {
        if self.segment_sec >= 0.0 {
            ctx.segment_sec = self.segment_sec;
        }
        if self.search_sec >= 0.0 {
            ctx.search_sec = self.search_sec;
        }
        if self.enc_window_sec >= 0.0 {
            let window_frames = (self.enc_window_sec * 100.0 + 0.5) as usize;
            ctx.config.enc_n_window_infer = window_frames.clamp(100, 800);
        }
        if self.stream_max_new_tokens > 0 {
            ctx.stream_max_new_tokens = self.stream_max_new_tokens;
        }
        if self.stream_chunk_sec > 0.0 {
            ctx.stream_chunk_sec = self.stream_chunk_sec;
        }
        if self.past_text_mode >= 0 {
            ctx.past_text_conditioning = self.past_text_mode == 1;
        }
        // 显式设置 skip_silence：AsrParams.default() 为 false，
        // 而 QwenCtx::load 默认为 true。必须总是设置才能让 AsrParams
        // 的 false 覆盖 QwenCtx::load 的默认 true，避免实时模式丢字。
        ctx.skip_silence = self.skip_silence;

        // 实时流式出字关键：QwenCtx::load 默认 stream_unfixed_chunks=99，
        // 意味着前 99 个 chunk 内 token_cb 永远不会被调用。
        // 设为 0 让第一个 chunk 就发射 token。
        ctx.stream_unfixed_chunks = 0;
        // rollback 控制每个 chunk 末尾保留多少 token 不发射（防抖动）。
        // 默认 5 太大。设为 1：最小延迟，仅保留最后 1 个 token 防抖。
        ctx.stream_rollback = 1;
    }

    /// Apply thread settings globally
    pub fn apply_threads(&self) {
        kernels::set_threads(self.n_threads as usize);
        if self.blas_threads > 0 {
            kernels::set_blas_threads(self.blas_threads as usize);
        }
        kernels::set_verbose(self.verbosity);
        if self.profile {
            kernels::set_profile(true);
            kernels::profile_reset();
        }
    }
}
