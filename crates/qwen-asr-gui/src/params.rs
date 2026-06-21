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
            stream_chunk_sec: -1.0,  // auto
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
        if self.skip_silence {
            ctx.skip_silence = true;
        }
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
