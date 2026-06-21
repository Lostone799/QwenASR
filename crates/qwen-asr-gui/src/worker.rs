//! Background worker thread for ASR inference
//!
//! Runs model loading and transcription on a separate thread,
//! communicating via shared state.

use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;
use crate::logger;
use crate::sync_ext::safe_lock;
use std::panic;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Status of the ASR worker
#[derive(Clone, Debug)]
pub enum WorkerStatus {
    Idle,
    LoadingModel(String),
    ModelLoaded(f64), // load time in seconds
    ModelLoadFailed(String),
    LoadingAudio(String),
    Transcribing,
    Done {
        text: String,
        total_ms: f64,
        encode_ms: f64,
        decode_ms: f64,
        text_tokens: i32,
    },
    Error(String),
}

/// Profile data from the last run
#[derive(Clone, Debug, Default)]
pub struct ProfileData {
    pub lines: Vec<String>,
}

/// Shared state between GUI and worker
pub struct WorkerState {
    pub status: Mutex<WorkerStatus>,
    pub profile: Mutex<ProfileData>,
    pub partial_text: Mutex<String>, // streaming text
}

impl WorkerState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(WorkerStatus::Idle),
            profile: Mutex::new(ProfileData::default()),
            partial_text: Mutex::new(String::new()),
        })
    }

    pub fn get_status(&self) -> WorkerStatus {
        // If the status mutex is poisoned (some other thread panicked while
        // holding it), recover transparently and report Idle so the GUI can
        // recover its UI rather than crashing on every frame.
        match self.status.lock() {
            Ok(g) => g.clone(),
            Err(p) => {
                logger::log_warn("WorkerState::status was poisoned; recovering");
                p.into_inner().clone()
            }
        }
    }

    pub fn set_status(&self, s: WorkerStatus) {
        // Best-effort write. If the mutex is poisoned we clear it and try
        // again, so a single panic upstream doesn't permanently freeze the
        // status indicator.
        match self.status.lock() {
            Ok(mut g) => *g = s,
            Err(p) => {
                logger::log_warn("WorkerState::status was poisoned on set; clearing");
                *p.into_inner() = s;
            }
        }
    }

    pub fn get_partial_text(&self) -> String {
        safe_lock(&self.partial_text).clone()
    }

    pub fn append_text(&self, token: &str) {
        safe_lock(&self.partial_text).push_str(token);
    }

    pub fn get_profile(&self) -> ProfileData {
        safe_lock(&self.profile).clone()
    }
}

/// The ASR worker handle
pub struct AsrWorker {
    pub state: Arc<WorkerState>,
    pub ctx: Arc<Mutex<Option<QwenCtx>>>,
    handle: Option<JoinHandle<()>>,
}

impl AsrWorker {
    pub fn new() -> Self {
        Self {
            state: WorkerState::new(),
            ctx: Arc::new(Mutex::new(None)),
            handle: None,
        }
    }

    /// Load model in background
    pub fn load_model(&mut self, model_dir: String, params: crate::params::AsrParams) {
        // P2 fix: actually join the previous task before dropping it. The old
        // code did `self.handle.take();` which silently abandoned the prior
        // worker thread. If that thread was mid-inference it could continue
        // to write to `self.state` and `self.ctx` after the new task started,
        // producing interleaved / corrupted status updates.
        self.cancel_in_flight();

        let state = self.state.clone();
        let ctx_arc = self.ctx.clone();

        self.handle = Some(thread::spawn(move || {
            // P1 fix: catch_unwind so a panic anywhere in the worker body
            // turns into a logged error + clean status update, not a poisoned
            // mutex that crashes the next frame.
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                state.set_status(WorkerStatus::LoadingModel(model_dir.clone()));
                logger::log_info(&format!("模型加载开始: {}", model_dir));

                // Apply thread settings
                params.apply_threads();

                let t0 = std::time::Instant::now();
                match QwenCtx::load(&model_dir) {
                    Some(mut ctx) => {
                        params.apply_to_ctx(&mut ctx);

                        // Set up token callback for streaming
                        let state_cb = state.clone();
                        ctx.token_cb = Some(Box::new(move |token: &str| {
                            state_cb.append_text(token);
                        }));

                        let load_sec = t0.elapsed().as_secs_f64();
                        match ctx_arc.lock() {
                            Ok(mut g) => *g = Some(ctx),
                            Err(p) => {
                                logger::log_warn("ctx mutex poisoned on load; clearing");
                                *p.into_inner() = Some(ctx);
                            }
                        }
                        state.set_status(WorkerStatus::ModelLoaded(load_sec));
                        logger::log_info(&format!("模型加载完成 ({:.1}s)", load_sec));
                    }
                    None => {
                        let err = format!("无法加载模型: {}", model_dir);
                        state.set_status(WorkerStatus::ModelLoadFailed(err.clone()));
                        logger::log_error(&format!("模型加载失败: {}", err));
                    }
                }
            }));

            if let Err(e) = result {
                // Convert the panic payload into a string for the log.
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                logger::log_error(&format!("模型加载线程 panic: {}", msg));
                state.set_status(WorkerStatus::ModelLoadFailed(format!("内部错误: {}", msg)));
            }
        }));
    }

    /// 从原始 f32 采样数据识别（麦克风模式）
    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: crate::params::AsrParams,
    ) {
        self.cancel_in_flight();

        let state = self.state.clone();
        let ctx_arc = self.ctx.clone();

        *safe_lock(&state.partial_text) = String::new();

        self.handle = Some(thread::spawn(move || {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                // Get context
                let mut ctx_guard = match ctx_arc.lock() {
                    Ok(g) => g,
                    Err(p) => {
                        logger::log_warn("ctx mutex poisoned on transcribe_samples; recovering");
                        p.into_inner()
                    }
                };
                let ctx = match ctx_guard.as_mut() {
                    Some(c) => c,
                    None => {
                        state.set_status(WorkerStatus::Error("模型未加载".into()));
                        logger::log_error("麦克风识别失败: 模型未加载");
                        return;
                    }
                };

                ctx.reset_perf();
                *safe_lock(&state.partial_text) = String::new();

                state.set_status(WorkerStatus::Transcribing);
                logger::log_info(&format!(
                    "麦克风识别开始 ({} 样本, {:.1}s)",
                    samples.len(),
                    samples.len() as f32 / 16000.0
                ));

                let result = transcribe::transcribe_audio(ctx, &samples);

                if params.profile {
                    let report = qwen_asr::kernels::profile_report_string();
                    let lines: Vec<String> = report.lines().map(|l| l.to_string()).collect();
                    *safe_lock(&state.profile) = ProfileData { lines };
                }

                match result {
                    Some(text) => {
                        let total_ms = ctx.perf_total_ms;
                        let text_tokens = ctx.perf_text_tokens;
                        state.set_status(WorkerStatus::Done {
                            text: text.clone(),
                            total_ms,
                            encode_ms: ctx.perf_encode_ms,
                            decode_ms: ctx.perf_decode_ms,
                            text_tokens,
                        });
                        logger::log_info(&format!(
                            "识别完成: \"{}\" ({:.0}ms, {} tokens)",
                            text, total_ms, text_tokens
                        ));
                    }
                    None => {
                        state.set_status(WorkerStatus::Error("识别失败".into()));
                        logger::log_error("麦克风识别失败: transcribe_audio 返回 None");
                    }
                }
            }));

            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                logger::log_error(&format!("麦克风识别线程 panic: {}", msg));
                state.set_status(WorkerStatus::Error(format!("内部错误: {}", msg)));
            }
        }));
    }

    /// Transcribe audio file in background
    pub fn transcribe(&mut self, audio_path: String, params: crate::params::AsrParams) {
        self.cancel_in_flight();

        let state = self.state.clone();
        let ctx_arc = self.ctx.clone();

        // Clear previous results
        *safe_lock(&state.partial_text) = String::new();

        self.handle = Some(thread::spawn(move || {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                state.set_status(WorkerStatus::LoadingAudio(audio_path.clone()));
                logger::log_info(&format!("文件识别开始: {}", audio_path));

                // Load audio
                let samples = qwen_asr::audio::load_wav(&audio_path);
                let samples = match samples {
                    Some(s) => s,
                    None => {
                        let err = format!("无法加载音频: {}", audio_path);
                        state.set_status(WorkerStatus::Error(err.clone()));
                        logger::log_error(&format!("音频加载失败: {}", err));
                        return;
                    }
                };

                // Get context
                let mut ctx_guard = match ctx_arc.lock() {
                    Ok(g) => g,
                    Err(p) => {
                        logger::log_warn("ctx mutex poisoned on transcribe; recovering");
                        p.into_inner()
                    }
                };
                let ctx = match ctx_guard.as_mut() {
                    Some(c) => c,
                    None => {
                        state.set_status(WorkerStatus::Error("模型未加载".into()));
                        logger::log_error("文件识别失败: 模型未加载");
                        return;
                    }
                };

                // Reset perf
                ctx.reset_perf();
                *safe_lock(&state.partial_text) = String::new();

                state.set_status(WorkerStatus::Transcribing);

                // Run transcription
                let result = transcribe::transcribe_audio(ctx, &samples);

                // Collect profile data
                if params.profile {
                    let report = qwen_asr::kernels::profile_report_string();
                    let lines: Vec<String> = report.lines().map(|l| l.to_string()).collect();
                    *safe_lock(&state.profile) = ProfileData { lines };
                }

                match result {
                    Some(text) => {
                        let total_ms = ctx.perf_total_ms;
                        let text_tokens = ctx.perf_text_tokens;
                        state.set_status(WorkerStatus::Done {
                            text: text.clone(),
                            total_ms,
                            encode_ms: ctx.perf_encode_ms,
                            decode_ms: ctx.perf_decode_ms,
                            text_tokens,
                        });
                        logger::log_info(&format!(
                            "识别完成: \"{}\" ({:.0}ms, {} tokens)",
                            text, total_ms, text_tokens
                        ));
                    }
                    None => {
                        state.set_status(WorkerStatus::Error("识别失败".into()));
                        logger::log_error("文件识别失败: transcribe_audio 返回 None");
                    }
                }
            }));

            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                logger::log_error(&format!("文件识别线程 panic: {}", msg));
                state.set_status(WorkerStatus::Error(format!("内部错误: {}", msg)));
            }
        }));
    }

    /// Cancel the in-flight task by joining the worker thread. This is the
    /// proper fix for P2: simply `take()`-ing the JoinHandle abandons the
    /// thread, leaving it free to race with the new task and write to shared
    /// state at unpredictable times.
    ///
    /// We do not have a hard cancel signal in the worker body today, so
    /// joining is best-effort bounded by whatever the worker happens to be
    /// doing. In practice the worker is in one of:
    ///   - QwenCtx::load (CPU-bound, ~1-2s on a 1.7B model)
    ///   - transcribe::transcribe_audio (a few seconds for a 30s clip)
    ///   - waiting for samples to arrive (mic mode — but we still join to
    ///     ensure it doesn't keep filling the buffer)
    ///
    /// and joining just blocks the caller for that long. The GUI button
    /// stays clickable via the render thread.
    fn cancel_in_flight(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Best-effort join. The worker's pending operations are not
            // interruptible in the current design, so we accept the wait.
            if let Err(e) = handle.join() {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                logger::log_warn(&format!("先前工作线程退出异常: {}", msg));
            }
        }
    }

    /// Check if worker is busy
    pub fn is_busy(&self) -> bool {
        matches!(
            self.state.get_status(),
            WorkerStatus::LoadingModel(_)
                | WorkerStatus::LoadingAudio(_)
                | WorkerStatus::Transcribing
        )
    }

    /// Get current status
    pub fn get_status(&self) -> WorkerStatus {
        self.state.get_status()
    }

    /// Get partial text (streaming)
    pub fn get_partial_text(&self) -> String {
        self.state.get_partial_text()
    }

    /// Get profile data
    pub fn get_profile(&self) -> ProfileData {
        self.state.get_profile()
    }
}
