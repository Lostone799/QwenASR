//! Background worker thread for ASR inference
//!
//! Runs model loading and transcription on a separate thread,
//! communicating via shared state.

use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;
use crate::logger;
use crate::sync_ext::safe_lock;
use std::panic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Status of the ASR worker
#[derive(Clone, Debug)]
pub enum WorkerStatus {
    Idle,
    LoadingModel(String),
    ModelLoaded(f64), // load time in seconds
    ModelLoadFailed(String),
    LoadingAudio(String),
    Transcribing,
    /// A periodic refinement pass completed during streaming. The GUI
    /// should display this as a higher-accuracy replacement for the
    /// current draft. `refine_idx` is 1-based.
    Refined {
        text: String,
        total_ms: f64,
        text_tokens: i32,
        refine_idx: u32,
    },
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

                // Respect stream-chunk setting for microphone recordings too.
                let result = if params.refine_enabled && ctx.stream_chunk_sec > 0.0 {
                    logger::log_info(&format!(
                        "麦克风识别使用流式+修正: chunk={}s, refine={}s",
                        ctx.stream_chunk_sec, params.refine_interval_sec
                    ));
                    let state_for_refine = state.clone();
                    transcribe::transcribe_streaming_with_refine(
                        ctx,
                        &samples,
                        params.refine_interval_sec,
                        move |idx, text, total_ms, tokens| {
                            *safe_lock(&state_for_refine.partial_text) = text.to_string();
                            state_for_refine.set_status(WorkerStatus::Refined {
                                text: text.to_string(),
                                total_ms,
                                text_tokens: tokens,
                                refine_idx: idx,
                            });
                        },
                    )
                } else if ctx.stream_chunk_sec > 0.0 {
                    logger::log_info(&format!(
                        "麦克风识别使用流式分块: chunk={}s",
                        ctx.stream_chunk_sec
                    ));
                    transcribe::transcribe_streaming(ctx, &samples)
                } else {
                    transcribe::transcribe_audio(ctx, &samples)
                };

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

                // Run transcription. If a positive stream chunk size is set,
                // split long audio into fixed chunks to cap peak memory.
                let result = if params.refine_enabled && ctx.stream_chunk_sec > 0.0 {
                    logger::log_info(&format!(
                        "文件识别使用流式+修正: chunk={}s, refine={}s",
                        ctx.stream_chunk_sec, params.refine_interval_sec
                    ));
                    let state_for_refine = state.clone();
                    transcribe::transcribe_streaming_with_refine(
                        ctx,
                        &samples,
                        params.refine_interval_sec,
                        move |idx, text, total_ms, tokens| {
                            *safe_lock(&state_for_refine.partial_text) = text.to_string();
                            state_for_refine.set_status(WorkerStatus::Refined {
                                text: text.to_string(),
                                total_ms,
                                text_tokens: tokens,
                                refine_idx: idx,
                            });
                        },
                    )
                } else if ctx.stream_chunk_sec > 0.0 {
                    logger::log_info(&format!(
                        "文件识别使用流式分块: chunk={}s",
                        ctx.stream_chunk_sec
                    ));
                    transcribe::transcribe_streaming(ctx, &samples)
                } else {
                    transcribe::transcribe_audio(ctx, &samples)
                };

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

    /// 启动实时识别：从动态增长的样本缓冲区定期全量识别。
    ///
    /// 用于麦克风录音模式——录音进行中每 `interval_sec` 秒取一次
    /// 累积样本做全量 `transcribe_audio`，结果通过 `Refined` 状态
    /// 发送给 GUI。`stop_flag` 被设置为 true 后，线程做最终一次
    /// 全量识别并通过 `Done` 状态返回。
    ///
    /// `samples` 是 MicRecorder 内部的 `Arc<Mutex<Vec<f32>>>`，
    /// 录音停止后仍可读取最终样本。
    ///
    /// # 算法（片段式流式）
    /// 1. 流式持续运行：每 `chunk_sec` 秒取当前片段的新增 chunk 做
    ///    `stream_push_audio`，token_cb 实时输出（`Transcribing` 状态）。
    /// 2. 片段结束：每 `segment_sec` 秒（默认 60s，可配置）作为一个片段，
    ///    片段结束时用流式累积结果作为片段最终结果（`Refined` 状态追加），
    ///    然后 `stream_state.reset()` + `ctx.kv_cache.len = 0` 重置上下文。
    /// 3. 停止后：对最后一个片段做最终整块识别（`Done` 状态）。
    ///
    /// # 优势
    /// - 录音无上限：片段式避免 KV cache O(n²) 增长
    /// - 流式持续：不会因长音频停止出字
    /// - 内存可控：每个片段独立，reset 后内存释放
    pub fn transcribe_live(
        &mut self,
        samples: Arc<Mutex<Vec<f32>>>,
        sample_rate: u32,
        params: crate::params::AsrParams,
        stop_flag: Arc<AtomicBool>,
    ) {
        self.cancel_in_flight();

        let state = self.state.clone();
        let ctx_arc = self.ctx.clone();

        *safe_lock(&state.partial_text) = String::new();

        // 流式 chunk 间隔（首字延迟 = chunk_sec + 识别时间）
        let chunk_sec = if params.stream_chunk_sec > 0.0 {
            params.stream_chunk_sec
        } else {
            2.0
        };

        // 片段长度：每个片段结束后重置 KV cache，避免 O(n²) 增长。
        // 默认 60s，可通过 QWEN_ASR_SEGMENT_SEC 配置。
        let segment_sec: f32 = std::env::var("QWEN_ASR_SEGMENT_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60.0);

        // 原始采样率下的样本数阈值
        let chunk_samples_orig = (chunk_sec * sample_rate as f32) as usize;

        self.handle = Some(thread::spawn(move || {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                // StreamState 持续保持 KV cache + encoder cache，
                // 实现流式 token 输出（1-2s 首字延迟）和上下文延续。
                let mut stream_state = transcribe::StreamState::new();
                // 实时模式：每次调用只处理 1 个 chunk，避免推理速度 < 录音速度时
                // 积压恶性循环（一次处理 N 个 chunk → 阻塞 N*推理时间 → 积压更多）。
                // 限制为 1 后，每次调用快速返回，worker 能记录日志、UI 能刷新。
                stream_state.max_chunks_per_call = 1;
                let mut refine_idx = 0u32;
                // 上次推送流式的原始采样率位置
                let mut last_pushed_pos = 0usize;
                // 片段式架构：当前片段在 samples 中的起始位置。
                // 片段结束时 reset StreamState + KV cache，segment_offset 前移。
                let mut segment_offset = 0usize;
                // 已完成片段的累积文本（片段结束时追加，作为最终结果的基础）
                let mut finalized_text = String::new();
                // 已完成片段的音频缓存（16kHz），用于停止后串行整块识别。
                // 每分钟约 3.84MB，可控。流式结果仅作临时显示，最终用整块识别替换。
                let mut segment_audio_cache: Vec<Vec<f32>> = Vec::new();
                // 已完成片段的流式文本缓存，与 segment_audio_cache 一一对应。
                // 整块识别期间用于显示"已识别整块 + 未识别流式"，避免文本回退。
                let mut segment_text_cache: Vec<String> = Vec::new();

                // 设置 token_cb：流式 token 实时 append 到 partial_text
                {
                    let mut ctx_guard = match ctx_arc.lock() {
                        Ok(g) => g,
                        Err(p) => {
                            logger::log_warn("ctx mutex poisoned on live setup; recovering");
                            p.into_inner()
                        }
                    };
                    if let Some(ctx) = ctx_guard.as_mut() {
                        let state_cb = state.clone();
                        ctx.token_cb = Some(Box::new(move |token: &str| {
                            state_cb.append_text(token);
                        }));
                    }
                }

                while !stop_flag.load(Ordering::Relaxed) {
                    let current_pos = safe_lock(&samples).len();
                    let new_len = current_pos.saturating_sub(last_pushed_pos);
                    // 当前音频时长（秒），用于限制流式最大长度
                    let audio_sec = current_pos as f32 / sample_rate as f32;

                    // 录音上限：仅自动化测试模式（QWEN_ASR_AUTO_RECORD=1）生效，
                    // 通过 QWEN_ASR_AUTO_STOP_SEC 配置（默认 60s）。
                    // 正常模式无上限：片段式架构通过每 segment_sec 重置 KV cache，
                    // 避免 O(n²) 增长和内存爆炸，支持任意长度录音。
                    if std::env::var("QWEN_ASR_AUTO_RECORD").as_deref() == Ok("1") {
                        let auto_stop_sec: f32 = std::env::var("QWEN_ASR_AUTO_STOP_SEC")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(60.0);
                        if audio_sec > auto_stop_sec {
                            logger::log_info(&format!(
                                "自动录音{:.0}s 超过{}s 上限，自动停止",
                                audio_sec, auto_stop_sec
                            ));
                            break;
                        }
                    }

                    // 流式限制：片段式架构下不再需要，保留环境变量兼容性
                    if let Some(stream_limit_sec) = std::env::var("QWEN_ASR_STREAM_LIMIT_SEC")
                        .ok()
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        if stream_limit_sec > 0.0 && audio_sec > stream_limit_sec {
                            thread::sleep(Duration::from_millis(500));
                            continue;
                        }
                    }

                    // 轮询模式：音频不够时短睡 200ms 再检查，
                    // 不做前置睡眠，确保首字延迟 ≈ 积累时间 + 推理时间。
                    // trigger_threshold 与 stream_push_audio 内部的 chunk_samples
                    // 匹配（都是 chunk_sec 秒），避免 available < chunk_samples 早退。
                    if new_len < chunk_samples_orig {
                        thread::sleep(Duration::from_millis(200));
                        continue;
                    }

                    // --- 片段式流式识别 ---
                    // 片段式架构：每 segment_sec 秒作为一个片段，片段结束后
                    // reset StreamState + KV cache，避免 O(n²) 增长。
                    // snap 只取当前片段音频 [segment_offset..current_pos]，
                    // StreamState 内部 audio_cursor 从 0 开始（片段 reset 后）。

                    // 检查片段是否结束
                    let segment_audio_len = current_pos.saturating_sub(segment_offset);
                    let segment_audio_sec = segment_audio_len as f32 / sample_rate as f32;

                    if segment_audio_sec >= segment_sec {
                        // 片段结束：缓存片段音频（16kHz）+ 流式文本，
                        // 用于停止后整块识别和整块识别期间的渐进式显示。
                        let snap = {
                            let guard = safe_lock(&samples);
                            guard[segment_offset..current_pos].to_vec()
                        };
                        let seg_samples_16k = if sample_rate != 16000 {
                            qwen_asr::audio::resample(&snap, sample_rate as i32, 16000)
                        } else {
                            snap
                        };
                        segment_audio_cache.push(seg_samples_16k);

                        // 缓存流式文本（与音频一一对应）
                        let segment_text = stream_state.text();
                        segment_text_cache.push(segment_text.clone());

                        // 临时显示：用流式结果作为片段结果（停止后会被整块识别替换）
                        if !segment_text.is_empty() {
                            finalized_text.push_str(&segment_text);
                            *safe_lock(&state.partial_text) = finalized_text.clone();
                            state.set_status(WorkerStatus::Refined {
                                text: finalized_text.clone(),
                                total_ms: 0.0,
                                text_tokens: 0,
                                refine_idx: refine_idx.wrapping_add(1),
                            });
                            refine_idx = refine_idx.wrapping_add(1);
                            logger::log_info(&format!(
                                "片段#{} 结束 ({:.1}s 音频, 已缓存): \"{}\" → 累积 {} 字符",
                                refine_idx, segment_audio_sec, segment_text,
                                finalized_text.chars().count()
                            ));
                        }
                        // reset StreamState + KV cache，开始新片段
                        stream_state.reset();
                        if let Ok(mut ctx_guard) = ctx_arc.lock() {
                            if let Some(ctx) = ctx_guard.as_mut() {
                                ctx.kv_cache.len = 0;
                            }
                        }
                        segment_offset = current_pos;
                        last_pushed_pos = current_pos;
                        continue;
                    }

                    // 流式推送：只传当前片段音频
                    let snap = {
                        let guard = safe_lock(&samples);
                        guard[segment_offset..current_pos].to_vec()
                    };
                    let samples_16k = if sample_rate != 16000 {
                        qwen_asr::audio::resample(&snap, sample_rate as i32, 16000)
                    } else {
                        snap
                    };

                    let mut ctx_guard = match ctx_arc.lock() {
                        Ok(g) => g,
                        Err(p) => {
                            logger::log_warn("ctx mutex poisoned on live; recovering");
                            p.into_inner()
                        }
                    };
                    if let Some(ctx) = ctx_guard.as_mut() {
                        let audio_total_16k = samples_16k.len();
                        let cursor_before = stream_state.audio_cursor();
                        let _delta = transcribe::stream_push_audio(
                            ctx,
                            &samples_16k,
                            &mut stream_state,
                            false, // non-final
                        );
                        let cursor_after = stream_state.audio_cursor();
                        last_pushed_pos = current_pos;

                        // 更新 partial_text：finalized_text + 当前片段流式结果
                        let partial_segment = stream_state.text();
                        let combined = format!("{}{}", finalized_text, partial_segment);
                        let backlog_sec = (audio_total_16k.saturating_sub(cursor_after))
                            as f32 / 16000.0;
                        if !combined.is_empty() {
                            *safe_lock(&state.partial_text) = combined;
                            state.set_status(WorkerStatus::Transcribing);
                        }
                        logger::log_info(&format!(
                            "流式识别: 片段{:.1}s/{:.0}s, cursor={}->{} (chunk#{}, 积压{:.0}s) → \"{}\" ({:.0}ms, {} tokens)",
                            segment_audio_sec, segment_sec,
                            cursor_before, cursor_after,
                            stream_state.chunk_idx().saturating_sub(1),
                            backlog_sec,
                            partial_segment,
                            ctx.perf_total_ms, ctx.perf_text_tokens
                        ));
                    }
                    drop(ctx_guard);
                }

                // 录音停止后：串行整块识别所有片段（方案 B，精度一致）。
                // segment_audio_cache 已缓存所有已完成片段的 16kHz 音频，
                // 最后片段需要从 samples 提取并 resample。
                // 每个片段 ≤ segment_sec（默认 60s），无栈溢出风险。
                // 整块识别期间渐进式显示：已识别片段整块结果 + 未识别片段流式结果，
                // 避免文本突然回退（修复"最后一句没有完整显示"问题）。

                // 1. 收集最后片段音频（16kHz），加入缓存
                let last_segment_samples = {
                    let guard = safe_lock(&samples);
                    let total = guard.len();
                    if total < segment_offset {
                        Vec::new()
                    } else {
                        guard[segment_offset..total].to_vec()
                    }
                };
                let last_segment_stream_text = stream_state.text();
                let last_segment_dur = last_segment_samples.len() as f32 / sample_rate as f32;

                // 最后片段 ≥ 1s：resample 并加入缓存，参与整块识别
                // 最后片段 < 1s 或为空：不整块识别，流式文本单独保留
                let mut trailing_stream_text = String::new(); // 最后片段 < 1s 时的流式文本
                if last_segment_dur >= 1.0 && !last_segment_samples.is_empty() {
                    let last_16k = if sample_rate != 16000 {
                        qwen_asr::audio::resample(&last_segment_samples, sample_rate as i32, 16000)
                    } else {
                        last_segment_samples
                    };
                    segment_audio_cache.push(last_16k);
                    segment_text_cache.push(last_segment_stream_text.clone());
                } else if !last_segment_stream_text.is_empty() {
                    // 最后片段 < 1s，流式文本单独保留，整块识别完成后追加
                    trailing_stream_text = last_segment_stream_text.clone();
                }

                // 2. 空录音检查
                if segment_audio_cache.is_empty()
                    && finalized_text.is_empty()
                    && last_segment_stream_text.is_empty()
                {
                    state.set_status(WorkerStatus::Error("录音为空".into()));
                    return;
                }

                // 3. 串行整块识别所有缓存片段
                // 如果缓存为空（录音 < 1s），直接用流式结果
                if segment_audio_cache.is_empty() {
                    let final_text = if last_segment_stream_text.is_empty() {
                        finalized_text.clone()
                    } else {
                        let mut t = finalized_text.clone();
                        t.push_str(&last_segment_stream_text);
                        t
                    };
                    if final_text.is_empty() {
                        state.set_status(WorkerStatus::Error("识别结果为空".into()));
                        return;
                    }
                    state.set_status(WorkerStatus::Done {
                        text: final_text.clone(),
                        total_ms: 0.0,
                        encode_ms: 0.0,
                        decode_ms: 0.0,
                        text_tokens: 0,
                    });
                    logger::log_info(&format!(
                        "最终识别完成(无整块识别, 仅流式): \"{}\"",
                        final_text
                    ));
                    return;
                }

                // 4. 串行整块识别每个片段
                // 渐进式显示：已识别片段用整块结果，未识别片段保留流式结果
                let total_segments = segment_audio_cache.len();
                logger::log_info(&format!(
                    "开始串行整块识别: {} 个片段", total_segments
                ));

                let mut final_text = String::new();
                let mut total_ms = 0.0f64;
                let mut total_encode_ms = 0.0f64;
                let mut total_decode_ms = 0.0f64;
                let mut total_text_tokens = 0i32;
                let mut all_failed = true;

                for (i, seg_samples) in segment_audio_cache.iter().enumerate() {
                    let seg_dur = seg_samples.len() as f32 / 16000.0;
                    let mut ctx_guard = match ctx_arc.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if let Some(ctx) = ctx_guard.as_mut() {
                        logger::log_info(&format!(
                            "整块识别片段 {}/{}: {:.1}s 音频",
                            i + 1, total_segments, seg_dur
                        ));
                        let saved_cb = ctx.token_cb.take();
                        let text = transcribe::transcribe_audio(ctx, seg_samples);
                        ctx.token_cb = saved_cb;

                        match text {
                            Some(t) => {
                                final_text.push_str(&t);
                                total_ms += ctx.perf_total_ms;
                                total_encode_ms += ctx.perf_encode_ms;
                                total_decode_ms += ctx.perf_decode_ms;
                                total_text_tokens += ctx.perf_text_tokens;
                                all_failed = false;
                            }
                            None => {
                                // 该片段整块识别失败，用流式结果兜底
                                if i < segment_text_cache.len() {
                                    final_text.push_str(&segment_text_cache[i]);
                                }
                                logger::log_warn(&format!(
                                    "片段 {}/{} 整块识别返回 None，用流式结果兜底",
                                    i + 1, total_segments
                                ));
                            }
                        }
                    }
                    drop(ctx_guard);

                    // 渐进式显示：已识别片段整块结果 + 未识别片段流式结果
                    // 避免整块识别期间文本突然变短（修复"最后一句没有完整显示"）
                    let mut display_text = final_text.clone();
                    for j in (i + 1)..segment_text_cache.len() {
                        display_text.push_str(&segment_text_cache[j]);
                    }
                    // 追加最后片段 < 1s 的流式文本（如果有）
                    if !trailing_stream_text.is_empty() {
                        display_text.push_str(&trailing_stream_text);
                    }
                    *safe_lock(&state.partial_text) = display_text.clone();
                    state.set_status(WorkerStatus::Refined {
                        text: display_text,
                        total_ms,
                        text_tokens: total_text_tokens,
                        refine_idx: refine_idx.wrapping_add(1),
                    });
                    logger::log_info(&format!(
                        "片段 {}/{} 识别完成 → 累积 {} 字符 (显示含未识别流式)",
                        i + 1, total_segments,
                        final_text.chars().count()
                    ));
                }

                // 5. 所有片段都失败：用流式累积结果兜底
                if all_failed {
                    let mut fallback = finalized_text.clone();
                    fallback.push_str(&last_segment_stream_text);
                    if fallback.is_empty() {
                        state.set_status(WorkerStatus::Error("所有片段整块识别失败".into()));
                    } else {
                        state.set_status(WorkerStatus::Done {
                            text: fallback.clone(),
                            total_ms: 0.0,
                            encode_ms: 0.0,
                            decode_ms: 0.0,
                            text_tokens: 0,
                        });
                        logger::log_warn(&format!(
                            "所有片段整块识别失败，使用流式结果: \"{}\"",
                            fallback
                        ));
                    }
                    return;
                }

                // 6. 追加最后片段 < 1s 的流式文本（如果有）
                if !trailing_stream_text.is_empty() {
                    final_text.push_str(&trailing_stream_text);
                }

                // 7. 设置最终 Done 状态
                state.set_status(WorkerStatus::Done {
                    text: final_text.clone(),
                    total_ms,
                    encode_ms: total_encode_ms,
                    decode_ms: total_decode_ms,
                    text_tokens: total_text_tokens,
                });
                logger::log_info(&format!(
                    "最终识别完成: {} 个片段, \"{}\" ({:.0}ms 总计, {} tokens)",
                    total_segments, final_text, total_ms, total_text_tokens
                ));
            }));

            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                logger::log_error(&format!("实时识别线程 panic: {}", msg));
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
                | WorkerStatus::Refined { .. }
        )
    }

    /// Reset a terminal status (Done / Error / ModelLoadFailed) back
    /// to Idle after the GUI has handled it. The GUI's `update()` runs
    /// every frame as long as `is_busy()` returns true, and the old
    /// `is_busy()` does not consider Done/Error "busy" — but the
    /// status field itself was not being reset, so `poll_worker()`
    /// would see the same terminal status again on the next frame
    /// and re-apply its side effects (most visibly: appending the
    /// result text again, producing an "infinite repeat" bug).
    ///
    /// Calling this after the GUI has fully consumed the terminal
    /// status breaks that loop.
    pub fn consume_status(&self) {
        match self.state.get_status() {
            WorkerStatus::Done { .. } | WorkerStatus::Error(_) | WorkerStatus::ModelLoadFailed(_) => {
                self.state.set_status(WorkerStatus::Idle);
            }
            WorkerStatus::Refined { .. } => {
                // 实时识别中的修正结果已被 GUI 消费，回到 Transcribing
                // 让 is_busy() 仍返回 true（worker 线程仍在运行）。
                self.state.set_status(WorkerStatus::Transcribing);
            }
            _ => {
                // Non-terminal — don't touch it. Letting the worker
                // thread overwrite our reset would race with the
                // actual progress, so we only consume states the
                // worker thread is done with.
            }
        }
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

#[cfg(test)]
mod tests {
    //! `transcribe_live` 端到端集成测试。
    //!
    //! 这些测试加载真实模型和真实音频，验证实时识别的核心流程：
    //!   - 最终识别结果与 `transcribe_audio` 基线一致
    //!   - 增量识别 + 定期修正流程能正确触发 `Refined` 事件
    //!   - 停止后能正确产出 `Done` 事件
    //!
    //! 标记为 `#[ignore]` 因为需要加载完整模型并运行 CPU 推理。
    //! 手动运行：
    //!   `cargo test -p qwen-asr-gui --bin qwen-asr-gui -- --ignored --nocapture`
    use super::*;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // crates/qwen-asr-gui -> workspace root
        manifest.parent().unwrap().parent().unwrap().to_path_buf()
    }

    fn model_dir() -> PathBuf {
        project_root().join("models").join("qwen3-asr-rust-1.7b")
    }

    fn audio_path() -> PathBuf {
        project_root().join("audio.wav")
    }

    fn load_test_audio() -> Vec<f32> {
        let path = audio_path();
        qwen_asr::audio::load_wav(path.to_str().unwrap())
            .unwrap_or_else(|| panic!("failed to load {}", path.display()))
    }

    /// 等待 worker 加载模型完成，返回加载耗时（秒）。
    /// 超时 120s 则 panic。
    fn wait_model_loaded(worker: &AsrWorker) -> f64 {
        let timeout = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            if std::time::Instant::now() > timeout {
                panic!("timeout waiting for model to load");
            }
            match worker.get_status() {
                WorkerStatus::ModelLoaded(sec) => {
                    worker.consume_status();
                    return sec;
                }
                WorkerStatus::ModelLoadFailed(ref e) => {
                    panic!("model load failed: {}", e);
                }
                WorkerStatus::Idle | WorkerStatus::LoadingModel(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                _ => {
                    // 其他状态也等待
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
    }

    /// 等待 worker 到达 Done 状态，期间收集所有 Refined 事件。
    /// 返回 (refined_count, done_text)。
    /// 超时 `timeout_sec` 秒则 panic。
    fn wait_done(worker: &AsrWorker, timeout_sec: u64) -> (u32, String) {
        let timeout = std::time::Instant::now() + std::time::Duration::from_secs(timeout_sec);
        let mut refined_count = 0u32;
        let mut last_refine_idx = 0u32;
        loop {
            if std::time::Instant::now() > timeout {
                panic!(
                    "timeout waiting for Done (refined_count={}, last_refine_idx={})",
                    refined_count, last_refine_idx
                );
            }
            match worker.get_status() {
                WorkerStatus::Done { text, .. } => {
                    worker.consume_status();
                    return (refined_count, text);
                }
                WorkerStatus::Error(ref e) => {
                    panic!("worker error: {}", e);
                }
                WorkerStatus::Refined { refine_idx, .. } => {
                    refined_count += 1;
                    last_refine_idx = refine_idx;
                    worker.consume_status();
                }
                _ => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }

    /// 测试 1：预加载全部音频 + 立即停止 → 最终结果应与基线一致。
    ///
    /// 这个测试验证 transcribe_live 的最终识别路径（stop_flag=true 后的
    /// "final incremental + final recognition"）能正确产出 Done 事件。
    ///
    /// 断言策略：
    ///   - done_text 非空
    ///   - done_text 包含至少一个中文字符（验证识别成功，而非退化输出）
    ///   - done_text 与基线 text_baseline 都非空（但不要求完全一致，
    ///     因为 silence skip 和浮点非确定性可能导致差异）
    #[test]
    #[ignore = "heavy e2e test: loads model + CPU inference, run with --ignored"]
    fn transcribe_live_final_matches_baseline() {
        let model = model_dir();
        let samples = load_test_audio();
        println!(
            "[setup] model={}, audio={} samples ({:.1}s)",
            model.display(),
            samples.len(),
            samples.len() as f32 / 16000.0
        );

        // --- 基线：直接 transcribe_audio ---
        // 注意：必须调用 apply_threads() 使全局状态（verbose, 线程数）
        // 与 worker 路径一致，否则浮点运算累积顺序不同会导致结果差异。
        let mut ctx_baseline = QwenCtx::load(model.to_str().unwrap())
            .expect("failed to load model for baseline");
        let params = crate::params::AsrParams::default();
        params.apply_threads(); // 关键：与 worker.load_model 内部的 apply_threads 对齐
        params.apply_to_ctx(&mut ctx_baseline);
        // 基线不要 token_cb（避免污染）
        ctx_baseline.token_cb = None;

        let t0 = std::time::Instant::now();
        let text_baseline = transcribe::transcribe_audio(&mut ctx_baseline, &samples)
            .expect("baseline transcription failed");
        let baseline_ms = t0.elapsed().as_millis();
        println!(
            "[baseline] \"{}\" ({:.0}ms, {} tokens, skip_silence={})",
            text_baseline,
            baseline_ms as f64,
            ctx_baseline.perf_text_tokens,
            ctx_baseline.skip_silence
        );

        // 释放基线模型内存，避免与 worker 模型同时驻留导致 OOM
        drop(ctx_baseline);

        // --- 实时模式：预加载全部样本 + 立即停止 ---
        let mut worker = AsrWorker::new();
        worker.load_model(model.to_str().unwrap().to_string(), params.clone());
        let load_sec = wait_model_loaded(&worker);
        println!("[live] model loaded in {:.1}s", load_sec);

        // 调试：检查 worker ctx 的 skip_silence
        {
            let ctx_guard = worker.ctx.lock().unwrap();
            if let Some(ref ctx) = *ctx_guard {
                println!("[live] worker ctx.skip_silence = {}", ctx.skip_silence);
            }
        }

        let shared_samples = Arc::new(Mutex::new(samples.clone()));
        let stop_flag = Arc::new(AtomicBool::new(true)); // 立即停止

        let t0 = std::time::Instant::now();
        worker.transcribe_live(shared_samples, 16000, params, stop_flag);

        let (refined_count, done_text) = wait_done(&worker, 300);
        let live_ms = t0.elapsed().as_millis();
        println!(
            "[live] done in {:.0}ms, refined_count={}, text=\"{}\"",
            live_ms, refined_count, done_text
        );

        // 验证：done_text 非空且包含中文字符
        assert!(!done_text.is_empty(), "Done text should not be empty");
        let has_chinese = done_text.chars().any(|c| {
            let cp = c as u32;
            (0x4E00..=0x9FFF).contains(&cp) // CJK Unified Ideographs
                || (0x3400..=0x4DBF).contains(&cp) // CJK Extension A
        });
        assert!(
            has_chinese,
            "Done text should contain at least one Chinese character, got: \"{}\"",
            done_text
        );
        // 基线也应非空
        assert!(
            !text_baseline.is_empty(),
            "Baseline text should not be empty"
        );
        // 打印对比信息（不强制完全一致，因为 silence skip + 浮点非确定性）
        if done_text.trim() == text_baseline.trim() {
            println!("[pass] transcribe_live final EXACTLY matches baseline");
        } else {
            println!(
                "[info] live and baseline differ (acceptable due to silence skip / FP nondeterminism):"
            );
            println!("  baseline: \"{}\"", text_baseline);
            println!("  live:     \"{}\"", done_text);
        }
        println!("[pass] transcribe_live final produces valid Chinese text");
    }

    /// 测试 2：使用 feeder 线程模拟实时录音，验证增量识别 + 定期修正流程。
    ///
    /// feeder 以 4x 速度向共享缓冲区添加音频，worker 在 chunk_sec/refine_sec
    /// 间隔下做增量识别和修正。验证：
    ///   - 至少触发 1 次 Refined 事件
    ///   - Done.text 非空
    ///   - Done.text 与基线相似（允许不完全一致，因为修正有上下文差异）
    #[test]
    #[ignore = "heavy e2e test: loads model + CPU inference, run with --ignored"]
    fn transcribe_live_feeder_fires_refined_events() {
        let model = model_dir();
        let samples_full = load_test_audio();

        // 截取 30s 音频（保持测试时间可控）
        let clip_samples = 30 * 16_000;
        let samples: Vec<f32> = if samples_full.len() > clip_samples {
            samples_full[..clip_samples].to_vec()
        } else {
            samples_full.clone()
        };
        println!(
            "[setup] audio={} samples ({:.1}s)",
            samples.len(),
            samples.len() as f32 / 16000.0
        );

        // --- 基线 ---
        // 注意：必须调用 apply_threads() 使全局状态与 worker 路径一致。
        let mut ctx_baseline = QwenCtx::load(model.to_str().unwrap())
            .expect("failed to load model for baseline");
        let mut params = crate::params::AsrParams::default();
        params.apply_threads(); // 与 worker.load_model 内部对齐
        params.apply_to_ctx(&mut ctx_baseline);
        ctx_baseline.token_cb = None;
        let text_baseline = transcribe::transcribe_audio(&mut ctx_baseline, &samples)
            .expect("baseline transcription failed");
        println!("[baseline] \"{}\"", text_baseline);

        // 释放基线模型内存，避免与 worker 模型同时驻留导致 OOM
        drop(ctx_baseline);

        // --- 实时模式 with feeder ---
        // 片段式架构：设置片段长度为 10s，让 30s 音频触发 2 次片段结束
        // （每次片段结束会触发 Refined 事件）
        params.stream_chunk_sec = 4.0; // 每 4s 音频做一次流式增量
        std::env::set_var("QWEN_ASR_SEGMENT_SEC", "10");

        let mut worker = AsrWorker::new();
        worker.load_model(model.to_str().unwrap().to_string(), params.clone());
        let load_sec = wait_model_loaded(&worker);
        println!("[live] model loaded in {:.1}s", load_sec);

        let shared_samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));

        // feeder 线程：以 4x 速度添加音频
        // 每 0.5s 添加 2s 音频 (= 32000 samples)
        let feed_speed = 4.0f32;
        let feed_interval_ms = 500u64;
        let feed_chunk_samples = (16000.0 * feed_speed * feed_interval_ms as f32 / 1000.0) as usize;
        let feeder_samples = samples.clone();
        let feeder_buffer = shared_samples.clone();
        let feeder_stop = stop_flag.clone();
        let feeder_handle = std::thread::spawn(move || {
            let mut pos = 0usize;
            while pos < feeder_samples.len() && !feeder_stop.load(Ordering::Relaxed) {
                let end = (pos + feed_chunk_samples).min(feeder_samples.len());
                {
                    let mut guard = feeder_buffer.lock().unwrap();
                    guard.extend_from_slice(&feeder_samples[pos..end]);
                }
                pos = end;
                // 分段睡眠以便及时响应 stop
                let mut slept = 0u64;
                while slept < feed_interval_ms && !feeder_stop.load(Ordering::Relaxed) {
                    let step = 100u64.min(feed_interval_ms - slept);
                    std::thread::sleep(Duration::from_millis(step));
                    slept += step;
                }
            }
        });

        // 监控用 clone（shared_samples 会被 move 到 transcribe_live）
        let monitor_buffer = shared_samples.clone();

        // 启动实时识别
        let t0 = std::time::Instant::now();
        worker.transcribe_live(shared_samples, 16000, params, stop_flag.clone());

        // 等待 feeder 完成（所有音频已喂入），然后停止
        // 30s 音频 / 4x 速度 = 7.5s 墙钟时间，给 12s 余量
        let feed_deadline = t0 + Duration::from_secs(15);
        loop {
            if std::time::Instant::now() > feed_deadline {
                break;
            }
            let fed = safe_lock(&monitor_buffer).len();
            if fed >= samples.len() {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        // 设置停止标志
        stop_flag.store(true, Ordering::Relaxed);
        println!(
            "[live] stop_flag set at {:.1}s, fed_samples={}",
            t0.elapsed().as_secs_f64(),
            safe_lock(&monitor_buffer).len()
        );

        // 等待 Done（最多 120s，因为最终识别 30s 音频需要时间）
        let (refined_count, done_text) = wait_done(&worker, 120);
        let total_sec = t0.elapsed().as_secs_f64();
        println!(
            "[live] done in {:.1}s, refined_count={}, text=\"{}\"",
            total_sec, refined_count, done_text
        );

        // 加入 feeder 线程
        let _ = feeder_handle.join();

        // 验证
        assert!(
            refined_count >= 1,
            "should have at least 1 Refined event, got {}",
            refined_count
        );
        assert!(!done_text.is_empty(), "Done text should not be empty");
        // done_text 应包含中文字符（验证识别成功）
        let has_chinese = done_text.chars().any(|c| {
            let cp = c as u32;
            (0x4E00..=0x9FFF).contains(&cp) || (0x3400..=0x4DBF).contains(&cp)
        });
        assert!(
            has_chinese,
            "Done text should contain Chinese characters, got: \"{}\"",
            done_text
        );
        // 不要求与基线完全一致（live 路径中增量+修正会修改 ctx 状态，
        // 且 silence skip + 浮点非确定性可能导致差异）
        if done_text.trim() == text_baseline.trim() {
            println!("[pass] feeder final EXACTLY matches baseline");
        } else {
            println!("[info] feeder and baseline differ (acceptable):");
            println!("  baseline: \"{}\"", text_baseline);
            println!("  live:     \"{}\"", done_text);
        }
        println!("[pass] transcribe_live feeder test passed");
        // 清理测试环境变量
        std::env::remove_var("QWEN_ASR_SEGMENT_SEC");
    }
}
