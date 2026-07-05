//! Main ASR GUI application

use crate::params::AsrParams;
use crate::recorder::MicRecorder;
use crate::sync_ext::safe_lock;
use crate::worker::{AsrWorker, WorkerStatus};
use crate::logger;
use eframe::egui;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct AsrApp {
    params: AsrParams,
    worker: AsrWorker,

    // UI state
    model_dir: String,
    model_loaded: bool,
    audio_path: String,
    result_text: String,
    status_text: String,
    profile_text: String,
    copied: bool,
    copy_timer: f32,
    /// Monotonic counter of completed transcription sessions, used as
    /// a label in the result textbox so the user can distinguish
    /// successive recordings (since v0.2 the textbox appends rather
    /// than replaces).
    session_idx: u32,
    /// 实时会话在 result_text 中的起始字节位置。
    /// Some(pos) = 实时模式，修正/最终结果截断到 pos 后替换；
    /// None = 非实时模式，Done 时追加。
    live_text_start: Option<usize>,

    // Input mode: file or microphone
    input_mode: InputMode,

    // Microphone recording state
    recorder: Option<MicRecorder>,
    /// 实时识别的停止标志，设置后通知 worker 线程做最终识别
    live_stop_flag: Option<Arc<AtomicBool>>,
    /// 录音错误信息（若有）
    mic_error: String,

    // Pending file dialog
    file_dialog_type: FileDialogType,
    /// True while waiting for async file dialog result
    dialog_pending: bool,
}

#[derive(PartialEq, Debug)]
enum InputMode {
    File,
    Microphone,
}

#[derive(PartialEq, Debug)]
enum FileDialogType {
    None,
    ModelDir,
    AudioFile,
}

impl AsrApp {
    pub fn new(model_path: Option<String>) -> Self {
        let params = AsrParams::default();
        let mut app = Self {
            params,
            worker: AsrWorker::new(),
            model_dir: String::new(),
            model_loaded: false,
            audio_path: String::new(),
            result_text: String::new(),
            status_text: "就绪。请加载模型。".into(),
            profile_text: String::new(),
            copied: false,
            copy_timer: 0.0,
            session_idx: 0,
            live_text_start: None,
            input_mode: InputMode::File,
            recorder: None,
            live_stop_flag: None,
            mic_error: String::new(),
            file_dialog_type: FileDialogType::None,
            dialog_pending: false,
        };

        // Auto-load model if path provided via --model argument
        if let Some(path) = model_path {
            app.model_dir = path.clone();
            app.status_text = format!("正在加载模型: {}...", path);
            logger::log_info(&format!("模型自动加载: {}", path));
            app.worker.load_model(path, app.params.clone());
        }

        app
    }

    fn open_file_dialog(&mut self, kind: FileDialogType) {
        self.file_dialog_type = kind;
    }

    fn handle_file_dialog(&mut self, _ctx: &egui::Context) {
        if self.file_dialog_type == FileDialogType::None {
            return;
        }

        let kind = std::mem::replace(&mut self.file_dialog_type, FileDialogType::None);
        self.dialog_pending = true;

        // Use AsyncFileDialog in a background thread to avoid COM conflicts
        // with winit's MTA initialization on Windows.
        // rfd's AsyncFileDialog internally spawns an STA thread for the dialog.
        match kind {
            FileDialogType::ModelDir => {
                std::thread::spawn(move || {
                    let future = rfd::AsyncFileDialog::new()
                        .set_title("选择模型目录")
                        .pick_folder();
                    if let Some(path) = pollster::block_on(future) {
                        *safe_lock(&PENDING_DIR) =
                            Some(path.path().to_string_lossy().to_string());
                    }
                });
            }
            FileDialogType::AudioFile => {
                std::thread::spawn(move || {
                    let future = rfd::AsyncFileDialog::new()
                        .set_title("选择音频文件")
                        .add_filter("音频文件", &["wav", "mp3", "flac", "m4a"])
                        .pick_file();
                    if let Some(path) = pollster::block_on(future) {
                        *safe_lock(&PENDING_AUDIO) =
                            Some(path.path().to_string_lossy().to_string());
                    }
                });
            }
            _ => {
                self.dialog_pending = false;
            }
        }
    }

    fn check_pending_dialogs(&mut self) {
        if let Some(dir) = safe_lock(&PENDING_DIR).take() {
            self.model_dir = dir.clone();
            self.model_loaded = false;
            self.dialog_pending = false;
            logger::log_info(&format!("用户选择模型目录: {}", dir));
            // Auto-trigger model loading after directory selection
            self.worker.load_model(dir, self.params.clone());
        }
        if let Some(path) = safe_lock(&PENDING_AUDIO).take() {
            self.audio_path = path.clone();
            self.dialog_pending = false;
            logger::log_info(&format!("用户选择音频文件: {}", path));
        }
    }

    /// 启动实时录音+识别（从"开始录音"按钮和自动测试模式两处调用）。
    /// `is_auto=true` 表示自动化测试模式触发，`false` 表示用户手动点击。
    fn start_live_recording(&mut self, is_auto: bool) {
        self.mic_error.clear();
        if is_auto {
            logger::log_info("自动录音模式: 开始录音（自动化测试）");
        }
        match MicRecorder::start() {
            Ok(rec) => {
                let sr = rec.sample_rate();
                let samples_arc = rec.samples_arc();
                logger::log_info(&format!(
                    "录音开始 (采样率: {}Hz), 启动实时识别",
                    sr
                ));
                let stop_flag = Arc::new(AtomicBool::new(false));
                self.live_stop_flag = Some(stop_flag.clone());
                self.worker.transcribe_live(
                    samples_arc,
                    sr,
                    self.params.clone(),
                    stop_flag,
                );
                self.live_text_start = Some(self.result_text.len());
                self.recorder = Some(rec);
                self.status_text = format!("录音中... 0.0s");
            }
            Err(e) => {
                logger::log_error(&format!("录音启动失败: {}", e));
                self.mic_error = e;
            }
        }
    }

    fn poll_worker(&mut self) {
        let status = self.worker.get_status();
        match status {
            WorkerStatus::Idle => {}
            WorkerStatus::LoadingModel(ref dir) => {
                self.status_text = format!("正在加载模型: {}...", dir);
            }
            WorkerStatus::ModelLoaded(sec) => {
                self.model_loaded = true;
                // Don't overwrite status while recording or transcribing
                if self.recorder.is_none() {
                    self.status_text = format!("模型加载完成 ({:.1}s)", sec);
                }
                // 自动录音模式：环境变量 QWEN_ASR_AUTO_RECORD=1 时，
                // 模型加载完成后自动开始录音（用于自动化测试）。
                if self.recorder.is_none()
                    && std::env::var("QWEN_ASR_AUTO_RECORD").as_deref() == Ok("1")
                {
                    self.start_live_recording(true);
                }
            }
            WorkerStatus::ModelLoadFailed(ref err) => {
                self.status_text = format!("错误: {}", err);
                self.model_loaded = false;
                // Terminal state — consume it so the next frame goes
                // back to Idle. Without this the GUI would re-display
                // the same error every frame.
                self.worker.consume_status();
            }
            WorkerStatus::LoadingAudio(ref path) => {
                self.status_text = format!("正在加载音频: {}...", path);
            }
            WorkerStatus::Transcribing => {
                // 实时文字进文本框（用户需求 #3）：partial_text 写入
                // result_text（替换式，靠 live_text_start 截断），状态栏
                // 只显示简短的"识别中..."。
                let partial = self.worker.get_partial_text();
                if !partial.is_empty() {
                    if let Some(pos) = self.live_text_start {
                        // 实时模式：截断到会话起始位置后追加 partial
                        self.result_text.truncate(pos);
                        if !self.result_text.is_empty() {
                            self.result_text.push_str("\n\n");
                        }
                        self.result_text.push_str(&partial);
                    }
                    // 非实时模式（live_text_start = None）：不修改
                    // result_text，由 Refined/Done 处理
                }
                self.status_text = "识别中...".into();
            }
            WorkerStatus::Refined {
                text,
                total_ms,
                text_tokens,
                refine_idx,
            } => {
                // 实时会话期间，每次修正替换本会话之前的内容（不堆积）。
                // live_text_start = Some(pos) 表示当前在实时会话中：
                //   截断 result_text 到 pos，再追加本次修正。
                // live_text_start = None 表示非实时模式（不应发生），退化为追加。
                let tok_s = if total_ms > 0.0 {
                    1000.0 * text_tokens as f64 / total_ms
                } else {
                    0.0
                };
                if !text.is_empty() {
                    if let Some(pos) = self.live_text_start {
                        // 替换式：截断到会话起始位置后追加本次修正
                        self.result_text.truncate(pos);
                        if !self.result_text.is_empty() {
                            self.result_text.push_str("\n\n");
                        }
                    } else if !self.result_text.is_empty() {
                        self.result_text.push_str("\n\n");
                    }
                    self.result_text.push_str(&format!(
                        "[修正 #{refine_idx}] {}\n({:.0}ms | {} tokens | {:.2} tok/s)",
                        text, total_ms, text_tokens, tok_s,
                    ));
                }
                self.status_text = format!(
                    "修正 #{refine_idx} ({:.0}ms | {} tokens | {:.2} tok/s)",
                    total_ms, text_tokens, tok_s,
                );
                // 消费此修正状态，回到 Transcribing 让 worker 继续
                self.worker.consume_status();
            }
            WorkerStatus::Done {
                text,
                total_ms,
                encode_ms,
                decode_ms,
                text_tokens,
            } => {
                // CRITICAL: consume the Done status before doing any
                // UI work. If we don't, poll_worker() runs every frame
                // (because is_busy() returns true while the worker is
                // in any of the non-Idle states, and we never put
                // ourselves back into Idle), each frame would see
                // Done{...} again, increment session_idx, and append
                // the same text into the result textbox — producing
                // the "infinite repeat" bug seen in v0.1 GUI build
                // 2026-06-21 23:39:37.
                self.worker.consume_status();

                // Append the new transcription to the result textbox
                // instead of replacing it. Each session is separated by
                // a blank line and a per-segment header with timing
                // info, so the user can see exactly which line came
                // from which recording.
                //
                // Session counter is a simple monotonic count of Done
                // events handled by the UI since startup — sufficient
                // for a GUI label without dragging in a chrono dep.
                self.session_idx += 1;
                if !text.is_empty() {
                    // 实时会话结束时：截断到 live_text_start 替换本会话内容；
                    // 非实时模式（文件识别）：直接追加。
                    if let Some(pos) = self.live_text_start.take() {
                        self.result_text.truncate(pos);
                        if !self.result_text.is_empty() {
                            self.result_text.push_str("\n\n");
                        }
                    } else if !self.result_text.is_empty() {
                        self.result_text.push_str("\n\n");
                    }
                    let tok_s = if total_ms > 0.0 {
                        1000.0 * text_tokens as f64 / total_ms
                    } else {
                        0.0
                    };
                    self.result_text.push_str(&format!(
                        "[#{}] {}\n({:.0}ms | {} tokens | {:.2} tok/s)",
                        self.session_idx, text, total_ms, text_tokens, tok_s,
                    ));
                } else {
                    // 即使最终结果为空，也要消费 live_text_start 以退出实时模式
                    self.live_text_start = None;
                }
                self.status_text = format!(
                    "完成: {:.0}ms (编码 {:.0}ms, 解码 {:.0}ms) | {} tokens ({:.2} tok/s)",
                    total_ms, encode_ms, decode_ms, text_tokens,
                    if total_ms > 0.0 { 1000.0 * text_tokens as f64 / total_ms } else { 0.0 }
                );

                // Update profile
                if self.params.profile {
                    let prof = self.worker.get_profile();
                    self.profile_text = prof.lines.join("\n");
                }
            }
            WorkerStatus::Error(ref err) => {
                self.status_text = format!("错误: {}", err);
                // Same reasoning as ModelLoadFailed: reset to Idle so
                // the error doesn't replay on every frame.
                self.worker.consume_status();
            }
        }
    }
}

// Global pending file dialog results (simple approach for async callbacks)
static PENDING_DIR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
static PENDING_AUDIO: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

impl eframe::App for AsrApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll worker status
        self.poll_worker();
        self.check_pending_dialogs();

        // Handle pending file dialogs
        self.handle_file_dialog(ctx);

        // Update recording status (before panels render)
        // 用户需求 #3：状态栏只显示简短状态（"录音中... X.Xs"），
        // 实时识别文字由 poll_worker() 的 Transcribing 分支写入 result_text。
        if self.recorder.is_some() {
            if let Some(ref rec) = self.recorder {
                let elapsed = rec.elapsed_sec();
                self.status_text = format!("录音中... {:.1}s", elapsed);
            }
        }

        // Copy notification timer
        if self.copied {
            self.copy_timer += ctx.input(|i| i.stable_dt);
            if self.copy_timer > 2.0 {
                self.copied = false;
                self.copy_timer = 0.0;
            }
        }

        // Request repaint while busy or waiting for dialog result
        if self.worker.is_busy() {
            ctx.request_repaint();
        }
        if self.dialog_pending {
            // Poll for async dialog result every 100ms
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if self.recorder.is_some() {
            // Update recording timer display ~10fps
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Top panel: title + status
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("Qwen3-ASR 语音识别");
                ui.label(format!("BLAS {}t", qwen_asr::kernels::get_blas_threads()));
            });
            ui.add_space(2.0);
        });

        // Bottom panel: status bar
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let status_color = if self.status_text.starts_with("错误") {
                    egui::Color32::from_rgb(220, 80, 80)
                } else if self.status_text.starts_with("完成") {
                    egui::Color32::from_rgb(80, 200, 80)
                } else {
                    egui::Color32::from_rgb(180, 180, 180)
                };
                ui.colored_label(status_color, &self.status_text);
            });
            ui.add_space(4.0);
        });

        // Left panel: model + audio + params
        egui::SidePanel::left("left_panel")
            .resizable(true)
            .width_range(320.0..=420.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);

                // Model section
                ui.group(|ui| {
                    ui.label(egui::RichText::new("模型").strong());
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        let busy = self.worker.is_busy();
                        if ui.add_enabled(!busy, egui::Button::new("加载模型"))
                            .on_hover_text("选择模型目录并加载")
                            .clicked()
                        {
                            self.open_file_dialog(FileDialogType::ModelDir);
                        }
                        if ui.button("浏览...").clicked() {
                            self.open_file_dialog(FileDialogType::ModelDir);
                        }
                    });
                    if !self.model_dir.is_empty() {
                        ui.label(format!("目录: {}", shorten_path(&self.model_dir, 40)));
                    }
                    if self.model_loaded {
                        ui.label(
                            egui::RichText::new("✓ 模型已加载")
                                .color(egui::Color32::from_rgb(80, 200, 80)),
                        );
                    }
                });

                ui.add_space(8.0);

                // Audio section
                ui.group(|ui| {
                    ui.label(egui::RichText::new("音频输入").strong());
                    ui.add_space(2.0);

                    // Input mode selector
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.input_mode, InputMode::File, "音频文件");
                        ui.radio_value(&mut self.input_mode, InputMode::Microphone, "麦克风");
                    });
                    ui.add_space(2.0);

                    let busy = self.worker.is_busy();
                    let recording = self.recorder.is_some();

                    match self.input_mode {
                        InputMode::File => {
                            ui.horizontal(|ui| {
                                if ui.add_enabled(
                                    self.model_loaded && !busy,
                                    egui::Button::new("选择音频"),
                                )
                                .on_hover_text("选择 WAV/MP3/FLAC 音频文件")
                                .clicked()
                                {
                                    self.open_file_dialog(FileDialogType::AudioFile);
                                }
                                let can_transcribe = self.model_loaded
                                    && !self.audio_path.is_empty()
                                    && !busy;
                                if ui.add_enabled(can_transcribe, egui::Button::new("开始识别"))
                                    .clicked()
                                {
                                    logger::log_info(&format!(
                                        "用户点击: 开始识别 (文件: {})",
                                        self.audio_path
                                    ));
                                    self.worker.transcribe(
                                        self.audio_path.clone(),
                                        self.params.clone(),
                                    );
                                }
                            });
                            if !self.audio_path.is_empty() {
                                ui.label(format!(
                                    "文件: {}",
                                    shorten_path(&self.audio_path, 40)
                                ));
                            }
                        }
                        InputMode::Microphone => {
                            // Show mic error if any
                            if !self.mic_error.is_empty() {
                                ui.label(
                                    egui::RichText::new(&self.mic_error)
                                        .color(egui::Color32::from_rgb(220, 80, 80)),
                                );
                            }

                            if !recording && !busy {
                                // Idle: show "开始录音" button
                                if ui
                                    .add_enabled(
                                        self.model_loaded,
                                        egui::Button::new("开始录音"),
                                    )
                                    .on_hover_text("点击后开始从麦克风录音，录音中自动实时识别")
                                    .clicked()
                                {
                                    logger::log_info("用户点击: 开始录音");
                                    self.start_live_recording(false);
                                }
                                if !self.model_loaded {
                                    ui.label(
                                        egui::RichText::new("请先加载模型")
                                            .color(egui::Color32::from_rgb(180, 180, 180)),
                                    );
                                }
                            } else if recording {
                                // Recording: show "停止录音" button
                                if ui
                                    .add(
                                        egui::Button::new("停止录音")
                                            .fill(egui::Color32::from_rgb(200, 80, 80)),
                                    )
                                    .on_hover_text("停止录音并做最终识别")
                                    .clicked()
                                {
                                    logger::log_info("用户点击: 停止录音");
                                    // 通知实时识别线程做最终识别
                                    if let Some(flag) = &self.live_stop_flag {
                                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    self.live_stop_flag = None;
                                    // 停止录音（drop stream）；worker 线程仍持有
                                    // samples Arc，可读取最终样本做最终识别
                                    if let Some(rec) = self.recorder.take() {
                                        let sr = rec.sample_rate();
                                        let _ = rec.stop(); // 停止采集
                                        logger::log_info(&format!(
                                            "录音停止 (采样率: {}Hz), 等待最终识别",
                                            sr
                                        ));
                                    }
                                    self.status_text = "录音已停止，正在做最终识别...".into();
                                }
                            }
                        }
                    }
                });

                ui.add_space(8.0);

                // Parameters section
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("参数设置").strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("恢复默认").on_hover_text("重置所有参数为推荐值").clicked() {
                                self.params = AsrParams::default();
                            }
                        });
                    });
                    ui.add_space(2.0);

                    egui::Grid::new("params_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("线程数");
                            ui.add(egui::DragValue::new(&mut self.params.n_threads).range(1..=64));
                            ui.end_row();

                            ui.label("BLAS线程");
                            ui.add(egui::DragValue::new(&mut self.params.blas_threads).range(1..=32));
                            ui.end_row();

                            ui.label("分段时长(s)");
                            ui.add(
                                egui::DragValue::new(&mut self.params.segment_sec)
                                    .range(-1.0..=60.0),
                            );
                            ui.end_row();

                            ui.label("编码窗口(s)");
                            ui.add(
                                egui::DragValue::new(&mut self.params.enc_window_sec)
                                    .range(-1.0..=30.0),
                            );
                            ui.end_row();

                            ui.label("最大Token");
                            ui.add(
                                egui::DragValue::new(&mut self.params.stream_max_new_tokens)
                                    .range(-1..=512),
                            );
                            ui.end_row();

                            ui.label("流式分块(s)");
                            ui.add(
                                egui::DragValue::new(&mut self.params.stream_chunk_sec)
                                    .range(-1.0..=30.0),
                            );
                            ui.end_row();

                            ui.label("二次修正");
                            ui.checkbox(&mut self.params.refine_enabled, "启用")
                                .on_hover_text(
                                    "流式草稿 + 定期整块重识别\n\
                                     兼顾实时性与精度",
                                );
                            ui.end_row();

                            ui.label("修正间隔(s)");
                            ui.add_enabled(
                                self.params.refine_enabled,
                                egui::DragValue::new(&mut self.params.refine_interval_sec)
                                    .range(5.0..=120.0),
                            );
                            ui.end_row();

                            ui.label("跳过静音");
                            ui.checkbox(&mut self.params.skip_silence, "");
                            ui.end_row();

                            ui.label("历史文本");
                            let mut pt = self.params.past_text_mode == 1;
                            ui.checkbox(&mut pt, "启用");
                            self.params.past_text_mode = if pt { 1 } else { 0 };
                            ui.end_row();

                            ui.label("性能分析");
                            ui.checkbox(&mut self.params.profile, "启用");
                            ui.end_row();

                            ui.label("日志级别");
                            ui.add(egui::DragValue::new(&mut self.params.verbosity).range(0..=3));
                            ui.end_row();
                        });

                    ui.add_space(4.0);
                    ui.collapsing("推荐参数说明", |ui| {
                        ui.label(
                            egui::RichText::new(
                                "BLAS线程: 4 (6C/12T 最优)\n\
                                 线程数: 自动检测性能核心\n\
                                 历史文本: 启用 (流式识别推荐)\n\
                                 性能分析: 用于定位瓶颈",
                            )
                            .color(egui::Color32::from_rgb(150, 150, 150)),
                        );
                    });
                });
            });

        // Central panel: results
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(4.0);

            // Results area
            let _busy = self.worker.is_busy();
            let recording = self.recorder.is_some();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("识别结果").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.copied {
                        ui.label(
                            egui::RichText::new("✓ 已复制")
                                .color(egui::Color32::from_rgb(80, 200, 80)),
                        );
                    }
                    if ui
                        .add_enabled(!recording, egui::Button::new("清空"))
                        .clicked()
                    {
                        logger::log_info("用户点击: 清空结果");
                        self.result_text.clear();
                    }
                    if ui
                        .add_enabled(!self.result_text.is_empty(), egui::Button::new("复制"))
                        .clicked()
                    {
                        logger::log_info("用户点击: 复制结果");
                        ui.ctx().copy_text(self.result_text.clone());
                        self.copied = true;
                        self.copy_timer = 0.0;
                    }
                });
            });

            ui.add_space(4.0);

            // Result text area — wrapped in a vertical ScrollArea so
            // long transcripts (multi-recording session) get a scroll
            // bar instead of growing past the viewport. The text
            // itself is editable, and the scroll position auto-sticks
            // to the bottom so the user always sees the latest result.
            let text_style = egui::TextStyle::Monospace;
            let mut text = self.result_text.clone();
            let mut scroll = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true);
            // Only auto-scroll when the user hasn't scrolled away
            // from the bottom; this preserves manual scroll position
            // when reviewing older results.
            scroll = scroll.animated(false);
            scroll.show(ui, |ui| {
                let text_edit = egui::TextEdit::multiline(&mut text)
                    .desired_width(f32::MAX)
                    .desired_rows(10)
                    .font(text_style)
                    .interactive(true);
                ui.add(text_edit);
            });
            // Update if user edited
            if text != self.result_text {
                self.result_text = text;
            }

            ui.add_space(8.0);

            // Profile section (collapsible)
            if self.params.profile || !self.profile_text.is_empty() {
                ui.collapsing("性能分析", |ui| {
                    if !self.profile_text.is_empty() {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.profile_text.clone())
                                .desired_width(f32::MAX)
                                .desired_rows(12)
                                .font(egui::TextStyle::Monospace)
                                .interactive(false),
                        );
                    } else {
                        ui.label("（启用性能分析后显示）");
                    }
                });
            }
        });
    }
}

/// Shorten a path string for display
fn shorten_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        let start = path.len().saturating_sub(max_len - 3);
        format!("...{}", &path[start..])
    }
}

#[cfg(test)]
mod tests {
    //! 文本框显示状态机的单元测试。
    //!
    //! 这些测试直接调用 `AsrApp::poll_worker()`，验证 `Refined`/`Done`
    //! 事件在实时模式（`live_text_start = Some`）和文件模式（`None`）下
    //! 对 `result_text` 的修改行为是否符合预期：
    //!   - 实时模式：修正/最终结果替换本会话之前的内容（不堆积）
    //!   - 文件模式：每次 Done 追加（会话之间分隔）
    //!   - 实时模式结束后 `live_text_start` 回到 `None`
    //!   - 之前会话的内容不被破坏
    use super::*;

    fn make_app() -> AsrApp {
        AsrApp::new(None)
    }

    /// 实时会话中多次 Refined 应该替换而非堆积。
    #[test]
    fn live_session_refined_events_replace_not_accumulate() {
        let mut app = make_app();
        // 模拟录音开始：result_text 为空，起始位置 = 0
        app.live_text_start = Some(0);

        // 第一次修正
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "alpha draft".into(),
            total_ms: 1000.0,
            text_tokens: 10,
            refine_idx: 1,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("alpha draft"),
            "第一次修正应出现: {:?}",
            app.result_text
        );
        assert!(
            !app.result_text.contains("beta draft"),
            "第二次修正不应出现: {:?}",
            app.result_text
        );
        let len_after_first = app.result_text.len();
        assert!(len_after_first > 0);

        // 第二次修正应替换第一次
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "beta draft".into(),
            total_ms: 2000.0,
            text_tokens: 20,
            refine_idx: 2,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("beta draft"),
            "第二次修正应出现: {:?}",
            app.result_text
        );
        assert!(
            !app.result_text.contains("alpha draft"),
            "第一次修正应被截断: {:?}",
            app.result_text
        );

        // Done 应替换第二次修正
        app.worker.state.set_status(WorkerStatus::Done {
            text: "gamma final".into(),
            total_ms: 3000.0,
            encode_ms: 1000.0,
            decode_ms: 2000.0,
            text_tokens: 30,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("gamma final"),
            "最终结果应出现: {:?}",
            app.result_text
        );
        assert!(
            !app.result_text.contains("beta draft"),
            "第二次修正应被 Done 截断: {:?}",
            app.result_text
        );
        assert!(
            app.live_text_start.is_none(),
            "Done 后 live_text_start 应为 None"
        );
    }

    /// 文件模式（非实时）：Done 应追加而非替换。
    #[test]
    fn file_mode_done_appends_not_replaces() {
        let mut app = make_app();
        // live_text_start 为 None（默认）= 文件模式

        // 第一次会话
        app.worker.state.set_status(WorkerStatus::Done {
            text: "file one".into(),
            total_ms: 1000.0,
            encode_ms: 500.0,
            decode_ms: 500.0,
            text_tokens: 10,
        });
        app.poll_worker();
        assert!(app.result_text.contains("file one"));

        // 第二次会话应追加
        app.worker.state.set_status(WorkerStatus::Done {
            text: "file two".into(),
            total_ms: 2000.0,
            encode_ms: 1000.0,
            decode_ms: 1000.0,
            text_tokens: 20,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("file one"),
            "文件模式应保留之前会话: {:?}",
            app.result_text
        );
        assert!(app.result_text.contains("file two"));
    }

    /// 实时会话应保留之前会话的内容，仅替换本会话部分。
    #[test]
    fn live_session_preserves_previous_content() {
        let mut app = make_app();

        // 先有一个文件模式会话留下内容
        app.worker.state.set_status(WorkerStatus::Done {
            text: "previous file".into(),
            total_ms: 1000.0,
            encode_ms: 500.0,
            decode_ms: 500.0,
            text_tokens: 10,
        });
        app.poll_worker();
        assert!(app.result_text.contains("previous file"));

        // 开始实时会话
        app.live_text_start = Some(app.result_text.len());

        // 第一次修正
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "live draft one".into(),
            total_ms: 1000.0,
            text_tokens: 10,
            refine_idx: 1,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("previous file"),
            "之前会话内容应保留: {:?}",
            app.result_text
        );
        assert!(app.result_text.contains("live draft one"));

        // 第二次修正只替换实时部分
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "live draft two".into(),
            total_ms: 2000.0,
            text_tokens: 20,
            refine_idx: 2,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("previous file"),
            "之前会话内容仍应保留: {:?}",
            app.result_text
        );
        assert!(app.result_text.contains("live draft two"));
        assert!(
            !app.result_text.contains("live draft one"),
            "第一次修正应被替换: {:?}",
            app.result_text
        );

        // Done
        app.worker.state.set_status(WorkerStatus::Done {
            text: "live final".into(),
            total_ms: 3000.0,
            encode_ms: 1000.0,
            decode_ms: 2000.0,
            text_tokens: 30,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("previous file"),
            "Done 后之前会话内容应保留: {:?}",
            app.result_text
        );
        assert!(app.result_text.contains("live final"));
        assert!(
            !app.result_text.contains("live draft two"),
            "修正应被 Done 替换: {:?}",
            app.result_text
        );
        assert!(app.live_text_start.is_none());
    }

    /// 实时模式下 Done 收到空文本应清除 live_text_start。
    #[test]
    fn live_session_done_empty_text_clears_live_state() {
        let mut app = make_app();
        app.live_text_start = Some(0);

        app.worker.state.set_status(WorkerStatus::Done {
            text: "".into(),
            total_ms: 0.0,
            encode_ms: 0.0,
            decode_ms: 0.0,
            text_tokens: 0,
        });
        app.poll_worker();
        assert!(
            app.live_text_start.is_none(),
            "空 Done 也应清除 live_text_start"
        );
        assert!(
            app.result_text.is_empty(),
            "空文本不应追加任何内容: {:?}",
            app.result_text
        );
    }

    /// 多次实时会话应各自独立替换，不互相干扰。
    #[test]
    fn multiple_live_sessions_are_independent() {
        let mut app = make_app();

        // 第一次实时会话
        app.live_text_start = Some(0);
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "session1 refine".into(),
            total_ms: 1000.0,
            text_tokens: 10,
            refine_idx: 1,
        });
        app.poll_worker();
        app.worker.state.set_status(WorkerStatus::Done {
            text: "session1 final".into(),
            total_ms: 2000.0,
            encode_ms: 1000.0,
            decode_ms: 1000.0,
            text_tokens: 20,
        });
        app.poll_worker();
        assert!(app.result_text.contains("session1 final"));
        assert!(app.live_text_start.is_none());

        // 第二次实时会话
        app.live_text_start = Some(app.result_text.len());
        app.worker.state.set_status(WorkerStatus::Refined {
            text: "session2 refine".into(),
            total_ms: 1000.0,
            text_tokens: 10,
            refine_idx: 1,
        });
        app.poll_worker();
        app.worker.state.set_status(WorkerStatus::Done {
            text: "session2 final".into(),
            total_ms: 2000.0,
            encode_ms: 1000.0,
            decode_ms: 1000.0,
            text_tokens: 20,
        });
        app.poll_worker();
        assert!(
            app.result_text.contains("session1 final"),
            "第一次会话最终结果应保留: {:?}",
            app.result_text
        );
        assert!(
            app.result_text.contains("session2 final"),
            "第二次会话最终结果应出现: {:?}",
            app.result_text
        );
        assert!(
            !app.result_text.contains("session1 refine"),
            "第一次会话的中间修正应已被替换: {:?}",
            app.result_text
        );
        assert!(
            !app.result_text.contains("session2 refine"),
            "第二次会话的中间修正应已被替换: {:?}",
            app.result_text
        );
    }
}
