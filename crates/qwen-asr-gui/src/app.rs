//! Main ASR GUI application

use crate::params::AsrParams;
use crate::recorder::MicRecorder;
use crate::sync_ext::safe_lock;
use crate::worker::{AsrWorker, WorkerStatus};
use crate::logger;
use eframe::egui;

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

    // Input mode: file or microphone
    input_mode: InputMode,

    // Microphone recording state
    recorder: Option<MicRecorder>,
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
            input_mode: InputMode::File,
            recorder: None,
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
            }
            WorkerStatus::ModelLoadFailed(ref err) => {
                self.status_text = format!("错误: {}", err);
                self.model_loaded = false;
            }
            WorkerStatus::LoadingAudio(ref path) => {
                self.status_text = format!("正在加载音频: {}...", path);
            }
            WorkerStatus::Transcribing => {
                let partial = self.worker.get_partial_text();
                if !partial.is_empty() {
                    self.status_text = format!("识别中... {}", partial);
                } else {
                    self.status_text = "识别中...".into();
                }
            }
            WorkerStatus::Done {
                text,
                total_ms,
                encode_ms,
                decode_ms,
                text_tokens,
            } => {
                self.result_text = text;
                let tok_s = if total_ms > 0.0 {
                    1000.0 * text_tokens as f64 / total_ms
                } else {
                    0.0
                };
                self.status_text = format!(
                    "完成: {:.0}ms (编码 {:.0}ms, 解码 {:.0}ms) | {} tokens ({:.2} tok/s)",
                    total_ms, encode_ms, decode_ms, text_tokens, tok_s
                );

                // Update profile
                if self.params.profile {
                    let prof = self.worker.get_profile();
                    self.profile_text = prof.lines.join("\n");
                }
            }
            WorkerStatus::Error(ref err) => {
                self.status_text = format!("错误: {}", err);
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
        if self.recorder.is_some() {
            if let Some(ref rec) = self.recorder {
                let elapsed = rec.elapsed_sec();
                let samples = rec.sample_count();
                let sr = rec.sample_rate();
                self.status_text = format!(
                    "录音中... {:.1}s ({}Hz, {} 样本)",
                    elapsed, sr, samples
                );
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
                                    .on_hover_text("点击后开始从麦克风录音")
                                    .clicked()
                                {
                                    self.mic_error.clear();
                                    logger::log_info("用户点击: 开始录音");
                                    match MicRecorder::start() {
                                        Ok(rec) => {
                                            logger::log_info(&format!(
                                                "录音开始 (采样率: {}Hz)",
                                                rec.sample_rate()
                                            ));
                                            self.recorder = Some(rec);
                                            self.status_text =
                                                format!("录音中... 0.0s");
                                        }
                                        Err(e) => {
                                            logger::log_error(&format!("录音启动失败: {}", e));
                                            self.mic_error = e;
                                        }
                                    }
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
                                        egui::Button::new("停止录音并识别")
                                            .fill(egui::Color32::from_rgb(200, 80, 80)),
                                    )
                                    .clicked()
                                {
                                    logger::log_info("用户点击: 停止录音并识别");
                                    if let Some(rec) = self.recorder.take() {
                                        let (samples, sr) = rec.stop();
                                        if samples.is_empty() {
                                            logger::log_warn("录音为空，无数据可识别");
                                            self.mic_error = "录音为空".into();
                                        } else {
                                            // Resample to 16kHz if needed
                                            let samples_16k = if sr != 16000 {
                                                logger::log_info(&format!(
                                                    "重采样: {}Hz -> 16000Hz",
                                                    sr
                                                ));
                                                qwen_asr::audio::resample(
                                                    &samples,
                                                    sr as i32,
                                                    16000,
                                                )
                                            } else {
                                                samples
                                            };
                                            let dur = samples_16k.len() as f32 / 16000.0;
                                            logger::log_info(&format!(
                                                "录音完成 ({:.1}s, {} 样本), 开始识别",
                                                dur,
                                                samples_16k.len()
                                            ));
                                            self.status_text =
                                                format!("录音完成 ({:.1}s)，正在识别...", dur);
                                            self.worker.transcribe_samples(
                                                samples_16k,
                                                self.params.clone(),
                                            );
                                        }
                                    }
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

            // Result text area
            let text_style = egui::TextStyle::Monospace;
            let mut text = self.result_text.clone();
            let text_edit = egui::TextEdit::multiline(&mut text)
                .desired_width(f32::MAX)
                .desired_rows(10)
                .font(text_style)
                .interactive(true);
            ui.add(text_edit);
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
