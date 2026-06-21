//! Qwen3-ASR GUI — egui-based desktop interface
//!
//! Features:
//! - Load model button with directory picker
//! - Parameter settings panel with recommended defaults
//! - Reset to defaults button
//! - Status display with copy/clear for results

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;

mod app;
mod logger;
mod params;
mod recorder;
#[cfg(windows)]
mod seh;
mod sync_ext;
mod worker;

use app::AsrApp;

fn main() -> eframe::Result {
    // Install Windows SEH crash reporter FIRST so that even panics during
    // logger init, or native exceptions (access violations, stack overflow,
    // etc.) from inside the kernels, are captured and reported. This is the
    // single most important fix for the GUI: without it, any access violation
    // inside an `unsafe` kernel disappears silently because the release binary
    // uses #![windows_subsystem = "windows"] and has no console to write to.
    #[cfg(windows)]
    seh::install();

    // 安装 panic hook（必须在日志系统初始化之前）
    logger::setup_panic_hook();

    // 初始化日志系统
    logger::init();

    // Trigger INT8 kernel selection early so the chosen path is recorded in
    // the GUI log *before* the first transcription. If a later crash happens
    // the SEH report + this log together pinpoint whether the AVX-VNNI
    // path was active (the most common cause of EXCEPTION_ILLEGAL_INSTRUCTION
    // on hybrid Intel CPUs and inside some VMs/WSL).
    if std::env::var("QWEN_ASR_DISABLE_VNNI").is_ok() {
        logger::log_info("QWEN_ASR_DISABLE_VNNI 已设置 → 强制走 AVX2 路径");
    } else {
        logger::log_info("QWEN_ASR_DISABLE_VNNI 未设置 → 按 CPUID 自动选择（AVX-VNNI/AVX2）");
    }

    // Parse --model <path> from command line for auto-loading
    let model_path: Option<String> = {
        let args: Vec<String> = std::env::args().collect();
        let mut path = None;
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--model" && i + 1 < args.len() {
                path = Some(args[i + 1].clone());
                break;
            }
            i += 1;
        }
        path
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 680.0])
            .with_min_inner_size([700.0, 500.0])
            .with_title("Qwen3-ASR 语音识别"),
        ..Default::default()
    };

    eframe::run_native(
        "Qwen3-ASR GUI",
        options,
        Box::new(|cc| {
            // Set look & feel
            egui_extras(cc);
            Ok(Box::new(AsrApp::new(model_path)))
        }),
    )
}

/// Configure egui visual style
fn egui_extras(cc: &eframe::CreationContext<'_>) {
    let mut style = egui::Style::default();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    cc.egui_ctx.set_style(style);

    let mut fonts = egui::FontDefinitions::default();
    // Use system CJK font if available
    if let Some(font) = load_cjk_font() {
        fonts.font_data.insert("cjk".to_owned(), font);
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "cjk".to_owned());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, "cjk".to_owned());
    }
    cc.egui_ctx.set_fonts(fonts);
}

/// Try to load a CJK-capable system font on Windows
fn load_cjk_font() -> Option<egui::FontData> {
    #[cfg(target_os = "windows")]
    {
        let font_paths = [
            "C:\\Windows\\Fonts\\msyh.ttc",    // Microsoft YaHei
            "C:\\Windows\\Fonts\\msyhbd.ttc",   // YaHei Bold
            "C:\\Windows\\Fonts\\simhei.ttf",   // SimHei
            "C:\\Windows\\Fonts\\simsun.ttc",  // SimSun
        ];
        for path in &font_paths {
            if let Ok(data) = std::fs::read(path) {
                return Some(egui::FontData::from_owned(data));
            }
        }
    }
    None
}
