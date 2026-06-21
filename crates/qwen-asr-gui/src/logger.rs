//! GUI 操作日志模块
//! 记录用户操作和系统事件到日志文件，供监控脚本分析

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;
use crate::sync_ext::safe_lock;

/// Windows SYSTEMTIME 结构体
#[cfg(windows)]
#[repr(C)]
struct SystemTime {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    milliseconds: u16,
}

#[cfg(windows)]
extern "system" {
    fn GetLocalTime(lpSystemTime: *mut SystemTime);
}

/// 获取本地时间
fn now() -> SystemTime {
    #[cfg(windows)]
    {
        let mut st = SystemTime {
            year: 1970,
            month: 1,
            day_of_week: 4,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            milliseconds: 0,
        };
        unsafe {
            GetLocalTime(&mut st);
        }
        st
    }
    #[cfg(not(windows))]
    {
        SystemTime {
            year: 1970,
            month: 1,
            day_of_week: 4,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            milliseconds: 0,
        }
    }
}

/// 格式化时间戳: YYYY-MM-DD HH:MM:SS.mmm
fn timestamp() -> String {
    let t = now();
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        t.year, t.month, t.day, t.hour, t.minute, t.second, t.milliseconds
    )
}

/// 获取日期字符串（用于日志文件名）: YYYYMMDD
fn date_str() -> String {
    let t = now();
    format!("{:04}{:02}{:02}", t.year, t.month, t.day)
}

/// 日志文件目录
static LOG_DIR: Mutex<Option<String>> = Mutex::new(None);

/// 初始化日志系统
/// 日志文件存放在 exe 同目录下的 logs/ 文件夹
pub fn init() {
    let exe_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let log_dir = exe_path.join("logs");
    if !log_dir.exists() {
        let _ = std::fs::create_dir_all(&log_dir);
    }
    *safe_lock(&LOG_DIR) = Some(log_dir.to_string_lossy().to_string());
    log_info("=== GUI 启动 ===");
}

/// 记录 INFO 级别日志
pub fn log_info(msg: &str) {
    log_event("INFO", msg);
}

/// 记录 ERROR 级别日志
pub fn log_error(msg: &str) {
    log_event("ERROR", msg);
}

/// 记录 WARN 级别日志
pub fn log_warn(msg: &str) {
    log_event("WARN", msg);
}

/// 安装 panic hook，捕获崩溃信息到日志文件
/// 必须在 main() 早期调用，确保所有 panic 都能被记录
pub fn setup_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // 记录 panic 到日志文件
        let msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());

        log_error(&format!("PANIC: {} at {}", msg, location));

        // 调用默认 hook（输出到 stderr）
        default_hook(panic_info);
    }));
}

/// 记录日志事件到文件
fn log_event(level: &str, msg: &str) {
    let ts = timestamp();
    let line = format!("[{}] [{}] {}", ts, level, msg);

    // 写入日志文件
    if let Some(ref dir) = *safe_lock(&LOG_DIR) {
        let filename = format!("gui_{}.log", date_str());
        let path = std::path::Path::new(dir).join(&filename);
        if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(&path) {
            let _ = writeln!(f, "{}", line);
        }
    }
}
