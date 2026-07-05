//! GUI 操作日志模块
//! 记录用户操作和系统事件到日志文件，供监控脚本分析

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};
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

/// Singleton buffered writer. We hold a single `BufWriter<File>` for
/// the lifetime of the process and reuse it across every log call,
/// instead of opening/closing the file on every call (the old design
/// cost ~50 µs per call = ~50 ms wasted over 1000 log events during a
/// 20-minute recognition run, plus the syscall overhead of
/// `OpenOptions::open` + `Drop` on Windows). The buffer is flushed
/// every call (`get_mut` + `flush`) — log volume in the GUI is low
/// (a few events per second) so this is cheaper than a periodic
/// timer-based flush, and it guarantees the line is on disk before
/// the next event can crash the process.
static LOG_WRITER: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();

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

    // Eagerly open the singleton writer so the first log_info call
    // doesn't have to do it (avoids a ~5 ms cold-start blip). If
    // the open fails we leave LOG_WRITER unset and log_event will
    // fall back to per-call open/close — strictly worse but still
    // functional.
    let filename = format!("gui_{}.log", date_str());
    let path = log_dir.join(&filename);
    if let Ok(file) = OpenOptions::new().append(true).create(true).open(&path) {
        let _ = LOG_WRITER.set(Mutex::new(Some(BufWriter::new(file))));
    }

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

    // Fast path: reuse the singleton buffered writer opened in
    // init(). The lock is uncontended in practice (worker thread +
    // panic hook + render thread can all log, but each is brief).
    if let Some(slot) = LOG_WRITER.get() {
        if let Ok(mut guard) = slot.lock() {
            if let Some(ref mut w) = *guard {
                if writeln!(w, "{}", line).is_ok() {
                    // Flush eagerly — log volume is low and we want
                    // the line on disk before any subsequent panic.
                    let _ = w.flush();
                    return;
                }
            }
        }
    }

    // Fallback: open/close the file on every call. Worse, but keeps
    // logging functional if init() couldn't open the file (e.g.
    // permission denied, full disk).
    if let Some(ref dir) = *safe_lock(&LOG_DIR) {
        let filename = format!("gui_{}.log", date_str());
        let path = std::path::Path::new(dir).join(&filename);
        if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(&path) {
            let _ = writeln!(f, "{}", line);
        }
    }
}
