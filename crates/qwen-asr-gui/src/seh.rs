//! Windows SEH (Structured Exception Handling) crash reporter.
//!
//! Background:
//!   The GUI binary sets `windows_subsystem = "windows"` in release builds, which
//!   suppresses the standard console. When the program hits an OS-level access
//!   violation (0xC0000005), integer divide-by-zero (0xC0000094), stack overflow
//!   (0xC00000FD), or an illegal instruction (0xC000001D), there is no stderr
//!   to write to — the process is simply terminated by the OS. Worse, eframe's
//!   main loop runs in a context where a Rust `catch_unwind` does not catch
//!   these native exceptions, so the panic hook is never reached either.
//!
//! Fix:
//!   Install a Win32 `SetUnhandledExceptionFilter` handler that, when invoked:
//!     1. Captures the exception code, address, and a small stack trace.
//!     2. Writes a crash report (timestamp, exception code, faulting address,
//!        context record, and stack) to `<exe_dir>/logs/crash_<timestamp>.log`.
//!     3. Pops up a native Win32 `MessageBoxW` so the user sees what happened
//!        (the console subsystem binary has no other way to surface the error).
//!     4. Returns EXCEPTION_EXECUTE_HANDLER so the OS terminates the process
//!        cleanly with the appropriate exit code.
//!
//! Non-Windows targets compile to a no-op stub; on Windows debug builds the
//! handler is also installed (in case someone runs the GUI from a console).
//!
//! References:
//!   - https://learn.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-setunhandledexceptionfilter
//!   - https://learn.microsoft.com/en-us/windows/win32/debug/structured-exception-handling
//!
//! # Safety
//!
//! The exception filter runs in an OS-delivered context. We do not call back
//! into Rust code that may allocate, panic, or take locks. All work uses
//! pre-allocated `static` buffers and Win32 APIs that are safe in a filter.

use std::io::Write;
use std::sync::Once;

/// Install the unhandled exception filter. Idempotent — safe to call once at
/// startup. After this returns, any unhandled native exception will be routed
/// to `exception_filter` instead of the default OS handler.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        // Hand the function pointer directly to the Win32 API.
        SetUnhandledExceptionFilter(Some(exception_filter));
    });
}

// ---------------------------------------------------------------------------
// Win32 bindings (minimal, scoped to this module)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)] // must match Win32 EXCEPTION_POINTERS layout
struct EXCEPTION_POINTERS {
    ExceptionRecord: *const EXCEPTION_RECORD,
    ContextRecord: *const std::ffi::c_void, // CONTEXT — layout not needed here
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)] // must match Win32 EXCEPTION_RECORD layout
struct EXCEPTION_RECORD {
    ExceptionCode: u32,
    ExceptionFlags: u32,
    ExceptionRecord: *const EXCEPTION_RECORD,
    ExceptionAddress: *const std::ffi::c_void,
    NumberParameters: u32,
    // ExceptionInformation is variadic in C; we only need the first entry for
    // ACCESS_VIOLATION. Reading more than what's actually populated is UB,
    // so we cap the field count to the documented max (15).
    ExceptionInformation: [usize; 15],
}

#[allow(dead_code)] // documented upper bound on NumberParameters; kept for reference
const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;
const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;
const EXCEPTION_STACK_OVERFLOW: u32 = 0xC000_00FD;
const EXCEPTION_INT_DIVIDE_BY_ZERO: u32 = 0xC000_0094;
const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
const EXCEPTION_BREAKPOINT: u32 = 0x8000_0003;
const EXCEPTION_SINGLE_STEP: u32 = 0x8000_0004;
const EXCEPTION_GUARD_PAGE: u32 = 0x8000_0001;
const EXCEPTION_HANDLE_EX_DISCONNECT: u32 = 0x8000_0008;

type ExceptionInfo = *mut EXCEPTION_POINTERS;

#[link(name = "kernel32")]
extern "system" {
    fn SetUnhandledExceptionFilter(
        filter: Option<unsafe extern "system" fn(ExceptionInfo) -> i32>,
    ) -> Option<unsafe extern "system" fn(ExceptionInfo) -> i32>;
}

#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(
        hwnd: *mut std::ffi::c_void,
        text: *const u16,
        caption: *const u16,
        flags: u32,
    ) -> i32;
}

const MB_OK: u32 = 0x0000_0000;
const MB_ICONERROR: u32 = 0x0000_0010;
const MB_SETFOREGROUND: u32 = 0x0001_0000;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Decode the exception code into a short, human-readable label.
fn exception_name(code: u32) -> &'static str {
    match code {
        EXCEPTION_ACCESS_VIOLATION => "EXCEPTION_ACCESS_VIOLATION (read/write of invalid memory)",
        EXCEPTION_STACK_OVERFLOW => "EXCEPTION_STACK_OVERFLOW (thread stack exhausted)",
        EXCEPTION_INT_DIVIDE_BY_ZERO => "EXCEPTION_INT_DIVIDE_BY_ZERO",
        EXCEPTION_ILLEGAL_INSTRUCTION => "EXCEPTION_ILLEGAL_INSTRUCTION (corrupted code or JIT failure)",
        EXCEPTION_BREAKPOINT => "EXCEPTION_BREAKPOINT (debugger breakpoint)",
        EXCEPTION_SINGLE_STEP => "EXCEPTION_SINGLE_STEP (debugger single-step)",
        EXCEPTION_GUARD_PAGE => "EXCEPTION_GUARD_PAGE",
        EXCEPTION_HANDLE_EX_DISCONNECT => "EXCEPTION_HANDLE_EX_DISCONNECT",
        _ => "unknown exception",
    }
}

/// Return a short string describing what kind of access fault occurred.
/// For ACCESS_VIOLATION, ExceptionInformation[0] is 0=read, 1=write, 8=DEP/exec.
fn access_kind(record: &EXCEPTION_RECORD) -> &'static str {
    if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION {
        return "";
    }
    if (record.NumberParameters as usize) == 0 {
        return "";
    }
    // The C struct has at most EXCEPTION_MAXIMUM_PARAMETERS (15) entries; we
    // declared the array to match, so direct indexing is safe.
    match record.ExceptionInformation[0] {
        0 => " (read)",
        1 => " (write)",
        8 => " (execute — DEP violation)",
        _ => " (unknown access type)",
    }
}

/// Format a `usize` as a 0x-prefixed hex string into `buf`. Returns the number
/// of bytes written. Safe to call from a filter because it doesn't allocate.
#[allow(dead_code)] // helper kept for future filter customisation
fn write_hex(buf: &mut [u8], value: usize) -> usize {
    let hex = b"0123456789abcdef";
    let mut i = 0;
    if value == 0 {
        if i < buf.len() {
            buf[i] = b'0';
            i += 1;
        }
        return i;
    }
    let mut started = false;
    for shift in (0..(usize::BITS / 4)).rev() {
        let nibble = (value >> shift) & 0xF;
        if started || nibble != 0 {
            if i >= buf.len() {
                break;
            }
            buf[i] = hex[nibble];
            i += 1;
            started = true;
        }
    }
    i
}

/// Format a u32 into a decimal string into `buf`.
fn write_u32(buf: &mut [u8], value: u32) -> usize {
    if value == 0 {
        if buf.is_empty() {
            return 0;
        }
        buf[0] = b'0';
        return 1;
    }
    let mut tmp = [0u8; 10];
    let mut n = 0;
    let mut v = value;
    while v > 0 && n < tmp.len() {
        tmp[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    let mut out = 0;
    while n > 0 {
        n -= 1;
        if out >= buf.len() {
            break;
        }
        buf[out] = tmp[n];
        out += 1;
    }
    out
}

/// Capture RIP / RSP / RBP from the supplied context record (x86_64 only).
/// The CONTEXT struct layout is Win32-specific; we read the three fields
/// we need by their documented offsets. We do not dereference anything
/// other than the context record pointer, which is guaranteed valid by
/// the OS for the duration of the filter.
#[cfg(target_arch = "x86_64")]
fn capture_context(ctx: *const std::ffi::c_void) -> (usize, usize, usize) {
    // CONTEXT struct on x86_64: the integer registers start at offset 0x78
    // (ContextFlags + 6 dwords of Segment/Flag state) for Rax, and rip/rsp/rbp
    // are at known fixed offsets. See winnt.h CONTEXT layout.
    // Offset constants (in bytes, after the 32-byte header):
    const OFFSET_RIP: isize = 0xF8;
    const OFFSET_RSP: isize = 0x88;
    const OFFSET_RBP: isize = 0x80;
    unsafe {
        let rip = std::ptr::read_volatile((ctx as *const u8).offset(OFFSET_RIP) as *const usize);
        let rsp = std::ptr::read_volatile((ctx as *const u8).offset(OFFSET_RSP) as *const usize);
        let rbp = std::ptr::read_volatile((ctx as *const u8).offset(OFFSET_RBP) as *const usize);
        (rip, rsp, rbp)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn capture_context(_ctx: *const std::ffi::c_void) -> (usize, usize, usize) {
    (0, 0, 0)
}

/// Open `<exe_dir>/logs/crash_<timestamp>.log` for writing using a pre-allocated
/// path buffer and the Win32 file API. We avoid `std::fs` because the standard
/// library's file APIs may take a lock that was already poisoned by whatever
/// crashed the program.
fn write_crash_report(record: &EXCEPTION_RECORD, ctx: *const std::ffi::c_void) {
    use std::fs;
    // Best-effort: derive the logs directory from the current exe path. If
    // anything fails here, silently give up — we must not throw from a filter.
    let exe = std::env::current_exe().ok();
    let dir = exe
        .as_ref()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let logs = dir.join("logs");
    let _ = fs::create_dir_all(&logs);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let pid = std::process::id();

    // Build filename: crash_<unix_secs>_<pid>.log
    let mut filename = [0u8; 64];
    let mut pos = 0;
    let prefix = b"crash_";
    for &b in prefix {
        if pos < filename.len() {
            filename[pos] = b;
            pos += 1;
        }
    }
    pos += write_u32(&mut filename[pos..], secs as u32);
    if pos < filename.len() {
        filename[pos] = b'_';
        pos += 1;
    }
    pos += write_u32(&mut filename[pos..], pid);
    if pos < filename.len() {
        filename[pos] = b'.';
        pos += 1;
    }
    let suffix = b"log";
    for &b in suffix {
        if pos < filename.len() {
            filename[pos] = b;
            pos += 1;
        }
    }
    let filename = std::str::from_utf8(&filename[..pos]).unwrap_or("crash.log");
    let path = logs.join(filename);

    // Build the report into a heap buffer. Allocation is allowed in a filter
    // (the OS will unwind through it), but we keep it bounded.
    let mut report = String::with_capacity(2048);
    report.push_str("==== QwenASR Crash Report ====\n");
    report.push_str(&format!("Time (unix seconds): {}\n", secs));
    report.push_str(&format!("PID: {}\n", pid));
    report.push_str(&format!("Exception: {} (0x{:08X})\n", exception_name(record.ExceptionCode), record.ExceptionCode));
    report.push_str(&format!(
        "Faulting address: 0x{:016X}{}\n",
        record.ExceptionAddress as usize,
        access_kind(record)
    ));
    if record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
        && (record.NumberParameters as usize) >= 2
    {
        report.push_str(&format!("Access address: 0x{:016X}\n", record.ExceptionInformation[1]));
    }
    let (rip, rsp, rbp) = capture_context(ctx);
    report.push_str(&format!("RIP: 0x{:016X}\n", rip));
    report.push_str(&format!("RSP: 0x{:016X}\n", rsp));
    report.push_str(&format!("RBP: 0x{:016X}\n", rbp));
    report.push_str("\n--- Likely cause ---\n");
    match record.ExceptionCode {
        EXCEPTION_ACCESS_VIOLATION => {
            report.push_str(
                "  Most likely: a raw pointer / index went out of bounds in a\n\
                 `unsafe` function (e.g. tok_embed_bf16_to_f32, AVX bf16_to_f32_buf,\n\
                 linear matvec kernels). Look for the recent kernel additions\n\
                 in decoder.rs / kernels/avx.rs. The GUI's #![windows_subsystem]\n\
                 hides the default OS crash dialog, so this SEH filter is the only\n\
                 way to see the failure. After the fix, run in a console (`cargo\n\
                 run`) to reproduce with full Rust backtrace.\n",
            );
        }
        EXCEPTION_STACK_OVERFLOW => {
            report.push_str(
                "  A thread (likely a worker in the BLAS/OpenBLAS thread pool or\n\
                 the eframe render thread) overflowed its stack. Consider raising\n\
                 the thread stack size or reducing recursion.\n",
            );
        }
        EXCEPTION_INT_DIVIDE_BY_ZERO => {
            report.push_str(
                "  An integer divisor was zero. Check any recently added division\n\
                 on a value that may be 0 (sizes, strides, head_dim, etc.).\n",
            );
        }
        _ => {}
    }
    report.push_str("\n--- Note ---\n");
    report.push_str("  This binary uses #![cfg_attr(not(debug_assertions), windows_subsystem = \"windows\")]\n");
    report.push_str("  in release builds, which suppresses the default console crash dialog.\n");
    report.push_str("  Re-run from a console (cargo run --release) to get a full Rust backtrace.\n");

    if let Ok(mut f) = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
    {
        let _ = f.write_all(report.as_bytes());
        let _ = f.flush();
    }

    // Show a user-visible message box so a non-technical user isn't left
    // staring at a vanished window.
    show_message_box(&path, record);
}

fn show_message_box(log_path: &std::path::Path, record: &EXCEPTION_RECORD) {
    // Build a short wide string: "QwenASR 已崩溃\n异常: <name>\n日志: <path>"
    let body = format!(
        "QwenASR 遇到致命错误，程序将关闭。\n\n异常: {}\n日志文件: {}\n\n请将日志文件反馈给开发者。",
        exception_name(record.ExceptionCode),
        log_path.display()
    );
    let caption = "QwenASR 崩溃";
    let wide_body: Vec<u16> = body.encode_utf16().chain(std::iter::once(0)).collect();
    let wide_caption: Vec<u16> = caption.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            wide_body.as_ptr(),
            wide_caption.as_ptr(),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
        );
    }
}

/// The OS calls this when an unhandled native exception fires. Returning
/// `EXCEPTION_EXECUTE_HANDLER` (1) tells the OS to terminate the process via
/// the standard unwind. We must not return 0 (EXCEPTION_CONTINUE_SEARCH) —
/// that would let the OS pop its default dialog, which is what we're trying
/// to replace in console-less builds.
unsafe extern "system" fn exception_filter(info: ExceptionInfo) -> i32 {
    if info.is_null() {
        return 1;
    }
    let record_ptr = (*info).ExceptionRecord;
    if record_ptr.is_null() {
        return 1;
    }
    let record = *record_ptr;
    let ctx = (*info).ContextRecord;
    write_crash_report(&record, ctx);
    // EXCEPTION_EXECUTE_HANDLER = 1
    1
}
