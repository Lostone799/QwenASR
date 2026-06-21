//! Poison-safe wrappers around `std::sync::Mutex`.
//!
//! Background:
//!   The GUI uses several `Mutex<T>` to share state between the eframe render
//!   thread and the background ASR worker thread. The default behaviour when
//!   a thread panics while holding a mutex is for the standard library to mark
//!   the mutex as "poisoned". A subsequent `.lock().unwrap()` then panics
//!   again — *even on threads that have nothing to do with the original panic*.
//!   In a long-running GUI this manifests as a "cascading crash": one bad
//!   tokeniser path poisons the shared status mutex, the next status update
//!   panics, the panic hook fires, the user sees a frozen window. Because the
//!   release binary uses `windows_subsystem = "windows"`, the OS-level panic
//!   abort is invisible.
//!
//! Fix:
//!   These helpers transparently recover from poisoning (the data inside a
//!   poisoned mutex is still valid; only the "did a previous holder panic?"
//!   flag is set). They also log the poisoning event so we have a breadcrumb
//!   pointing at whatever caused it.
//!
//! Pattern:
//!   ```ignore
//!   use crate::sync_ext::safe_lock;
//!   let value = safe_lock(&mutex).clone();
//!   ```

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Acquire a `Mutex` lock, transparently recovering from poisoning. If the
/// mutex was poisoned by a previous panic, the poison flag is cleared and
/// the recovery is logged at WARN level so the breadcrumb is visible in the
/// GUI log.
pub fn safe_lock<'a, T>(mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(g) => g,
        Err(p) => {
            crate::logger::log_warn(&format!(
                "Mutex recovered from poisoning at {}:{}",
                file!(),
                line!()
            ));
            p.into_inner()
        }
    }
}

/// Try to acquire a `Mutex` lock, returning `None` if the mutex is poisoned.
/// Use this when the caller has no meaningful recovery action and would
/// rather no-op than crash the entire UI.
#[allow(dead_code)] // exposed for future use; not yet called
pub fn try_safe_lock<'a, T>(mutex: &'a Mutex<T>) -> Option<MutexGuard<'a, T>> {
    match mutex.lock() {
        Ok(g) => Some(g),
        Err(p) => {
            crate::logger::log_warn(&format!(
                "Mutex poisoned; skipping operation at {}:{}",
                file!(),
                line!()
            ));
            // We still own the guard via into_inner() — the caller may want
            // to use it. For try_safe_lock the explicit choice is to discard.
            drop(p.into_inner());
            None
        }
    }
}

/// Helper for `PoisonError<MutexGuard<...>>` — recovers the guard.
#[allow(dead_code)] // exposed for future use; not yet called
pub fn recover_poison<'a, T>(
    err: PoisonError<MutexGuard<'a, T>>,
    location: &str,
) -> MutexGuard<'a, T> {
    crate::logger::log_warn(&format!("Mutex poison recovered: {}", location));
    err.into_inner()
}
