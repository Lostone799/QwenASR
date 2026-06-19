//! BLAS/vDSP bindings, thread pool, and SIMD kernel dispatch.

pub mod generic;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod avx;

use std::thread;

// BLAS extern bindings
#[cfg(all(feature = "blas", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

// vDSP/vForce bindings (macOS Accelerate)
#[cfg(all(feature = "vdsp", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vDSP_dotpr(
        a: *const f32, a_stride: i32,
        b: *const f32, b_stride: i32,
        result: *mut f32,
        n: u64,
    );
    fn vDSP_vsmul(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vDSP_vsma(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        b: *const f32, b_stride: i32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vvexpf(dst: *mut f32, src: *const f32, n: *const i32);
}

#[cfg(all(feature = "blas", not(target_vendor = "apple")))]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(feature = "blas")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(feature = "blas")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(feature = "blas")]
const CBLAS_TRANS: i32 = 112;

// Verbose flag
static VERBOSE: AtomicI32 = AtomicI32::new(0);

// ========================================================================
// Profiling support
// ========================================================================

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::time::Instant;

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_profile(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn is_profiling() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

macro_rules! define_profile_counters {
    ($($name:ident),+) => {
        pub struct ProfileCounters {
            $(pub $name: (AtomicU64, AtomicU64),)+ // (total_ns, call_count)
        }

        impl ProfileCounters {
            pub const fn new() -> Self {
                ProfileCounters {
                    $($name: (AtomicU64::new(0), AtomicU64::new(0)),)+
                }
            }
        }

        impl Default for ProfileCounters {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ProfileCounters {
            pub fn reset(&self) {
                $(
                    self.$name.0.store(0, Ordering::Relaxed);
                    self.$name.1.store(0, Ordering::Relaxed);
                )+
            }

            pub fn report(&self) {
                $(
                    let ns = self.$name.0.load(Ordering::Relaxed);
                    let calls = self.$name.1.load(Ordering::Relaxed);
                    if calls > 0 {
                        let ms = ns as f64 / 1_000_000.0;
                        let avg = ms / calls as f64;
                        eprintln!("[profile] {}: {:.1}ms ({} calls, {:.2}ms avg)",
                                  stringify!($name), ms, calls, avg);
                    }
                )+
            }
        }
    }
}

define_profile_counters!(
    rms_norm, layer_norm, gelu, swiglu,
    bf16_matvec, bf16_to_f32_conv, attention_bidir, attention_causal,
    sgemm, conv2d_op, rope, add_inplace_op,
    model_load, encoder_load, decoder_load, tokenizer_load, audio_load, mel_compute,
    lm_head, rms_norm_per_head_op, kv_cache_write, dec_forward_total, enc_forward_total, dec_prefill_total
);

pub static PROF: ProfileCounters = ProfileCounters::new();

pub struct ProfileGuard {
    start: Instant,
    counter: &'static (AtomicU64, AtomicU64),
}

impl ProfileGuard {
    #[inline]
    pub fn new(counter: &'static (AtomicU64, AtomicU64)) -> Option<Self> {
        if PROFILE_ENABLED.load(Ordering::Relaxed) {
            Some(ProfileGuard { start: Instant::now(), counter })
        } else {
            None
        }
    }
}

impl Drop for ProfileGuard {
    #[inline]
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.counter.0.fetch_add(ns, Ordering::Relaxed);
        self.counter.1.fetch_add(1, Ordering::Relaxed);
    }
}

// Convenience: unused ProfileTimer alias removed

pub fn profile_reset() { PROF.reset(); }
pub fn profile_report() { PROF.report(); }

pub fn set_verbose(v: i32) {
    VERBOSE.store(v, Ordering::Relaxed);
}

pub fn verbose() -> i32 {
    VERBOSE.load(Ordering::Relaxed)
}

// ========================================================================
// Thread Pool (persistent, mutex+condvar, matches C approach)
// ========================================================================

use std::sync::{Mutex, Condvar, Arc, OnceLock};

const MAX_THREADS: usize = 16;

struct ThreadPool {
    // Mutex+condvar only used as slow-path fallback when spin-wait misses
    state: Mutex<bool>, // shutdown flag only
    work_cv: Condvar,
    // All dispatch data is lock-free via atomics
    gen_atomic: AtomicU64,
    done_atomic: AtomicUsize,
    fn_ptr_atomic: AtomicUsize,
    fn_call_atomic: AtomicUsize,
    n_threads_atomic: AtomicUsize,
}

static THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

fn get_pool() -> &'static Arc<ThreadPool> {
    THREAD_POOL.get_or_init(|| {
        Arc::new(ThreadPool {
            state: Mutex::new(false),
            work_cv: Condvar::new(),
            gen_atomic: AtomicU64::new(0),
            done_atomic: AtomicUsize::new(0),
            fn_ptr_atomic: AtomicUsize::new(0),
            fn_call_atomic: AtomicUsize::new(0),
            n_threads_atomic: AtomicUsize::new(1),
        })
    })
}

fn pool_worker(pool: Arc<ThreadPool>, tid: usize) {
    let mut last_gen: u64 = 0;
    loop {
        // Fast path: spin briefly on atomic generation counter
        let mut found = false;
        for _ in 0..512 {
            let gen = pool.gen_atomic.load(Ordering::Acquire);
            if gen != last_gen {
                last_gen = gen;
                found = true;
                break;
            }
            core::hint::spin_loop();
        }

        if !found {
            // Slow path: condvar wait (mutex only protects shutdown flag)
            let mut shutdown = match pool.state.lock() {
                Ok(s) => s,
                Err(p) => p.into_inner(),
            };
            while !*shutdown && pool.gen_atomic.load(Ordering::Relaxed) == last_gen {
                shutdown = match pool.work_cv.wait(shutdown) {
                    Ok(s) => s,
                    Err(p) => p.into_inner(),
                };
            }
            if *shutdown {
                return;
            }
            last_gen = pool.gen_atomic.load(Ordering::Acquire);
        }

        // Read dispatch data from atomics (ordered by gen_atomic Acquire)
        let fn_ptr = pool.fn_ptr_atomic.load(Ordering::Relaxed) as *const ();
        let fn_call: fn(*const (), usize, usize) = unsafe {
            core::mem::transmute(pool.fn_call_atomic.load(Ordering::Relaxed))
        };
        let n_threads = pool.n_threads_atomic.load(Ordering::Relaxed);

        fn_call(fn_ptr, tid, n_threads);
        pool.done_atomic.fetch_add(1, Ordering::Release);
    }
}

static SPAWNED_THREADS: AtomicUsize = AtomicUsize::new(0);

fn ensure_workers(pool: &Arc<ThreadPool>, n_threads: usize) {
    let spawned = SPAWNED_THREADS.load(Ordering::Relaxed);
    if spawned >= n_threads - 1 {
        return;
    }
    let start = spawned + 1;
    for tid in start..n_threads {
        let p = pool.clone();
        thread::Builder::new()
            .name(format!("qwen-worker-{}", tid))
            .spawn(move || pool_worker(p, tid))
            .expect("failed to spawn worker thread");
    }
    SPAWNED_THREADS.store(n_threads - 1, Ordering::Relaxed);
}

static THREAD_POOL_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_threads(n: usize) {
    let n = n.clamp(1, MAX_THREADS);
    THREAD_POOL_THREADS.store(n, Ordering::Relaxed);
    if n > 1 {
        let pool = get_pool();
        ensure_workers(pool, n);
    }
    // Set OpenBLAS thread count via env var (DLL only exports cblas_sgemm).
    // Benchmark: 8 OpenBLAS threads optimal with 12-pool (avoids
    // oversubscription contention with bf16_matvec custom kernels).
    #[cfg(all(feature = "blas", not(target_vendor = "apple")))]
    {
        let blas_threads = (n * 2 / 3).max(1).min(8);
        // SAFETY: single-threaded init path before any BLAS call
        unsafe { std::env::set_var("OPENBLAS_NUM_THREADS", blas_threads.to_string()); }
    }
    if verbose() >= 2 {
        eprintln!("Thread pool: {} threads", n);
    }
}

pub fn get_num_threads() -> usize {
    THREAD_POOL_THREADS.load(Ordering::Relaxed)
}

pub fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Number of high-performance physical cores. On Apple Silicon this is
/// `hw.perflevel0.physicalcpu` (P-cores). With the batched-attention decode
/// path, both the encoder and (especially) the bandwidth-bound single-token
/// decode run faster on the P-cores alone — adding efficiency cores increases
/// dispatch and memory-bus contention and slows every measured mode. Falls back
/// to total CPU count on other platforms / on failure.
pub fn get_num_perf_cpus() -> usize {
    #[cfg(target_os = "macos")]
    {
        let mut out: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        let name = b"hw.perflevel0.physicalcpu\0";
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                &mut out as *mut i32 as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && out > 0 {
            return out as usize;
        }
    }
    get_num_cpus()
}

/// Run a closure in parallel using the persistent thread pool.
/// The closure takes (thread_id, n_threads).
fn parallel_for<F: Fn(usize, usize) + Send + Sync>(f: F) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        f(0, 1);
        return;
    }

    let pool = get_pool();

    // Trampoline: cast *const () back to &F and call it
    fn trampoline<F: Fn(usize, usize) + Send + Sync>(ptr: *const (), tid: usize, nt: usize) {
        let f = unsafe { &*(ptr as *const F) };
        f(tid, nt);
    }

    // Publish dispatch data via atomics (Relaxed OK: gen_atomic Release provides ordering)
    pool.done_atomic.store(0, Ordering::Relaxed);
    pool.fn_ptr_atomic.store(&f as *const F as *const () as usize, Ordering::Relaxed);
    pool.fn_call_atomic.store(trampoline::<F> as usize, Ordering::Relaxed);
    pool.n_threads_atomic.store(n_threads, Ordering::Relaxed);
    // Release: ensures all stores above are visible to workers that Acquire gen_atomic
    pool.gen_atomic.fetch_add(1, Ordering::Release);

    // Wake workers that fell through to condvar wait
    // Lock scope is minimal: just notify, no data to write
    {
        let _guard = match pool.state.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        pool.work_cv.notify_all();
    }

    // Main thread does tid=0
    f(0, n_threads);

    // Wait for workers: spin on atomic done counter
    let expected = n_threads - 1;
    loop {
        if pool.done_atomic.load(Ordering::Acquire) >= expected {
            break;
        }
        core::hint::spin_loop();
    }
}

// ========================================================================
// Parallel dispatch helpers — eliminate repeated chunk-splitting boilerplate
// ========================================================================

/// Compute the `[start, end)` range for thread `tid` when splitting `total` items
/// across `nt` threads using div-ceil chunking. Returns `None` if the thread has
/// no work (start >= end).
#[inline(always)]
fn parallel_chunk_range(total: usize, tid: usize, nt: usize) -> Option<(usize, usize)> {
    let chunk = total.div_ceil(nt);
    let start = tid * chunk;
    let end = (start + chunk).min(total);
    if start >= end { None } else { Some((start, end)) }
}

/// Reduce per-thread argmax results into a single global argmax.
/// `indices` and `vals` must have at least `n` elements.
#[inline(always)]
fn reduce_argmax(indices: &[usize], vals: &[f32], n: usize) -> usize {
    let mut best = indices[0];
    let mut best_val = vals[0];
    for i in 1..n {
        if vals[i] > best_val {
            best_val = vals[i];
            best = indices[i];
        }
    }
    best
}

// ========================================================================
// INT8 kernel dispatch — unify aarch64 (NEON) and x86_64 (AVX2/VNNI) paths
// ========================================================================

/// INT8 matvec kernel function signature.
/// Computes `y[out_dim] = W_int8[out_dim, in_dim] @ x_int8[in_dim] * x_scale * w_scales[out_dim] + bias`,
/// applying the XOR-0x80 sign-flip correction using precomputed `w_sums`.
type Int8MatvecFn = unsafe fn(&mut [f32], *const i8, f32, *const i8, &[f32], &[i32], Option<&[f32]>, usize, usize);

/// INT8 argmax kernel function signature.
/// Returns `(argmax_index, max_value)` over rows `[start, end)` of `W_int8`.
type Int8ArgmaxFn = unsafe fn(*const i8, f32, *const i8, &[f32], &[i32], usize, usize, usize) -> (usize, f32);

/// Select the best INT8 matvec kernel for the current CPU.
#[cfg(target_arch = "aarch64")]
#[inline]
fn select_int8_matvec_fn() -> Int8MatvecFn { neon::matvec_int8 }

/// Select the best INT8 matvec kernel for the current CPU (AVX2 or AVX-VNNI).
/// Returns `None` if no AVX2 support (caller must fall back to f32 path).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn select_int8_matvec_fn() -> Option<Int8MatvecFn> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    Some(if is_x86_feature_detected!("avxvnni") {
        avx::matvec_int8_vnni
    } else {
        avx::matvec_int8_avx2
    })
}

/// Select the best INT8 argmax kernel for the current CPU.
#[cfg(target_arch = "aarch64")]
#[inline]
fn select_int8_argmax_fn() -> Int8ArgmaxFn { neon::argmax_int8_range }

/// Select the best INT8 argmax kernel for the current CPU (AVX2 or AVX-VNNI).
/// Returns `None` if no AVX2 support.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn select_int8_argmax_fn() -> Option<Int8ArgmaxFn> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    Some(if is_x86_feature_detected!("avxvnni") {
        avx::argmax_int8_range_vnni
    } else {
        avx::argmax_int8_range_avx2
    })
}

/// Fallback: dequantize INT8 weights to f32 and run f32 linear matvec.
fn int8_fallback_matvec(y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let mut w_f32 = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let scale = w_scales[row];
        for col in 0..in_dim {
            w_f32[row * in_dim + col] = w_int8[row * in_dim + col] as f32 * scale;
        }
    }
    linear(y, x, &w_f32, bias, 1, in_dim, out_dim);
}

/// Dequantize INT8 weights to f32 (used by QKV/SwiGLU fallback paths).
fn int8_to_f32_weights(w_int8: &[i8], w_scales: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let scale = w_scales[r];
        for c in 0..cols {
            out[r * cols + c] = w_int8[r * cols + c] as f32 * scale;
        }
    }
    out
}

// ========================================================================
// Dispatch helpers - pick NEON/AVX/generic at compile time
// ========================================================================

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

pub fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_to_f32_buf(dst, src); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_to_f32_buf(dst, src); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..src.len() {
        dst[i] = bf16_to_f32(src[i]);
    }
}

fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
}

fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::argmax_bf16_range(x, w_bf16, in_dim, start, end)
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        let mut result = 0.0f32;
        unsafe { vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, n as u64); }
        result
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::dot_f32(a, b, n) } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::dot_f32(a, b, n) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::dot_f32(a, b, n)
}

#[inline]
pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsmul(dst.as_ptr(), 1, &scale, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_scale_inplace(dst, scale, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_scale_inplace(dst, scale, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_scale_inplace(dst, scale, n);
}

#[inline]
pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsma(src.as_ptr(), 1, &alpha, dst.as_ptr(), 1, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_axpy_inplace(dst, src, alpha, n);
}

#[inline]
pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_scale_add(dst, src, correction, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_scale_add(dst, src, correction, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_scale_add(dst, src, correction, n);
}

// ========================================================================
// Basic Operations
// ========================================================================

pub fn add_inplace(a: &mut [f32], b: &[f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.add_inplace_op);
    for i in 0..n { a[i] += b[i]; }
}

// ========================================================================
// Matrix Operations
// ========================================================================

/// C = A @ B (no transpose): `A[M,K]`, `B[K,N]`, `C[M,N]`
pub fn matmul_nn(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ki * n + ni];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// C = A @ B^T: `A[M,K]`, `B[N,K]`, `C[M,N]`
pub fn matmul_t(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ni * k + ki];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// y = x @ W^T + b: `x[seq,in]`, `W[out,in]`, `b[out]`, `y[seq,out]`
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            0.0, y.as_mut_ptr(), out_dim as i32,
        );
        if let Some(b) = b {
            for s in 0..seq_len {
                for o in 0..out_dim {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |b| b[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] = sum;
            }
        }
    }
}

pub fn linear_nobias(y: &mut [f32], x: &[f32], w: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) {
    linear(y, x, w, None, seq_len, in_dim, out_dim);
}

/// y += bias + x @ w.T  (accumulate into existing y, fusing residual add)
pub fn linear_accumulate(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        // Add bias to y first (y already has residual)
        if let Some(b) = b {
            for s in 0..seq_len {
                let row = &mut y[s * out_dim..(s + 1) * out_dim];
                for o in 0..out_dim {
                    row[o] += b[o];
                }
            }
        }
        // y = 1.0 * x @ w.T + 1.0 * y  (accumulate matmul into y)
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            1.0, y.as_mut_ptr(), out_dim as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |bb| bb[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] += sum;
            }
        }
    }
}

fn bf16_to_f32_view(src: *const u16, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; n];
    let src_slice = unsafe { std::slice::from_raw_parts(src, n) };
    bf16_to_f32_buf(&mut buf, src_slice);
    buf
}

/// Threaded bf16 matvec
fn bf16_matvec_threaded(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
        return;
    }

    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();
    let w_ptr = w_bf16;
    let bias_ptr = bias.map(|b| b.as_ptr());

    // SAFETY: Each thread writes to non-overlapping segments of y
    let y_send = y_ptr as usize;
    let x_send = x_ptr as usize;
    let w_send = w_ptr as usize;
    let bias_send = bias_ptr.map(|p| p as usize);

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(out_dim, tid, nt) {
            Some(r) => r,
            None => return,
        };

        let y_local = unsafe { std::slice::from_raw_parts_mut((y_send as *mut f32).add(start), end - start) };
        let x_local = unsafe { std::slice::from_raw_parts(x_send as *const f32, in_dim) };
        let w_local = unsafe { (w_send as *const u16).add(start * in_dim) };
        let bias_local = bias_send.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), end - start) });

        bf16_matvec_fused(y_local, x_local, w_local, bias_local, in_dim, end - start);
    });
}

/// Like linear_nobias_bf16 for seq_len=1, but ADDS to the destination: `y[i] += W[i] @ x`.
/// Achieves fused residual add by passing y as its own "bias".
pub fn linear_nobias_bf16_addto(y: &mut [f32], x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    // SAFETY: bf16_matvec_fused reads bias[i] before writing y[i], so aliasing y as bias is safe.
    let bias = unsafe { std::slice::from_raw_parts(y.as_ptr(), out_dim) };
    bf16_matvec_threaded(y, x, w_bf16, Some(bias), in_dim, out_dim);
}

pub fn linear_nobias_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear_nobias(y, x, &w_f32, seq_len, in_dim, out_dim);
}

/// Like linear_nobias_bf16 but reuses a caller-provided scratch buffer for bf16→f32 conversion.
/// # Safety
/// Caller must ensure w_bf16 points to at least out_dim * in_dim valid bf16 values.
pub unsafe fn linear_nobias_bf16_scratch(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize, scratch: &mut [f32]) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let n = out_dim * in_dim;
    let src = unsafe { std::slice::from_raw_parts(w_bf16, n) };
    bf16_to_f32_buf(&mut scratch[..n], src);
    linear_nobias(y, x, &scratch[..n], seq_len, in_dim, out_dim);
}

pub fn linear_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, b, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear(y, x, &w_f32, b, seq_len, in_dim, out_dim);
}

/// Fused Q/K/V matvec for single-token decode
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_bf16_qkv(
    q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq: *const u16, wk: *const u16, wv: *const u16,
    in_dim: usize, q_dim: usize, kv_dim: usize,
) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(q, x, wq, None, in_dim, q_dim);
        bf16_matvec_fused(k, x, wk, None, in_dim, kv_dim);
        bf16_matvec_fused(v, x, wv, None, in_dim, kv_dim);
        return;
    }

    let total_dim = q_dim + 2 * kv_dim;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;
    let wq_ptr = wq as usize;
    let wk_ptr = wk as usize;
    let wv_ptr = wv as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(total_dim, tid, nt) {
            Some(r) => r,
            None => return,
        };

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let q_end = q_dim;
        let k_end = q_end + kv_dim;

        // Q range
        if start < q_end {
            let s = start;
            let e = end.min(q_end);
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wq_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // K range
        if end > q_end && start < k_end {
            let s = start.saturating_sub(q_end);
            let e_abs = end.min(k_end);
            let e = e_abs - q_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wk_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // V range
        if end > k_end {
            let s = start.saturating_sub(k_end);
            let e_abs = end.min(total_dim);
            let e = e_abs - k_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wv_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }
    });
}

/// Fused gate_up matvec + SwiGLU for single-token decode.
/// Computes: `ffn_out[j] = silu(gate[j]) * up[j]` where gate/up come from interleaved gate_up_fused matvec.
/// Keeps gate_up output in L1 cache for the SwiGLU operation.
pub fn linear_nobias_bf16_swiglu(
    ffn_out: &mut [f32],
    x: &[f32],
    gate_up_bf16: *const u16,
    in_dim: usize,
    intermediate: usize,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let n_threads = get_num_threads();

    if n_threads <= 1 {
        // Single-threaded: compute gate_up, then SwiGLU inline
        let mut gate_buf = vec![0.0f32; 2 * intermediate];
        bf16_matvec_fused(&mut gate_buf, x, gate_up_bf16, None, in_dim, 2 * intermediate);
        for j in 0..intermediate {
            let g = gate_buf[2 * j];
            let u = gate_buf[2 * j + 1];
            ffn_out[j] = g / (1.0 + (-g).exp()) * u;
        }
        return;
    }

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = gate_up_bf16 as usize;
    let ffn_ptr = ffn_out.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(intermediate, tid, nt) {
            Some(r) => r,
            None => return,
        };
        let n_rows = end - start;

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let w_local = unsafe { (w_ptr as *const u16).add(2 * start * in_dim) };

        // Compute gate_up for this chunk (thread-local stack buffer)
        let mut gate_up_local = vec![0.0f32; 2 * n_rows];
        bf16_matvec_fused(&mut gate_up_local, x_local, w_local, None, in_dim, 2 * n_rows);

        // Apply SwiGLU inline while data is hot in L1
        let ffn_local = unsafe { std::slice::from_raw_parts_mut((ffn_ptr as *mut f32).add(start), n_rows) };
        for j in 0..n_rows {
            let g = gate_up_local[2 * j];
            let u = gate_up_local[2 * j + 1];
            ffn_local[j] = g / (1.0 + (-g).exp()) * u;
        }
    });
}

/// INT8 threaded matvec: y = W_int8 @ x + bias  (x is f32, quantized on the fly)
fn int8_matvec_threaded(y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], w_sums: &[i32], bias: Option<&[f32]>, in_dim: usize, out_dim: usize, x_int8_buf: &mut Vec<i8>) {
    let x_scale = quantize_f32_to_int8_into(x, x_int8_buf);
    let n_threads = get_num_threads();

    // Select kernel; fall back to f32 if no SIMD support
    let matvec_fn = match select_int8_matvec_fn() {
        Some(f) => f,
        None => {
            let _ = (x_scale, n_threads, w_sums);
            int8_fallback_matvec(y, x, w_int8, w_scales, bias, in_dim, out_dim);
            return;
        }
    };

    if n_threads <= 1 {
        unsafe {
            matvec_fn(y, x_int8_buf.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, w_sums, bias, in_dim, out_dim);
        }
        return;
    }

    let x_int8_ptr = x_int8_buf.as_ptr() as usize;
    let w_int8_ptr = w_int8.as_ptr() as usize;
    let w_scales_ptr = w_scales.as_ptr() as usize;
    let w_sums_ptr = w_sums.as_ptr() as usize;
    let y_ptr = y.as_mut_ptr() as usize;
    let bias_ptr = bias.map(|b| b.as_ptr() as usize);

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(out_dim, tid, nt) {
            Some(r) => r,
            None => return,
        };
        let len = end - start;

        let y_local = unsafe { std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(start), len) };
        let w_local = unsafe { (w_int8_ptr as *const i8).add(start * in_dim) };
        let w_scales_local = unsafe { std::slice::from_raw_parts((w_scales_ptr as *const f32).add(start), len) };
        let w_sums_local = unsafe { std::slice::from_raw_parts((w_sums_ptr as *const i32).add(start), len) };
        let bias_local = bias_ptr.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), len) });

        unsafe {
            matvec_fn(y_local, x_int8_ptr as *const i8, x_scale, w_local, w_scales_local, w_sums_local, bias_local, in_dim, len);
        }
    });
}

/// INT8 GEMM kernel function pointer type.
/// Processes 4 output rows simultaneously for cache efficiency.
type Int8Gemm4RowsFn = unsafe fn(
    *mut f32, *const i8, f32, *const i8, *const f32, *const i32, usize
);

/// Select optimal INT8 GEMM kernel based on CPU features.
/// Priority: AVX-VNNI > AVX2 > scalar fallback.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn select_int8_gemm_4rows_fn() -> Int8Gemm4RowsFn {
    if !is_x86_feature_detected!("avx2") {
        // Fallback: use 4x scalar rows (delegates to int8_fallback_matvec per row)
        int8_gemm_4rows_fallback
    } else if is_x86_feature_detected!("avxvnni") {
        avx::int8_gemm_4rows_vnni
    } else {
        avx::int8_gemm_4rows_avx2
    }
}

/// Scalar fallback for INT8 GEMM 4 rows (when no AVX2).
/// Delegates to int8_fallback_matvec for each row.
unsafe fn int8_gemm_4rows_fallback(
    y: *mut f32,
    x_int8: *const i8,
    x_scale: f32,
    w_int8: *const i8,
    w_scales: *const f32,
    w_sums: *const i32,
    in_dim: usize,
) {
    for r in 0..4 {
        let mut sum = 0i32;
        for i in 0..in_dim {
            sum += (*x_int8.add(i) as i32) * (*w_int8.add(r * in_dim + i) as i32);
        }
        *y.add(r) = sum as f32 * x_scale * *w_scales.add(r);
    }
}

/// INT8 GEMM for prefill: y = x @ w_int8.T * w_scales
/// Uses 4-row batched kernel for cache efficiency (shares x_int8 loads).
/// Multi-threaded: parallelizes across output dimension.
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_int8_prefill(
    y: &mut [f32],
    x: &[f32],
    w_int8: &[i8],
    w_scales: &[f32],
    w_sums: &[i32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) {
    let _pg = ProfileGuard::new(&PROF.sgemm);

    let gemm_fn = select_int8_gemm_4rows_fn();
    let n_threads = get_num_threads();

    for s in 0..seq_len {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        let y_row = &mut y[s * out_dim..(s + 1) * out_dim];

        // Quantize input once per token (shared across all threads and output rows)
        let (x_int8, x_scale) = quantize_f32_to_int8(x_row);
        let x_int8_ptr = x_int8.as_ptr() as usize;

        if n_threads <= 1 {
            // Single-threaded path
            let mut o = 0usize;
            while o + 4 <= out_dim {
                unsafe {
                    gemm_fn(
                        y_row.as_mut_ptr().add(o),
                        x_int8_ptr as *const i8,
                        x_scale,
                        w_int8.as_ptr().add(o * in_dim),
                        w_scales.as_ptr().add(o),
                        w_sums.as_ptr().add(o),
                        in_dim,
                    );
                }
                o += 4;
            }
            // Handle remaining rows
            while o < out_dim {
                let mut sum = 0i32;
                let w_row = &w_int8[o * in_dim..(o + 1) * in_dim];
                for i in 0..in_dim {
                    sum += x_int8[i] as i32 * w_row[i] as i32;
                }
                y_row[o] = sum as f32 * x_scale * w_scales[o];
                o += 1;
            }
        } else {
            // Multi-threaded: parallelize across output dimension
            // Each thread processes a contiguous block of output rows
            let y_ptr = y_row.as_mut_ptr() as usize;
            let w_ptr = w_int8.as_ptr() as usize;
            let ws_ptr = w_scales.as_ptr() as usize;
            let wsum_ptr = w_sums.as_ptr() as usize;

            parallel_for(|tid, nt| {
                let chunk = out_dim.div_ceil(nt);
                let start = (tid * chunk).min(out_dim);
                let end = ((tid + 1) * chunk).min(out_dim);

                let mut o = start;
                // Process 4 rows at a time
                while o + 4 <= end {
                    unsafe {
                        gemm_fn(
                            (y_ptr as *mut f32).add(o),
                            x_int8_ptr as *const i8,
                            x_scale,
                            (w_ptr as *const i8).add(o * in_dim),
                            (ws_ptr as *const f32).add(o),
                            (wsum_ptr as *const i32).add(o),
                            in_dim,
                        );
                    }
                    o += 4;
                }
                // Handle remaining rows in this thread's block
                while o < end {
                    let mut sum = 0i32;
                    for i in 0..in_dim {
                        unsafe {
                            sum += *(x_int8_ptr as *const i8).add(i) as i32
                                * (*(w_ptr as *const i8).add(o * in_dim + i)) as i32;
                        }
                    }
                    unsafe {
                        *(y_ptr as *mut f32).add(o) =
                            sum as f32 * x_scale * *(ws_ptr as *const f32).add(o);
                    }
                    o += 1;
                }
            });
        }
    }
}

/// Core INT8 GEMM with pre-quantized input: y = x_int8 @ w_int8.T * (x_scales * w_scales)
/// Assumes x_int8 (seq_len × in_dim) and x_scales (seq_len) are already computed.
/// This is the shared GEMM kernel used by both single and fused (QKV) paths.
/// Uses 4-row batched kernel for cache efficiency (shares x_int8 loads).
#[allow(clippy::too_many_arguments)]
fn int8_gemm_tiled_core(
    y: &mut [f32],
    x_int8: &[i8],
    x_scales: &[f32],
    w_int8: &[i8],
    w_scales: &[f32],
    w_sums: &[i32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) {
    let gemm4_fn = select_int8_gemm_4rows_fn();
    let n_threads = get_num_threads();

    if n_threads <= 1 {
        // Single-threaded: 4-row blocks, then scalar tail.
        let mut o = 0usize;
        // 4-row blocks
        while o + 4 <= out_dim {
            for s in 0..seq_len {
                unsafe {
                    gemm4_fn(
                        y.as_mut_ptr().add(s * out_dim + o),
                        x_int8.as_ptr().add(s * in_dim),
                        x_scales[s],
                        w_int8.as_ptr().add(o * in_dim),
                        w_scales.as_ptr().add(o),
                        w_sums.as_ptr().add(o),
                        in_dim,
                    );
                }
            }
            o += 4;
        }
        // Scalar tail
        while o < out_dim {
            let w_row = &w_int8[o * in_dim..(o + 1) * in_dim];
            for s in 0..seq_len {
                let x_row = &x_int8[s * in_dim..(s + 1) * in_dim];
                let mut sum = 0i32;
                for i in 0..in_dim {
                    sum += x_row[i] as i32 * w_row[i] as i32;
                }
                y[s * out_dim + o] = sum as f32 * x_scales[s] * w_scales[o];
            }
            o += 1;
        }
    } else {
        // Multi-threaded: parallelize across o_blocks.
        let y_ptr = y.as_mut_ptr() as usize;
        let x_int8_ptr = x_int8.as_ptr() as usize;
        let x_scales_ptr = x_scales.as_ptr() as usize;
        let w_ptr = w_int8.as_ptr() as usize;
        let ws_ptr = w_scales.as_ptr() as usize;
        let wsum_ptr = w_sums.as_ptr() as usize;

        parallel_for(move |tid, nt| {
            // 4-row aligned partitioning
            let align = 4;
            let o_per_thread = (out_dim / nt / align) * align;
            let start = (tid * o_per_thread).min(out_dim);
            let end = if tid == nt - 1 {
                out_dim
            } else {
                ((tid + 1) * o_per_thread).min(out_dim)
            };

            let mut o = start;
            // 4-row blocks
            while o + 4 <= end {
                for s in 0..seq_len {
                    unsafe {
                        gemm4_fn(
                            (y_ptr as *mut f32).add(s * out_dim + o),
                            (x_int8_ptr as *const i8).add(s * in_dim),
                            *(x_scales_ptr as *const f32).add(s),
                            (w_ptr as *const i8).add(o * in_dim),
                            (ws_ptr as *const f32).add(o),
                            (wsum_ptr as *const i32).add(o),
                            in_dim,
                        );
                    }
                }
                o += 4;
            }
            // Scalar tail
            while o < end {
                for s in 0..seq_len {
                    let mut sum = 0i32;
                    let w_row = unsafe {
                        std::slice::from_raw_parts((w_ptr as *const i8).add(o * in_dim), in_dim)
                    };
                    let x_row = unsafe {
                        std::slice::from_raw_parts(
                            (x_int8_ptr as *const i8).add(s * in_dim),
                            in_dim,
                        )
                    };
                    for i in 0..in_dim {
                        sum += x_row[i] as i32 * w_row[i] as i32;
                    }
                    unsafe {
                        *(y_ptr as *mut f32).add(s * out_dim + o) =
                            sum as f32 * *(x_scales_ptr as *const f32).add(s)
                                * *(ws_ptr as *const f32).add(o);
                    }
                }
                o += 1;
            }
        });
    }
}

/// Quantize all tokens upfront (once, shared across all output rows).
/// Returns (x_int8_all, x_scales) — caller owns the buffers.
fn quantize_tokens_for_gemm(x: &[f32], seq_len: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut x_int8_all = vec![0i8; seq_len * in_dim];
    let mut x_scales = vec![0.0f32; seq_len];
    for s in 0..seq_len {
        let (q, scale) = quantize_f32_to_int8(&x[s * in_dim..(s + 1) * in_dim]);
        x_int8_all[s * in_dim..(s + 1) * in_dim].copy_from_slice(&q);
        x_scales[s] = scale;
    }
    (x_int8_all, x_scales)
}

/// INT8 GEMM for prefill (tiled): y = x @ w_int8.T * w_scales
/// Optimized loop order: outer = o_block (4 output rows), inner = s (token).
/// Weights are loaded once per o_block and reused across all tokens, reducing
/// weight memory traffic from seq_len × out_dim × in_dim to out_dim × in_dim.
/// All tokens are quantized upfront (once) instead of per-token inside the loop.
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_int8_prefill_tiled(
    y: &mut [f32],
    x: &[f32],
    w_int8: &[i8],
    w_scales: &[f32],
    w_sums: &[i32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) {
    let _pg = ProfileGuard::new(&PROF.sgemm);

    // 1. Quantize all tokens upfront (once, shared across all output rows).
    let (x_int8_all, x_scales) = quantize_tokens_for_gemm(x, seq_len, in_dim);

    // 2. GEMM with pre-quantized input.
    int8_gemm_tiled_core(y, &x_int8_all, &x_scales, w_int8, w_scales, w_sums, seq_len, in_dim, out_dim);
}

/// INT8 fused QKV matvec for single-token decode
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_int8_qkv(
    q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq_int8: &[i8], wq_scales: &[f32], wq_sums: &[i32],
    wk_int8: &[i8], wk_scales: &[f32], wk_sums: &[i32],
    wv_int8: &[i8], wv_scales: &[f32], wv_sums: &[i32],
    in_dim: usize, q_dim: usize, kv_dim: usize,
    x_int8_buf: &mut Vec<i8>,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let x_scale = quantize_f32_to_int8_into(x, x_int8_buf);
    let n_threads = get_num_threads();

    // Select kernel; fall back to f32 if no SIMD support
    let matvec_fn = match select_int8_matvec_fn() {
        Some(f) => f,
        None => {
            let _ = (x_scale, n_threads);
            let wq_f32 = int8_to_f32_weights(wq_int8, wq_scales, q_dim, in_dim);
            let wk_f32 = int8_to_f32_weights(wk_int8, wk_scales, kv_dim, in_dim);
            let wv_f32 = int8_to_f32_weights(wv_int8, wv_scales, kv_dim, in_dim);
            linear_nobias(q, x, &wq_f32, 1, in_dim, q_dim);
            linear_nobias(k, x, &wk_f32, 1, in_dim, kv_dim);
            linear_nobias(v, x, &wv_f32, 1, in_dim, kv_dim);
            return;
        }
    };

    if n_threads <= 1 {
        unsafe {
            matvec_fn(q, x_int8_buf.as_ptr(), x_scale, wq_int8.as_ptr(), wq_scales, wq_sums, None, in_dim, q_dim);
            matvec_fn(k, x_int8_buf.as_ptr(), x_scale, wk_int8.as_ptr(), wk_scales, wk_sums, None, in_dim, kv_dim);
            matvec_fn(v, x_int8_buf.as_ptr(), x_scale, wv_int8.as_ptr(), wv_scales, wv_sums, None, in_dim, kv_dim);
        }
        return;
    }

    let total_dim = q_dim + 2 * kv_dim;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;
    let x_int8_ptr = x_int8_buf.as_ptr() as usize;
    let wq_ptr = wq_int8.as_ptr() as usize;
    let wk_ptr = wk_int8.as_ptr() as usize;
    let wv_ptr = wv_int8.as_ptr() as usize;
    let wq_scales_ptr = wq_scales.as_ptr() as usize;
    let wk_scales_ptr = wk_scales.as_ptr() as usize;
    let wv_scales_ptr = wv_scales.as_ptr() as usize;
    let wq_sums_ptr = wq_sums.as_ptr() as usize;
    let wk_sums_ptr = wk_sums.as_ptr() as usize;
    let wv_sums_ptr = wv_sums.as_ptr() as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(total_dim, tid, nt) {
            Some(r) => r,
            None => return,
        };

        let q_end = q_dim;
        let k_end = q_end + kv_dim;

        // Q range [start, end) ∩ [0, q_end)
        if start < q_end {
            let s = start;
            let e = end.min(q_end);
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wq_ptr as *const i8).add(s * in_dim) };
                let scales_local = unsafe { std::slice::from_raw_parts((wq_scales_ptr as *const f32).add(s), e - s) };
                let sums_local = unsafe { std::slice::from_raw_parts((wq_sums_ptr as *const i32).add(s), e - s) };
                unsafe { matvec_fn(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, sums_local, None, in_dim, e - s); }
            }
        }
        // K range [start, end) ∩ [q_end, k_end)
        if start < k_end && end > q_end {
            let s = start.max(q_end) - q_end;
            let e = end.min(k_end) - q_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wk_ptr as *const i8).add(s * in_dim) };
                let scales_local = unsafe { std::slice::from_raw_parts((wk_scales_ptr as *const f32).add(s), e - s) };
                let sums_local = unsafe { std::slice::from_raw_parts((wk_sums_ptr as *const i32).add(s), e - s) };
                unsafe { matvec_fn(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, sums_local, None, in_dim, e - s); }
            }
        }
        // V range [start, end) ∩ [k_end, total_dim)
        if end > k_end {
            let s = start.max(k_end) - k_end;
            let e = end.min(total_dim) - k_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wv_ptr as *const i8).add(s * in_dim) };
                let scales_local = unsafe { std::slice::from_raw_parts((wv_scales_ptr as *const f32).add(s), e - s) };
                let sums_local = unsafe { std::slice::from_raw_parts((wv_sums_ptr as *const i32).add(s), e - s) };
                unsafe { matvec_fn(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, sums_local, None, in_dim, e - s); }
            }
        }
    });
}

/// INT8 fused gate_up + SwiGLU
pub fn linear_nobias_int8_swiglu(
    ffn_out: &mut [f32], x: &[f32],
    w_int8: &[i8], w_scales: &[f32], w_sums: &[i32],
    in_dim: usize, intermediate: usize,
    x_int8_buf: &mut Vec<i8>, gate_buf: &mut [f32],
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let x_scale = quantize_f32_to_int8_into(x, x_int8_buf);
    let n_threads = get_num_threads();

    // Select kernel; fall back to f32 if no SIMD support
    let matvec_fn = match select_int8_matvec_fn() {
        Some(f) => f,
        None => {
            let _ = (x_scale, n_threads);
            let w_f32 = int8_to_f32_weights(w_int8, w_scales, 2 * intermediate, in_dim);
            linear_nobias(gate_buf, x, &w_f32, 1, in_dim, 2 * intermediate);
            for j in 0..intermediate {
                let g = gate_buf[2 * j];
                let u = gate_buf[2 * j + 1];
                ffn_out[j] = g / (1.0 + (-g).exp()) * u;
            }
            return;
        }
    };

    if n_threads <= 1 {
        unsafe {
            matvec_fn(gate_buf, x_int8_buf.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, w_sums, None, in_dim, 2 * intermediate);
        }
        for j in 0..intermediate {
            let g = gate_buf[2 * j];
            let u = gate_buf[2 * j + 1];
            ffn_out[j] = g / (1.0 + (-g).exp()) * u;
        }
        return;
    }

    let x_int8_ptr = x_int8_buf.as_ptr() as usize;
    let w_int8_ptr = w_int8.as_ptr() as usize;
    let w_scales_ptr = w_scales.as_ptr() as usize;
    let w_sums_ptr = w_sums.as_ptr() as usize;
    let ffn_ptr = ffn_out.as_mut_ptr() as usize;
    let gate_ptr = gate_buf.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(intermediate, tid, nt) {
            Some(r) => r,
            None => return,
        };
        let n_rows = end - start;

        let w_local = unsafe { (w_int8_ptr as *const i8).add(2 * start * in_dim) };
        let w_scales_local = unsafe { std::slice::from_raw_parts((w_scales_ptr as *const f32).add(2 * start), 2 * n_rows) };
        let w_sums_local = unsafe { std::slice::from_raw_parts((w_sums_ptr as *const i32).add(2 * start), 2 * n_rows) };

        // Use pre-allocated gate_buf slice [2*start .. 2*end] — no heap allocation
        let gate_up_local = unsafe { std::slice::from_raw_parts_mut((gate_ptr as *mut f32).add(2 * start), 2 * n_rows) };
        unsafe {
            matvec_fn(gate_up_local, x_int8_ptr as *const i8, x_scale, w_local, w_scales_local, w_sums_local, None, in_dim, 2 * n_rows);
        }

        let ffn_local = unsafe { std::slice::from_raw_parts_mut((ffn_ptr as *mut f32).add(start), n_rows) };
        for j in 0..n_rows {
            let g = gate_up_local[2 * j];
            let u = gate_up_local[2 * j + 1];
            ffn_local[j] = g / (1.0 + (-g).exp()) * u;
        }
    });
}

/// INT8 matvec with fused residual add: y += W_int8 @ x  (y acts as bias)
pub fn linear_nobias_int8_addto(y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], w_sums: &[i32], in_dim: usize, out_dim: usize, x_int8_buf: &mut Vec<i8>) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let bias = unsafe { std::slice::from_raw_parts(y.as_ptr(), out_dim) };
    int8_matvec_threaded(y, x, w_int8, w_scales, w_sums, Some(bias), in_dim, out_dim, x_int8_buf);
}

pub fn matmul_t_bf16(c: &mut [f32], a: &[f32], b_bf16: *const u16, m: usize, k: usize, n: usize) {
    if m == 1 {
        bf16_matvec_threaded(c, a, b_bf16, None, k, n);
    } else {
        let b_f32 = bf16_to_f32_view(b_bf16, n * k);
        matmul_t(c, a, &b_f32, m, k, n);
    }
}

// ========================================================================
// 2D Convolution (im2col + BLAS sgemm)
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn im2col(input: &[f32], cols: &mut [f32], c_in: usize, h_in: usize, w_in: usize,
          kh: usize, kw: usize, stride: usize, padding: usize, h_out: usize, w_out: usize) {
    let col_len = h_out * w_out;
    for ic in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let col_row = (ic * kh + ki) * kw + kj;
                for oh in 0..h_out {
                    let ih = oh * stride + ki;
                    let ih = ih as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = ow * stride + kj;
                        let iw = iw as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            input[ic * h_in * w_in + ih as usize * w_in + iw as usize]
                        } else {
                            0.0
                        };
                        cols[col_row * col_len + oh * w_out + ow] = val;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
              c_in: usize, c_out: usize, h_in: usize, w_in: usize,
              kh: usize, kw: usize, stride: usize, padding: usize) {
    let _pg = ProfileGuard::new(&PROF.conv2d_op);
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;

    let mut cols = vec![0.0f32; patch_size * spatial_out];
    conv2d_impl(out, input, weight, bias, &mut cols, c_in, c_out, h_in, w_in, kh, kw, stride, padding);
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_with_cols(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
                        cols: &mut Vec<f32>,
                        c_in: usize, c_out: usize, h_in: usize, w_in: usize,
                        kh: usize, kw: usize, stride: usize, padding: usize) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    cols.resize(patch_size * spatial_out, 0.0);
    conv2d_impl(out, input, weight, bias, cols, c_in, c_out, h_in, w_in, kh, kw, stride, padding);
}

#[allow(clippy::too_many_arguments)]
fn conv2d_impl(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
               cols: &mut [f32],
               c_in: usize, c_out: usize, h_in: usize, w_in: usize,
               kh: usize, kw: usize, stride: usize, padding: usize) {
    let _pg = ProfileGuard::new(&PROF.conv2d_op);
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    let cols = &mut cols[..patch_size * spatial_out];

    // Thread im2col across col_rows (each row is independent)
    let n_threads = get_num_threads();
    if n_threads > 1 && patch_size >= 16 {
        let input_ptr = input.as_ptr() as usize;
        let cols_ptr = cols.as_mut_ptr() as usize;
        parallel_for(|tid, nt| {
            let (start, end) = match parallel_chunk_range(patch_size, tid, nt) {
                Some(r) => r,
                None => return,
            };
            for col_row in start..end {
                let ic = col_row / (kh * kw);
                let rem = col_row % (kh * kw);
                let ki = rem / kw;
                let kj = rem % kw;
                for oh in 0..h_out {
                    let ih = (oh * stride + ki) as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = (ow * stride + kj) as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            unsafe { *(input_ptr as *const f32).add(ic * h_in * w_in + ih as usize * w_in + iw as usize) }
                        } else {
                            0.0
                        };
                        unsafe { *(cols_ptr as *mut f32).add(col_row * spatial_out + oh * w_out + ow) = val; }
                    }
                }
            }
        });
    } else {
        im2col(input, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);
    }

    // GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out]
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            c_out as i32, spatial_out as i32, patch_size as i32,
            1.0, weight.as_ptr(), patch_size as i32,
            cols.as_ptr(), spatial_out as i32,
            0.0, out.as_mut_ptr(), spatial_out as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for oc in 0..c_out {
            for s in 0..spatial_out {
                let mut sum = 0.0f32;
                for p in 0..patch_size {
                    sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
                }
                out[oc * spatial_out + s] = sum;
            }
        }
    }

    if let Some(bias) = bias {
        for oc in 0..c_out {
            let b = bias[oc];
            for s in 0..spatial_out {
                out[oc * spatial_out + s] += b;
            }
        }
    }
}

/// INT8 conv2d: same as conv2d_impl but uses INT8 quantized weights.
/// im2col output is transposed to [spatial_out, patch_size] for INT8 GEMM compatibility.
/// Output is transposed back to [c_out, spatial_out].
#[allow(clippy::too_many_arguments)]
fn conv2d_int8_impl(
    out: &mut [f32], input: &[f32],
    w_int8: &[i8], w_scales: &[f32], w_sums: &[i32], bias: Option<&[f32]>,
    cols: &mut [f32],  // scratch buffer, size = patch_size * spatial_out
    c_in: usize, c_out: usize, h_in: usize, w_in: usize,
    kh: usize, kw: usize, stride: usize, padding: usize,
) {
    let _pg = ProfileGuard::new(&PROF.conv2d_op);
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    let cols = &mut cols[..patch_size * spatial_out];

    // im2col with TRANSPOSED layout: cols[spatial_out, patch_size]
    // Each row is a spatial position's patch (compatible with INT8 GEMM input)
    let n_threads = get_num_threads();
    if n_threads > 1 && patch_size >= 16 {
        let input_ptr = input.as_ptr() as usize;
        let cols_ptr = cols.as_mut_ptr() as usize;
        parallel_for(move |tid, nt| {
            let (start, end) = match parallel_chunk_range(patch_size, tid, nt) {
                Some(r) => r,
                None => return,
            };
            for col_row in start..end {
                let ic = col_row / (kh * kw);
                let rem = col_row % (kh * kw);
                let ki = rem / kw;
                let kj = rem % kw;
                for oh in 0..h_out {
                    let ih = (oh * stride + ki) as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = (ow * stride + kj) as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            unsafe { *(input_ptr as *const f32).add(ic * h_in * w_in + ih as usize * w_in + iw as usize) }
                        } else {
                            0.0
                        };
                        // Transposed: cols[(oh*w_out+ow) * patch_size + col_row] = val
                        unsafe { *(cols_ptr as *mut f32).add((oh * w_out + ow) * patch_size + col_row) = val; }
                    }
                }
            }
        });
    } else {
        // Single-threaded im2col (transposed)
        for col_row in 0..patch_size {
            let ic = col_row / (kh * kw);
            let rem = col_row % (kh * kw);
            let ki = rem / kw;
            let kj = rem % kw;
            for oh in 0..h_out {
                let ih = (oh * stride + ki) as isize - padding as isize;
                for ow in 0..w_out {
                    let iw = (ow * stride + kj) as isize - padding as isize;
                    let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                        input[ic * h_in * w_in + ih as usize * w_in + iw as usize]
                    } else {
                        0.0
                    };
                    cols[(oh * w_out + ow) * patch_size + col_row] = val;
                }
            }
        }
    }

    // INT8 GEMM: y[spatial_out, c_out] = cols[spatial_out, patch_size] @ w_int8[c_out, patch_size]^T
    // linear_nobias_int8_prefill_tiled quantizes cols internally and calls int8_gemm_tiled_core
    // Need separate output buffer since GEMM reads from cols (input) and writes to output
    let mut gemm_out = vec![0.0f32; spatial_out * c_out];
    linear_nobias_int8_prefill_tiled(&mut gemm_out, cols, w_int8, w_scales, w_sums, spatial_out, patch_size, c_out);

    // Transpose output from [spatial_out, c_out] to [c_out, spatial_out] and add bias
    if n_threads > 1 && c_out >= 16 {
        let gemm_ptr = gemm_out.as_ptr() as usize;
        let out_ptr = out.as_mut_ptr() as usize;
        parallel_for(move |tid, nt| {
            let (start, end) = match parallel_chunk_range(c_out, tid, nt) {
                Some(r) => r,
                None => return,
            };
            for oc in start..end {
                let b = bias.map_or(0.0, |bias| bias[oc]);
                for s in 0..spatial_out {
                    unsafe { *(out_ptr as *mut f32).add(oc * spatial_out + s) = *(gemm_ptr as *const f32).add(s * c_out + oc) + b; }
                }
            }
        });
    } else {
        for oc in 0..c_out {
            let b = bias.map_or(0.0, |bias| bias[oc]);
            for s in 0..spatial_out {
                out[oc * spatial_out + s] = gemm_out[s * c_out + oc] + b;
            }
        }
    }
}

/// Public wrapper for INT8 conv2d with external cols buffer.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_int8_with_cols(out: &mut [f32], input: &[f32],
    w_int8: &[i8], w_scales: &[f32], w_sums: &[i32], bias: Option<&[f32]>,
    cols: &mut Vec<f32>,
    c_in: usize, c_out: usize, h_in: usize, w_in: usize,
    kh: usize, kw: usize, stride: usize, padding: usize,
) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    // Need max(patch_size, c_out) * spatial_out for buffer reuse
    let buf_size = patch_size.max(c_out) * spatial_out;
    cols.resize(buf_size, 0.0);
    conv2d_int8_impl(out, input, w_int8, w_scales, w_sums, bias, cols,
        c_in, c_out, h_in, w_in, kh, kw, stride, padding);
}

// ========================================================================
// Normalization
// ========================================================================

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32],
                  seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.layer_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mean: f32 = x_row.iter().sum::<f32>() / hidden as f32;

            let var: f32 = x_row.iter().map(|&v| {
                let d = v - mean;
                d * d
            }).sum::<f32>() / hidden as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..hidden {
                out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
            }
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.rms_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
            let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                out_row[i] = x_row[i] * rms_inv * weight[i];
            }
        }
    }
}

pub fn rms_norm_per_head(x: &mut [f32], weight: &[f32], seq_len: usize, n_heads: usize, head_dim: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.rms_norm_per_head_op);
    let hidden = n_heads * head_dim;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let off = s * hidden + h * head_dim;

            #[cfg(target_arch = "aarch64")]
            {
                let vec = &mut x[off..off + head_dim];
                unsafe { neon::rms_norm_inplace(vec, weight, head_dim, eps); }
                continue;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let vec = &mut x[off..off + head_dim];
                let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
                let rms_inv = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
                for d in 0..head_dim {
                    vec[d] = vec[d] * rms_inv * weight[d];
                }
            }
        }
    }
}

// ========================================================================
// Activation Functions
// ========================================================================

pub fn silu(x: &mut [f32], n: usize) {
    for val in x.iter_mut().take(n) {
        *val = *val / (1.0 + (-*val).exp());
    }
}

pub fn gelu(x: &mut [f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.gelu);
    let n_threads = get_num_threads();
    // Thread GELU for large buffers (encoder FFN: ~320K floats)
    if n_threads > 1 && n > 4096 {
        let x_ptr = x.as_mut_ptr() as usize;
        parallel_for(|tid, nt| {
            let (start, end) = match parallel_chunk_range(n, tid, nt) {
                Some(r) => r,
                None => return,
            };
            let x_local = unsafe { std::slice::from_raw_parts_mut((x_ptr as *mut f32).add(start), end - start) };
            #[cfg(target_arch = "aarch64")]
            unsafe { neon::gelu_inplace(x_local, end - start); }
            #[cfg(target_arch = "x86_64")]
            unsafe { avx::gelu_inplace(x_local, end - start); }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for i in 0..(end - start) {
                let val = x_local[i];
                let x3 = val * val * val;
                let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
                x_local[i] = 0.5 * val * (1.0 + inner.tanh());
            }
        });
        return;
    }
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::gelu_inplace(x, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::gelu_inplace(x, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
    }
}

pub fn swiglu_multiply(out: &mut [f32], gate_up: &[f32], seq_len: usize, intermediate: usize) {
    let _pg = ProfileGuard::new(&PROF.swiglu);
    let total = seq_len * intermediate;
    let n_threads = get_num_threads();

    // Thread SwiGLU for large prefill buffers
    if n_threads > 1 && total > 4096 {
        let out_ptr = out.as_mut_ptr() as usize;
        let gu_ptr = gate_up.as_ptr() as usize;
        parallel_for(|tid, nt| {
            let (start, end) = match parallel_chunk_range(seq_len, tid, nt) {
                Some(r) => r,
                None => return,
            };
            for s in start..end {
                let gu = unsafe { std::slice::from_raw_parts((gu_ptr as *const f32).add(s * 2 * intermediate), 2 * intermediate) };
                let o = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(s * intermediate), intermediate) };
                #[cfg(target_arch = "aarch64")]
                { unsafe { neon::swiglu_interleaved(o, gu, intermediate); } continue; }
                #[cfg(target_arch = "x86_64")]
                { unsafe { avx::swiglu_interleaved(o, gu, intermediate); } continue; }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                for j in 0..intermediate {
                    let g = gu[2 * j];
                    let u = gu[2 * j + 1];
                    o[j] = g / (1.0 + (-g).exp()) * u;
                }
            }
        });
        return;
    }

    for s in 0..seq_len {
        let gu = &gate_up[s * 2 * intermediate..s * 2 * intermediate + 2 * intermediate];
        let o = &mut out[s * intermediate..(s + 1) * intermediate];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        for j in 0..intermediate {
            let g = gu[2 * j];
            let u = gu[2 * j + 1];
            let g_silu = g / (1.0 + (-g).exp());
            o[j] = g_silu * u;
        }
    }
}

pub fn softmax(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for val in row.iter_mut().take(cols) {
            *val -= max_val;
        }

        #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
        {
            let n = cols as i32;
            unsafe { vvexpf(row.as_mut_ptr(), row.as_ptr(), &n); }
        }
        #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
        {
            #[cfg(target_arch = "aarch64")]
            { unsafe { neon::exp_inplace(row); } }

            #[cfg(target_arch = "x86_64")]
            { unsafe { avx::exp_inplace(row); } }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for c in 0..cols {
                row[c] = row[c].exp();
            }
        }

        let mut sum = 0.0f32;
        for val in row.iter().take(cols) {
            sum += val;
        }
        let inv_sum = 1.0 / sum;
        for val in row.iter_mut().take(cols) {
            *val *= inv_sum;
        }
    }
}

// ========================================================================
// Attention Operations
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn bidirectional_attention_heads(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                                  n_heads: usize, head_dim: usize, scale: f32,
                                  window_starts: &[i32], n_windows: usize,
                                  head_start: usize, head_end: usize) {
    let hidden = n_heads * head_dim;

    for h in head_start..head_end {
        for w in 0..n_windows {
            let ws = window_starts[w] as usize;
            let we = window_starts[w + 1] as usize;

            for i in ws..we {
                let q_off = i * hidden + h * head_dim;
                let q_row = &q[q_off..q_off + head_dim];
                let o_row = &mut out[i * hidden + h * head_dim..i * hidden + h * head_dim + head_dim];

                let mut max_score = -1e30f32;
                let mut sum_exp = 0.0f32;
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

                for j in ws..we {
                    let k_off = j * hidden + h * head_dim;
                    let v_off = j * hidden + h * head_dim;
                    let k_row = &k[k_off..k_off + head_dim];
                    let v_row = &v[v_off..v_off + head_dim];

                    let score = dot_f32(q_row, k_row, head_dim) * scale;

                    if score > max_score {
                        let correction = (max_score - score).exp();
                        sum_exp = sum_exp * correction + 1.0;
                        vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        let wt = (score - max_score).exp();
                        sum_exp += wt;
                        vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if sum_exp > 0.0 {
                    let inv_sum = 1.0 / sum_exp;
                    vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bidirectional_attention(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                               seq: usize, n_heads: usize, head_dim: usize, scale: f32,
                               window_starts: &[i32], n_windows: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_bidir);
    let n_threads = get_num_threads();
    let hidden = n_heads * head_dim;

    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k.as_ptr() as usize;
        let v_ptr = v.as_ptr() as usize;
        let ws_ptr = window_starts.as_ptr() as usize;

        parallel_for(|tid, nt| {
            let (h0, h1) = match parallel_chunk_range(n_heads, tid, nt) {
                Some(r) => r,
                None => return,
            };

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq * hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq * hidden) };
            let k_local = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, seq * hidden) };
            let v_local = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, seq * hidden) };
            let ws_local = unsafe { std::slice::from_raw_parts(ws_ptr as *const i32, n_windows + 1) };

            bidirectional_attention_heads(out_local, q_local, k_local, v_local,
                                         n_heads, head_dim, scale,
                                         ws_local, n_windows, h0, h1);
        });
        return;
    }

    bidirectional_attention_heads(out, q, k, v, n_heads, head_dim, scale,
                                 window_starts, n_windows, 0, n_heads);
}

/// Two-pass causal attention using BLAS sgemm with head-contiguous KV cache.
/// K/V layout: `[head][pos][head_dim]` — each head's data is contiguous across positions.
///
/// Single-token (seq_q=1): online softmax with NEON dot products — avoids BLAS overhead,
/// scores allocation, and fuses all 3 passes into a single scan over KV positions.
///
/// Multi-token (seq_q>1): 3-pass BLAS sgemm approach.
#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    // Single-token path: online softmax without allocation or BLAS
    if seq_q == 1 {
        for h in head_start..head_end {
            let kv_h = h / heads_per_kv;
            let k_head = unsafe { k_base.add(kv_h * head_stride) };
            let v_head = unsafe { v_base.add(kv_h * head_stride) };
            let q_off = h * head_dim;
            let o_row = &mut out[q_off..q_off + head_dim];
            let k_end = (q_offset + 1).min(seq_k);

            if k_end == 0 {
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }
                continue;
            }

            let q_row = &q[q_off..q_off + head_dim];

            // Online softmax: single pass over KV positions
            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

            for j in 0..k_end {
                let k_row = unsafe { std::slice::from_raw_parts(k_head.add(j * head_dim), head_dim) };
                let v_row = unsafe { std::slice::from_raw_parts(v_head.add(j * head_dim), head_dim) };

                let score = dot_f32(q_row, k_row, head_dim) * scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
        return;
    }

    // Multi-token path: batched per-head GEMMs.
    // Per head: S[seq_q, seq_k] = scale * Q_h @ K_h^T, then causal-masked
    // row softmax, then O[seq_q, head_dim] = S @ V_h. This replaces the
    // 2*seq_q tiny (N=1) BLAS calls per head with two real GEMMs.
    let mut scores = vec![0.0f32; seq_q * seq_k];

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;
        let k_head = unsafe { k_base.add(kv_h * head_stride) };
        let v_head = unsafe { v_base.add(kv_h * head_stride) };

        // S = scale * Q_h[seq_q, head_dim] @ K_h[seq_k, head_dim]^T.
        // Q_h rows are strided by q_hidden inside `q`; K_h is contiguous.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                seq_q as i32, seq_k as i32, head_dim as i32,
                scale,
                q.as_ptr().add(h * head_dim), q_hidden as i32,
                k_head, head_dim as i32,
                0.0,
                scores.as_mut_ptr(), seq_k as i32,
            );
        }

        // Causal-masked row softmax: query i attends keys 0..=(q_offset+i).
        for i in 0..seq_q {
            let k_end = (q_offset + i + 1).min(seq_k);
            let row = &mut scores[i * seq_k..i * seq_k + seq_k];
            if k_end == 0 {
                for v in row.iter_mut() { *v = 0.0; }
                continue;
            }

            let mut max_s = row[0];
            for j in 1..k_end { if row[j] > max_s { max_s = row[j]; } }
            for j in 0..k_end { row[j] -= max_s; }

            #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
            {
                let n = k_end as i32;
                unsafe { vvexpf(row.as_mut_ptr(), row.as_ptr(), &n); }
            }
            #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
            {
                for j in 0..k_end { row[j] = row[j].exp(); }
            }

            let mut sum_exp = 0.0f32;
            for j in 0..k_end { sum_exp += row[j]; }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for j in 0..k_end { row[j] *= inv; }
            }
            // Zero the masked (future) keys so the O = S @ V GEMM ignores them.
            for j in k_end..seq_k { row[j] = 0.0; }
        }

        // O[seq_q, head_dim] = S[seq_q, seq_k] @ V_h[seq_k, head_dim].
        // O rows are strided by q_hidden inside `out`.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                seq_q as i32, head_dim as i32, seq_k as i32,
                1.0,
                scores.as_ptr(), seq_k as i32,
                v_head, head_dim as i32,
                0.0,
                out.as_mut_ptr().add(h * head_dim), q_hidden as i32,
            );
        }
    }
}

/// Fallback: online softmax causal attention (no BLAS), head-contiguous KV layout.
#[cfg(not(feature = "blas"))]
#[allow(clippy::too_many_arguments)]
fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;

        for i in 0..seq_q {
            let q_off = i * q_hidden + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let o_row = &mut out[i * q_hidden + h * head_dim..i * q_hidden + h * head_dim + head_dim];
            let global_pos = q_offset + i;
            let k_end = (global_pos + 1).min(seq_k);

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

            for j in 0..k_end {
                let k_row = unsafe { std::slice::from_raw_parts(k_base.add(kv_h * head_stride + j * head_dim), head_dim) };
                let v_row = unsafe { std::slice::from_raw_parts(v_base.add(kv_h * head_stride + j * head_dim), head_dim) };

                let score = dot_f32(q_row, k_row, head_dim) * scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attention(out: &mut [f32], q: &[f32],
                         k_base: *const f32, v_base: *const f32,
                         head_stride: usize,
                         seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                         head_dim: usize, scale: f32, q_offset: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_causal);
    let n_threads = get_num_threads();
    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k_base as usize;
        let v_ptr = v_base as usize;
        let q_hidden = n_heads * head_dim;

        parallel_for(|tid, nt| {
            let (h0, h1) = match parallel_chunk_range(n_heads, tid, nt) {
                Some(r) => r,
                None => return,
            };

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq_q * q_hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq_q * q_hidden) };

            causal_attention_heads(out_local, q_local,
                                   k_ptr as *const f32, v_ptr as *const f32,
                                   head_stride,
                                   seq_q, seq_k, n_heads, n_kv_heads,
                                   head_dim, scale, q_offset, h0, h1);
        });
        return;
    }

    causal_attention_heads(out, q, k_base, v_base, head_stride,
                            seq_q, seq_k, n_heads, n_kv_heads,
                            head_dim, scale, q_offset, 0, n_heads);
}

// ========================================================================
// Position Embeddings
// ========================================================================

pub fn sinusoidal_pe(pe: &mut [f32], n_pos: usize, d_model: usize) {
    let half = d_model / 2;
    let log_timescale = (10000.0f32).ln() / (half - 1) as f32;

    for p in 0..n_pos {
        let row = &mut pe[p * d_model..(p + 1) * d_model];
        for d in 0..half {
            let inv_timescale = (-(d as f32) * log_timescale).exp();
            let angle = p as f32 * inv_timescale;
            row[d] = angle.sin();
            row[half + d] = angle.cos();
        }
    }
}

pub fn compute_rope_neox(cos_out: &mut [f32], sin_out: &mut [f32], positions: &[i32],
                          seq: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;

    for s in 0..seq {
        let pos = positions[s] as f32;
        for d in 0..half {
            let freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
            let angle = pos * freq;
            let c = angle.cos();
            let sn = angle.sin();
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

pub fn apply_rope_neox(x: &mut [f32], cos_vals: &[f32], sin_vals: &[f32],
                        seq: usize, n_heads: usize, head_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.rope);
    let half = head_dim / 2;
    let hidden = n_heads * head_dim;

    for s in 0..seq {
        let c = &cos_vals[s * head_dim..];
        let sn = &sin_vals[s * head_dim..];

        for h in 0..n_heads {
            let base = s * hidden + h * head_dim;
            let vec = &mut x[base..base + head_dim];

            #[cfg(target_arch = "aarch64")]
            {
                let mut d = 0usize;
                while d + 4 <= half {
                    unsafe {
                        use core::arch::aarch64::*;
                        let x1 = vld1q_f32(vec.as_ptr().add(d));
                        let x2 = vld1q_f32(vec.as_ptr().add(half + d));
                        let cv = vld1q_f32(c.as_ptr().add(d));
                        let sv = vld1q_f32(sn.as_ptr().add(d));
                        // vec[d] = x1*cos - x2*sin
                        let new1 = vfmsq_f32(vmulq_f32(x1, cv), x2, sv);
                        // vec[half+d] = x2*cos + x1*sin (cos[half+d]==cos[d])
                        let new2 = vfmaq_f32(vmulq_f32(x2, cv), x1, sv);
                        vst1q_f32(vec.as_mut_ptr().add(d), new1);
                        vst1q_f32(vec.as_mut_ptr().add(half + d), new2);
                    }
                    d += 4;
                }
                while d < half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d] - x2 * sn[d];
                    vec[half + d] = x2 * c[d] + x1 * sn[d];
                    d += 1;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                for d in 0..half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                    vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
                }
            }
        }
    }
}

/// Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits.
/// Quantize x (f32) to int8 with absmax scaling. Returns (x_int8, scale).
pub fn quantize_f32_to_int8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in x { max_abs = max_abs.max(v.abs()); }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 127.0 / max_abs.max(1e-10);
    let int8: Vec<i8> = x.iter().map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
    (int8, scale)
}

/// Quantize x (f32) into a pre-allocated buffer (avoids heap allocation).
/// Returns scale. Buffer is resized if needed.
pub fn quantize_f32_to_int8_into(x: &[f32], buf: &mut Vec<i8>) -> f32 {
    let mut max_abs = 0.0f32;
    for &v in x { max_abs = max_abs.max(v.abs()); }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 127.0 / max_abs.max(1e-10);
    buf.resize(x.len(), 0);
    for (i, &v) in x.iter().enumerate() {
        buf[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Quantize BF16 weights to INT8 per-row. Returns (int8_data, per_row_scales, per_row_w_sums).
pub fn quantize_bf16_weights_to_int8(w_bf16: *const u16, out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    #[cfg(target_arch = "aarch64")]
    unsafe { return neon::quantize_bf16_to_int8(w_bf16, out_dim, in_dim); }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut int8_data = vec![0i8; out_dim * in_dim];
        let mut scales = vec![0.0f32; out_dim];
        let mut w_sums = vec![0i32; out_dim];
        let src = unsafe { std::slice::from_raw_parts(w_bf16, out_dim * in_dim) };
        for row in 0..out_dim {
            let mut max_abs = 0.0f32;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16).abs();
                if v > max_abs { max_abs = v; }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 127.0 / max_abs.max(1e-10);
            scales[row] = scale;
            let mut row_sum: i32 = 0;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16);
                let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                int8_data[row * in_dim + k] = q;
                row_sum += q as i32;
            }
            w_sums[row] = row_sum;
        }
        (int8_data, scales, w_sums)
    }
}

/// Quantize F32 weights to INT8 per-row. Returns (int8_data, per_row_scales, per_row_w_sums).
/// Used for encoder weights which are already in F32 format.
pub fn quantize_f32_weights_to_int8(w: &[f32], out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let mut int8_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];
    let mut w_sums = vec![0i32; out_dim];
    for row in 0..out_dim {
        let mut max_abs = 0.0f32;
        for k in 0..in_dim {
            let v = w[row * in_dim + k].abs();
            if v > max_abs { max_abs = v; }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 127.0 / max_abs.max(1e-10);
        scales[row] = scale;
        let mut row_sum: i32 = 0;
        for k in 0..in_dim {
            let q = (w[row * in_dim + k] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            int8_data[row * in_dim + k] = q;
            row_sum += q as i32;
        }
        w_sums[row] = row_sum;
    }
    (int8_data, scales, w_sums)
}

/// INT8 linear with bias: y = x @ W^T + b (beta=0, overwrites y).
/// Uses 4-row batched INT8 GEMM kernel for cache efficiency.
#[allow(clippy::too_many_arguments)]
pub fn linear_int8(
    y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], w_sums: &[i32],
    b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize,
) {
    // ProfileGuard is in linear_nobias_int8_prefill_tiled
    linear_nobias_int8_prefill_tiled(y, x, w_int8, w_scales, w_sums, seq_len, in_dim, out_dim);
    if let Some(b) = b {
        for s in 0..seq_len {
            for o in 0..out_dim {
                y[s * out_dim + o] += b[o];
            }
        }
    }
}

/// INT8 linear accumulate: y += b; y += x @ W^T (beta=1, accumulates into y).
/// Uses a scratch buffer to compute INT8 GEMM result, then adds to y.
#[allow(clippy::too_many_arguments)]
pub fn linear_accumulate_int8(
    y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], w_sums: &[i32],
    b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize,
    scratch: &mut [f32],
) {
    // ProfileGuard is in linear_nobias_int8_prefill_tiled
    // Add bias to y first (y already has residual)
    if let Some(b) = b {
        for s in 0..seq_len {
            for o in 0..out_dim {
                y[s * out_dim + o] += b[o];
            }
        }
    }
    // Compute tmp = x @ w^T (INT8 GEMM, beta=0)
    linear_nobias_int8_prefill_tiled(scratch, x, w_int8, w_scales, w_sums, seq_len, in_dim, out_dim);
    // y += tmp
    let n = seq_len * out_dim;
    for i in 0..n {
        y[i] += scratch[i];
    }
}

/// Fused INT8 QKV projection for prefill: quantize x ONCE, compute Q/K/V.
/// Saves 2/3 of quantization cost (memory alloc + compute) vs 3 separate
/// linear_int8 calls, since Q/K/V share the same input x.
/// Each output gets its own bias added.
#[allow(clippy::too_many_arguments)]
pub fn linear_int8_qkv_prefill_fused(
    q: &mut [f32], k: &mut [f32], v: &mut [f32],
    x: &[f32],
    wq_int8: &[i8], wq_scales: &[f32], wq_sums: &[i32], wq_bias: &[f32],
    wk_int8: &[i8], wk_scales: &[f32], wk_sums: &[i32], wk_bias: &[f32],
    wv_int8: &[i8], wv_scales: &[f32], wv_sums: &[i32], wv_bias: &[f32],
    seq_len: usize, in_dim: usize, out_dim: usize,
) {
    let _pg = ProfileGuard::new(&PROF.sgemm);

    // 1. Quantize x ONCE (reused for Q, K, V — saves 2/3 quantization cost)
    let (x_int8_all, x_scales) = quantize_tokens_for_gemm(x, seq_len, in_dim);

    // 2. Q = x @ Wq^T + bq
    int8_gemm_tiled_core(q, &x_int8_all, &x_scales, wq_int8, wq_scales, wq_sums, seq_len, in_dim, out_dim);
    for s in 0..seq_len {
        for o in 0..out_dim {
            q[s * out_dim + o] += wq_bias[o];
        }
    }

    // 3. K = x @ Wk^T + bk
    int8_gemm_tiled_core(k, &x_int8_all, &x_scales, wk_int8, wk_scales, wk_sums, seq_len, in_dim, out_dim);
    for s in 0..seq_len {
        for o in 0..out_dim {
            k[s * out_dim + o] += wk_bias[o];
        }
    }

    // 4. V = x @ Wv^T + bv
    int8_gemm_tiled_core(v, &x_int8_all, &x_scales, wv_int8, wv_scales, wv_sums, seq_len, in_dim, out_dim);
    for s in 0..seq_len {
        for o in 0..out_dim {
            v[s * out_dim + o] += wv_bias[o];
        }
    }
}

/// Fused INT8 QKV projection for decoder prefill: quantize x ONCE, compute Q/K/V.
/// Unlike encoder version: no bias, Q outputs q_dim, K/V output kv_dim (GQA).
/// Saves 2/3 of quantization cost vs 3 separate linear_nobias_int8_prefill calls.
#[allow(clippy::too_many_arguments)]
pub fn linear_int8_qkv_prefill_fused_nobias(
    q: &mut [f32], k: &mut [f32], v: &mut [f32],
    x: &[f32],
    wq_int8: &[i8], wq_scales: &[f32], wq_sums: &[i32],
    wk_int8: &[i8], wk_scales: &[f32], wk_sums: &[i32],
    wv_int8: &[i8], wv_scales: &[f32], wv_sums: &[i32],
    seq_len: usize, in_dim: usize, q_dim: usize, kv_dim: usize,
) {
    let _pg = ProfileGuard::new(&PROF.sgemm);

    // 1. Quantize x ONCE (reused for Q, K, V — saves 2/3 quantization cost)
    let (x_int8_all, x_scales) = quantize_tokens_for_gemm(x, seq_len, in_dim);

    // 2. Q = x @ Wq^T (q_dim outputs)
    int8_gemm_tiled_core(q, &x_int8_all, &x_scales, wq_int8, wq_scales, wq_sums, seq_len, in_dim, q_dim);

    // 3. K = x @ Wk^T (kv_dim outputs)
    int8_gemm_tiled_core(k, &x_int8_all, &x_scales, wk_int8, wk_scales, wk_sums, seq_len, in_dim, kv_dim);

    // 4. V = x @ Wv^T (kv_dim outputs)
    int8_gemm_tiled_core(v, &x_int8_all, &x_scales, wv_int8, wv_scales, wv_sums, seq_len, in_dim, kv_dim);
}

/// INT8 threaded argmax: find argmax(x @ W.T) using INT8 quantized weights.
pub fn argmax_matvec_int8(x: &[f32], w_int8: &[i8], w_scales: &[f32], w_sums: &[i32], in_dim: usize, out_dim: usize) -> usize {
    let _pg = ProfileGuard::new(&PROF.lm_head);
    let (x_int8, x_scale) = quantize_f32_to_int8(x);
    let n_threads = get_num_threads();

    // Select kernel; fall back to f32 if no SIMD support
    let argmax_fn = match select_int8_argmax_fn() {
        Some(f) => f,
        None => {
            let _ = (x_int8, x_scale, n_threads, w_sums);
            let mut w_f32 = vec![0.0f32; out_dim * in_dim];
            for row in 0..out_dim {
                for col in 0..in_dim {
                    w_f32[row * in_dim + col] = w_int8[row * in_dim + col] as f32 * w_scales[row];
                }
            }
            let mut logits = vec![0.0f32; out_dim];
            linear_nobias(&mut logits, x, &w_f32, 1, in_dim, out_dim);
            let mut best = 0usize;
            let mut best_val = logits[0];
            for i in 1..out_dim {
                if logits[i] > best_val {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }
    };

    if n_threads <= 1 {
        let (best, _) = unsafe {
            argmax_fn(x_int8.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, w_sums, in_dim, 0, out_dim)
        };
        return best;
    }

    let mut best_indices = [0usize; MAX_THREADS];
    let mut best_vals = [-1e30f32; MAX_THREADS];

    let x_int8_ptr = x_int8.as_ptr() as usize;
    let w_int8_ptr = w_int8.as_ptr() as usize;
    let w_scales_ptr = w_scales.as_ptr() as usize;
    let w_sums_ptr = w_sums.as_ptr() as usize;
    let bi_ptr = best_indices.as_mut_ptr() as usize;
    let bv_ptr = best_vals.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(out_dim, tid, nt) {
            Some(r) => r,
            None => {
                unsafe {
                    *(bv_ptr as *mut f32).add(tid) = -1e30;
                    *(bi_ptr as *mut usize).add(tid) = 0;
                }
                return;
            }
        };

        let w_scales_local = unsafe { std::slice::from_raw_parts(w_scales_ptr as *const f32, out_dim) };
        let w_sums_local = unsafe { std::slice::from_raw_parts(w_sums_ptr as *const i32, out_dim) };
        let (best, best_val) = unsafe {
            argmax_fn(x_int8_ptr as *const i8, x_scale, w_int8_ptr as *const i8, w_scales_local, w_sums_local, in_dim, start, end)
        };
        unsafe {
            *(bi_ptr as *mut usize).add(tid) = best;
            *(bv_ptr as *mut f32).add(tid) = best_val;
        }
    });

    reduce_argmax(&best_indices, &best_vals, n_threads)
}

pub fn argmax_matvec_bf16(x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) -> usize {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        let (best, _) = argmax_bf16_range(x, w_bf16, in_dim, 0, out_dim);
        return best;
    }

    let mut best_indices = vec![0usize; n_threads];
    let mut best_vals = vec![-1e30f32; n_threads];

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = w_bf16 as usize;
    let bi_ptr = best_indices.as_mut_ptr() as usize;
    let bv_ptr = best_vals.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let (start, end) = match parallel_chunk_range(out_dim, tid, nt) {
            Some(r) => r,
            None => {
                unsafe {
                    *(bv_ptr as *mut f32).add(tid) = -1e30;
                    *(bi_ptr as *mut usize).add(tid) = 0;
                }
                return;
            }
        };

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let (best, best_val) = argmax_bf16_range(x_local, w_ptr as *const u16, in_dim, start, end);
        unsafe {
            *(bi_ptr as *mut usize).add(tid) = best;
            *(bv_ptr as *mut f32).add(tid) = best_val;
        }
    });

    reduce_argmax(&best_indices, &best_vals, n_threads)
}
