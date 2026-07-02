//! oneDNN dynamic-loading convolution backend (Windows prototype).
//!
//! This module loads `dnnl.dll` at runtime so that end-user binaries can run
//! without oneDNN when the DLL is not present. If the DLL cannot be loaded,
//! or if any primitive creation step fails, the caller should fall back to the
//! existing im2col + sgemm path.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// oneDNN C API constants (oneDNN v3.x)
// ---------------------------------------------------------------------------

const DNNL_STATUS_SUCCESS: i32 = 0;
const DNNL_ENGINE_CPU: i32 = 1;
const DNNL_PROP_KIND_FORWARD_INFERENCE: i32 = 96;
const DNNL_ALG_KIND_CONVOLUTION_DIRECT: i32 = 0x1;
// dnnl_data_type_t: undef=0, f16=1, bf16=2, f32=3, s32=4, s8=5, u8=6, f64=7
// (verified from dnnl_common_types.h)
const DNNL_DATA_TYPE_F32: i32 = 3;
// dnnl_format_tag_t: undef=0, any=1, a=2, ab=3, abc=4, abcd=5.
// NCHW / OIHW are aliases for the plain 4D tag abcd.
const DNNL_FORMAT_TAG_A: i32 = 2;
const DNNL_FORMAT_TAG_NCHW: i32 = 5;
const DNNL_STREAM_IN_ORDER: u32 = 0x1;

// DNNL_ARG_* values (verified from dnnl_types.h)
const DNNL_ARG_SRC: i32 = 1;
const DNNL_ARG_WEIGHTS: i32 = 33;
const DNNL_ARG_BIAS: i32 = 41;
const DNNL_ARG_DST: i32 = 17;

// dnnl_query_t: dnnl_query_exec_arg_md = 255
// (used with index = DNNL_ARG_* to query actual memory descriptor from pd)
const DNNL_QUERY_EXEC_ARG_MD: i32 = 255;

// ---------------------------------------------------------------------------
// Function pointer types for the dynamically-loaded oneDNN symbols.
// ---------------------------------------------------------------------------

type DnnlStatus = i32;
type DnnlEngine = c_void;
type DnnlStream = c_void;
type DnnlMemory = c_void;
type DnnlMemoryDesc = c_void;
type DnnlPrimitiveDesc = c_void;
type DnnlPrimitive = c_void;

type EngineCreateFn = unsafe extern "C" fn(*mut *mut DnnlEngine, i32, usize) -> DnnlStatus;
type EngineDestroyFn = unsafe extern "C" fn(*mut DnnlEngine) -> DnnlStatus;

type StreamCreateFn = unsafe extern "C" fn(*mut *mut DnnlStream, *mut DnnlEngine, u32) -> DnnlStatus;
type StreamWaitFn = unsafe extern "C" fn(*mut DnnlStream) -> DnnlStatus;
type StreamDestroyFn = unsafe extern "C" fn(*mut DnnlStream) -> DnnlStatus;

type MemoryDescCreateWithTagFn = unsafe extern "C" fn(
    *mut *mut DnnlMemoryDesc,
    i32,
    *const i64,
    i32,
    i32,
) -> DnnlStatus;
type MemoryDescDestroyFn = unsafe extern "C" fn(*mut DnnlMemoryDesc) -> DnnlStatus;

type MemoryCreateFn = unsafe extern "C" fn(
    *mut *mut DnnlMemory,
    *const DnnlMemoryDesc,
    *mut DnnlEngine,
    *mut c_void,
) -> DnnlStatus;
type MemoryDestroyFn = unsafe extern "C" fn(*mut DnnlMemory) -> DnnlStatus;

type ConvolutionForwardPrimitiveDescCreateFn = unsafe extern "C" fn(
    *mut *mut DnnlPrimitiveDesc,
    *mut DnnlEngine,
    i32, // prop_kind
    i32, // alg_kind
    *const DnnlMemoryDesc, // src_desc
    *const DnnlMemoryDesc, // weights_desc
    *const DnnlMemoryDesc, // bias_desc
    *const DnnlMemoryDesc, // dst_desc
    *const i64,            // strides
    *const i64,            // dilates
    *const i64,            // padding_l
    *const i64,            // padding_r
    *const c_void,         // attr
) -> DnnlStatus;
type PrimitiveDescDestroyFn = unsafe extern "C" fn(*mut DnnlPrimitiveDesc) -> DnnlStatus;

/// Queries a memory descriptor from a primitive descriptor.
/// Returns a pointer to the queried md (or NULL on error).
/// `what` should be DNNL_QUERY_EXEC_ARG_MD (255), `index` is the DNNL_ARG_* value.
type PrimitiveDescQueryMdFn =
    unsafe extern "C" fn(*const DnnlPrimitiveDesc, i32, i32) -> *const DnnlMemoryDesc;

type PrimitiveCreateFn =
    unsafe extern "C" fn(*mut *mut DnnlPrimitive, *const DnnlPrimitiveDesc) -> DnnlStatus;
type PrimitiveDestroyFn = unsafe extern "C" fn(*mut DnnlPrimitive) -> DnnlStatus;
type PrimitiveExecuteFn = unsafe extern "C" fn(
    *const DnnlPrimitive,
    *mut DnnlStream,
    i32,
    *const DnnlExecArg,
) -> DnnlStatus;

#[repr(C)]
struct DnnlExecArg {
    arg: i32,
    memory: *mut DnnlMemory,
}

// ---------------------------------------------------------------------------
// Loaded library + required symbols.
// ---------------------------------------------------------------------------

type SetVerboseFn = unsafe extern "C" fn(i32) -> DnnlStatus;

struct OnednnLib {
    _lib: libloading::Library,
    engine_create: EngineCreateFn,
    engine_destroy: EngineDestroyFn,
    stream_create: StreamCreateFn,
    stream_wait: StreamWaitFn,
    stream_destroy: StreamDestroyFn,
    memory_desc_create_with_tag: MemoryDescCreateWithTagFn,
    memory_desc_destroy: MemoryDescDestroyFn,
    memory_create: MemoryCreateFn,
    memory_destroy: MemoryDestroyFn,
    convolution_forward_primitive_desc_create: ConvolutionForwardPrimitiveDescCreateFn,
    primitive_desc_destroy: PrimitiveDescDestroyFn,
    primitive_desc_query_md: PrimitiveDescQueryMdFn,
    primitive_create: PrimitiveCreateFn,
    primitive_destroy: PrimitiveDestroyFn,
    primitive_execute: PrimitiveExecuteFn,
    set_verbose: SetVerboseFn,
}

impl OnednnLib {
    fn load(path: &Path) -> Option<Self> {
        let lib = unsafe { libloading::Library::new(path).ok()? };

        fn get_sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Option<T> {
            let sym: libloading::Symbol<T> = unsafe { lib.get(name).ok()? };
            Some(*sym)
        }

        let engine_create = get_sym(&lib, b"dnnl_engine_create\0")?;
        let engine_destroy = get_sym(&lib, b"dnnl_engine_destroy\0")?;
        let stream_create = get_sym(&lib, b"dnnl_stream_create\0")?;
        let stream_wait = get_sym(&lib, b"dnnl_stream_wait\0")?;
        let stream_destroy = get_sym(&lib, b"dnnl_stream_destroy\0")?;
        let memory_desc_create_with_tag = get_sym(&lib, b"dnnl_memory_desc_create_with_tag\0")?;
        let memory_desc_destroy = get_sym(&lib, b"dnnl_memory_desc_destroy\0")?;
        let memory_create = get_sym(&lib, b"dnnl_memory_create\0")?;
        let memory_destroy = get_sym(&lib, b"dnnl_memory_destroy\0")?;
        let convolution_forward_primitive_desc_create = get_sym(
            &lib,
            b"dnnl_convolution_forward_primitive_desc_create\0",
        )?;
        let primitive_desc_destroy = get_sym(&lib, b"dnnl_primitive_desc_destroy\0")?;
        let primitive_desc_query_md = get_sym(&lib, b"dnnl_primitive_desc_query_md\0")?;
        let primitive_create = get_sym(&lib, b"dnnl_primitive_create\0")?;
        let primitive_destroy = get_sym(&lib, b"dnnl_primitive_destroy\0")?;
        let primitive_execute = get_sym(&lib, b"dnnl_primitive_execute\0")?;
        let set_verbose = get_sym(&lib, b"dnnl_set_verbose\0")?;

        Some(Self {
            _lib: lib,
            engine_create,
            engine_destroy,
            stream_create,
            stream_wait,
            stream_destroy,
            memory_desc_create_with_tag,
            memory_desc_destroy,
            memory_create,
            memory_destroy,
            convolution_forward_primitive_desc_create,
            primitive_desc_destroy,
            primitive_desc_query_md,
            primitive_create,
            primitive_destroy,
            primitive_execute,
            set_verbose,
        })
    }
}

/// Convolution shape/configuration key used to cache oneDNN primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConvKey {
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
}

struct OnednnCtx {
    lib: OnednnLib,
    engine: *mut DnnlEngine,
    stream: *mut DnnlStream,
    primitive_cache: Mutex<HashMap<ConvKey, CachedConv>>,
}

/// Cached convolution primitive + all associated handles.
///
/// The input memory descriptors (`src_md`, `w_md`, `b_md`, `dst_md`) are kept
/// alive because the query API (`dnnl_primitive_desc_query`) is broken on the
/// pip-provided oneDNN 2024.2.1 build — it returns INVALID_ARGS for weights
/// and NULL for dst. Reusing the exact same md objects that were passed to
/// `convolution_forward_primitive_desc_create` is the only reliable way to
/// create compatible memory objects on this build.
struct CachedConv {
    primitive: *mut DnnlPrimitive,
    pd: *mut DnnlPrimitiveDesc,
    src_md: *mut DnnlMemoryDesc,
    w_md: *mut DnnlMemoryDesc,
    b_md: *mut DnnlMemoryDesc,
    dst_md: *mut DnnlMemoryDesc,
}

// The raw pointers are owned by this module and never exposed or aliased.
// oneDNN engine/stream are accessed only through the module's functions.
unsafe impl Send for OnednnCtx {}
unsafe impl Sync for OnednnCtx {}

impl Drop for OnednnCtx {
    fn drop(&mut self) {
        unsafe {
            // Order: primitive → mds → primitive_desc → stream → engine.
            if let Ok(cache) = self.primitive_cache.lock() {
                for (_, c) in cache.iter() {
                    let _ = (self.lib.primitive_destroy)(c.primitive);
                    let _ = (self.lib.memory_desc_destroy)(c.src_md);
                    let _ = (self.lib.memory_desc_destroy)(c.w_md);
                    let _ = (self.lib.memory_desc_destroy)(c.b_md);
                    let _ = (self.lib.memory_desc_destroy)(c.dst_md);
                    let _ = (self.lib.primitive_desc_destroy)(c.pd);
                }
            }
            let _ = (self.lib.stream_destroy)(self.stream);
            let _ = (self.lib.engine_destroy)(self.engine);
        }
    }
}

// ---------------------------------------------------------------------------
// Global initialization.
// ---------------------------------------------------------------------------

static ONEDNN_CTX: OnceLock<Option<OnednnCtx>> = OnceLock::new();

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn workspace_root() -> Option<PathBuf> {
    // Crate manifest is at <workspace>/crates/qwen-asr/Cargo.toml.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
}

fn candidate_sdk_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("ONE_DNN_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(root) = workspace_root() {
        // The SDK may live next to the workspace (e.g. CI layout) or inside it.
        let candidates = [
            root.join("onednn_sdk_2024_2_1"),
            root.join("onednn_sdk_devel"),
            root.join("onednn_sdk"),
        ];
        for sdk in candidates {
            dirs.push(sdk.clone());
            if let Some(parent) = root.parent() {
                dirs.push(parent.join(sdk.file_name().unwrap_or_default()));
            }
        }
        // Locally-built oneDNN 3.7.1 from source: dnnl.dll is in
        // <clawd>/onednn_build/src/Release/ (sibling of QwenASR workspace).
        if let Some(parent) = root.parent() {
            dirs.push(parent.join("onednn_build").join("src").join("Release"));
        }
    }
    // Also allow discovery from the executable / working directory tree.
    for base in std::env::current_exe()
        .ok()
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .into_iter()
        .chain(std::env::current_dir().ok())
    {
        dirs.push(base);
    }
    dirs
}

fn locate_dnnl_dll() -> Option<PathBuf> {
    for sdk in candidate_sdk_dirs() {
        let p = sdk.join("Library").join("bin").join("dnnl.dll");
        if p.is_file() {
            return Some(p);
        }
        // Also permit a flat layout where dnnl.dll sits directly in the SDK root.
        let p = sdk.join("dnnl.dll");
        if p.is_file() {
            return Some(p);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("dnnl.dll");
            if p.is_file() {
                return Some(p);
            }
        }
    }

    // Do not fall back to a bare PATH search: an incompatible dnnl.dll on PATH
    // can crash during engine creation. Users may set ONE_DNN_DIR explicitly.
    None
}

/// Set up the runtime environment required by the pip-provided oneDNN builds.
///
/// These builds use the DPC++ runtime for the CPU engine. The OpenCL CPU
/// runtime (`intelocl64.dll`) and its dependencies live in the SDK bin
/// directory. We point the OpenCL ICD loader at it and make sure the bin
/// directory is on PATH so transitive dependencies can be resolved.
fn setup_onednn_runtime_env(bin_dir: &Path) {
    let intelocl = bin_dir.join("intelocl64.dll");
    if intelocl.is_file() {
        if std::env::var("OCL_ICD_FILENAMES").is_err() {
            if let Some(s) = intelocl.to_str() {
                std::env::set_var("OCL_ICD_FILENAMES", s);
            }
        }
    }

    if let Some(bin_str) = bin_dir.to_str() {
        let current = std::env::var("PATH").unwrap_or_default();
        let sep = ";";
        if !current.split(sep).any(|s| s.eq_ignore_ascii_case(bin_str)) {
            let mut updated = String::with_capacity(bin_str.len() + 1 + current.len());
            updated.push_str(bin_str);
            updated.push_str(sep);
            updated.push_str(&current);
            std::env::set_var("PATH", updated);
        }
    }
}

fn init() -> Option<OnednnCtx> {
    let path = locate_dnnl_dll()?;
    if let Some(bin_dir) = path.parent() {
        setup_onednn_runtime_env(bin_dir);
    }
    let lib = OnednnLib::load(&path)?;

    let mut engine: *mut DnnlEngine = std::ptr::null_mut();
    unsafe {
        // Enable verbose output when DNNL_VERBOSE env var is set (for debugging).
        // Do NOT call set_verbose() unconditionally — it prevents oneDNN from
        // reading the DNNL_VERBOSE env var lazily on first use.
        if std::env::var("DNNL_VERBOSE").is_err() {
            // Only set level 2 if env var is not already set by the user.
            // Comment out the next line to disable verbose by default.
            // (lib.set_verbose)(2);
        }
        let ec = (lib.engine_create)(&mut engine, DNNL_ENGINE_CPU, 0);
        if ec != DNNL_STATUS_SUCCESS {
            return None;
        }
        let mut stream: *mut DnnlStream = std::ptr::null_mut();
        let sc = (lib.stream_create)(&mut stream, engine, DNNL_STREAM_IN_ORDER);
        if sc != DNNL_STATUS_SUCCESS {
            let _ = (lib.engine_destroy)(engine);
            return None;
        }
        Some(OnednnCtx {
            lib,
            engine,
            stream,
            primitive_cache: Mutex::new(HashMap::new()),
        })
    }
}

fn get_ctx() -> Option<&'static OnednnCtx> {
    ONEDNN_CTX.get_or_init(init).as_ref()
}

/// Whether oneDNN is available and was successfully initialized.
pub fn onednn_available() -> bool {
    get_ctx().is_some()
}

/// Whether the oneDNN path is allowed.
///
/// Default is **disabled** (Phase 4 A/B showed oneDNN 15.5% slower than P1 AVX2,
/// and the FFI path causes access violations on some dnnl.dll builds). Enable
/// explicitly via `QWEN_ASR_ENABLE_ONEDNN=1` for re-evaluation on VNNI CPUs or
/// larger-L3 hardware (see docs/research/failed-optimizations-backup-paths.md).
pub fn onednn_enabled() -> bool {
    std::env::var("QWEN_ASR_ENABLE_ONEDNN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Convolution wrapper.
// ---------------------------------------------------------------------------

/// Try to compute a 2D NCHW/OIHW convolution using oneDNN.
/// Returns `true` on success; on failure the caller must fall back.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_onednn(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> bool {
    let _pg = super::ProfileGuard::new(&super::PROF.conv2d_op);

    if !onednn_enabled() {
        return false;
    }

    let ctx = match get_ctx() {
        Some(c) => c,
        None => return false,
    };

    let bias = match bias {
        Some(b) => b,
        // Minimal prototype: require bias (all encoder conv layers use bias).
        None => return false,
    };

    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let spatial_out = h_out * w_out;

    if out.len() < c_out * spatial_out
        || input.len() < c_in * h_in * w_in
        || weight.len() < c_out * c_in * kh * kw
        || bias.len() < c_out
    {
        return false;
    }

    unsafe { run_conv(ctx, out, input, weight, bias, c_in, c_out, h_in, w_in, kh, kw, stride, padding, h_out, w_out) }
}

macro_rules! conv_err {
    ($c_in:expr, $h_in:expr, $w_in:expr, $kh:expr, $c_out:expr, $h_out:expr, $w_out:expr, $kw:expr, $stride:expr, $padding:expr, $step:expr $(, $ret:expr)?) => {
        {
            $(
                let _ = $ret;
            )?
            eprintln!(
                "[onednn] conv {}x{}x{}x{} -> {}x{}x{}x{} s={} p={} failed at {}",
                $c_in, $h_in, $w_in, $kh, $c_out, $h_out, $w_out, $kw, $stride, $padding, $step
            );
        }
    };
}

/// Create a oneDNN convolution primitive for a fixed shape (NCHW/OIHW, bias).
///
/// Returns a `CachedConv` containing the primitive, its descriptor, and the
/// four input memory descriptors. All handles are owned by the caller and
/// must be destroyed (primitive → mds → pd) before the engine is destroyed.
///
/// The input mds are kept alive (not destroyed) because the query API is
/// broken on this oneDNN build — we reuse the exact same md objects at
/// execute time to create compatible memory objects.
#[allow(clippy::too_many_arguments)]
unsafe fn create_conv_primitive(
    ctx: &OnednnCtx,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
) -> Option<CachedConv> {
    let lib = &ctx.lib;

    let src_dims: [i64; 4] = [1, c_in as i64, h_in as i64, w_in as i64];
    let w_dims: [i64; 4] = [c_out as i64, c_in as i64, kh as i64, kw as i64];
    let bias_dims: [i64; 1] = [c_out as i64];
    let dst_dims: [i64; 4] = [1, c_out as i64, h_out as i64, w_out as i64];

    let strides: [i64; 2] = [stride as i64, stride as i64];
    let padding_l: [i64; 2] = [padding as i64, padding as i64];
    let padding_r: [i64; 2] = [padding as i64, padding as i64];
    let dilates: [i64; 2] = [0, 0];

    let mut src_md: *mut DnnlMemoryDesc = std::ptr::null_mut();
    let mut w_md: *mut DnnlMemoryDesc = std::ptr::null_mut();
    let mut b_md: *mut DnnlMemoryDesc = std::ptr::null_mut();
    let mut dst_md: *mut DnnlMemoryDesc = std::ptr::null_mut();

    macro_rules! ok {
        ($e:expr, $step:literal) => {
            if $e != DNNL_STATUS_SUCCESS {
                conv_err!(c_in, h_in, w_in, kh, c_out, h_out, w_out, kw, stride, padding, $step);
                return None;
            }
        };
    }

    ok!((lib.memory_desc_create_with_tag)(
        &mut src_md,
        4,
        src_dims.as_ptr(),
        DNNL_DATA_TYPE_F32,
        DNNL_FORMAT_TAG_NCHW,
    ), "src_md");
    ok!((lib.memory_desc_create_with_tag)(
        &mut w_md,
        4,
        w_dims.as_ptr(),
        DNNL_DATA_TYPE_F32,
        DNNL_FORMAT_TAG_NCHW,
    ), "w_md");
    ok!((lib.memory_desc_create_with_tag)(
        &mut b_md,
        1,
        bias_dims.as_ptr(),
        DNNL_DATA_TYPE_F32,
        DNNL_FORMAT_TAG_A,
    ), "b_md");
    ok!((lib.memory_desc_create_with_tag)(
        &mut dst_md,
        4,
        dst_dims.as_ptr(),
        DNNL_DATA_TYPE_F32,
        DNNL_FORMAT_TAG_NCHW,
    ), "dst_md");

    let mut pd: *mut DnnlPrimitiveDesc = std::ptr::null_mut();
    let pd_ok = (lib.convolution_forward_primitive_desc_create)(
        &mut pd,
        ctx.engine,
        DNNL_PROP_KIND_FORWARD_INFERENCE,
        DNNL_ALG_KIND_CONVOLUTION_DIRECT,
        src_md,
        w_md,
        b_md,
        dst_md,
        strides.as_ptr(),
        dilates.as_ptr(),
        padding_l.as_ptr(),
        padding_r.as_ptr(),
        std::ptr::null(),
    );

    if pd_ok != DNNL_STATUS_SUCCESS {
        conv_err!(c_in, h_in, w_in, kh, c_out, h_out, w_out, kw, stride, padding, "pd_create");
        let _ = (lib.memory_desc_destroy)(src_md);
        let _ = (lib.memory_desc_destroy)(w_md);
        let _ = (lib.memory_desc_destroy)(b_md);
        let _ = (lib.memory_desc_destroy)(dst_md);
        return None;
    }

    let mut primitive: *mut DnnlPrimitive = std::ptr::null_mut();
    let prim_ok = (lib.primitive_create)(&mut primitive, pd);
    if prim_ok != DNNL_STATUS_SUCCESS {
        conv_err!(c_in, h_in, w_in, kh, c_out, h_out, w_out, kw, stride, padding, "primitive_create");
        let _ = (lib.primitive_desc_destroy)(pd);
        let _ = (lib.memory_desc_destroy)(src_md);
        let _ = (lib.memory_desc_destroy)(w_md);
        let _ = (lib.memory_desc_destroy)(b_md);
        let _ = (lib.memory_desc_destroy)(dst_md);
        return None;
    }

    // Keep ALL handles alive — run_conv reuses the mds at execute time.
    Some(CachedConv { primitive, pd, src_md, w_md, b_md, dst_md })
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_conv(
    ctx: &OnednnCtx,
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
) -> bool {
    let lib = &ctx.lib;

    let key = ConvKey {
        c_in,
        c_out,
        h_in,
        w_in,
        kh,
        kw,
        stride,
        padding,
        h_out,
        w_out,
    };

    let cached = {
        let mut cache = match ctx.primitive_cache.lock() {
            Ok(c) => c,
            Err(_) => return false,
        };
        if let Some(c) = cache.get(&key) {
            // Clone the handle values out of the cache (pointers are Copy).
            CachedConv {
                primitive: c.primitive,
                pd: c.pd,
                src_md: c.src_md,
                w_md: c.w_md,
                b_md: c.b_md,
                dst_md: c.dst_md,
            }
        } else {
            match create_conv_primitive(ctx, c_in, c_out, h_in, w_in, kh, kw, stride, padding, h_out, w_out) {
                Some(c) => {
                    cache.insert(key, CachedConv {
                        primitive: c.primitive,
                        pd: c.pd,
                        src_md: c.src_md,
                        w_md: c.w_md,
                        b_md: c.b_md,
                        dst_md: c.dst_md,
                    });
                    c
                }
                None => return false,
            }
        }
    };

    // Reuse the exact same md objects that were passed to pd creation.
    // The query API is broken on this oneDNN build, and independently-created
    // equivalent mds are rejected at execute (INVALID_ARGS) — but the original
    // md objects that the primitive desc was created with are accepted.
    let src_md = cached.src_md;
    let w_md = cached.w_md;
    let b_md = cached.b_md;
    let dst_md = cached.dst_md;
    let primitive = cached.primitive;

    macro_rules! ok {
        ($e:expr, $step:literal) => {
            if $e != DNNL_STATUS_SUCCESS {
                conv_err!(c_in, h_in, w_in, kh, c_out, h_out, w_out, kw, stride, padding, $step);
                return false;
            }
        };
    }

    let mut src_mem: *mut DnnlMemory = std::ptr::null_mut();
    let mut w_mem: *mut DnnlMemory = std::ptr::null_mut();
    let mut b_mem: *mut DnnlMemory = std::ptr::null_mut();
    let mut dst_mem: *mut DnnlMemory = std::ptr::null_mut();

    ok!((lib.memory_create)(
        &mut src_mem, src_md, ctx.engine, input.as_ptr() as *mut c_void,
    ), "src_mem");
    ok!((lib.memory_create)(
        &mut w_mem, w_md, ctx.engine, weight.as_ptr() as *mut c_void,
    ), "w_mem");
    ok!((lib.memory_create)(
        &mut b_mem, b_md, ctx.engine, bias.as_ptr() as *mut c_void,
    ), "b_mem");
    ok!((lib.memory_create)(
        &mut dst_mem, dst_md, ctx.engine, out.as_mut_ptr() as *mut c_void,
    ), "dst_mem");

    let args = [
        DnnlExecArg { arg: DNNL_ARG_SRC, memory: src_mem },
        DnnlExecArg { arg: DNNL_ARG_WEIGHTS, memory: w_mem },
        DnnlExecArg { arg: DNNL_ARG_BIAS, memory: b_mem },
        DnnlExecArg { arg: DNNL_ARG_DST, memory: dst_mem },
    ];

    let exec_ok = (lib.primitive_execute)(primitive, ctx.stream, args.len() as i32, args.as_ptr());

    let _ = (lib.memory_destroy)(src_mem);
    let _ = (lib.memory_destroy)(w_mem);
    let _ = (lib.memory_destroy)(b_mem);
    let _ = (lib.memory_destroy)(dst_mem);
    // NOTE: src_md/w_md/b_md/dst_md are owned by the cache — do NOT destroy.

    if exec_ok != DNNL_STATUS_SUCCESS {
        eprintln!(
            "[onednn] conv {}x{}x{}x{} -> {}x{}x{}x{} s={} p={} execute failed: status={}",
            c_in, h_in, w_in, kh, c_out, h_out, w_out, kw, stride, padding, exec_ok
        );
        return false;
    }

    ok!((lib.stream_wait)(ctx.stream), "stream_wait");
    true
}

// env_or is reserved for future use (e.g. explicit ONE_DNN_PATH override).
#[allow(dead_code)]
fn _env_or(name: &str, default: &str) -> String {
    env_or(name, default)
}
