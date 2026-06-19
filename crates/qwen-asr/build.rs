fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Check if blas feature is enabled via CARGO_FEATURE_BLAS env var
    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        match target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                println!("cargo:rustc-link-lib=openblas");
            }
            "windows" => {
                // OpenBLAS for Windows: link against openblas.lib (import lib for libopenblas.dll)
                // The DLL name is libopenblas.dll, but we link with the import lib named openblas.lib
                println!("cargo:rustc-link-lib=dylib=openblas");
                // Add search path for the OpenBLAS library
                let openblas_dir = std::env::var("OPENBLAS_DIR").unwrap_or_else(|_| {
                    // Default path
                    "C:\\Users\\Administrator\\clawd\\openblas".to_string()
                });
                println!("cargo:rustc-link-search=native={}", openblas_dir);
            }
            _ => {
                // No BLAS available, will use fallback matmul
            }
        }
    }
}
