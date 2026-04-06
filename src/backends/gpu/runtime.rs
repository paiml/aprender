//! Cross-platform async runtime helpers for GPU operations.
//!
//! - Native: Uses `pollster::block_on` for sync wrappers
//! - WASM: Sync wrappers unavailable; use async methods directly

/// Block on async code (native only).
///
/// On WASM, this function is not available - use async methods directly
/// with `wasm_bindgen_futures::spawn_local` or await.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    pollster::block_on(f)
}

/// Check if sync GPU operations are available.
///
/// Returns `true` on native platforms, `false` on WASM.
#[cfg(not(target_arch = "wasm32"))]
pub const fn sync_available() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
pub const fn sync_available() -> bool {
    false
}

/// Spawn async task for WASM.
#[cfg(all(feature = "gpu-wasm", target_arch = "wasm32"))]
pub fn spawn_local<F>(f: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(f);
}

/// Log to console (WASM).
#[cfg(all(feature = "gpu-wasm", target_arch = "wasm32"))]
pub fn console_log(s: &str) {
    web_sys::console::log_1(&s.into());
}

#[cfg(not(target_arch = "wasm32"))]
pub fn console_log(s: &str) {
    eprintln!("{}", s);
}
