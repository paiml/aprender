//! CUDA Driver Tests (PMAT-018: 95% Coverage Strike)
//!
//! These tests REQUIRE CUDA hardware. They WILL NOT SKIP.
//! The RTX 4090 is present. Execute the tests.

#![cfg(all(test, feature = "cuda"))]

use super::context::{cuda_available, device_count, get_driver, CudaContext};
use super::graph::{CaptureMode, CudaGraph};
use super::memory::GpuBuffer;
use super::module::CudaModule;
use super::stream::CudaStream;
use super::types::LaunchConfig;
use std::ffi::c_void;

/// Serialises graph capture against context-wide synchronization.
///
/// GPU-ORD-4: `cuCtxSynchronize` is illegal while *any* stream in the context
/// is capturing a graph, and the attempt also invalidates the capture — so the
/// two tests destroy each other, in both directions at once:
/// `test_context_synchronize` panicking with
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (900) while
/// `test_cuda_graph_capture_modes` panicked with
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` (901) in the same run. Every test in
/// this crate shares one primary context, so this is not avoidable by using a
/// different context.
///
/// This is the CUDA rule, not a defect in either test, so the fix is to stop
/// running them at the same time. Capture mode is irrelevant here: `Global` vs
/// `ThreadLocal` changes how other threads' *unsafe actions* are policed, not
/// whether a context-wide sync may proceed during a capture.
///
/// CONTRACT: a test that opens a stream capture, or that calls
/// `CudaContext::synchronize()`, MUST hold this lock — and must also be listed
/// in the `gpu-exclusive` group in `.config/nextest.toml`, because under
/// nextest each test is its own process and an in-process lock excludes
/// nothing.
pub(super) static CAPTURE_VS_CTX_SYNC: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`CAPTURE_VS_CTX_SYNC`], recovering from poisoning.
///
/// The lock only orders test bodies; it guards no invariant a panic could
/// leave broken, so recovering stops one unrelated failure from cascading.
pub(super) fn capture_vs_ctx_sync() -> std::sync::MutexGuard<'static, ()> {
    CAPTURE_VS_CTX_SYNC
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

mod cuda_graph_tests;
mod driver_and_context;
mod gpu_buffer;
mod module_tests;
mod streams;
mod stress_and_advanced;
