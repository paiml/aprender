mod activations;
mod basic_ops;
mod matmul_and_convolve;
mod scalar_matching;
mod tiled_ops;

use super::*;
use std::sync::OnceLock;

/// Shared GPU backend for fast test execution (initialized once)
static SHARED_GPU: OnceLock<Option<GpuBackend>> = OnceLock::new();

/// Get shared GPU backend (fast) or None if unavailable
fn get_shared_gpu() -> Option<GpuBackend> {
    SHARED_GPU
        .get_or_init(|| if GpuBackend::is_available() { Some(GpuBackend::new()) } else { None })
        .clone()
}
