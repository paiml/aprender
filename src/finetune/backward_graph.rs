//! PMAT-464/477: CUDA graph capture for NF4 backward pass.
//!
//! With fused LoRA gradient clipping (PMAT-477, zero D2H sync), the entire
//! backward loop is capturable: backward + fused_clip + optimizer are all
//! async GPU kernel launches with no host-device synchronization.
//!
//! This module implements capture/replay of the 28-layer backward loop,
//! eliminating per-step kernel launch overhead.
//!
//! # Contract: cuda-graph-backward-v1.yaml
//!
//! - F-GRAPH-BWD-001: Loss trajectory matches ungraphed within 0.1
//! - F-GRAPH-BWD-002: Graph capture succeeds (no CUDA_ERROR)
//! - F-GRAPH-BWD-003: Throughput >= 1.10x ungraphed at batch=4

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CaptureMode, CudaGraphExec, CudaStream};

/// Cached backward graph state.
#[cfg(feature = "cuda")]
pub(crate) struct BackwardGraphState {
    /// Cached CUDA graph executable for backward replay
    pub exec: CudaGraphExec,
    /// seq_len this graph was captured at (invalidate on change)
    pub cached_seq_len: usize,
}

/// Check if backward graph capture is enabled via environment variable.
#[cfg(feature = "cuda")]
pub(crate) fn use_backward_graph() -> bool {
    static USE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *USE.get_or_init(|| std::env::var("CUDA_GRAPH").as_deref() == Ok("1"))
}

/// Try to capture the backward loop into a CUDA graph.
///
/// Called on the first backward at a given seq_len. Records all kernel
/// launches (backward + fused_clip + optimizer per layer) into a graph.
///
/// # Returns
///
/// `Some(BackwardGraphState)` on successful capture, `None` on failure.
#[cfg(feature = "cuda")]
pub(crate) fn try_capture_backward<F>(
    stream: &CudaStream,
    seq_len: usize,
    backward_fn: F,
) -> Option<BackwardGraphState>
where
    F: FnOnce() -> Option<()>,
{
    // Pre-allocate cuBLAS workspace must have happened before this point (PMAT-063)
    stream
        .begin_capture(CaptureMode::ThreadLocal)
        .map_err(|e| eprintln!("[CUDA] Backward graph capture begin failed: {e}"))
        .ok()?;

    let result = backward_fn();

    if result.is_none() {
        // Backward failed during capture — abort
        let _ = stream.end_capture();
        eprintln!("[CUDA] Backward graph capture aborted: backward failed");
        return None;
    }

    match stream.end_capture() {
        Ok(graph) => match graph.instantiate() {
            Ok(exec) => {
                eprintln!("[CUDA] Backward graph captured: seq_len={seq_len}");
                Some(BackwardGraphState { exec, cached_seq_len: seq_len })
            }
            Err(e) => {
                eprintln!("[CUDA] Backward graph instantiate failed: {e}");
                None
            }
        },
        Err(e) => {
            eprintln!("[CUDA] Backward graph end_capture failed: {e}");
            None
        }
    }
}

/// Replay a previously captured backward graph.
///
/// Must be called with the same seq_len the graph was captured at.
/// All buffer pointers must remain valid (pre-allocated training state).
#[cfg(feature = "cuda")]
pub(crate) fn replay_backward(state: &BackwardGraphState, stream: &CudaStream) -> Option<()> {
    state
        .exec
        .launch(stream.raw())
        .map_err(|e| eprintln!("[CUDA] Backward graph replay failed: {e}"))
        .ok()
}
