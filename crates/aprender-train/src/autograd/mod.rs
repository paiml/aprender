//! Tape-based autograd engine
//!
//! Provides automatic differentiation using a computational graph with gradient tape.
//!
//! ## CUDA Acceleration (SPEC-FT-001 v3.0.0)
//!
//! When the `cuda` feature is enabled, use `CudaTensor` for GPU-accelerated training:
//!
//! ```ignore
//! use entrenar::autograd::{CudaDevice, CudaTensor};
//!
//! let device = CudaDevice::default_device()?;
//! let tensor = CudaTensor::from_vec(&device, vec![1.0, 2.0, 3.0], true)?;
//! ```
//!
//! ## Gradient Checkpointing
//!
//! For memory-efficient training of large models, use the `checkpoint` module:
//!
//! ```ignore
//! use entrenar::autograd::checkpoint::{checkpoint, CheckpointConfig};
//!
//! let output = checkpoint(|x| layer.forward(x), &input);
//! ```

mod backward;
pub mod checkpoint;
mod context;
#[cfg(feature = "cuda")]
pub mod cuda_backward;
#[cfg(feature = "cuda")]
pub mod cuda_forward;
#[cfg(feature = "cuda")]
pub mod cuda_optim;
pub mod cuda_tensor;
pub mod cuda_training;
pub mod graph_opt;
pub(crate) mod ops;
pub mod precision;
mod tensor;
#[cfg(feature = "gpu")]
pub mod wgpu_backward;
#[cfg(feature = "gpu")]
pub mod wgpu_block;
#[cfg(feature = "gpu")]
pub mod wgpu_cross_entropy;
#[cfg(feature = "gpu")]
pub mod wgpu_training;

#[cfg(test)]
mod tests;

pub use backward::BackwardOp;
pub use checkpoint::{
    checkpoint, checkpoint_if, estimate_memory_savings, estimate_policy_tradeoff,
    optimal_checkpoints, BinomialCheckpointing, CheckpointConfig, CheckpointManager,
    CheckpointPolicy, CheckpointedSegment, CustomPolicy, MemoryBudget, OperationInfo,
    PolicyCheckpointManager, SaveAll, SaveMatmuls, SaveNothing, SaveUnbatchedMatmuls,
};
pub use context::Context;
pub use cuda_training::{cuda_training_available, CudaTrainer};
pub use graph_opt::{
    traced_binary_op, CommonSubexprElimination, ComputeGraph, ConstantFolding, DeadCodeElimination,
    GraphOptimizer, NodeId, OpType, OptimizationPass, OptimizationReport, ShapeError, ShapeTracker,
    TracedTensor, TracedValue,
};
pub use ops::*;
pub use precision::{
    bf16_to_f32, bf16_truncate, f32_to_bf16, f32_to_fp16, fp16_to_f32, gemm_bf16_reference,
    GradScaler, MixedPrecisionConfig, Precision,
};
pub use tensor::Tensor;

/// Process-wide counter of device-side backward kernel launches observed so
/// far — cuBLAS backward GEMM calls (`cuda_forward::matmul`/`matmul_f16`).
///
/// PMAT-991 (#2906): this counter, not the caller's request string, is the
/// ONLY source of truth for whether a cuBLAS backward path actually engaged.
/// The old CLI banner was derived from the request alone and could claim GPU
/// training while every backward ran on the CPU.
static BACKWARD_KERNEL_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Number of device-side backward kernel launches observed since process
/// start (or the last [`reset_backward_kernel_launches`]).
pub fn backward_kernel_launches() -> u64 {
    BACKWARD_KERNEL_LAUNCHES.load(std::sync::atomic::Ordering::SeqCst)
}

/// Record one device-side backward kernel launch (PMAT-991). Called from
/// every cuBLAS backward GEMM call site in `cuda_forward::matmul` /
/// `cuda_forward::matmul_f16`, only after the launch itself succeeded.
///
/// Only reachable when the `cuda` feature is compiled in — every call site
/// lives behind `#[cfg(feature = "cuda")]`, so a `cpu-fallback` build never
/// calls this and must not be warned/denied as dead code for it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_backward_kernel_launch() {
    BACKWARD_KERNEL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Derive the "cuBLAS backward engaged" training banner strictly from
/// observed device-side backward kernel launches (PMAT-991, #2906) — never
/// from `requested` alone. A banner claiming a cuBLAS backward path with
/// zero launched backward kernels is exactly the defect this forecloses.
/// `launches_at_start` is the caller's snapshot of [`backward_kernel_launches`]
/// taken before its training run; there is deliberately no reset.
pub fn training_backend_banner(requested: &str, launches_at_start: u64) -> Option<String> {
    if requested != "cuda" {
        return None;
    }
    // Since-start semantics (R-3 review quorum 2026-09-06, 3/3): the counter is
    // process-wide and monotonic, so a second fine-tune in the same process, or
    // a long-lived server, must not inherit an earlier run's launches. The
    // caller snapshots `backward_kernel_launches()` before its run and the
    // banner speaks only about launches observed since then.
    let launches = backward_kernel_launches().saturating_sub(launches_at_start);
    if launches == 0 {
        return None;
    }
    Some(format!(
        "[gpu-backend] CUDA — device-side backward engaged ({launches} device-side backward launches this run)"
    ))
}

/// Perform backward pass on a tensor
pub fn backward(tensor: &mut Tensor, grad_output: Option<ndarray::Array1<f32>>) {
    if let Some(grad) = grad_output {
        tensor.set_grad(grad);
    } else {
        // Initialize with ones for scalar loss
        let ones = ndarray::Array1::ones(tensor.data().len());
        tensor.set_grad(ones);
    }

    if let Some(op) = tensor.backward_op() {
        op.backward();
    }
}
