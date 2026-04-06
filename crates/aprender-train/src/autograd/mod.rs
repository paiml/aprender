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
