//! Backward Pass Kernels for Training
//!
//! This module provides GPU kernels for computing gradients during backpropagation.
//! These kernels are essential for training neural networks with CUDA acceleration.
//!
//! ## Available Backward Kernels
//!
//! - **Activations**: ReLU, GELU, SiLU backward passes
//! - **Softmax**: Warp-parallel softmax gradient with shuffle reductions
//! - **RMSNorm**: RMS normalization gradients (input, gamma)
//! - **LayerNorm**: Layer normalization gradients (input, gamma, beta)
//! - **GEMM**: Matrix multiplication gradients for A and B
//!
//! ## Planned Kernels (Issue #85)
//!
//! - **FlashAttention**: Efficient attention backward with LSE reuse
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trueno_gpu::kernels::backward::{ReluBackwardKernel, GeluBackwardKernel};
//! use trueno_gpu::kernels::Kernel;
//!
//! // Create and emit backward kernels
//! let relu_bwd = ReluBackwardKernel::new(4096);
//! let ptx = relu_bwd.emit_ptx();
//!
//! let gelu_bwd = GeluBackwardKernel::new(4096);
//! let ptx = gelu_bwd.emit_ptx();
//! ```
//!
//! ## Mathematical Guarantees
//!
//! All backward kernels satisfy:
//! 1. **Correctness**: Analytical gradient matches finite-difference within ε < 1e-4
//! 2. **Numerical Stability**: No NaN/Inf for valid inputs
//! 3. **Barrier Safety**: Pass PARITY-114 validation
//! 4. **Determinism**: Same inputs produce same outputs
//!
//! ## Integration with entrenar
//!
//! These kernels enable speedup for fine-tuning in the `entrenar` training library
//! by moving backward passes from CPU (ndarray) to GPU (CUDA).

mod activations;
mod cross_entropy;
mod gemm;
mod layer_norm;
mod nf4_tensor_core;
mod rms_norm;
mod softmax;

// Re-export backward kernels
pub use activations::{GeluBackwardKernel, ReluBackwardKernel, SiluBackwardKernel};
pub use cross_entropy::{FusedCausalCrossEntropyKernel, FusedCrossEntropyKernel};
pub use gemm::{GemmBackwardAKernel, GemmBackwardBKernel};
pub use layer_norm::LayerNormBackwardKernel;
pub use nf4_tensor_core::Nf4TensorCoreGemmBackwardAKernel;
pub use rms_norm::{BatchedRmsNormBackwardKernel, RmsNormBackwardKernel};
pub use softmax::{BatchedSoftmaxBackwardKernel, SoftmaxBackwardKernel};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernel;

    #[test]
    fn test_all_backward_kernels_compile() {
        // Activation backward
        let relu = ReluBackwardKernel::new(1024);
        let gelu = GeluBackwardKernel::new(1024);
        let silu = SiluBackwardKernel::new(1024);

        assert!(relu.emit_ptx().contains(".entry"));
        assert!(gelu.emit_ptx().contains(".entry"));
        assert!(silu.emit_ptx().contains(".entry"));

        // Softmax backward
        let softmax = SoftmaxBackwardKernel::new(64, 32);
        assert!(softmax.emit_ptx().contains(".entry"));

        // Batched softmax backward (handles row_size > 32)
        let batched_softmax = BatchedSoftmaxBackwardKernel::new(64, 128);
        assert!(batched_softmax.emit_ptx().contains(".entry"));

        // RMSNorm backward
        let rms_norm = RmsNormBackwardKernel::new(64, 32, 1e-6);
        assert!(rms_norm.emit_ptx().contains(".entry"));

        // Batched RMSNorm backward (handles hidden_dim > 32)
        let batched_rms_norm = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
        assert!(batched_rms_norm.emit_ptx().contains(".entry"));

        // LayerNorm backward
        let layer_norm = LayerNormBackwardKernel::new(64, 32);
        assert!(layer_norm.emit_ptx().contains(".entry"));

        // GEMM backward
        let gemm_a = GemmBackwardAKernel::new(64, 64, 64);
        assert!(gemm_a.emit_ptx().contains(".entry"));
        let gemm_b = GemmBackwardBKernel::new(64, 64, 64);
        assert!(gemm_b.emit_ptx().contains(".entry"));
    }

    #[test]
    fn test_all_backward_kernels_barrier_safe() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(ReluBackwardKernel::new(1024)),
            Box::new(GeluBackwardKernel::new(1024)),
            Box::new(SiluBackwardKernel::new(1024)),
            Box::new(SoftmaxBackwardKernel::new(64, 32)),
            Box::new(BatchedSoftmaxBackwardKernel::new(64, 128)),
            Box::new(RmsNormBackwardKernel::new(64, 32, 1e-6)),
            Box::new(BatchedRmsNormBackwardKernel::new(64, 128, 1e-6)),
            Box::new(LayerNormBackwardKernel::new(64, 32)),
            Box::new(GemmBackwardAKernel::new(64, 64, 64)),
            Box::new(GemmBackwardBKernel::new(64, 64, 64)),
            Box::new(FusedCausalCrossEntropyKernel::new(32000)),
        ];

        for kernel in kernels {
            let result = kernel.analyze_barrier_safety();
            // Verify barrier safety and display name for diagnostics
            let name = kernel.name();
            assert!(result.is_safe, "Kernel {name} should be barrier-safe");
            assert!(result.violations.is_empty());
        }
    }

    #[test]
    fn test_reexported_kernel_names() {
        // Verify all re-exported kernels have correct names
        assert_eq!(ReluBackwardKernel::new(128).name(), "relu_backward");
        assert_eq!(GeluBackwardKernel::new(128).name(), "gelu_backward");
        assert_eq!(SiluBackwardKernel::new(128).name(), "silu_backward");
        assert_eq!(SoftmaxBackwardKernel::new(16, 16).name(), "softmax_backward");
        assert_eq!(BatchedSoftmaxBackwardKernel::new(16, 64).name(), "batched_softmax_backward");
        assert_eq!(RmsNormBackwardKernel::new(16, 16, 1e-5).name(), "rms_norm_backward");
        assert_eq!(
            BatchedRmsNormBackwardKernel::new(16, 64, 1e-5).name(),
            "batched_rms_norm_backward"
        );
        assert_eq!(LayerNormBackwardKernel::new(16, 16).name(), "layer_norm_backward");
        assert_eq!(GemmBackwardAKernel::new(32, 32, 32).name(), "gemm_backward_a");
        assert_eq!(GemmBackwardBKernel::new(32, 32, 32).name(), "gemm_backward_b");
    }

    #[test]
    fn test_activation_kernels_clone_and_debug() {
        // Test Clone trait on activation kernels
        let relu = ReluBackwardKernel::new(512);
        let relu_clone = relu.clone();
        assert_eq!(relu.n, relu_clone.n);

        let gelu = GeluBackwardKernel::new(512);
        let gelu_clone = gelu.clone();
        assert_eq!(gelu.n, gelu_clone.n);

        let silu = SiluBackwardKernel::new(512);
        let silu_clone = silu.clone();
        assert_eq!(silu.n, silu_clone.n);

        // Test Debug trait
        let debug_str = format!("{relu:?}");
        assert!(debug_str.contains("ReluBackwardKernel"));
        assert!(debug_str.contains("512"));
    }

    #[test]
    fn test_softmax_kernel_clone_and_debug() {
        let kernel = SoftmaxBackwardKernel::new(128, 32);
        let cloned = kernel.clone();
        assert_eq!(kernel.num_rows, cloned.num_rows);
        assert_eq!(kernel.row_size, cloned.row_size);

        let debug_str = format!("{kernel:?}");
        assert!(debug_str.contains("SoftmaxBackwardKernel"));
        assert!(debug_str.contains("128"));
        assert!(debug_str.contains("32"));
    }

    #[test]
    fn test_rms_norm_kernel_clone_and_debug() {
        let kernel = RmsNormBackwardKernel::new(64, 16, 1e-6);
        let cloned = kernel.clone();
        assert_eq!(kernel.num_rows, cloned.num_rows);
        assert_eq!(kernel.hidden_dim, cloned.hidden_dim);
        assert!((kernel.eps - cloned.eps).abs() < 1e-10);

        let debug_str = format!("{kernel:?}");
        assert!(debug_str.contains("RmsNormBackwardKernel"));
        assert!(debug_str.contains("64"));
        assert!(debug_str.contains("16"));

        let batched = BatchedRmsNormBackwardKernel::new(32, 128, 1e-5);
        let batched_clone = batched.clone();
        assert_eq!(batched.num_rows, batched_clone.num_rows);
        assert_eq!(batched.hidden_dim, batched_clone.hidden_dim);
        assert!((batched.eps - batched_clone.eps).abs() < 1e-10);
        let debug_str = format!("{batched:?}");
        assert!(debug_str.contains("BatchedRmsNormBackwardKernel"));
    }

    #[test]
    fn test_layer_norm_kernel_clone_and_debug() {
        let kernel = LayerNormBackwardKernel::new(256, 8);
        let cloned = kernel.clone();
        assert_eq!(kernel.num_rows, cloned.num_rows);
        assert_eq!(kernel.hidden_dim, cloned.hidden_dim);

        let debug_str = format!("{kernel:?}");
        assert!(debug_str.contains("LayerNormBackwardKernel"));
        assert!(debug_str.contains("256"));
        assert!(debug_str.contains('8'));
    }

    #[test]
    fn test_gemm_kernels_clone_and_debug() {
        let gemm_a = GemmBackwardAKernel::new(128, 256, 64);
        let gemm_a_clone = gemm_a.clone();
        assert_eq!(gemm_a.m, gemm_a_clone.m);
        assert_eq!(gemm_a.n, gemm_a_clone.n);
        assert_eq!(gemm_a.k, gemm_a_clone.k);

        let gemm_b = GemmBackwardBKernel::new(128, 256, 64);
        let gemm_b_clone = gemm_b.clone();
        assert_eq!(gemm_b.m, gemm_b_clone.m);
        assert_eq!(gemm_b.n, gemm_b_clone.n);
        assert_eq!(gemm_b.k, gemm_b_clone.k);

        let debug_a = format!("{gemm_a:?}");
        assert!(debug_a.contains("GemmBackwardAKernel"));
        assert!(debug_a.contains("128"));

        let debug_b = format!("{gemm_b:?}");
        assert!(debug_b.contains("GemmBackwardBKernel"));
        assert!(debug_b.contains("256"));
    }

    #[test]
    fn test_ptx_contains_correct_params() {
        // Verify PTX contains expected parameter declarations
        let relu_ptx = ReluBackwardKernel::new(64).emit_ptx();
        assert!(relu_ptx.contains("input_ptr"));
        assert!(relu_ptx.contains("grad_output_ptr"));
        assert!(relu_ptx.contains("grad_input_ptr"));

        let softmax_ptx = SoftmaxBackwardKernel::new(32, 16).emit_ptx();
        assert!(softmax_ptx.contains("output_ptr"));
        assert!(softmax_ptx.contains("num_rows"));
        assert!(softmax_ptx.contains("row_size"));

        let batched_softmax_ptx = BatchedSoftmaxBackwardKernel::new(32, 64).emit_ptx();
        assert!(batched_softmax_ptx.contains("output_ptr"));
        assert!(batched_softmax_ptx.contains("grad_output_ptr"));
        assert!(batched_softmax_ptx.contains("total_rows"));
        assert!(batched_softmax_ptx.contains("row_size"));

        let rms_norm_ptx = RmsNormBackwardKernel::new(32, 16, 1e-5).emit_ptx();
        assert!(rms_norm_ptx.contains("gamma_ptr"));
        assert!(rms_norm_ptx.contains("rms_ptr"));

        let batched_rms_ptx = BatchedRmsNormBackwardKernel::new(32, 64, 1e-5).emit_ptx();
        assert!(batched_rms_ptx.contains("input_ptr"));
        assert!(batched_rms_ptx.contains("gamma_ptr"));
        assert!(batched_rms_ptx.contains("grad_output_ptr"));
        assert!(batched_rms_ptx.contains("grad_input_ptr"));
        assert!(batched_rms_ptx.contains("grad_gamma_ptr"));
        assert!(batched_rms_ptx.contains("num_rows"));
        assert!(batched_rms_ptx.contains("hidden_dim"));
        assert!(batched_rms_ptx.contains(".param .f32 eps"));

        let layer_norm_ptx = LayerNormBackwardKernel::new(32, 16).emit_ptx();
        assert!(layer_norm_ptx.contains("mean_ptr"));
        assert!(layer_norm_ptx.contains("rstd_ptr"));

        let gemm_a_ptx = GemmBackwardAKernel::new(16, 16, 16).emit_ptx();
        assert!(gemm_a_ptx.contains("grad_c_ptr"));
        assert!(gemm_a_ptx.contains("b_ptr"));
        assert!(gemm_a_ptx.contains("grad_a_ptr"));

        let gemm_b_ptx = GemmBackwardBKernel::new(16, 16, 16).emit_ptx();
        assert!(gemm_b_ptx.contains("a_ptr"));
        assert!(gemm_b_ptx.contains("grad_b_ptr"));
    }

    #[test]
    fn test_backward_kernels_with_various_sizes() {
        // Test with minimum viable sizes
        let relu_small = ReluBackwardKernel::new(1);
        assert!(relu_small.emit_ptx().contains(".entry"));

        let gelu_small = GeluBackwardKernel::new(1);
        assert!(gelu_small.emit_ptx().contains(".entry"));

        let silu_small = SiluBackwardKernel::new(1);
        assert!(silu_small.emit_ptx().contains(".entry"));

        // Test with edge case sizes
        let softmax_min = SoftmaxBackwardKernel::new(1, 1);
        assert!(softmax_min.emit_ptx().contains(".entry"));

        let rms_norm_min = RmsNormBackwardKernel::new(1, 1, 1e-8);
        assert!(rms_norm_min.emit_ptx().contains(".entry"));

        let layer_norm_min = LayerNormBackwardKernel::new(1, 1);
        assert!(layer_norm_min.emit_ptx().contains(".entry"));

        let gemm_min = GemmBackwardAKernel::new(1, 1, 1);
        assert!(gemm_min.emit_ptx().contains(".entry"));

        let gemm_b_min = GemmBackwardBKernel::new(1, 1, 1);
        assert!(gemm_b_min.emit_ptx().contains(".entry"));
    }

    #[test]
    fn test_backward_kernels_large_sizes() {
        // Test with large (but reasonable) sizes
        let relu_large = ReluBackwardKernel::new(1_000_000);
        assert!(relu_large.emit_ptx().contains(".entry"));
        assert_eq!(relu_large.n, 1_000_000);

        let gelu_large = GeluBackwardKernel::new(1_000_000);
        assert!(gelu_large.emit_ptx().contains(".entry"));
        assert_eq!(gelu_large.n, 1_000_000);

        let silu_large = SiluBackwardKernel::new(1_000_000);
        assert!(silu_large.emit_ptx().contains(".entry"));
        assert_eq!(silu_large.n, 1_000_000);
    }

    #[test]
    fn test_backward_ptx_version_and_target() {
        // All kernels should generate valid PTX with version info
        let kernels_ptx: Vec<String> = vec![
            ReluBackwardKernel::new(64).emit_ptx(),
            GeluBackwardKernel::new(64).emit_ptx(),
            SiluBackwardKernel::new(64).emit_ptx(),
            SoftmaxBackwardKernel::new(16, 16).emit_ptx(),
            BatchedSoftmaxBackwardKernel::new(16, 64).emit_ptx(),
            RmsNormBackwardKernel::new(16, 16, 1e-6).emit_ptx(),
            BatchedRmsNormBackwardKernel::new(16, 64, 1e-6).emit_ptx(),
            LayerNormBackwardKernel::new(16, 16).emit_ptx(),
            GemmBackwardAKernel::new(16, 16, 16).emit_ptx(),
            GemmBackwardBKernel::new(16, 16, 16).emit_ptx(),
        ];

        for ptx in kernels_ptx {
            assert!(ptx.contains(".version"), "PTX should contain version");
            assert!(ptx.contains(".target"), "PTX should contain target");
            assert!(ptx.contains("ret;"), "PTX should contain return instruction");
        }
    }
}
