// =============================================================================
// QWEN-009: FUSED RMSNORM + GATE+UP Q4K GEMV + SWIGLU KERNEL (3-WAY FUSION)
// =============================================================================

mod build_ptx;

/// Fused RMSNorm + Gate+Up Q4_K GEMV + SwiGLU kernel for FFN optimization
///
/// This kernel eliminates all intermediate global memory roundtrips in FFN:
/// - Standard flow: RMSNorm -> gate GEMV -> up GEMV -> SwiGLU (4 kernels)
/// - Fused flow: Single kernel with shared memory (1 kernel)
///
/// Memory bandwidth savings:
/// - Eliminates: 3x hidden_size x 4 bytes intermediate writes/reads
/// - For Qwen 3B (hidden=3584): saves 42KB per FFN call
///
/// # Grid Configuration
///
/// - Block: 256 threads (8 warps)
/// - Grid: intermediate_size blocks (one per output element)
/// - Shared memory: hidden_size x 4 bytes for normalized input cache
#[derive(Debug, Clone)]
pub struct FusedRmsNormGateUpSwigluQ4KKernel {
    /// K dimension (hidden size, input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (intermediate size, output dimension)
    pub n: u32,
    /// Epsilon for RMSNorm numerical stability
    pub epsilon: f32,
}

impl FusedRmsNormGateUpSwigluQ4KKernel {
    /// Create a new fused RMSNorm + Gate+Up + SwiGLU Q4_K kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self {
            k,
            n,
            epsilon: 1e-6, // Qwen uses 1e-6 epsilon
        }
    }

    /// Set custom epsilon value for RMSNorm
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

#[cfg(test)]
mod tests_3way_fusion {
    use super::*;
    use crate::kernels::Kernel;

    #[test]
    fn test_fused_rmsnorm_gate_up_swiglu_q4k_kernel_builds() {
        let kernel = FusedRmsNormGateUpSwigluQ4KKernel::new(3584, 18944);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_rmsnorm_gate_up_swiglu_q4k"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_fused_rmsnorm_gate_up_swiglu_q4k_kernel_name() {
        let kernel = FusedRmsNormGateUpSwigluQ4KKernel::new(1024, 4096);
        assert_eq!(kernel.name(), "fused_rmsnorm_gate_up_swiglu_q4k");
    }

    #[test]
    fn test_fused_rmsnorm_gate_up_swiglu_q4k_kernel_clone() {
        let kernel = FusedRmsNormGateUpSwigluQ4KKernel::new(2048, 8192);
        let cloned = kernel.clone();
        assert_eq!(cloned.k, kernel.k);
        assert_eq!(cloned.n, kernel.n);
        assert_eq!(cloned.epsilon, kernel.epsilon);
    }

    #[test]
    fn test_fused_rmsnorm_gate_up_swiglu_q4k_with_epsilon() {
        let kernel = FusedRmsNormGateUpSwigluQ4KKernel::new(2048, 8192).with_epsilon(1e-5);
        assert_eq!(kernel.epsilon, 1e-5);
    }

    #[test]
    fn test_fused_rmsnorm_gate_up_swiglu_q4k_kernel_debug() {
        let kernel = FusedRmsNormGateUpSwigluQ4KKernel::new(2048, 8192);
        let debug = format!("{:?}", kernel);
        assert!(debug.contains("FusedRmsNormGateUpSwigluQ4KKernel"));
        assert!(debug.contains("2048"));
        assert!(debug.contains("8192"));
    }
}
