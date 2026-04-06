//! Gradient Clipping Kernels
//!
//! GPU kernels for gradient norm clipping to prevent exploding gradients.
//!
//! ## Clipping Formula
//!
//! ```text
//! if ‖g‖ > max_norm:
//!     g = g * (max_norm / ‖g‖)
//! ```
//!
//! This is L2 norm clipping, which scales all gradients uniformly.
//!
//! ## Usage Pattern
//!
//! 1. Compute gradient L2 norm on host (sum of squares, then sqrt)
//! 2. Compute scale = min(1.0, max_norm / norm)
//! 3. Apply scale to all gradients using `GradientClipKernel`
//!
//! The norm computation stays on host because it requires a global reduction,
//! which is more efficient with cuBLAS/host-side than a custom kernel for
//! typical gradient sizes.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Gradient clipping kernel (apply pre-computed scale)
///
/// This kernel applies a clipping scale factor to all gradients.
/// The scale is pre-computed on the host as `min(1.0, max_norm / grad_norm)`.
///
/// # Parameters
/// - `grads`: gradient tensor (updated in-place)
/// - `scale`: clipping scale factor (pre-computed)
/// - `n`: number of elements
///
/// # Example
///
/// ```ignore
/// // On host:
/// let norm = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
/// let scale = (max_norm / norm).min(1.0);
///
/// // Launch kernel:
/// gradient_clip_kernel.launch(grads, scale, n);
/// ```
#[derive(Debug, Clone)]
pub struct GradientClipKernel {
    /// Number of gradient elements
    pub n: u32,
}

impl GradientClipKernel {
    /// Create a new gradient clipping kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for GradientClipKernel {
    fn name(&self) -> &str {
        "gradient_clip"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gradient_clip")
            .param(PtxType::U64, "grads_ptr")
            .param(PtxType::F32, "scale")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load n and bounds check
                let n = ctx.load_param_u32("n");
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Load pointer and scale
                let grads_ptr = ctx.load_param_u64("grads_ptr");
                let scale = ctx.load_param_f32("scale");

                // Calculate byte offset
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let grad_addr = ctx.add_u64(grads_ptr, offset);

                // Load gradient
                let grad = ctx.ld_global_f32(grad_addr);

                // Scale gradient: grad_new = grad * scale
                let grad_new = ctx.mul_f32(grad, scale);

                // Store updated gradient
                ctx.st_global_f32(grad_addr, grad_new);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_clip_kernel_name() {
        let kernel = GradientClipKernel::new(1024);
        assert_eq!(kernel.name(), "gradient_clip");
    }

    #[test]
    fn test_gradient_clip_ptx_generation() {
        let kernel = GradientClipKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry gradient_clip"));

        // Verify parameters
        assert!(ptx.contains(".param .u64 grads_ptr"));
        assert!(ptx.contains(".param .f32 scale"));

        // Verify multiplication for scaling
        assert!(ptx.contains("mul.f32"));
    }
}
