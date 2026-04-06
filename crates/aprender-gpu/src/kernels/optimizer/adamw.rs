//! AdamW Optimizer Kernels
//!
//! Fused GPU kernels for Adam and AdamW optimizers.
//!
//! ## AdamW Update Formula
//!
//! ```text
//! m = β1 * m + (1 - β1) * g           // First moment update
//! v = β2 * v + (1 - β2) * g²          // Second moment update
//! m̂ = m / (1 - β1^t)                  // Bias-corrected first moment
//! v̂ = v / (1 - β2^t)                  // Bias-corrected second moment
//! θ = θ - lr * (m̂ / (√v̂ + ε) + λ * θ) // Weight update with decay
//! ```
//!
//! For vanilla Adam, λ = 0 (no weight decay).

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused AdamW optimizer step kernel
///
/// Performs in-place weight update with:
/// - First and second moment updates
/// - Bias correction
/// - Weight decay (L2 regularization, decoupled)
///
/// # Parameters (kernel args)
/// - `params`: weight tensor (updated in-place)
/// - `grads`: gradient tensor
/// - `m`: first moment state (updated in-place)
/// - `v`: second moment state (updated in-place)
/// - `lr`: learning rate
/// - `beta1`: first moment decay (typically 0.9)
/// - `beta2`: second moment decay (typically 0.999)
/// - `eps`: numerical stability (typically 1e-8)
/// - `weight_decay`: L2 penalty coefficient
/// - `step`: current step (for bias correction)
/// - `n`: number of parameters
#[derive(Debug, Clone)]
pub struct AdamWStepKernel {
    /// Number of parameters
    pub n: u32,
}

impl AdamWStepKernel {
    /// Create a new AdamW step kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for AdamWStepKernel {
    fn name(&self) -> &str {
        "adamw_step"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("adamw_step")
            .param(PtxType::U64, "params_ptr")
            .param(PtxType::U64, "grads_ptr")
            .param(PtxType::U64, "m_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::F32, "lr")
            .param(PtxType::F32, "beta1")
            .param(PtxType::F32, "beta2")
            .param(PtxType::F32, "eps")
            .param(PtxType::F32, "weight_decay")
            .param(PtxType::F32, "bias_correction1") // Pre-computed: 1 / (1 - β1^t)
            .param(PtxType::F32, "bias_correction2") // Pre-computed: 1 / (1 - β2^t)
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

                // Load pointers
                let params_ptr = ctx.load_param_u64("params_ptr");
                let grads_ptr = ctx.load_param_u64("grads_ptr");
                let m_ptr = ctx.load_param_u64("m_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");

                // Load hyperparameters
                let lr = ctx.load_param_f32("lr");
                let beta1 = ctx.load_param_f32("beta1");
                let beta2 = ctx.load_param_f32("beta2");
                let eps = ctx.load_param_f32("eps");
                let weight_decay = ctx.load_param_f32("weight_decay");
                let bias_corr1 = ctx.load_param_f32("bias_correction1");
                let bias_corr2 = ctx.load_param_f32("bias_correction2");

                // Calculate byte offset (4 bytes per f32)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);

                // Calculate addresses
                let param_addr = ctx.add_u64(params_ptr, offset);
                let grad_addr = ctx.add_u64(grads_ptr, offset);
                let m_addr = ctx.add_u64(m_ptr, offset);
                let v_addr = ctx.add_u64(v_ptr, offset);

                // Load current values
                let param = ctx.ld_global_f32(param_addr);
                let grad = ctx.ld_global_f32(grad_addr);
                let m = ctx.ld_global_f32(m_addr);
                let v = ctx.ld_global_f32(v_addr);

                // Compute 1 - beta1, 1 - beta2
                let one = ctx.mov_f32_imm(1.0);
                let one_minus_beta1 = ctx.sub_f32(one, beta1);
                let one_minus_beta2 = ctx.sub_f32(one, beta2);

                // m_new = beta1 * m + (1 - beta1) * grad
                let m_scaled = ctx.mul_f32(beta1, m);
                let grad_scaled = ctx.mul_f32(one_minus_beta1, grad);
                let m_new = ctx.add_f32(m_scaled, grad_scaled);

                // v_new = beta2 * v + (1 - beta2) * grad^2
                let v_scaled = ctx.mul_f32(beta2, v);
                let grad_sq = ctx.mul_f32(grad, grad);
                let grad_sq_scaled = ctx.mul_f32(one_minus_beta2, grad_sq);
                let v_new = ctx.add_f32(v_scaled, grad_sq_scaled);

                // Bias-corrected estimates
                // m_hat = m_new * bias_correction1 (pre-computed as 1/(1-β1^t))
                let m_hat = ctx.mul_f32(m_new, bias_corr1);
                // v_hat = v_new * bias_correction2
                let v_hat = ctx.mul_f32(v_new, bias_corr2);

                // sqrt(v_hat) + eps
                let sqrt_v = ctx.sqrt_f32(v_hat);
                let denom = ctx.add_f32(sqrt_v, eps);

                // Adam update: m_hat / denom
                let adam_update = ctx.div_f32(m_hat, denom);

                // Weight decay term: weight_decay * param
                let decay_term = ctx.mul_f32(weight_decay, param);

                // Combined update: adam_update + decay_term
                let total_update = ctx.add_f32(adam_update, decay_term);

                // param_new = param - lr * total_update
                let lr_update = ctx.mul_f32(lr, total_update);
                let param_new = ctx.sub_f32(param, lr_update);

                // Store updated values
                ctx.st_global_f32(param_addr, param_new);
                ctx.st_global_f32(m_addr, m_new);
                ctx.st_global_f32(v_addr, v_new);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Fused Adam optimizer step kernel (no weight decay)
///
/// Same as AdamW but without the decoupled weight decay term.
/// Use this for vanilla Adam optimization.
#[derive(Debug, Clone)]
pub struct AdamStepKernel {
    /// Number of parameters
    pub n: u32,
}

impl AdamStepKernel {
    /// Create a new Adam step kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for AdamStepKernel {
    fn name(&self) -> &str {
        "adam_step"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("adam_step")
            .param(PtxType::U64, "params_ptr")
            .param(PtxType::U64, "grads_ptr")
            .param(PtxType::U64, "m_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::F32, "lr")
            .param(PtxType::F32, "beta1")
            .param(PtxType::F32, "beta2")
            .param(PtxType::F32, "eps")
            .param(PtxType::F32, "bias_correction1")
            .param(PtxType::F32, "bias_correction2")
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

                // Load pointers
                let params_ptr = ctx.load_param_u64("params_ptr");
                let grads_ptr = ctx.load_param_u64("grads_ptr");
                let m_ptr = ctx.load_param_u64("m_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");

                // Load hyperparameters
                let lr = ctx.load_param_f32("lr");
                let beta1 = ctx.load_param_f32("beta1");
                let beta2 = ctx.load_param_f32("beta2");
                let eps = ctx.load_param_f32("eps");
                let bias_corr1 = ctx.load_param_f32("bias_correction1");
                let bias_corr2 = ctx.load_param_f32("bias_correction2");

                // Calculate byte offset
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);

                // Calculate addresses
                let param_addr = ctx.add_u64(params_ptr, offset);
                let grad_addr = ctx.add_u64(grads_ptr, offset);
                let m_addr = ctx.add_u64(m_ptr, offset);
                let v_addr = ctx.add_u64(v_ptr, offset);

                // Load current values
                let param = ctx.ld_global_f32(param_addr);
                let grad = ctx.ld_global_f32(grad_addr);
                let m = ctx.ld_global_f32(m_addr);
                let v = ctx.ld_global_f32(v_addr);

                // Compute 1 - beta1, 1 - beta2
                let one = ctx.mov_f32_imm(1.0);
                let one_minus_beta1 = ctx.sub_f32(one, beta1);
                let one_minus_beta2 = ctx.sub_f32(one, beta2);

                // m_new = beta1 * m + (1 - beta1) * grad
                let m_scaled = ctx.mul_f32(beta1, m);
                let grad_scaled = ctx.mul_f32(one_minus_beta1, grad);
                let m_new = ctx.add_f32(m_scaled, grad_scaled);

                // v_new = beta2 * v + (1 - beta2) * grad^2
                let v_scaled = ctx.mul_f32(beta2, v);
                let grad_sq = ctx.mul_f32(grad, grad);
                let grad_sq_scaled = ctx.mul_f32(one_minus_beta2, grad_sq);
                let v_new = ctx.add_f32(v_scaled, grad_sq_scaled);

                // Bias-corrected estimates
                let m_hat = ctx.mul_f32(m_new, bias_corr1);
                let v_hat = ctx.mul_f32(v_new, bias_corr2);

                // sqrt(v_hat) + eps
                let sqrt_v = ctx.sqrt_f32(v_hat);
                let denom = ctx.add_f32(sqrt_v, eps);

                // Adam update: lr * m_hat / denom
                let update = ctx.div_f32(m_hat, denom);
                let lr_update = ctx.mul_f32(lr, update);

                // param_new = param - lr_update
                let param_new = ctx.sub_f32(param, lr_update);

                // Store updated values
                ctx.st_global_f32(param_addr, param_new);
                ctx.st_global_f32(m_addr, m_new);
                ctx.st_global_f32(v_addr, v_new);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step_kernel_name() {
        let kernel = AdamWStepKernel::new(1024);
        assert_eq!(kernel.name(), "adamw_step");
    }

    #[test]
    fn test_adamw_ptx_generation() {
        let kernel = AdamWStepKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry adamw_step"));

        // Verify it has the required parameters
        assert!(ptx.contains(".param .u64 params_ptr"));
        assert!(ptx.contains(".param .u64 grads_ptr"));
        assert!(ptx.contains(".param .f32 lr"));
        assert!(ptx.contains(".param .f32 weight_decay"));

        // Verify key operations
        assert!(ptx.contains("sqrt.rn.f32")); // sqrt for v_hat
        assert!(ptx.contains("div.rn.f32")); // division for adam update
    }

    #[test]
    fn test_adam_step_kernel_name() {
        let kernel = AdamStepKernel::new(1024);
        assert_eq!(kernel.name(), "adam_step");
    }

    #[test]
    fn test_adam_ptx_generation() {
        let kernel = AdamStepKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry adam_step"));

        // No weight_decay parameter in vanilla Adam
        assert!(!ptx.contains("weight_decay"));

        // Verify key operations
        assert!(ptx.contains("sqrt.rn.f32"));
        assert!(ptx.contains("div.rn.f32"));
    }
}
