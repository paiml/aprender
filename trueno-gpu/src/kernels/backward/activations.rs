//! Activation Function Backward Kernels
//!
//! Backward (gradient) kernels for activation functions used in training.
//!
//! ## Mathematical Specifications
//!
//! - **ReLU**: `grad_input = grad_output * (input > 0 ? 1 : 0)`
//! - **GELU**: `grad_input = grad_output * gelu'(input)` (derivative of GELU)
//! - **SiLU**: `grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))`
//!
//! ## Falsifiable Prediction (P-ACT-BACK-001)
//!
//! All activation backward kernels match finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// ReLU Backward Kernel
///
/// Forward:  `y = max(0, x)`
/// Backward: `∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)`
///
/// # Parameters
/// - `input_ptr`: Original input to forward pass (needed for gradient mask)
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient (∂L/∂x)
/// - `n`: Number of elements
#[derive(Debug, Clone)]
pub struct ReluBackwardKernel {
    /// Number of elements
    pub n: u32,
}

impl ReluBackwardKernel {
    /// Create a new ReLU backward kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ReluBackwardKernel {
    fn name(&self) -> &str {
        "relu_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("relu_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate addresses
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let grad_out_addr = ctx.add_u64(grad_output_ptr, offset);
                let grad_in_addr = ctx.add_u64(grad_input_ptr, offset);

                // Load input and grad_output
                let x = ctx.ld_global_f32(in_addr);
                let grad_out = ctx.ld_global_f32(grad_out_addr);

                // ReLU backward: grad_input = grad_output * (x > 0 ? 1 : 0)
                let zero = ctx.mov_f32_imm(0.0);
                let is_positive = ctx.setp_gt_f32(x, zero);

                // Select between grad_out and 0 based on condition
                // selp_f32(pred, true_val, false_val): d = pred ? true_val : false_val
                let grad_in = ctx.selp_f32(is_positive, grad_out, zero);

                // Store result
                ctx.st_global_f32(grad_in_addr, grad_in);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// GELU Backward Kernel (approximate)
///
/// Forward:  `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
/// Backward: `∂L/∂x = ∂L/∂y * gelu'(x)`
///
/// Where `gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx`
/// and `u = sqrt(2/π) * (x + 0.044715 * x³)`, `du/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)`
///
/// Simplified form using tanh derivative: `gelu'(x) = 0.5 * (1 + tanh(u) + x * (1 - tanh²(u)) * du/dx)`
#[derive(Debug, Clone)]
pub struct GeluBackwardKernel {
    /// Number of elements
    pub n: u32,
}

impl GeluBackwardKernel {
    /// Create a new GELU backward kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for GeluBackwardKernel {
    fn name(&self) -> &str {
        "gelu_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gelu_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate addresses
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let grad_out_addr = ctx.add_u64(grad_output_ptr, offset);
                let grad_in_addr = ctx.add_u64(grad_input_ptr, offset);

                // Load input and grad_output
                let x = ctx.ld_global_f32(in_addr);
                let grad_out = ctx.ld_global_f32(grad_out_addr);

                // Constants
                let sqrt_2_pi = ctx.mov_f32_imm(0.797_884_6); // sqrt(2/π)
                let c = ctx.mov_f32_imm(0.044_715);
                let c3 = ctx.mov_f32_imm(0.134_145); // 3 * 0.044715
                let half = ctx.mov_f32_imm(0.5);
                let one = ctx.mov_f32_imm(1.0);
                let two = ctx.mov_f32_imm(2.0);
                let zero = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // Compute x² and x³
                let x2 = ctx.mul_f32(x, x);
                let x3 = ctx.mul_f32(x2, x);

                // u = sqrt(2/π) * (x + 0.044715 * x³)
                let cx3 = ctx.mul_f32(c, x3);
                let inner = ctx.add_f32(x, cx3);
                let u = ctx.mul_f32(sqrt_2_pi, inner);

                // tanh(u) via sigmoid: tanh(x) = 2*sigmoid(2x) - 1
                let two_u = ctx.mul_f32(two, u);
                let neg_two_u = ctx.sub_f32(zero, two_u);
                let scaled_exp = ctx.mul_f32(neg_two_u, log2_e);
                let exp_neg = ctx.ex2_f32(scaled_exp);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let two_sigmoid = ctx.mul_f32(two, sigmoid);
                let tanh_u = ctx.sub_f32(two_sigmoid, one);

                // du/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
                let c3x2 = ctx.mul_f32(c3, x2);
                let du_inner = ctx.add_f32(one, c3x2);
                let du_dx = ctx.mul_f32(sqrt_2_pi, du_inner);

                // sech²(u) = 1 - tanh²(u)
                let tanh_sq = ctx.mul_f32(tanh_u, tanh_u);
                let sech_sq = ctx.sub_f32(one, tanh_sq);

                // gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx
                // First term: 0.5 * (1 + tanh(u))
                let one_plus_tanh = ctx.add_f32(one, tanh_u);
                let term1 = ctx.mul_f32(half, one_plus_tanh);

                // Second term: 0.5 * x * sech²(u) * du/dx
                let x_sech_sq = ctx.mul_f32(x, sech_sq);
                let x_sech_sq_du = ctx.mul_f32(x_sech_sq, du_dx);
                let term2 = ctx.mul_f32(half, x_sech_sq_du);

                // gelu'(x) = term1 + term2
                let gelu_prime = ctx.add_f32(term1, term2);

                // grad_input = grad_output * gelu'(x)
                let grad_in = ctx.mul_f32(grad_out, gelu_prime);

                // Store result
                ctx.st_global_f32(grad_in_addr, grad_in);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// SiLU (Swish) Backward Kernel
///
/// Forward:  `y = x * sigmoid(x)`
/// Backward: `∂L/∂x = ∂L/∂y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))`
///         = `∂L/∂y * (y + sigmoid(x) * (1 - y))`
///         = `∂L/∂y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))`
///
/// Simplified: `silu'(x) = sigmoid(x) * (1 + x - y)` where `y = x * sigmoid(x)`
#[derive(Debug, Clone)]
pub struct SiluBackwardKernel {
    /// Number of elements
    pub n: u32,
}

impl SiluBackwardKernel {
    /// Create a new SiLU backward kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for SiluBackwardKernel {
    fn name(&self) -> &str {
        "silu_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("silu_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate addresses
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let grad_out_addr = ctx.add_u64(grad_output_ptr, offset);
                let grad_in_addr = ctx.add_u64(grad_input_ptr, offset);

                // Load input and grad_output
                let x = ctx.ld_global_f32(in_addr);
                let grad_out = ctx.ld_global_f32(grad_out_addr);

                // Constants
                let one = ctx.mov_f32_imm(1.0);
                let zero = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // sigmoid(x) = 1 / (1 + exp(-x))
                let neg_x = ctx.sub_f32(zero, x);
                let scaled = ctx.mul_f32(neg_x, log2_e);
                let exp_neg_x = ctx.ex2_f32(scaled);
                let denom = ctx.add_f32(one, exp_neg_x);
                let sigmoid_x = ctx.div_f32(one, denom);

                // y = x * sigmoid(x) (forward output)
                let y = ctx.mul_f32(x, sigmoid_x);

                // silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                //          = sigmoid(x) * (1 + x - y)
                let one_plus_x = ctx.add_f32(one, x);
                let one_plus_x_minus_y = ctx.sub_f32(one_plus_x, y);
                let silu_prime = ctx.mul_f32(sigmoid_x, one_plus_x_minus_y);

                // grad_input = grad_output * silu'(x)
                let grad_in = ctx.mul_f32(grad_out, silu_prime);

                // Store result
                ctx.st_global_f32(grad_in_addr, grad_in);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_backward_name() {
        let kernel = ReluBackwardKernel::new(2048);
        assert_eq!(kernel.name(), "relu_backward");
    }

    #[test]
    fn test_relu_backward_ptx_generation() {
        let kernel = ReluBackwardKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry relu_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify comparison (x > 0)
        assert!(ptx.contains("setp.gt.f32"));
        // Verify conditional select
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn test_gelu_backward_name() {
        let kernel = GeluBackwardKernel::new(2048);
        assert_eq!(kernel.name(), "gelu_backward");
    }

    #[test]
    fn test_gelu_backward_ptx_generation() {
        let kernel = GeluBackwardKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry gelu_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify tanh computation via exp
        assert!(ptx.contains("ex2.approx.f32"));
        // Verify multiplication for derivative
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_silu_backward_name() {
        let kernel = SiluBackwardKernel::new(2048);
        assert_eq!(kernel.name(), "silu_backward");
    }

    #[test]
    fn test_silu_backward_ptx_generation() {
        let kernel = SiluBackwardKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry silu_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify sigmoid computation (exp and division)
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("div.rn.f32"));
    }

    #[test]
    fn test_relu_backward_barrier_safety() {
        let kernel = ReluBackwardKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "ReLU backward should be barrier-safe: {:?}", result.violations);
    }

    #[test]
    fn test_gelu_backward_barrier_safety() {
        let kernel = GeluBackwardKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "GELU backward should be barrier-safe: {:?}", result.violations);
    }

    #[test]
    fn test_silu_backward_barrier_safety() {
        let kernel = SiluBackwardKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "SiLU backward should be barrier-safe: {:?}", result.violations);
    }
}
