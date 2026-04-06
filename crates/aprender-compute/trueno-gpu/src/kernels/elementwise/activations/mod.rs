//! Activation Function Kernels
//!
//! GPU kernels for activation functions used in transformer FFN blocks.
//!
//! - `ReluKernel`: Rectified Linear Unit
//! - `SiluKernel`: Sigmoid Linear Unit (SiLU/Swish)
//! - `GeluKernel`: Gaussian Error Linear Unit
//! - `ElementwiseMulKernel`: Element-wise multiplication
//! - `ScaleKernel`: Scalar multiplication

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// ReLU Activation Kernel: output = max(0, x)
///
/// Rectified Linear Unit activation function.
/// ReLU(x) = max(0, x)
///
/// # Issue #88: Forward kernel for training pipelines
#[derive(Debug, Clone)]
pub struct ReluKernel {
    /// Number of elements
    pub n: u32,
}

impl ReluKernel {
    /// Create a new ReLU activation kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ReluKernel {
    fn name(&self) -> &str {
        "relu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("relu")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
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
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load x
                let x = ctx.ld_global_f32(in_addr);

                // Compute ReLU: max(0, x)
                let zero = ctx.mov_f32_imm(0.0);
                let result = ctx.max_f32(x, zero);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// SiLU (Swish) Activation Kernel: output = x * sigmoid(x)
///
/// Sigmoid Linear Unit activation function used in LLaMA/TinyLlama FFN.
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// # PAR-023: Used in GPU-resident FFN block
#[derive(Debug, Clone)]
pub struct SiluKernel {
    /// Number of elements
    pub n: u32,
}

impl SiluKernel {
    /// Create a new SiLU activation kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for SiluKernel {
    fn name(&self) -> &str {
        "silu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("silu")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
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
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load x
                let x = ctx.ld_global_f32(in_addr);

                // Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                // Step 1: neg_x = -x (0 - x)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_x = ctx.sub_f32(zero, x);
                // Step 2: exp_neg_x = exp(-x) using ex2 (base-2 exp)
                // exp(x) = 2^(x * log2(e)) where log2(e) ≈ 1.4426950408889634
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(neg_x, log2_e);
                let exp_neg_x = ctx.ex2_f32(scaled);
                // Step 3: denom = 1 + exp(-x)
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg_x);
                // Step 4: sigmoid = 1 / denom (using division)
                let sigmoid = ctx.div_f32(one, denom);
                // Step 5: result = x * sigmoid
                let result = ctx.mul_f32(x, sigmoid);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// GELU Activation Kernel (approximate): output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Gaussian Error Linear Unit activation function used in GPT/BERT models.
///
/// # PAR-023: Used in GPU-resident FFN block for models using GELU
#[derive(Debug, Clone)]
pub struct GeluKernel {
    /// Number of elements
    pub n: u32,
}

impl GeluKernel {
    /// Create a new GELU activation kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for GeluKernel {
    fn name(&self) -> &str {
        "gelu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gelu")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
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
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load x
                let x = ctx.ld_global_f32(in_addr);

                // GELU approximation:
                // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                // sqrt(2/π) ≈ 0.7978845608
                let sqrt_2_pi = ctx.mov_f32_imm(0.797_884_6);
                let c = ctx.mov_f32_imm(0.044_715);
                let half = ctx.mov_f32_imm(0.5);
                let one = ctx.mov_f32_imm(1.0);

                // x³
                let x2 = ctx.mul_f32(x, x);
                let x3 = ctx.mul_f32(x2, x);

                // 0.044715 * x³
                let cx3 = ctx.mul_f32(c, x3);

                // x + 0.044715 * x³
                let inner = ctx.add_f32(x, cx3);

                // sqrt(2/π) * (x + 0.044715 * x³)
                let scaled = ctx.mul_f32(sqrt_2_pi, inner);

                // tanh approximation using (exp(2x) - 1) / (exp(2x) + 1)
                // For better precision, use: tanh(x) = 2*sigmoid(2x) - 1
                let two = ctx.mov_f32_imm(2.0);
                let zero = ctx.mov_f32_imm(0.0);
                let two_x = ctx.mul_f32(two, scaled);
                let neg_two_x = ctx.sub_f32(zero, two_x);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled_exp = ctx.mul_f32(neg_two_x, log2_e);
                let exp_neg = ctx.ex2_f32(scaled_exp);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                // tanh = 2*sigmoid - 1
                let two_sigmoid = ctx.mul_f32(two, sigmoid);
                let tanh = ctx.sub_f32(two_sigmoid, one);

                // 1 + tanh(...)
                let one_plus_tanh = ctx.add_f32(one, tanh);

                // 0.5 * x
                let half_x = ctx.mul_f32(half, x);

                // result = 0.5 * x * (1 + tanh(...))
                let result = ctx.mul_f32(half_x, one_plus_tanh);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Element-wise Multiply Kernel: output = input1 * input2
///
/// Used for gated activations in SwiGLU: silu(gate) * up
///
/// # PAR-023: Used in GPU-resident FFN block
#[derive(Debug, Clone)]
pub struct ElementwiseMulKernel {
    /// Number of elements
    pub n: u32,
}

impl ElementwiseMulKernel {
    /// Create a new element-wise multiply kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ElementwiseMulKernel {
    fn name(&self) -> &str {
        "elementwise_mul"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("elementwise_mul")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let addr1 = ctx.add_u64(input1_ptr, offset);
                let addr2 = ctx.add_u64(input2_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load both values
                let val1 = ctx.ld_global_f32(addr1);
                let val2 = ctx.ld_global_f32(addr2);

                // Multiply
                let result = ctx.mul_f32(val1, val2);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Scale Kernel: output = input * scale (scalar constant)
///
/// Multiplies each element by a constant scale factor.
/// Used for attention score scaling (1/sqrt(d_k)).
#[derive(Debug, Clone)]
pub struct ScaleKernel {
    /// Number of elements
    pub n: u32,
}

impl ScaleKernel {
    /// Create a new scale kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ScaleKernel {
    fn name(&self) -> &str {
        "scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("scale")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::F32, "scale")
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
                let output_ptr = ctx.load_param_u64("output_ptr");
                let scale = ctx.load_param_f32("scale");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load input value
                let val = ctx.ld_global_f32(in_addr);

                // Multiply by scale
                let result = ctx.mul_f32(val, scale);

                // Store result
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests;
