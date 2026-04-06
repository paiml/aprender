//! Precise RoPE kernels for CPU/GPU bit-exactness

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// CORRECTNESS-013: Precise RoPE Kernel for CPU/GPU bit-exactness
///
/// Uses polynomial sin/cos approximations instead of hardware `sin.approx.f32`
/// and `cos.approx.f32` which have ~2^-21 error. For Qwen 2.5 with theta=1M,
/// the high-frequency components are very sensitive to trig precision.
#[derive(Debug, Clone)]
pub struct PreciseRopeKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0 or 1000000.0 for Qwen2.5)
    pub theta: f32,
}

impl PreciseRopeKernel {
    /// Create a new precise RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            theta,
        }
    }
}

impl Kernel for PreciseRopeKernel {
    fn name(&self) -> &str {
        "rope_precise"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;

        PtxKernel::new("rope_precise")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U32, "pos")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // CORRECTNESS-013: Use precise polynomial sin/cos
                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// CORRECTNESS-013: Precise RoPE Indirect Kernel for CUDA graph compatibility
///
/// Same as PreciseRopeKernel but reads position from a GPU buffer.
#[derive(Debug, Clone)]
pub struct PreciseRopeIndirectKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base
    pub theta: f32,
}

impl PreciseRopeIndirectKernel {
    /// Create a new precise RoPE indirect kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            theta,
        }
    }
}

impl Kernel for PreciseRopeIndirectKernel {
    fn name(&self) -> &str {
        "rope_precise_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        let half_dim = head_dim / 2;

        PtxKernel::new("rope_precise_indirect")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "pos_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // NEOX style pairing for Qwen2.5 compatibility
                let elem0 = pair_idx;
                let elem1 = ctx.add_u32_reg(pair_idx, half_dim_reg);

                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32_precise(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
