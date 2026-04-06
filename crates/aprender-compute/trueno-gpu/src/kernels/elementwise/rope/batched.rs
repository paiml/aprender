//! Batched RoPE kernel for multi-sequence processing

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// ============================================================================
// PAR-114: Batched RoPE Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched RoPE Kernel: Apply rotary position embeddings to M sequences
///
/// Processes M sequences in parallel using Grid.y for batch index.
/// Each sequence can have a different position.
///
/// # Grid Configuration
///
/// - Grid: (num_heads, batch_size, 1)
/// - Block: (head_dim / 2, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedRopeKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Rope theta base (typically 10000.0)
    pub theta: f32,
}

impl BatchedRopeKernel {
    /// Create a new batched RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, batch_size: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, batch_size, theta }
    }
}

impl Kernel for BatchedRopeKernel {
    fn name(&self) -> &str {
        "batched_rope"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let theta = self.theta;

        PtxKernel::new("batched_rope")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "positions_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);

                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let positions_ptr = ctx.load_param_u64("positions_ptr");

                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let four = ctx.mov_u32_imm(4);
                let pos_byte_offset = ctx.mul_lo_u32(batch_idx, four);
                let pos_byte_offset_64 = ctx.cvt_u64_u32(pos_byte_offset);
                let pos_addr = ctx.add_u64(positions_ptr, pos_byte_offset_64);
                let pos = ctx.ld_global_u32(pos_addr);

                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                let heads_per_seq = ctx.mov_u32_imm(num_heads);
                let dim = ctx.mov_u32_imm(head_dim);
                let seq_stride = ctx.mul_lo_u32(heads_per_seq, dim);
                let batch_offset = ctx.mul_lo_u32(batch_idx, seq_stride);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let base_offset = ctx.add_u32_reg(batch_offset, head_offset);
                let offset0 = ctx.add_u32_reg(base_offset, elem0);
                let offset1 = ctx.add_u32_reg(base_offset, elem1);

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

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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

// ============================================================================
// ALB-106: Batched RoPE Backward Kernel (inverse rotation for gradient flow)
// ============================================================================

/// Batched RoPE Backward Kernel: Apply inverse rotary embeddings to gradients
///
/// Identical to `BatchedRopeKernel` but applies the transpose rotation R^T(-θ):
/// - Forward: x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos
/// - Backward: x0' = x0*cos + x1*sin, x1' = -x0*sin + x1*cos
///
/// # Grid Configuration
///
/// - Grid: (num_heads, batch_size, 1)
/// - Block: (head_dim / 2, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedRopeBackwardKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Rope theta base (typically 10000.0)
    pub theta: f32,
}

impl BatchedRopeBackwardKernel {
    /// Create a new batched RoPE backward kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, batch_size: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, batch_size, theta }
    }
}

impl Kernel for BatchedRopeBackwardKernel {
    fn name(&self) -> &str {
        "batched_rope_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let theta = self.theta;

        PtxKernel::new("batched_rope_backward")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "positions_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);

                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let positions_ptr = ctx.load_param_u64("positions_ptr");

                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let four = ctx.mov_u32_imm(4);
                let pos_byte_offset = ctx.mul_lo_u32(batch_idx, four);
                let pos_byte_offset_64 = ctx.cvt_u64_u32(pos_byte_offset);
                let pos_addr = ctx.add_u64(positions_ptr, pos_byte_offset_64);
                let pos = ctx.ld_global_u32(pos_addr);

                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                let heads_per_seq = ctx.mov_u32_imm(num_heads);
                let dim = ctx.mov_u32_imm(head_dim);
                let seq_stride = ctx.mul_lo_u32(heads_per_seq, dim);
                let batch_offset = ctx.mul_lo_u32(batch_idx, seq_stride);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let base_offset = ctx.add_u32_reg(batch_offset, head_offset);
                let offset0 = ctx.add_u32_reg(base_offset, elem0);
                let offset1 = ctx.add_u32_reg(base_offset, elem1);

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

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Inverse rotation: negate sin terms (R^T = R(-θ))
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.add_f32(x0_cos, x1_sin); // + instead of -

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.sub_f32(x1_cos, x0_sin); // swapped: cos*x1 - sin*x0

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
