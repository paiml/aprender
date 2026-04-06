//! Layout Conversion Kernels
//!
//! GPU kernels for tensor layout transformations in multi-head attention.
//!
//! - [`InterleavedToBatchedKernel`]: Convert interleaved to batched layout
//! - [`ExtractSingleHeadKernel`]: Extract one head from interleaved tensor
//! - [`CopySingleHeadKernel`]: Copy to head position in interleaved tensor
//! - [`BatchedToInterleavedKernel`]: Convert batched to interleaved layout

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Interleaved to Batched Kernel: [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
#[derive(Debug, Clone)]
pub struct InterleavedToBatchedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl InterleavedToBatchedKernel {
    /// Create a new interleaved-to-batched kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for InterleavedToBatchedKernel {
    fn name(&self) -> &str {
        "interleaved_to_batched"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        // All dimensions passed as runtime .param — NO baked immediates.
        PtxKernel::new("interleaved_to_batched")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "n_heads")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "total_elems")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.load_param_u32("total_elems");
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let seq_len = ctx.load_param_u32("seq_len");
                let head_dim = ctx.load_param_u32("head_dim");

                // d_model = n_heads * head_dim (computed from runtime params)
                let n_heads = ctx.load_param_u32("n_heads");
                let d_model = ctx.mul_lo_u32(n_heads, head_dim);

                let s = ctx.div_u32_reg(gid, d_model);
                let remainder = ctx.rem_u32_reg(gid, d_model);
                let h = ctx.div_u32_reg(remainder, head_dim);
                let d = ctx.rem_u32_reg(remainder, head_dim);

                // seq_head = seq_len * head_dim
                let seq_head = ctx.mul_lo_u32(seq_len, head_dim);
                let out_base = ctx.mul_lo_u32(h, seq_head);
                let out_row = ctx.mad_lo_u32(s, head_dim, d);
                let out_idx = ctx.add_u32_reg(out_base, out_row);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Extract Single Head Kernel: extract head h from interleaved [seq_len, n_heads * head_dim]
#[derive(Debug, Clone)]
pub struct ExtractSingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl ExtractSingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for ExtractSingleHeadKernel {
    fn name(&self) -> &str {
        "extract_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let output_size = seq_len * head_dim;

        PtxKernel::new("extract_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(output_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let in_idx = ctx.add_u32_reg(row_offset, head_offset);
                let in_idx = ctx.add_u32_reg(in_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Copy Single Head Kernel: copy [seq_len, head_dim] to head position in interleaved output
#[derive(Debug, Clone)]
pub struct CopySingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl CopySingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for CopySingleHeadKernel {
    fn name(&self) -> &str {
        "copy_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let input_size = seq_len * head_dim;

        PtxKernel::new("copy_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(input_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let out_idx = ctx.add_u32_reg(row_offset, head_offset);
                let out_idx = ctx.add_u32_reg(out_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched to Interleaved Kernel: [n_heads, seq_len, head_dim] -> [seq_len, n_heads * head_dim]
#[derive(Debug, Clone)]
pub struct BatchedToInterleavedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl BatchedToInterleavedKernel {
    /// Create a new batched-to-interleaved kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for BatchedToInterleavedKernel {
    fn name(&self) -> &str {
        "batched_to_interleaved"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Contract: dimension-independent-kernels-v1.yaml (FALSIFY-DIM-004)
        // All dimensions passed as runtime .param — NO baked immediates.
        PtxKernel::new("batched_to_interleaved")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "n_heads")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "total_elems")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.load_param_u32("total_elems");
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let seq_len = ctx.load_param_u32("seq_len");
                let head_dim = ctx.load_param_u32("head_dim");
                let n_heads = ctx.load_param_u32("n_heads");

                // d_model = n_heads * head_dim
                let d_model = ctx.mul_lo_u32(n_heads, head_dim);

                let s = ctx.div_u32_reg(gid, d_model);
                let remainder = ctx.rem_u32_reg(gid, d_model);
                let h = ctx.div_u32_reg(remainder, head_dim);
                let d = ctx.rem_u32_reg(remainder, head_dim);

                // seq_head = seq_len * head_dim
                let seq_head = ctx.mul_lo_u32(seq_len, head_dim);
                let in_base = ctx.mul_lo_u32(h, seq_head);
                let in_row = ctx.mad_lo_u32(s, head_dim, d);
                let in_idx = ctx.add_u32_reg(in_base, in_row);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
