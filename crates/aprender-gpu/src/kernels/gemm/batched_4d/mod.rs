//! 4D Batched GEMM Kernel (for multi-head attention)
//!
//! Implements C[b,h] = A[b,h] @ B[b,h] for attention computations
//! with batch and head dimensions.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// 4D Batched GEMM configuration for multi-head attention
#[derive(Debug, Clone)]
pub struct Batched4DGemmConfig {
    /// Batch size
    pub batch: u32,
    /// Number of attention heads
    pub heads: u32,
    /// M dimension (rows of A and C, typically sequence length)
    pub m: u32,
    /// N dimension (cols of B and C, typically sequence length or head_dim)
    pub n: u32,
    /// K dimension (cols of A, rows of B, typically head_dim)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
}

impl Default for Batched4DGemmConfig {
    fn default() -> Self {
        Self {
            batch: 1,
            heads: 8,
            m: 512,
            n: 512,
            k: 64,
            tile_size: 16,
        }
    }
}

/// Batched 4D GEMM kernel for attention patterns (Q @ K^T, attn @ V)
/// Grid: ((m+tile-1)/tile, (n+tile-1)/tile, batch * heads)
#[derive(Debug, Clone)]
pub struct Batched4DGemmKernel {
    /// Kernel configuration
    pub config: Batched4DGemmConfig,
}

impl Batched4DGemmKernel {
    /// Create a new 4D batched GEMM kernel for attention
    /// Pattern: [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
    #[must_use]
    pub fn new(batch: u32, heads: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: Batched4DGemmConfig {
                batch,
                heads,
                m,
                n,
                k,
                ..Default::default()
            },
        }
    }

    /// Create with custom tile size
    #[must_use]
    pub fn with_tile_size(batch: u32, heads: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: Batched4DGemmConfig {
                batch,
                heads,
                m,
                n,
                k,
                tile_size,
            },
        }
    }

    fn build_kernel(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;
        let heads_val = self.config.heads;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_4d_gemm")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "heads")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // z-dimension encodes batch * heads
                // batch_head_idx = ctaid.z
                // batch_idx = batch_head_idx / heads
                // head_idx = batch_head_idx % heads
                let batch_head_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);
                let batch_idx = ctx.div_u32(batch_head_idx, heads_val);
                let head_idx = ctx.rem_u32(batch_head_idx, heads_val);

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters
                let batch_param = ctx.load_param_u32("batch");
                let heads_param = ctx.load_param_u32("heads");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Validity predicates
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let head_valid = ctx.setp_lt_u32(head_idx, heads_param);
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate 4D offsets using immediate strides
                // A: [batch, heads, m, k] -> stride: [heads*m*k, m*k, k, 1]
                // B: [batch, heads, k, n] -> stride: [heads*k*n, k*n, n, 1]
                // C: [batch, heads, m, n] -> stride: [heads*m*n, m*n, n, 1]
                let a_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * m_val * k_val * 4);
                let a_head_off = ctx.mul_wide_u32(head_idx, m_val * k_val * 4);
                let a_base = ctx.add_u64(a_ptr, a_batch_off);
                let a_base = ctx.add_u64(a_base, a_head_off);

                let b_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * k_val * n_val * 4);
                let b_head_off = ctx.mul_wide_u32(head_idx, k_val * n_val * 4);
                let b_base = ctx.add_u64(b_ptr, b_batch_off);
                let b_base = ctx.add_u64(b_base, b_head_off);

                let c_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * m_val * n_val * 4);
                let c_head_off = ctx.mul_wide_u32(head_idx, m_val * n_val * 4);
                let c_base = ctx.add_u64(c_ptr, c_batch_off);
                let c_base = ctx.add_u64(c_base, c_head_off);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop
                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Shared memory offsets
                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset);

                // Load A tile
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                ctx.branch_if_not(batch_valid, "skip_a_load");
                ctx.branch_if_not(head_valid, "skip_a_load");
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");

                let row_offset_a = ctx.mul_wide_u32(row, k_val * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_ptr = ctx.add_u64(a_base, row_offset_a);
                let a_addr = ctx.add_u64(a_row_ptr, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);

                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(batch_valid, "skip_b_load");
                ctx.branch_if_not(head_valid, "skip_b_load");
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");

                let row_offset_b = ctx.mul_wide_u32(b_row, n_val * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_ptr = ctx.add_u64(b_base, row_offset_b);
                let b_addr = ctx.add_u64(b_row_ptr, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);

                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // Inner loop
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let as_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let as_addr = ctx.mul_u32(as_idx, 4);
                let a_shared = ctx.ld_shared_f32(as_addr);

                let bs_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let bs_idx_bytes = ctx.mul_u32(bs_idx, 4);
                let bs_addr = ctx.add_u32_reg(smem_b_base, bs_idx_bytes);
                let b_shared = ctx.ld_shared_f32(bs_addr);

                ctx.fma_f32_inplace(acc, a_shared, b_shared);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                // PARITY-114: Bounds check after all barriers
                ctx.branch_if_not(batch_valid, "exit");
                ctx.branch_if_not(head_valid, "exit");
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_ptr = ctx.add_u64(c_base, c_row_offset);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

impl Kernel for Batched4DGemmKernel {
    fn name(&self) -> &str {
        "batched_4d_gemm"
    }

    fn build_ptx(&self) -> PtxKernel {
        self.build_kernel()
    }
}

#[cfg(test)]
mod tests;
