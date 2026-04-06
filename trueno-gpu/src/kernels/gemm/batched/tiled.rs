//! Tiled batched GEMM variant with shared memory.

#![allow(clippy::similar_names)]

use super::BatchedGemmKernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

impl BatchedGemmKernel {
    pub(super) fn build_tiled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Get batch index from ctaid.z
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Global row and column
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters - DON'T exit early (PARITY-114)
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid output
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets using immediate values
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop counter
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
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");

                let row_offset_a = ctx.mul_wide_u32(row, k_val * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_batch_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);

                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(batch_valid, "skip_b_load");
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");

                let row_offset_b = ctx.mul_wide_u32(b_row, n_val * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_batch_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);

                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // Inner loop: accumulate from shared memory
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

                // PARITY-114: Bounds check after tile loop
                ctx.branch_if_not(batch_valid, "exit");
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_batch_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
