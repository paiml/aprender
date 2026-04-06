//! Tiled GEMM with 4x unrolled inner loop (WAPR-PERF-009)

#![allow(clippy::similar_names)]

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

use super::GemmKernel;

impl GemmKernel {
    /// Tiled GEMM with 4x unrolled inner loop (WAPR-PERF-009)
    ///
    /// Reduces loop overhead from 12:1 to ~3:1 instructions per FMA.
    /// The inner K loop processes 4 elements at a time, reducing the
    /// number of branch/compare instructions by 4x.
    #[allow(clippy::too_many_lines)]
    pub(super) fn build_tiled_unrolled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles, f32
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;

        // Unroll factor: process 4 elements per inner loop iteration
        let unroll_factor = 4u32;
        // Number of unrolled iterations (tile_size must be divisible by 4)
        let unrolled_iters = tile_size / unroll_factor;

        PtxKernel::new("gemm_tiled_unrolled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

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

                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");
                let row_offset_a = ctx.mul_wide_u32(row, self.config.k * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);
                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");
                let row_offset_b = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // ========================================
                // 4x UNROLLED INNER LOOP (WAPR-PERF-009)
                // ========================================
                let inner_k = ctx.mov_u32_imm(0);
                let unrolled_iters_reg = ctx.mov_u32_imm(unrolled_iters);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, unrolled_iters_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let k_base = ctx.mul_u32(inner_k, unroll_factor);

                // === Iteration 0: k = k_base + 0 ===
                let k0 = k_base;
                let as_idx0 = ctx.mad_lo_u32(tid_y, tile_size_reg, k0);
                let as_addr0 = ctx.mul_u32(as_idx0, 4);
                let a_shared0 = ctx.ld_shared_f32(as_addr0);

                let bs_idx0 = ctx.mad_lo_u32(k0, tile_size_reg, tid_x);
                let bs_idx_bytes0 = ctx.mul_u32(bs_idx0, 4);
                let bs_addr0 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes0);
                let b_shared0 = ctx.ld_shared_f32(bs_addr0);

                ctx.fma_f32_inplace(acc, a_shared0, b_shared0);

                // === Iteration 1: k = k_base + 1 ===
                let k1 = ctx.add_u32(k_base, 1);
                let as_idx1 = ctx.mad_lo_u32(tid_y, tile_size_reg, k1);
                let as_addr1 = ctx.mul_u32(as_idx1, 4);
                let a_shared1 = ctx.ld_shared_f32(as_addr1);

                let bs_idx1 = ctx.mad_lo_u32(k1, tile_size_reg, tid_x);
                let bs_idx_bytes1 = ctx.mul_u32(bs_idx1, 4);
                let bs_addr1 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes1);
                let b_shared1 = ctx.ld_shared_f32(bs_addr1);

                ctx.fma_f32_inplace(acc, a_shared1, b_shared1);

                // === Iteration 2: k = k_base + 2 ===
                let k2 = ctx.add_u32(k_base, 2);
                let as_idx2 = ctx.mad_lo_u32(tid_y, tile_size_reg, k2);
                let as_addr2 = ctx.mul_u32(as_idx2, 4);
                let a_shared2 = ctx.ld_shared_f32(as_addr2);

                let bs_idx2 = ctx.mad_lo_u32(k2, tile_size_reg, tid_x);
                let bs_idx_bytes2 = ctx.mul_u32(bs_idx2, 4);
                let bs_addr2 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes2);
                let b_shared2 = ctx.ld_shared_f32(bs_addr2);

                ctx.fma_f32_inplace(acc, a_shared2, b_shared2);

                // === Iteration 3: k = k_base + 3 ===
                let k3 = ctx.add_u32(k_base, 3);
                let as_idx3 = ctx.mad_lo_u32(tid_y, tile_size_reg, k3);
                let as_addr3 = ctx.mul_u32(as_idx3, 4);
                let a_shared3 = ctx.ld_shared_f32(as_addr3);

                let bs_idx3 = ctx.mad_lo_u32(k3, tile_size_reg, tid_x);
                let bs_idx_bytes3 = ctx.mul_u32(bs_idx3, 4);
                let bs_addr3 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes3);
                let b_shared3 = ctx.ld_shared_f32(bs_addr3);

                ctx.fma_f32_inplace(acc, a_shared3, b_shared3);

                // inner_k++ (increments by 1, actual k increments by 4)
                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
