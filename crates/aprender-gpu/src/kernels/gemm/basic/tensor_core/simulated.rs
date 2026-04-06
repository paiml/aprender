//! Simulated Tensor Core GEMM using 16x16 tiles
//!
//! Uses 16 threads per block (one thread per output row), each computing
//! one row of the 16x16 output tile with explicit shared memory tiling.

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

use super::super::GemmKernel;

impl GemmKernel {
    pub(crate) fn build_tensor_core(&self) -> PtxKernel {
        // Tensor Core GEMM using 16x16 tiles
        // This kernel uses 16 threads per block (one thread per output row)
        // Each thread computes one row of the 16x16 output tile
        //
        // Launch config: grid_2d((m+15)/16, (n+15)/16, 16, 1)

        // Shared memory for two 16x16 tiles (A and B) in fp32
        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 4 * 2; // Two fp32 tiles
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_tensor_core")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let tile_row = ctx.mul_u32(ctaid_y, tile_size);
                let tile_col = ctx.mul_u32(ctaid_x, tile_size);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let my_row = ctx.add_u32_reg(tile_row, tid_x);
                let my_row_valid = ctx.setp_lt_u32(my_row, m_param);

                // Initialize 16 accumulators (one per output column)
                let acc0 = ctx.mov_f32_imm(0.0);
                let acc1 = ctx.mov_f32_imm(0.0);
                let acc2 = ctx.mov_f32_imm(0.0);
                let acc3 = ctx.mov_f32_imm(0.0);
                let acc4 = ctx.mov_f32_imm(0.0);
                let acc5 = ctx.mov_f32_imm(0.0);
                let acc6 = ctx.mov_f32_imm(0.0);
                let acc7 = ctx.mov_f32_imm(0.0);
                let acc8 = ctx.mov_f32_imm(0.0);
                let acc9 = ctx.mov_f32_imm(0.0);
                let acc10 = ctx.mov_f32_imm(0.0);
                let acc11 = ctx.mov_f32_imm(0.0);
                let acc12 = ctx.mov_f32_imm(0.0);
                let acc13 = ctx.mov_f32_imm(0.0);
                let acc14 = ctx.mov_f32_imm(0.0);
                let acc15 = ctx.mov_f32_imm(0.0);

                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);

                ctx.label("k_tile_loop");

                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32(k_tile_idx, tile_size);

                // Load A tile row
                let a_row_offset = ctx.mul_wide_u32(my_row, self.config.k * 4);
                let a_base = ctx.add_u64(a_ptr, a_row_offset);

                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("load_a_loop");
                let a_load_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(a_load_done, "load_a_end");

                let a_smem_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, inner_k);
                let a_smem_offset = ctx.mul_u32(a_smem_idx, 4);
                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(a_smem_offset, zero_a);

                let k_idx = ctx.add_u32_reg(k_offset, inner_k);
                let k_valid = ctx.setp_lt_u32(k_idx, k_param);

                ctx.branch_if_not(my_row_valid, "skip_a_load");
                ctx.branch_if_not(k_valid, "skip_a_load");

                let a_elem_offset = ctx.mul_wide_u32(k_idx, 4);
                let a_addr = ctx.add_u64(a_base, a_elem_offset);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(a_smem_offset, a_val);

                ctx.label("skip_a_load");
                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("load_a_loop");
                ctx.label("load_a_end");

                // Load B tile column
                let b_col = ctx.add_u32_reg(tile_col, tid_x);
                let b_col_valid = ctx.setp_lt_u32(b_col, n_param);
                let inner_k2 = ctx.mov_u32_imm(0);

                ctx.label("load_b_loop");
                let b_load_done = ctx.setp_ge_u32(inner_k2, tile_size_reg);
                ctx.branch_if(b_load_done, "load_b_end");

                let b_smem_idx = ctx.mad_lo_u32(inner_k2, tile_size_reg, tid_x);
                let b_smem_offset = ctx.mul_u32(b_smem_idx, 4);
                let b_smem_addr = ctx.add_u32_reg(smem_b_base, b_smem_offset);
                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(b_smem_addr, zero_b);

                let k_idx2 = ctx.add_u32_reg(k_offset, inner_k2);
                let k_valid2 = ctx.setp_lt_u32(k_idx2, k_param);

                ctx.branch_if_not(k_valid2, "skip_b_load");
                ctx.branch_if_not(b_col_valid, "skip_b_load");

                let b_row_offset = ctx.mul_wide_u32(k_idx2, self.config.n * 4);
                let b_col_offset = ctx.mul_wide_u32(b_col, 4);
                let b_base = ctx.add_u64(b_ptr, b_row_offset);
                let b_addr = ctx.add_u64(b_base, b_col_offset);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(b_smem_addr, b_val);

                ctx.label("skip_b_load");
                ctx.add_u32_inplace(inner_k2, 1);
                ctx.branch("load_b_loop");
                ctx.label("load_b_end");

                ctx.bar_sync(0);

                // Compute: for each k in 0..16, acc[j] += A_shared[tid_x,k] * B_shared[k,j]
                let compute_k = ctx.mov_u32_imm(0);

                ctx.label("compute_loop");
                let compute_done = ctx.setp_ge_u32(compute_k, tile_size_reg);
                ctx.branch_if(compute_done, "compute_end");

                let a_compute_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, compute_k);
                let a_compute_offset = ctx.mul_u32(a_compute_idx, 4);
                let a_compute_val = ctx.ld_shared_f32(a_compute_offset);

                // Unrolled for all 16 columns
                let b0_idx = ctx.mul_u32_reg(compute_k, tile_size_reg);
                let b0_offset = ctx.mul_u32(b0_idx, 4);
                let b0_addr = ctx.add_u32_reg(smem_b_base, b0_offset);
                let b0_val = ctx.ld_shared_f32(b0_addr);
                ctx.fma_f32_inplace(acc0, a_compute_val, b0_val);

                let b1_idx = ctx.add_u32(b0_idx, 1);
                let b1_offset = ctx.mul_u32(b1_idx, 4);
                let b1_addr = ctx.add_u32_reg(smem_b_base, b1_offset);
                let b1_val = ctx.ld_shared_f32(b1_addr);
                ctx.fma_f32_inplace(acc1, a_compute_val, b1_val);

                let b2_idx = ctx.add_u32(b0_idx, 2);
                let b2_offset = ctx.mul_u32(b2_idx, 4);
                let b2_addr = ctx.add_u32_reg(smem_b_base, b2_offset);
                let b2_val = ctx.ld_shared_f32(b2_addr);
                ctx.fma_f32_inplace(acc2, a_compute_val, b2_val);

                let b3_idx = ctx.add_u32(b0_idx, 3);
                let b3_offset = ctx.mul_u32(b3_idx, 4);
                let b3_addr = ctx.add_u32_reg(smem_b_base, b3_offset);
                let b3_val = ctx.ld_shared_f32(b3_addr);
                ctx.fma_f32_inplace(acc3, a_compute_val, b3_val);

                let b4_idx = ctx.add_u32(b0_idx, 4);
                let b4_offset = ctx.mul_u32(b4_idx, 4);
                let b4_addr = ctx.add_u32_reg(smem_b_base, b4_offset);
                let b4_val = ctx.ld_shared_f32(b4_addr);
                ctx.fma_f32_inplace(acc4, a_compute_val, b4_val);

                let b5_idx = ctx.add_u32(b0_idx, 5);
                let b5_offset = ctx.mul_u32(b5_idx, 4);
                let b5_addr = ctx.add_u32_reg(smem_b_base, b5_offset);
                let b5_val = ctx.ld_shared_f32(b5_addr);
                ctx.fma_f32_inplace(acc5, a_compute_val, b5_val);

                let b6_idx = ctx.add_u32(b0_idx, 6);
                let b6_offset = ctx.mul_u32(b6_idx, 4);
                let b6_addr = ctx.add_u32_reg(smem_b_base, b6_offset);
                let b6_val = ctx.ld_shared_f32(b6_addr);
                ctx.fma_f32_inplace(acc6, a_compute_val, b6_val);

                let b7_idx = ctx.add_u32(b0_idx, 7);
                let b7_offset = ctx.mul_u32(b7_idx, 4);
                let b7_addr = ctx.add_u32_reg(smem_b_base, b7_offset);
                let b7_val = ctx.ld_shared_f32(b7_addr);
                ctx.fma_f32_inplace(acc7, a_compute_val, b7_val);

                let b8_idx = ctx.add_u32(b0_idx, 8);
                let b8_offset = ctx.mul_u32(b8_idx, 4);
                let b8_addr = ctx.add_u32_reg(smem_b_base, b8_offset);
                let b8_val = ctx.ld_shared_f32(b8_addr);
                ctx.fma_f32_inplace(acc8, a_compute_val, b8_val);

                let b9_idx = ctx.add_u32(b0_idx, 9);
                let b9_offset = ctx.mul_u32(b9_idx, 4);
                let b9_addr = ctx.add_u32_reg(smem_b_base, b9_offset);
                let b9_val = ctx.ld_shared_f32(b9_addr);
                ctx.fma_f32_inplace(acc9, a_compute_val, b9_val);

                let b10_idx = ctx.add_u32(b0_idx, 10);
                let b10_offset = ctx.mul_u32(b10_idx, 4);
                let b10_addr = ctx.add_u32_reg(smem_b_base, b10_offset);
                let b10_val = ctx.ld_shared_f32(b10_addr);
                ctx.fma_f32_inplace(acc10, a_compute_val, b10_val);

                let b11_idx = ctx.add_u32(b0_idx, 11);
                let b11_offset = ctx.mul_u32(b11_idx, 4);
                let b11_addr = ctx.add_u32_reg(smem_b_base, b11_offset);
                let b11_val = ctx.ld_shared_f32(b11_addr);
                ctx.fma_f32_inplace(acc11, a_compute_val, b11_val);

                let b12_idx = ctx.add_u32(b0_idx, 12);
                let b12_offset = ctx.mul_u32(b12_idx, 4);
                let b12_addr = ctx.add_u32_reg(smem_b_base, b12_offset);
                let b12_val = ctx.ld_shared_f32(b12_addr);
                ctx.fma_f32_inplace(acc12, a_compute_val, b12_val);

                let b13_idx = ctx.add_u32(b0_idx, 13);
                let b13_offset = ctx.mul_u32(b13_idx, 4);
                let b13_addr = ctx.add_u32_reg(smem_b_base, b13_offset);
                let b13_val = ctx.ld_shared_f32(b13_addr);
                ctx.fma_f32_inplace(acc13, a_compute_val, b13_val);

                let b14_idx = ctx.add_u32(b0_idx, 14);
                let b14_offset = ctx.mul_u32(b14_idx, 4);
                let b14_addr = ctx.add_u32_reg(smem_b_base, b14_offset);
                let b14_val = ctx.ld_shared_f32(b14_addr);
                ctx.fma_f32_inplace(acc14, a_compute_val, b14_val);

                let b15_idx = ctx.add_u32(b0_idx, 15);
                let b15_offset = ctx.mul_u32(b15_idx, 4);
                let b15_addr = ctx.add_u32_reg(smem_b_base, b15_offset);
                let b15_val = ctx.ld_shared_f32(b15_addr);
                ctx.fma_f32_inplace(acc15, a_compute_val, b15_val);

                ctx.add_u32_inplace(compute_k, 1);
                ctx.branch("compute_loop");
                ctx.label("compute_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                ctx.branch_if_not(my_row_valid, "exit");
                ctx.branch_if_not(tile_col_valid, "exit");

                // Store 16 accumulators
                let c_row_offset = ctx.mul_wide_u32(my_row, self.config.n * 4);
                let c_base = ctx.add_u64(c_ptr, c_row_offset);

                let c0_col = ctx.add_u32(tile_col, 0);
                let c0_offset = ctx.mul_wide_u32(c0_col, 4);
                let c0_addr = ctx.add_u64(c_base, c0_offset);
                ctx.st_global_f32(c0_addr, acc0);

                let c1_col = ctx.add_u32(tile_col, 1);
                let c1_offset = ctx.mul_wide_u32(c1_col, 4);
                let c1_addr = ctx.add_u64(c_base, c1_offset);
                ctx.st_global_f32(c1_addr, acc1);

                let c2_col = ctx.add_u32(tile_col, 2);
                let c2_offset = ctx.mul_wide_u32(c2_col, 4);
                let c2_addr = ctx.add_u64(c_base, c2_offset);
                ctx.st_global_f32(c2_addr, acc2);

                let c3_col = ctx.add_u32(tile_col, 3);
                let c3_offset = ctx.mul_wide_u32(c3_col, 4);
                let c3_addr = ctx.add_u64(c_base, c3_offset);
                ctx.st_global_f32(c3_addr, acc3);

                let c4_col = ctx.add_u32(tile_col, 4);
                let c4_offset = ctx.mul_wide_u32(c4_col, 4);
                let c4_addr = ctx.add_u64(c_base, c4_offset);
                ctx.st_global_f32(c4_addr, acc4);

                let c5_col = ctx.add_u32(tile_col, 5);
                let c5_offset = ctx.mul_wide_u32(c5_col, 4);
                let c5_addr = ctx.add_u64(c_base, c5_offset);
                ctx.st_global_f32(c5_addr, acc5);

                let c6_col = ctx.add_u32(tile_col, 6);
                let c6_offset = ctx.mul_wide_u32(c6_col, 4);
                let c6_addr = ctx.add_u64(c_base, c6_offset);
                ctx.st_global_f32(c6_addr, acc6);

                let c7_col = ctx.add_u32(tile_col, 7);
                let c7_offset = ctx.mul_wide_u32(c7_col, 4);
                let c7_addr = ctx.add_u64(c_base, c7_offset);
                ctx.st_global_f32(c7_addr, acc7);

                let c8_col = ctx.add_u32(tile_col, 8);
                let c8_offset = ctx.mul_wide_u32(c8_col, 4);
                let c8_addr = ctx.add_u64(c_base, c8_offset);
                ctx.st_global_f32(c8_addr, acc8);

                let c9_col = ctx.add_u32(tile_col, 9);
                let c9_offset = ctx.mul_wide_u32(c9_col, 4);
                let c9_addr = ctx.add_u64(c_base, c9_offset);
                ctx.st_global_f32(c9_addr, acc9);

                let c10_col = ctx.add_u32(tile_col, 10);
                let c10_offset = ctx.mul_wide_u32(c10_col, 4);
                let c10_addr = ctx.add_u64(c_base, c10_offset);
                ctx.st_global_f32(c10_addr, acc10);

                let c11_col = ctx.add_u32(tile_col, 11);
                let c11_offset = ctx.mul_wide_u32(c11_col, 4);
                let c11_addr = ctx.add_u64(c_base, c11_offset);
                ctx.st_global_f32(c11_addr, acc11);

                let c12_col = ctx.add_u32(tile_col, 12);
                let c12_offset = ctx.mul_wide_u32(c12_col, 4);
                let c12_addr = ctx.add_u64(c_base, c12_offset);
                ctx.st_global_f32(c12_addr, acc12);

                let c13_col = ctx.add_u32(tile_col, 13);
                let c13_offset = ctx.mul_wide_u32(c13_col, 4);
                let c13_addr = ctx.add_u64(c_base, c13_offset);
                ctx.st_global_f32(c13_addr, acc13);

                let c14_col = ctx.add_u32(tile_col, 14);
                let c14_offset = ctx.mul_wide_u32(c14_col, 4);
                let c14_addr = ctx.add_u64(c_base, c14_offset);
                ctx.st_global_f32(c14_addr, acc14);

                let c15_col = ctx.add_u32(tile_col, 15);
                let c15_offset = ctx.mul_wide_u32(c15_col, 4);
                let c15_addr = ctx.add_u64(c_base, c15_offset);
                ctx.st_global_f32(c15_addr, acc15);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
