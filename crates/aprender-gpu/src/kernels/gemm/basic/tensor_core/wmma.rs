//! True WMMA FP16 GEMM using hardware Tensor Core PTX intrinsics
//!
//! Uses wmma.load, wmma.mma, wmma.store for hardware Tensor Core acceleration.
//! Requires sm_70+ (Volta or later). Input is FP32, converted to FP16 internally.
//! Launch config: grid_2d((m+15)/16, (n+15)/16, 32, 1) - one warp per 16x16 output tile

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType, WmmaLayout};

use super::super::GemmKernel;

impl GemmKernel {
    pub(crate) fn build_wmma_fp16(&self) -> PtxKernel {
        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 2 * 2; // Two FP16 tiles
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_wmma_fp16")
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

                let tile_row_valid = ctx.setp_lt_u32(tile_row, m_param);
                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                let smem_a_base = ctx.mov_u32_imm(0);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 2); // After A tile (FP16)

                let frag_c = ctx.wmma_init_c_zero();

                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                let k_offset = ctx.mul_u32_reg(k_tile_idx, tile_size_reg);

                // Load A tile to shared memory (FP32 global -> FP16 shared)
                let elements_per_thread = ctx.mov_u32_imm(8);
                let my_start = ctx.mul_u32_reg(tid_x, elements_per_thread);

                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop");
                let load_done = ctx.setp_ge_u32(load_idx, elements_per_thread);
                ctx.branch_if(load_done, "load_a_end");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                let row_in_tile = ctx.div_u32(elem_idx, 16);
                let col_in_tile = ctx.rem_u32(elem_idx, 16);

                let smem_a_offset = ctx.mul_u32(elem_idx, 2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_offset);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let zero_f16 = ctx.cvt_f16_f32(zero_f32);
                ctx.st_shared_f16(smem_a_addr, zero_f16);

                let a_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let a_col = ctx.add_u32_reg(k_offset, col_in_tile);
                let a_row_valid = ctx.setp_lt_u32(a_row, m_param);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                ctx.branch_if_not(a_row_valid, "skip_wmma_a_load");
                ctx.branch_if_not(a_col_valid, "skip_wmma_a_load");

                let k_reg = ctx.mov_u32_imm(self.config.k);
                let a_idx = ctx.mad_lo_u32(a_row, k_reg, a_col);
                let a_byte_offset = ctx.mul_wide_u32(a_idx, 4);
                let a_addr = ctx.add_u64(a_ptr, a_byte_offset);

                let a_val_f32 = ctx.ld_global_f32(a_addr);
                let a_val_f16 = ctx.cvt_f16_f32(a_val_f32);
                ctx.st_shared_f16(smem_a_addr, a_val_f16);

                ctx.label("skip_wmma_a_load");
                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_a_loop");
                ctx.label("load_a_end");

                // Load B tile to shared memory
                let load_idx_b = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop");
                let load_b_done = ctx.setp_ge_u32(load_idx_b, elements_per_thread);
                ctx.branch_if(load_b_done, "load_b_end");

                let elem_idx_b = ctx.add_u32_reg(my_start, load_idx_b);
                let row_in_tile_b = ctx.div_u32(elem_idx_b, 16);
                let col_in_tile_b = ctx.rem_u32(elem_idx_b, 16);

                let smem_b_offset = ctx.mul_u32(elem_idx_b, 2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_offset);
                let zero_b_f32 = ctx.mov_f32_imm(0.0);
                let zero_b_f16 = ctx.cvt_f16_f32(zero_b_f32);
                ctx.st_shared_f16(smem_b_addr, zero_b_f16);

                let b_row = ctx.add_u32_reg(k_offset, row_in_tile_b);
                let b_col = ctx.add_u32_reg(tile_col, col_in_tile_b);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);
                let b_col_valid = ctx.setp_lt_u32(b_col, n_param);

                ctx.branch_if_not(b_row_valid, "skip_wmma_b_load");
                ctx.branch_if_not(b_col_valid, "skip_wmma_b_load");

                let n_reg = ctx.mov_u32_imm(self.config.n);
                let b_idx = ctx.mad_lo_u32(b_row, n_reg, b_col);
                let b_byte_offset = ctx.mul_wide_u32(b_idx, 4);
                let b_addr = ctx.add_u64(b_ptr, b_byte_offset);

                let b_val_f32 = ctx.ld_global_f32(b_addr);
                let b_val_f16 = ctx.cvt_f16_f32(b_val_f32);
                ctx.st_shared_f16(smem_b_addr, b_val_f16);

                ctx.label("skip_wmma_b_load");
                ctx.add_u32_inplace(load_idx_b, 1);
                ctx.branch("load_b_loop");
                ctx.label("load_b_end");

                ctx.bar_sync(0);

                // WMMA matrix multiply
                let smem_generic_base = ctx.shared_base_addr();

                let frag_a = ctx.wmma_load_a_f16(smem_generic_base, 16, WmmaLayout::RowMajor);

                let smem_b_offset_u64 = ctx.cvt_u64_u32(smem_b_base);
                let smem_b_ptr = ctx.add_u64(smem_generic_base, smem_b_offset_u64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_ptr, 16, WmmaLayout::RowMajor);

                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                ctx.branch_if_not(tile_row_valid, "exit");
                ctx.branch_if_not(tile_col_valid, "exit");

                let c_row_offset = ctx.mul_wide_u32(tile_row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(tile_col, 4);
                let c_tile_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_tile_base, c_col_offset);

                ctx.wmma_store_d_f32(c_addr, &frag_c, self.config.n, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
