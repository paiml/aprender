//! WMMA FP16 batched GEMM variant using Tensor Core PTX intrinsics (WAPR-PERF-011).

#![allow(clippy::similar_names)]

use super::BatchedGemmKernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType, WmmaLayout};

impl BatchedGemmKernel {
    /// Build WMMA FP16 batched GEMM kernel using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Each batch is processed by a separate grid slice in the z-dimension.
    /// Uses cvta.shared.u64 pattern from WAPR-PERF-010 for correct WMMA loads.
    /// Launch config: grid_3d((m+15)/16, (n+15)/16, batch, 32, 1, 1)
    pub(super) fn build_wmma_fp16(&self) -> PtxKernel {
        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 2 * 2; // Two FP16 tiles (A and B)
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_wmma_fp16")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // WAPR-PERF-011: Batched WMMA for multi-head attention
                // Grid z-dimension indexes batch, x/y index 16x16 output tiles
                // One warp (32 threads) processes one output tile per batch

                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Calculate output tile position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let tile_row = ctx.mul_u32(ctaid_y, tile_size);
                let tile_col = ctx.mul_u32(ctaid_x, tile_size);

                // Load parameters
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid tile
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let tile_row_valid = ctx.setp_lt_u32(tile_row, m_param);
                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets
                // A batch offset = batch_idx * m * k * 4
                // B batch offset = batch_idx * k * n * 4
                // C batch offset = batch_idx * m * n * 4
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Shared memory base addresses
                let smem_a_base = ctx.mov_u32_imm(0);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 2); // After A tile (FP16)

                // Initialize accumulator fragments
                let frag_c = ctx.wmma_init_c_zero();

                // Loop counter for K tiles
                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                // K offset for this tile
                let k_offset = ctx.mul_u32_reg(k_tile_idx, tile_size_reg);

                // === Load A tile to shared memory (FP32 global → FP16 shared) ===
                let elements_per_thread = ctx.mov_u32_imm(8);
                let my_start = ctx.mul_u32_reg(tid_x, elements_per_thread);

                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop_batched");
                let load_done = ctx.setp_ge_u32(load_idx, elements_per_thread);
                ctx.branch_if(load_done, "load_a_end_batched");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                let row_in_tile = ctx.div_u32(elem_idx, 16);
                let col_in_tile = ctx.rem_u32(elem_idx, 16);

                // Store 0 first (default for out-of-bounds)
                let smem_a_offset = ctx.mul_u32(elem_idx, 2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_offset);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let zero_f16 = ctx.cvt_f16_f32(zero_f32);
                ctx.st_shared_f16(smem_a_addr, zero_f16);

                // Check bounds: a_row < m AND a_col < k
                let a_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let a_col = ctx.add_u32_reg(k_offset, col_in_tile);
                let a_row_valid = ctx.setp_lt_u32(a_row, m_param);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                ctx.branch_if_not(a_row_valid, "skip_a_load_batched");
                ctx.branch_if_not(a_col_valid, "skip_a_load_batched");
                ctx.branch_if_not(batch_valid, "skip_a_load_batched");

                // Global A address: A[batch, tile_row + row_in_tile, k_offset + col_in_tile]
                let k_reg = ctx.mov_u32_imm(k_val);
                let a_idx = ctx.mad_lo_u32(a_row, k_reg, a_col);
                let a_byte_offset = ctx.mul_wide_u32(a_idx, 4);
                let a_addr = ctx.add_u64(a_batch_ptr, a_byte_offset);

                let a_val_f32 = ctx.ld_global_f32(a_addr);
                let a_val_f16 = ctx.cvt_f16_f32(a_val_f32);
                ctx.st_shared_f16(smem_a_addr, a_val_f16);

                ctx.label("skip_a_load_batched");
                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_a_loop_batched");
                ctx.label("load_a_end_batched");

                // === Load B tile to shared memory ===
                let load_idx_b = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop_batched");
                let load_b_done = ctx.setp_ge_u32(load_idx_b, elements_per_thread);
                ctx.branch_if(load_b_done, "load_b_end_batched");

                let elem_idx_b = ctx.add_u32_reg(my_start, load_idx_b);
                let row_in_tile_b = ctx.div_u32(elem_idx_b, 16);
                let col_in_tile_b = ctx.rem_u32(elem_idx_b, 16);

                let smem_b_offset = ctx.mul_u32(elem_idx_b, 2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_offset);
                let zero_b_f32 = ctx.mov_f32_imm(0.0);
                let zero_b_f16 = ctx.cvt_f16_f32(zero_b_f32);
                ctx.st_shared_f16(smem_b_addr, zero_b_f16);

                // Check bounds: b_row < k AND b_col < n
                let b_row = ctx.add_u32_reg(k_offset, row_in_tile_b);
                let b_col = ctx.add_u32_reg(tile_col, col_in_tile_b);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);
                let b_col_valid = ctx.setp_lt_u32(b_col, n_param);

                ctx.branch_if_not(b_row_valid, "skip_b_load_batched");
                ctx.branch_if_not(b_col_valid, "skip_b_load_batched");
                ctx.branch_if_not(batch_valid, "skip_b_load_batched");

                // Global B address: B[batch, k_offset + row_in_tile, tile_col + col_in_tile]
                let n_reg = ctx.mov_u32_imm(n_val);
                let b_idx = ctx.mad_lo_u32(b_row, n_reg, b_col);
                let b_byte_offset = ctx.mul_wide_u32(b_idx, 4);
                let b_addr = ctx.add_u64(b_batch_ptr, b_byte_offset);

                let b_val_f32 = ctx.ld_global_f32(b_addr);
                let b_val_f16 = ctx.cvt_f16_f32(b_val_f32);
                ctx.st_shared_f16(smem_b_addr, b_val_f16);

                ctx.label("skip_b_load_batched");
                ctx.add_u32_inplace(load_idx_b, 1);
                ctx.branch("load_b_loop_batched");
                ctx.label("load_b_end_batched");

                // Synchronize before WMMA
                ctx.bar_sync(0);

                // === WMMA matrix multiply ===
                // WAPR-PERF-010 FIX: Use cvta.shared.u64 to get generic pointer
                let smem_generic_base = ctx.shared_base_addr();

                // Load A fragment from shared memory
                let frag_a = ctx.wmma_load_a_f16(smem_generic_base, 16, WmmaLayout::RowMajor);

                // Load B fragment from shared memory
                // WAPR-PERF-014 FIX: B is stored row-major in shared memory, so use RowMajor
                let smem_b_offset_u64 = ctx.cvt_u64_u32(smem_b_base);
                let smem_b_ptr = ctx.add_u64(smem_generic_base, smem_b_offset_u64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_ptr, 16, WmmaLayout::RowMajor);

                // Matrix multiply-accumulate: D = A * B + C
                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // WAPR-PERF-010 FIX: Copy D → C for accumulation across K tiles
                // The MMA instruction outputs to new registers, so we must copy
                // the result back to the accumulator for the next iteration
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                // Synchronize after WMMA (before next tile load)
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");

                ctx.label("k_tile_end");

                // Store result to global memory (only valid threads)
                ctx.branch_if_not(batch_valid, "exit_batched");
                ctx.branch_if_not(tile_row_valid, "exit_batched");
                ctx.branch_if_not(tile_col_valid, "exit_batched");

                // C output address with batch offset
                let c_tile_row_offset = ctx.mul_wide_u32(tile_row, n_val * 4);
                let c_tile_col_offset = ctx.mul_wide_u32(tile_col, 4);
                let c_tile_base = ctx.add_u64(c_batch_ptr, c_tile_row_offset);
                let c_tile_addr = ctx.add_u64(c_tile_base, c_tile_col_offset);

                ctx.wmma_store_d_f32(c_tile_addr, &frag_c, n_val, WmmaLayout::RowMajor);

                ctx.label("exit_batched");
                ctx.ret();
            })
    }
}
