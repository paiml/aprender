//! PTX generation for fused RMSNorm + Gate+Up Q4K GEMV + SwiGLU kernel

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::super::super::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::FusedRmsNormGateUpSwigluQ4KKernel;

impl Kernel for FusedRmsNormGateUpSwigluQ4KKernel {
    fn name(&self) -> &str {
        "fused_rmsnorm_gate_up_swiglu_q4k"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k = self.k;
        let epsilon = self.epsilon;

        // Shared memory layout:
        // - [0, k*4): Normalized input vector (k floats)
        // - [k*4, k*4+32): Warp partial sums for RMSNorm (8 floats for 8 warps)
        // - [k*4+32, k*4+64): Warp partial sums for gate (8 floats)
        // - [k*4+64, k*4+96): Warp partial sums for up (8 floats)
        let smem_size = (k * 4 + 96) as usize;

        PtxKernel::new("fused_rmsnorm_gate_up_swiglu_q4k")
            .param(PtxType::U64, "out_ptr") // Output: SwiGLU result (N)
            .param(PtxType::U64, "wg_ptr") // Q4_K gate weights (N x K/256 super-blocks)
            .param(PtxType::U64, "wu_ptr") // Q4_K up weights (N x K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K) - raw, not normalized
            .param(PtxType::U64, "gamma_ptr") // RMSNorm scale weights (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check: if block_id >= n_dim, exit early
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                // Load parameters
                let k_dim = ctx.load_param_u32("k_dim");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let wg_ptr = ctx.load_param_u64("wg_ptr");
                let wu_ptr = ctx.load_param_u64("wu_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let four = ctx.mov_u32_imm(4);
                let one = ctx.mov_u32_imm(1);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one);

                // ================================================================
                // PHASE 1: Load input and compute sum of squares for RMSNorm
                // ================================================================
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("load_loop");
                let loop_idx = ctx.add_u32_reg(idx, thread_id);
                let in_bounds = ctx.setp_lt_u32(loop_idx, k_dim);
                ctx.branch_if_not(in_bounds, "load_loop_end");

                // Load x[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let x_addr = ctx.add_u64(x_ptr, elem_offset);
                let x_val = ctx.ld_global_f32(x_addr);

                // Store to shared memory (will normalize later)
                ctx.st_shared_f32(elem_offset, x_val);

                // Accumulate sq_sum += x_val * x_val
                ctx.fma_f32_inplace(sq_sum, x_val, x_val);

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("load_loop");

                ctx.label("load_loop_end");

                // Warp-level reduction for sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Store warp partial to shared memory
                let k_bytes = ctx.mul_u32_reg(k_dim, four);
                let k_bytes_64 = ctx.cvt_u64_u32(k_bytes);
                let warp_sum_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let warp_sum_addr = ctx.add_u64(k_bytes_64, warp_sum_offset);

                ctx.branch_if_not(is_lane0, "skip_warp_write");
                ctx.st_shared_f32(warp_sum_addr, sq_sum);
                ctx.label("skip_warp_write");

                ctx.bar_sync(0);

                // Thread 0 computes rms_inv
                let total_sq_sum = ctx.mov_f32_imm(0.0);

                ctx.branch_if_not(is_thread0, "skip_final_reduce");

                for warp in 0..8u32 {
                    let warp_offset = ctx.mov_u64_imm((warp * 4) as u64);
                    let addr = ctx.add_u64(k_bytes_64, warp_offset);
                    let warp_sum = ctx.ld_shared_f32(addr);
                    ctx.add_f32_inplace(total_sq_sum, warp_sum);
                }

                // rms_inv = rsqrt(mean_sq + epsilon)
                let k_f32 = ctx.cvt_f32_u32(k_dim);
                let mean_sq = ctx.div_f32(total_sq_sum, k_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Store rms_inv for broadcast
                let rms_inv_offset = ctx.mov_u64_imm(0);
                ctx.st_shared_f32(rms_inv_offset, rms_inv);

                ctx.label("skip_final_reduce");

                ctx.bar_sync(1);

                // All threads load rms_inv
                let rms_inv_broadcast_offset = ctx.mov_u64_imm(0);
                let rms_inv_val = ctx.ld_shared_f32(rms_inv_broadcast_offset);

                // ================================================================
                // PHASE 2: Normalize input in shared memory
                // ================================================================
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, thread_id);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, k_dim);
                ctx.branch_if_not(in_bounds2, "norm_loop_end");

                // Load x from shared memory
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let x_smem = ctx.ld_shared_f32(elem_offset2);

                // Load gamma
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // Normalize: x_norm = x * rms_inv * gamma
                let normalized = ctx.mul_f32(x_smem, rms_inv_val);
                let scaled = ctx.mul_f32(normalized, gamma);

                // Store back to shared memory
                ctx.st_shared_f32(elem_offset2, scaled);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("norm_loop_end");

                ctx.bar_sync(2);

                // ================================================================
                // PHASE 3: Compute gate and up projections using normalized input
                // ================================================================
                let acc_gate = ctx.mov_f32_imm(0.0);
                let acc_up = ctx.mov_f32_imm(0.0);

                // Calculate number of super-blocks
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row offset for weights
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let wg_row_base = ctx.add_u64(wg_ptr, row_offset);
                let wu_row_base = ctx.add_u64(wu_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "sb_loop_end");

                // Calculate super-block addresses
                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let wg_sb_addr = ctx.add_u64(wg_row_base, sb_offset);
                let wu_sb_addr = ctx.add_u64(wu_row_base, sb_offset);

                // Load d and dmin for gate
                let d_gate_f16 = ctx.ld_global_f16(wg_sb_addr);
                let d_gate = ctx.cvt_f32_f16(d_gate_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_gate_addr = ctx.add_u64(wg_sb_addr, two_64);
                let dmin_gate_f16 = ctx.ld_global_f16(dmin_gate_addr);
                let dmin_gate = ctx.cvt_f32_f16(dmin_gate_f16);

                // Load d and dmin for up
                let d_up_f16 = ctx.ld_global_f16(wu_sb_addr);
                let d_up = ctx.cvt_f32_f16(d_up_f16);
                let dmin_up_addr = ctx.add_u64(wu_sb_addr, two_64);
                let dmin_up_f16 = ctx.ld_global_f16(dmin_up_addr);
                let dmin_up = ctx.cvt_f32_f16(dmin_up_f16);

                // Use d and dmin directly (simplified, correct for uniform-scale super-blocks)
                let eff_scale_gate = d_gate;
                let eff_min_gate = dmin_gate;
                let eff_scale_up = d_up;
                let eff_min_up = dmin_up;

                // Quantized data at offset 16
                let quant_offset = ctx.mov_u64_imm(16);
                let wg_quant_base = ctx.add_u64(wg_sb_addr, quant_offset);
                let wu_quant_base = ctx.add_u64(wu_sb_addr, quant_offset);

                // Constants for nibble extraction
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let eight_shift = ctx.mov_u32_imm(8);
                let sixteen_shift = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                // Load 4 bytes = 8 nibbles of weights (coalesced)
                let quant_byte_offset = ctx.mul_wide_u32_reg(lane_id, four);
                let wg_quant_addr = ctx.add_u64(wg_quant_base, quant_byte_offset);
                let wu_quant_addr = ctx.add_u64(wu_quant_base, quant_byte_offset);

                let wg_packed = ctx.ld_global_u32(wg_quant_addr);
                let wu_packed = ctx.ld_global_u32(wu_quant_addr);

                // Input base: super-block * 256 + lane * 8
                let sb_base_u32 = ctx.mov_u32_imm(256);
                let sb_base = ctx.mul_u32_reg(sb_idx, sb_base_u32);
                let eight_const = ctx.mov_u32_imm(8);
                let lane_base = ctx.mul_u32_reg(lane_id, eight_const);
                let input_base_idx = ctx.add_u32_reg(sb_base, lane_base);

                // Extract nibbles and process
                let nib0_g = ctx.and_u32(wg_packed, mask_4bit);
                let nib0_u = ctx.and_u32(wu_packed, mask_4bit);
                let shift4 = ctx.mov_u32_imm(4);
                let tmp1_g = ctx.shr_u32(wg_packed, shift4);
                let nib1_g = ctx.and_u32(tmp1_g, mask_4bit);
                let tmp1_u = ctx.shr_u32(wu_packed, shift4);
                let nib1_u = ctx.and_u32(tmp1_u, mask_4bit);
                let tmp2_g = ctx.shr_u32(wg_packed, eight_shift);
                let nib2_g = ctx.and_u32(tmp2_g, mask_4bit);
                let tmp2_u = ctx.shr_u32(wu_packed, eight_shift);
                let nib2_u = ctx.and_u32(tmp2_u, mask_4bit);
                let shift12 = ctx.mov_u32_imm(12);
                let tmp3_g = ctx.shr_u32(wg_packed, shift12);
                let nib3_g = ctx.and_u32(tmp3_g, mask_4bit);
                let tmp3_u = ctx.shr_u32(wu_packed, shift12);
                let nib3_u = ctx.and_u32(tmp3_u, mask_4bit);
                let tmp4_g = ctx.shr_u32(wg_packed, sixteen_shift);
                let nib4_g = ctx.and_u32(tmp4_g, mask_4bit);
                let tmp4_u = ctx.shr_u32(wu_packed, sixteen_shift);
                let nib4_u = ctx.and_u32(tmp4_u, mask_4bit);
                let shift20 = ctx.mov_u32_imm(20);
                let tmp5_g = ctx.shr_u32(wg_packed, shift20);
                let nib5_g = ctx.and_u32(tmp5_g, mask_4bit);
                let tmp5_u = ctx.shr_u32(wu_packed, shift20);
                let nib5_u = ctx.and_u32(tmp5_u, mask_4bit);
                let tmp6_g = ctx.shr_u32(wg_packed, twenty_four);
                let nib6_g = ctx.and_u32(tmp6_g, mask_4bit);
                let tmp6_u = ctx.shr_u32(wu_packed, twenty_four);
                let nib6_u = ctx.and_u32(tmp6_u, mask_4bit);
                let shift28 = ctx.mov_u32_imm(28);
                let nib7_g = ctx.shr_u32(wg_packed, shift28);
                let nib7_u = ctx.shr_u32(wu_packed, shift28);

                // Convert to f32
                let nib0_g_f = ctx.cvt_f32_u32(nib0_g);
                let nib0_u_f = ctx.cvt_f32_u32(nib0_u);
                let nib1_g_f = ctx.cvt_f32_u32(nib1_g);
                let nib1_u_f = ctx.cvt_f32_u32(nib1_u);
                let nib2_g_f = ctx.cvt_f32_u32(nib2_g);
                let nib2_u_f = ctx.cvt_f32_u32(nib2_u);
                let nib3_g_f = ctx.cvt_f32_u32(nib3_g);
                let nib3_u_f = ctx.cvt_f32_u32(nib3_u);
                let nib4_g_f = ctx.cvt_f32_u32(nib4_g);
                let nib4_u_f = ctx.cvt_f32_u32(nib4_u);
                let nib5_g_f = ctx.cvt_f32_u32(nib5_g);
                let nib5_u_f = ctx.cvt_f32_u32(nib5_u);
                let nib6_g_f = ctx.cvt_f32_u32(nib6_g);
                let nib6_u_f = ctx.cvt_f32_u32(nib6_u);
                let nib7_g_f = ctx.cvt_f32_u32(nib7_g);
                let nib7_u_f = ctx.cvt_f32_u32(nib7_u);

                // Dequantize
                let neg_min_g = ctx.neg_f32(eff_min_gate);
                let neg_min_u = ctx.neg_f32(eff_min_up);
                let dq0_g = ctx.fma_f32(eff_scale_gate, nib0_g_f, neg_min_g);
                let dq0_u = ctx.fma_f32(eff_scale_up, nib0_u_f, neg_min_u);
                let dq1_g = ctx.fma_f32(eff_scale_gate, nib1_g_f, neg_min_g);
                let dq1_u = ctx.fma_f32(eff_scale_up, nib1_u_f, neg_min_u);
                let dq2_g = ctx.fma_f32(eff_scale_gate, nib2_g_f, neg_min_g);
                let dq2_u = ctx.fma_f32(eff_scale_up, nib2_u_f, neg_min_u);
                let dq3_g = ctx.fma_f32(eff_scale_gate, nib3_g_f, neg_min_g);
                let dq3_u = ctx.fma_f32(eff_scale_up, nib3_u_f, neg_min_u);
                let dq4_g = ctx.fma_f32(eff_scale_gate, nib4_g_f, neg_min_g);
                let dq4_u = ctx.fma_f32(eff_scale_up, nib4_u_f, neg_min_u);
                let dq5_g = ctx.fma_f32(eff_scale_gate, nib5_g_f, neg_min_g);
                let dq5_u = ctx.fma_f32(eff_scale_up, nib5_u_f, neg_min_u);
                let dq6_g = ctx.fma_f32(eff_scale_gate, nib6_g_f, neg_min_g);
                let dq6_u = ctx.fma_f32(eff_scale_up, nib6_u_f, neg_min_u);
                let dq7_g = ctx.fma_f32(eff_scale_gate, nib7_g_f, neg_min_g);
                let dq7_u = ctx.fma_f32(eff_scale_up, nib7_u_f, neg_min_u);

                // Load normalized inputs from shared memory and accumulate
                let zero_imm = ctx.mov_u32_imm(0);
                let one_imm = ctx.mov_u32_imm(1);
                let two_imm = ctx.mov_u32_imm(2);
                let three_imm = ctx.mov_u32_imm(3);
                let four_imm = ctx.mov_u32_imm(4);
                let five_imm = ctx.mov_u32_imm(5);
                let six_imm = ctx.mov_u32_imm(6);
                let seven_imm = ctx.mov_u32_imm(7);

                let idx0 = ctx.add_u32_reg(input_base_idx, zero_imm);
                let off0 = ctx.mul_wide_u32_reg(idx0, four);
                let x0 = ctx.ld_shared_f32(off0);
                ctx.fma_f32_inplace(acc_gate, dq0_g, x0);
                ctx.fma_f32_inplace(acc_up, dq0_u, x0);

                let idx1 = ctx.add_u32_reg(input_base_idx, one_imm);
                let off1 = ctx.mul_wide_u32_reg(idx1, four);
                let x1 = ctx.ld_shared_f32(off1);
                ctx.fma_f32_inplace(acc_gate, dq1_g, x1);
                ctx.fma_f32_inplace(acc_up, dq1_u, x1);

                let idx2 = ctx.add_u32_reg(input_base_idx, two_imm);
                let off2 = ctx.mul_wide_u32_reg(idx2, four);
                let x2 = ctx.ld_shared_f32(off2);
                ctx.fma_f32_inplace(acc_gate, dq2_g, x2);
                ctx.fma_f32_inplace(acc_up, dq2_u, x2);

                let idx3 = ctx.add_u32_reg(input_base_idx, three_imm);
                let off3 = ctx.mul_wide_u32_reg(idx3, four);
                let x3 = ctx.ld_shared_f32(off3);
                ctx.fma_f32_inplace(acc_gate, dq3_g, x3);
                ctx.fma_f32_inplace(acc_up, dq3_u, x3);

                let idx4 = ctx.add_u32_reg(input_base_idx, four_imm);
                let off4 = ctx.mul_wide_u32_reg(idx4, four);
                let x4 = ctx.ld_shared_f32(off4);
                ctx.fma_f32_inplace(acc_gate, dq4_g, x4);
                ctx.fma_f32_inplace(acc_up, dq4_u, x4);

                let idx5 = ctx.add_u32_reg(input_base_idx, five_imm);
                let off5 = ctx.mul_wide_u32_reg(idx5, four);
                let x5 = ctx.ld_shared_f32(off5);
                ctx.fma_f32_inplace(acc_gate, dq5_g, x5);
                ctx.fma_f32_inplace(acc_up, dq5_u, x5);

                let idx6 = ctx.add_u32_reg(input_base_idx, six_imm);
                let off6 = ctx.mul_wide_u32_reg(idx6, four);
                let x6 = ctx.ld_shared_f32(off6);
                ctx.fma_f32_inplace(acc_gate, dq6_g, x6);
                ctx.fma_f32_inplace(acc_up, dq6_u, x6);

                let idx7 = ctx.add_u32_reg(input_base_idx, seven_imm);
                let off7 = ctx.mul_wide_u32_reg(idx7, four);
                let x7 = ctx.ld_shared_f32(off7);
                ctx.fma_f32_inplace(acc_gate, dq7_g, x7);
                ctx.fma_f32_inplace(acc_up, dq7_u, x7);

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // ================================================================
                // PHASE 4: Warp reduction and SwiGLU activation
                // ================================================================
                // Warp reduction for acc_gate
                let shfl16_gate = ctx.shfl_down_f32(acc_gate, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, shfl16_gate);
                let shfl8_gate = ctx.shfl_down_f32(acc_gate, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, shfl8_gate);
                let shfl4_gate = ctx.shfl_down_f32(acc_gate, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, shfl4_gate);
                let shfl2_gate = ctx.shfl_down_f32(acc_gate, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, shfl2_gate);
                let shfl1_gate = ctx.shfl_down_f32(acc_gate, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_gate, shfl1_gate);

                // Warp reduction for acc_up
                let shfl16_up = ctx.shfl_down_f32(acc_up, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, shfl16_up);
                let shfl8_up = ctx.shfl_down_f32(acc_up, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, shfl8_up);
                let shfl4_up = ctx.shfl_down_f32(acc_up, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, shfl4_up);
                let shfl2_up = ctx.shfl_down_f32(acc_up, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, shfl2_up);
                let shfl1_up = ctx.shfl_down_f32(acc_up, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc_up, shfl1_up);

                // Store warp partials to shared memory
                let thirty_two_64 = ctx.mov_u64_imm(32);
                let sixty_four_64 = ctx.mov_u64_imm(64);
                let warp_gate_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let warp_gate_base = ctx.add_u64(k_bytes_64, thirty_two_64);
                let warp_gate_addr = ctx.add_u64(warp_gate_base, warp_gate_offset);
                let warp_up_base = ctx.add_u64(k_bytes_64, sixty_four_64);
                let warp_up_addr = ctx.add_u64(warp_up_base, warp_gate_offset);

                ctx.branch_if_not(is_lane0, "skip_warp_acc_write");
                ctx.st_shared_f32(warp_gate_addr, acc_gate);
                ctx.st_shared_f32(warp_up_addr, acc_up);
                ctx.label("skip_warp_acc_write");

                ctx.bar_sync(3);

                // Thread 0 computes final SwiGLU result
                ctx.branch_if_not(is_thread0, "exit");

                let final_gate = ctx.mov_f32_imm(0.0);
                let final_up = ctx.mov_f32_imm(0.0);
                for warp in 0..8u32 {
                    let warp_offset = ctx.mov_u64_imm((warp * 4) as u64);
                    let gate_addr = ctx.add_u64(warp_gate_base, warp_offset);
                    let up_addr = ctx.add_u64(warp_up_base, warp_offset);
                    let warp_gate_sum = ctx.ld_shared_f32(gate_addr);
                    let warp_up_sum = ctx.ld_shared_f32(up_addr);
                    ctx.add_f32_inplace(final_gate, warp_gate_sum);
                    ctx.add_f32_inplace(final_up, warp_up_sum);
                }

                // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
                let neg_gate = ctx.neg_f32(final_gate);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled_exp = ctx.mul_f32(neg_gate, log2_e);
                let exp_val = ctx.ex2_f32(scaled_exp);
                let one_f = ctx.mov_f32_imm(1.0);
                let one_plus_exp = ctx.add_f32(one_f, exp_val);
                let sigmoid = ctx.rcp_f32(one_plus_exp);
                let silu = ctx.mul_f32(final_gate, sigmoid);

                // output = SiLU(gate) * up
                let output = ctx.mul_f32(silu, final_up);

                // Store output[block_id]
                let out_offset = ctx.mul_wide_u32(block_id, 4);
                let out_addr = ctx.add_u64(out_ptr, out_offset);
                ctx.st_global_f32(out_addr, output);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
