use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Wide Q4_K GEMV kernel with 256 threads (8 warps) and cross-warp reduction (PAR-082)
///
/// Root cause of 3x Ollama performance gap:
/// - CoalescedQ4KGemvKernel uses 32 threads (1 warp) per output row
/// - Only 33% occupancy on RTX 4090 SMs (16 blocks × 1 warp = 16/48 warps)
/// - Cannot hide memory latency — SM idles during ~400 cycle global loads
///
/// Fix: 8 warps per block, each processes every 8th super-block, then reduce via shared memory
/// - 256 threads per block → 67-100% occupancy (6 blocks × 8 warps = 48 warps/SM)
/// - 8x more warps to hide memory latency
/// - Shared memory: only 32 bytes (8 partial sums)
///
/// # Performance Target
/// - Memory bandwidth: 500+ GB/s (vs ~190 GB/s with 1 warp)
/// - Decode: 100+ tok/s 7B Q4K (vs 39 tok/s with 1 warp)
///
/// # References
/// - llama.cpp `dequantize_mul_mat_vec_q4_K` uses 256 threads
/// - Pope et al. (2023) "Efficiently Scaling Transformer Inference"
#[derive(Debug, Clone)]
pub struct WideQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 8 = 256 threads)
    pub num_warps: u32,
}

impl WideQ4KGemvKernel {
    /// Create a new wide Q4_K GEMV kernel with 8 warps (256 threads)
    ///
    /// Empirically measured: 8 warps (67 tok/s) > 4 warps (61 tok/s) on RTX 4090
    /// The extra SM occupancy from 8 warps hides memory latency better
    /// than the reduced cross-warp reduction overhead of 4 warps.
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 8 }
    }
}

impl Kernel for WideQ4KGemvKernel {
    fn name(&self) -> &str {
        "wide_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;

        // Shared memory for cross-warp reduction: num_warps × 4 bytes
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("wide_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                // Bounds check (block_id = output row index)
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "wide_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each warp starts at its own super-block index, strides by num_warps
                let sb_idx_init = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_init, warp_id);

                let num_warps_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("wide_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "wide_sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (all lanes)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ========================================================
                // Scale loading: Lane 0 loads 12 bytes as 3 x u32, broadcasts
                // (Same as CoalescedQ4KGemvKernel)
                // ========================================================
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "wide_skip_scale_load");

                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);

                ctx.label("wide_skip_scale_load");

                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract scale bytes
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                let s0_32 = ctx.and_u32(scales_0_3_bcast, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3_bcast, eight);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3_bcast, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3_bcast, twenty_four);

                let s4_32 = ctx.and_u32(scales_4_7_bcast, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7_bcast, eight);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7_bcast, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7_bcast, twenty_four);

                let s8_32 = ctx.and_u32(scales_8_11_bcast, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11_bcast, eight);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11_bcast, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11_bcast, twenty_four);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Block 0-3
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);
                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);
                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);
                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);
                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);

                // Block 4-7
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);
                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);
                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);
                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);
                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);

                // Precompute d*scale and dmin*min
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // qs base
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process 8 values per thread (same as CoalescedQ4K)
                let offsets_and_blocks: [(u32, u32); 8] = [
                    (0, 0),
                    (32, 1),
                    (64, 2),
                    (96, 3),
                    (128, 4),
                    (160, 5),
                    (192, 6),
                    (224, 7),
                ];

                for (offset, block_idx) in offsets_and_blocks {
                    let (ds, dm) = match block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);
                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);

                // Stride by num_warps (not 1)
                ctx.add_u32_reg_inplace(sb_idx, num_warps_reg);
                ctx.branch("wide_sb_loop");

                ctx.label("wide_sb_loop_end");

                // ====================================================
                // Phase 1: Intra-warp reduction (warp shuffle)
                // ====================================================
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // ====================================================
                // Phase 2: Cross-warp reduction via shared memory
                // Lane 0 of each warp writes partial sum to shared[warp_id]
                // ====================================================
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0_final = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0_final, "wide_skip_smem_write");

                // shared[warp_id] = acc
                let four_smem = ctx.mov_u32_imm(4);
                let warp_offset = ctx.mul_u32_reg(warp_id, four_smem); // 4 bytes per f32
                let warp_smem_addr = ctx.cvt_u64_u32(warp_offset);
                ctx.st_shared_f32(warp_smem_addr, acc);

                ctx.label("wide_skip_smem_write");

                // Full block synchronization
                ctx.bar_sync(0);

                // Thread 0 reads all partial sums and reduces
                let is_thread0 = ctx.setp_eq_u32(thread_id, zero_u32);
                ctx.branch_if_not(is_thread0, "wide_exit");

                let final_sum = ctx.mov_f32_imm(0.0);
                // Unrolled loop over num_warps (8) partial sums
                for w in 0..num_warps {
                    let w_offset = ctx.mov_u64_imm(u64::from(w * 4));
                    let partial = ctx.ld_shared_f32(w_offset);
                    ctx.add_f32_inplace(final_sum, partial);
                }

                // Write final result to global memory
                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, final_sum);

                ctx.label("wide_exit");
                ctx.ret();
            })
    }
}
