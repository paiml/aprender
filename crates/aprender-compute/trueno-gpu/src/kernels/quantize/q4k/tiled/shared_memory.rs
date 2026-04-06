//! Tiled Q4_K GEMV kernel with shared memory input caching
//!
//! Input vector is cached in shared memory and shared by multiple outputs.
//! Addresses inefficiency in `Q4KGemvKernel` where each warp loads entire
//! input vector from global memory.

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Tiled Q4_K GEMV kernel with shared memory input caching
///
/// Addresses inefficiency in `Q4KGemvKernel`:
/// - Original: Each warp loads entire input vector from global memory
/// - Tiled: Input vector cached in shared memory, shared by multiple outputs
pub struct TiledQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of outputs per block (default: 4)
    pub outputs_per_block: u32,
}

impl TiledQ4KGemvKernel {
    /// Create a new tiled Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self {
            k,
            n,
            outputs_per_block: 4, // Default: 4 outputs per block (128 threads = 4 warps)
        }
    }

    /// Set number of outputs computed per block
    #[must_use]
    pub const fn with_outputs_per_block(mut self, outputs_per_block: u32) -> Self {
        self.outputs_per_block = outputs_per_block;
        self
    }
}

impl Kernel for TiledQ4KGemvKernel {
    fn name(&self) -> &str {
        "tiled_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k = self.k;
        let outputs_per_block = self.outputs_per_block;

        // Shared memory for input vector: K floats
        let smem_size = (k * 4) as usize;

        PtxKernel::new("tiled_q4k_gemv")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q4_K weights (N × K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Constants
                let four = ctx.mov_u32_imm(4);
                let outputs_per_block_reg = ctx.mov_u32_imm(outputs_per_block);

                // ================================================================
                // PHASE 1: Cooperatively load input vector into shared memory
                // ================================================================
                let idx = ctx.mov_u32_imm(0);

                ctx.label("load_loop");
                let loop_idx = ctx.add_u32_reg(idx, thread_id);
                let in_bounds = ctx.setp_lt_u32(loop_idx, k_dim);
                ctx.branch_if_not(in_bounds, "load_loop_end");

                // Load x[loop_idx] from global memory
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let x_addr = ctx.add_u64(x_ptr, elem_offset);
                let x_val = ctx.ld_global_f32(x_addr);

                // GH-37 FIX: Use direct .shared addressing (u32 offset) instead of
                // generic addressing (cvta.shared.u64 + ld/st). Direct .shared is
                // more efficient: fewer registers, no cvta instruction, hardware
                // knows the address space.
                let smem_offset = ctx.mul_u32_reg(loop_idx, four);
                ctx.st_shared_f32(smem_offset, x_val);

                ctx.add_u32_inplace(idx, 32 * outputs_per_block); // stride by block size
                ctx.branch("load_loop");

                ctx.label("load_loop_end");

                // Synchronize: ensure input is fully loaded
                ctx.bar_sync(0);

                // ================================================================
                // PHASE 2: Compute multiple outputs using cached input
                // ================================================================
                // Each warp computes one output element
                // With 8 warps per block, we compute up to 8 outputs per block
                let warp_id = ctx.div_u32(thread_id, 32);
                let lane_id = ctx.rem_u32(thread_id, 32);

                // Calculate which output this warp is computing
                let base_output = ctx.mul_u32_reg(block_id, outputs_per_block_reg);
                let output_idx = ctx.add_u32_reg(base_output, warp_id);

                // Check if this warp has work to do
                let warp_oob = ctx.setp_ge_u32(output_idx, n_dim);
                ctx.branch_if(warp_oob, "exit");

                // Also check if warp_id < outputs_per_block
                let warp_beyond_block = ctx.setp_ge_u32(warp_id, outputs_per_block_reg);
                ctx.branch_if(warp_beyond_block, "exit");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate number of super-blocks: ceil(K / 256) for GGUF
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Calculate base address for this row's weights
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(output_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Super-block loop
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Each thread in warp processes 8 elements (256 per super-block / 32 threads)
                let thread_partial = ctx.mov_f32_imm(0.0);

                for offset in [0u32, 32, 64, 96, 128, 160, 192, 224] {
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Determine sub-block (0-7)
                    let sub_block = ctx.div_u32(val_idx, 32);

                    // Load scale bytes (simplified - could be optimized further)
                    let four_64 = ctx.mov_u64_imm(4);
                    let scales_base = ctx.add_u64(sb_addr, four_64);

                    // Simple scale/min extraction for sub-blocks 0-3
                    let sub_block_lt_4 = ctx.mov_u32_imm(4);
                    let is_simple = ctx.setp_lt_u32(sub_block, sub_block_lt_4);

                    let sub_block_64 = ctx.cvt_u64_u32(sub_block);
                    let scale_byte_addr = ctx.add_u64(scales_base, sub_block_64);
                    let scale_byte = ctx.ld_global_u8(scale_byte_addr);
                    let scale_byte_32 = ctx.cvt_u32_u8(scale_byte);

                    let four_reg = ctx.mov_u32_imm(4);
                    let sub_block_plus_4 = ctx.add_u32_reg(sub_block, four_reg);
                    let sub_block_plus_4_64 = ctx.cvt_u64_u32(sub_block_plus_4);
                    let min_byte_addr = ctx.add_u64(scales_base, sub_block_plus_4_64);
                    let min_byte = ctx.ld_global_u8(min_byte_addr);
                    let min_byte_32 = ctx.cvt_u32_u8(min_byte);

                    let mask_6bit = ctx.mov_u32_imm(0x3F);
                    let mask_4bit = ctx.mov_u32_imm(0x0F);
                    let six = ctx.mov_u32_imm(6);

                    let scale_simple = ctx.and_u32(scale_byte_32, mask_6bit);
                    let min_simple = ctx.and_u32(min_byte_32, mask_6bit);

                    // Complex path for blocks 4-7
                    // CORRECTNESS-001: Fixed scale/min extraction per GGML Q4_K spec
                    // CPU reference (extract_scale_min at realizar/quantize.rs:6589):
                    //   scale = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
                    //   min   = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
                    let eight_64 = ctx.mov_u64_imm(8);
                    let scales_8_base = ctx.add_u64(scales_base, eight_64);
                    // Safe subtraction: for sub_block < 4, use 0 to avoid underflow
                    // (the loaded value won't be used anyway due to selp)
                    let sub_block_minus_4_raw = ctx.sub_u32_reg(sub_block, four_reg);
                    let zero_safe = ctx.mov_u32_imm(0);
                    let sub_block_minus_4 =
                        ctx.selp_u32(is_simple, zero_safe, sub_block_minus_4_raw);
                    let sub_block_minus_4_64 = ctx.cvt_u64_u32(sub_block_minus_4);
                    let scales_8_addr = ctx.add_u64(scales_8_base, sub_block_minus_4_64);
                    let s8_byte = ctx.ld_global_u8(scales_8_addr);
                    let s8_byte_32 = ctx.cvt_u32_u8(s8_byte);

                    // Load scales[sub_block - 4] for scale high bits (not scales[sub_block]!)
                    let scale_hi_src_addr = ctx.add_u64(scales_base, sub_block_minus_4_64);
                    let scale_hi_src_byte = ctx.ld_global_u8(scale_hi_src_addr);
                    let scale_hi_src_32 = ctx.cvt_u32_u8(scale_hi_src_byte);

                    // scale = (scales[sub_block + 4] & 0x0F) | ((scales[sub_block - 4] >> 6) << 4)
                    let s8_lo = ctx.and_u32(s8_byte_32, mask_4bit);
                    let s0_hi = ctx.shr_u32(scale_hi_src_32, six);
                    let s0_hi_shifted = ctx.shl_u32(s0_hi, four_reg);
                    let scale_complex = ctx.or_u32(s8_lo, s0_hi_shifted);

                    // min = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4)
                    // Note: use scale_byte_32 (scales[sub_block]) NOT min_byte_32 (scales[sub_block + 4])
                    let s8_hi = ctx.shr_u32(s8_byte_32, four_reg);
                    let s4_hi = ctx.shr_u32(scale_byte_32, six);
                    let s4_hi_shifted = ctx.shl_u32(s4_hi, four_reg);
                    let min_complex = ctx.or_u32(s8_hi, s4_hi_shifted);

                    let scale = ctx.selp_u32(is_simple, scale_simple, scale_complex);
                    let min = ctx.selp_u32(is_simple, min_simple, min_complex);

                    let scale_f = ctx.cvt_f32_u32(scale);
                    let min_f = ctx.cvt_f32_u32(min);
                    let ds = ctx.mul_f32(d, scale_f);
                    let dm = ctx.mul_f32(dmin, min_f);

                    // Load quantized value
                    let sixteen_64 = ctx.mov_u64_imm(16);
                    let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_reg);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit);

                    // Dequantize
                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Load activation from SHARED MEMORY (the key optimization!)
                    // GH-37 FIX: Use direct .shared addressing (u32 offset, ld.shared.f32)
                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_smem_offset = ctx.mul_u32_reg(x_idx, four);
                    let x_cached = ctx.ld_shared_f32(x_smem_offset);

                    ctx.fma_f32_inplace(thread_partial, x_cached, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp shuffle reduction
                let shfl16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl16);
                let shfl8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl8);
                let shfl4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl4);
                let shfl2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl2);
                let shfl1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, shfl1);

                // Only lane 0 of each warp writes
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);
                ctx.branch_if_not(is_lane0, "exit");

                // Store y[output_idx]
                let y_offset = ctx.mul_wide_u32_reg(output_idx, four);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
