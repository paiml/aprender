//! Q5_K FUSED GEMM KERNEL (PARITY-116)

use super::super::{Kernel, Q5K_SUPER_BLOCK_BYTES, Q5K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q5_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct Q5KKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by 256
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
}

impl Q5KKernel {
    /// Create a new Q5_K quantized GEMM kernel
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
        }
    }

    /// Set output tile size
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k / Q5K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q5KKernel {
    fn name(&self) -> &str {
        "q5k_gemm_ggml"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = Q5K_SUPER_BLOCK_SIZE * 4; // 256 f32 values

        PtxKernel::new("q5k_gemm_ggml")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_quant_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_quant_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate output position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check predicates
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp to valid range for memory safety
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks (K / 256)
                let num_k_super_blocks = ctx.div_u32(k_param, Q5K_SUPER_BLOCK_SIZE);

                // Super-block loop
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Calculate super-block address
                let sb_per_row = num_k_super_blocks;
                let row_sb_offset = ctx.mul_u32_reg(clamped_col, sb_per_row);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let byte_offset = ctx.mul_wide_u32(total_sb_offset, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load d (f16 at offset 0)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load dmin (f16 at offset 2)
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Process 8 sub-blocks of 32 values each
                let sub_block_idx = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, eight);
                ctx.branch_if(sub_done, "sub_block_done");

                // Extract 6-bit scale and min (same as Q4_K)
                let bit_offset = ctx.mul_u32(sub_block_idx, 12);
                let byte_idx = ctx.div_u32(bit_offset, 8);
                let bit_in_byte = ctx.rem_u32(bit_offset, 8);

                let four = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let scales_addr = ctx.add_u64(scales_base, byte_idx_64);
                let scale_b0 = ctx.ld_global_u8(scales_addr);
                let one_64 = ctx.mov_u64_imm(1);
                let scales_addr1 = ctx.add_u64(scales_addr, one_64);
                let scale_b1 = ctx.ld_global_u8(scales_addr1);

                let b0_32 = ctx.cvt_u32_u8(scale_b0);
                let b1_32 = ctx.cvt_u32_u8(scale_b1);
                let eight_shift = ctx.mov_u32_imm(8);
                let b1_shifted = ctx.shl_u32(b1_32, eight_shift);
                let combined = ctx.or_u32(b0_32, b1_shifted);
                let bits_12 = ctx.shr_u32(combined, bit_in_byte);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let scale_6bit = ctx.and_u32(bits_12, mask_6bit);
                let six_shift = ctx.mov_u32_imm(6);
                let min_shifted = ctx.shr_u32(bits_12, six_shift);
                let min_6bit = ctx.and_u32(min_shifted, mask_6bit);

                let scale_f32 = ctx.cvt_f32_u32(scale_6bit);
                let min_f32 = ctx.cvt_f32_u32(min_6bit);
                let inv_63 = ctx.mov_f32_imm(1.0 / 63.0);
                let scale_norm = ctx.mul_f32(scale_f32, inv_63);
                let min_norm = ctx.mul_f32(min_f32, inv_63);

                // Thread's lane within sub-block
                let lane = ctx.rem_u32(tid, 32);

                // Load low 4-bit value from qs (offset 16 + sub_block_idx * 16 + lane/2)
                let sixteen = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen);
                let sub_block_offset = ctx.mul_u32(sub_block_idx, 16);
                let sub_block_offset_64 = ctx.cvt_u64_u32(sub_block_offset);
                let qs_sub_base = ctx.add_u64(qs_base, sub_block_offset_64);

                let byte_in_sub = ctx.div_u32(lane, 2);
                let nibble_idx = ctx.rem_u32(lane, 2);
                let byte_in_sub_64 = ctx.cvt_u64_u32(byte_in_sub);
                let qs_addr = ctx.add_u64(qs_sub_base, byte_in_sub_64);
                let packed_ql = ctx.ld_global_u8(qs_addr);

                let shift_amt = ctx.mul_u32(nibble_idx, 4);
                let packed_ql_32 = ctx.cvt_u32_u8(packed_ql);
                let shifted_ql = ctx.shr_u32(packed_ql_32, shift_amt);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let ql = ctx.and_u32(shifted_ql, mask_4bit);

                // Load high bit from qh (offset 144 + (sub_block_idx * 32 + lane) / 8)
                let qh_base_offset = ctx.mov_u64_imm(144);
                let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                let global_bit_idx = ctx.mul_u32(sub_block_idx, 32);
                let global_bit_idx_full = ctx.add_u32_reg(global_bit_idx, lane);
                let qh_byte_idx = ctx.div_u32(global_bit_idx_full, 8);
                let qh_bit_idx = ctx.rem_u32(global_bit_idx_full, 8);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_byte = ctx.ld_global_u8(qh_addr);
                let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);
                let qh_shifted = ctx.shr_u32(qh_byte_32, qh_bit_idx);
                let mask_1bit = ctx.mov_u32_imm(1);
                let qh = ctx.and_u32(qh_shifted, mask_1bit);

                // Combine: quant = ql + 16 * qh (5-bit value: 0-31)
                let sixteen_u32 = ctx.mov_u32_imm(16);
                let qh_scaled = ctx.mul_u32_reg(qh, sixteen_u32);
                let quant = ctx.add_u32_reg(ql, qh_scaled);

                // Dequantize: val = d * scale * quant - dmin * min
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let d_scale = ctx.mul_f32(d, scale_norm);
                let scaled = ctx.mul_f32(d_scale, quant_f32);
                let dmin_min = ctx.mul_f32(dmin, min_norm);
                let dequant = ctx.sub_f32(scaled, dmin_min);

                // Load activation and accumulate
                let two_fifty_six = ctx.mov_u32_imm(256);
                let sb_k_offset = ctx.mul_u32_reg(sb_idx, two_fifty_six);
                let sub_k_offset = ctx.mul_u32_reg(sub_block_idx, thirty_two);
                let k_offset = ctx.add_u32_reg(sb_k_offset, sub_k_offset);
                let k_offset_full = ctx.add_u32_reg(k_offset, lane);

                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset_full);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce
                let shuffled_16 = ctx.shfl_down_f32(prod, 16, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_16);
                let shuffled_8 = ctx.shfl_down_f32(prod_1, 8, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(prod_2, 4, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(prod_3, 2, 0xFFFF_FFFF);
                let prod_4 = ctx.add_f32(prod_3, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(prod_4, 1, 0xFFFF_FFFF);
                let sub_block_sum = ctx.add_f32(prod_4, shuffled_1);

                let broadcast_sum = ctx.shfl_idx_f32(sub_block_sum, 0, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, broadcast_sum);

                ctx.add_u32_inplace(sub_block_idx, 1);
                ctx.branch("sub_block_loop");

                ctx.label("sub_block_done");

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_done");

                // Store result
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);

                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
