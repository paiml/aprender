//! Q6_K fused GEMM kernel (PARITY-117).

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// Q6_K FUSED GEMM KERNEL (PARITY-117)
// =============================================================================
//
// Q6_K Super-block Layout (210 bytes for 256 values):
// - Offset 0-127: ql (128 bytes, 256 x 4-bit low values packed)
// - Offset 128-191: qh (64 bytes, 256 x 2-bit high values packed)
// - Offset 192-207: scales (16 bytes, 16 x 8-bit scales for 16 sub-blocks of 16)
// - Offset 208-209: d (f16 super-block scale)
//
// Dequantization: val = d x scale_b x (ql + 4*qh - 32)
// Where ql is 4-bit (0-15), qh is 2-bit (0-3), giving 6-bit signed range (-32 to 31)

/// Q6_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct Q6KKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by 256
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
}

impl Q6KKernel {
    /// Create a new Q6_K quantized GEMM kernel
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
        self.k / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q6KKernel {
    fn name(&self) -> &str {
        "q6k_gemm_ggml"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = Q6K_SUPER_BLOCK_SIZE * 4; // 256 f32 values

        PtxKernel::new("q6k_gemm_ggml")
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

                // Clamp to valid range
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks (K / 256)
                let num_k_super_blocks = ctx.div_u32(k_param, Q6K_SUPER_BLOCK_SIZE);

                // Super-block loop
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Calculate super-block address
                let sb_per_row = num_k_super_blocks;
                let row_sb_offset = ctx.mul_u32_reg(clamped_col, sb_per_row);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let byte_offset = ctx.mul_wide_u32(total_sb_offset, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Process 16 sub-blocks of 16 values each (Q6_K uses 16-element sub-blocks)
                let sub_block_idx = ctx.mov_u32_imm(0);
                let sixteen_blocks = ctx.mov_u32_imm(16);
                let sixteen_values = ctx.mov_u32_imm(16);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, sixteen_blocks);
                ctx.branch_if(sub_done, "sub_block_done");

                // Load 8-bit scale for this sub-block (offset 192 + sub_block_idx)
                let scales_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_offset);
                let sub_block_idx_64 = ctx.cvt_u64_u32(sub_block_idx);
                let scale_addr = ctx.add_u64(scales_base, sub_block_idx_64);
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                // Q6_K scales are signed 8-bit, center at 32 for proper range
                let scale_f32 = ctx.cvt_f32_u32(scale_u32);

                // Thread's lane within sub-block (0-15)
                let lane = ctx.rem_u32(tid, 16);

                // Global value index within super-block
                let global_val_idx = ctx.mul_u32(sub_block_idx, 16);
                let global_val_idx_full = ctx.add_u32_reg(global_val_idx, lane);

                // Load low 4-bit value from ql (offset 0 + global_val_idx / 2)
                let ql_byte_idx = ctx.div_u32(global_val_idx_full, 2);
                let ql_nibble_idx = ctx.rem_u32(global_val_idx_full, 2);
                let ql_byte_idx_64 = ctx.cvt_u64_u32(ql_byte_idx);
                let ql_addr = ctx.add_u64(sb_addr, ql_byte_idx_64);
                let ql_packed = ctx.ld_global_u8(ql_addr);
                let ql_packed_32 = ctx.cvt_u32_u8(ql_packed);
                let four = ctx.mov_u32_imm(4);
                let ql_shift = ctx.mul_u32_reg(ql_nibble_idx, four);
                let ql_shifted = ctx.shr_u32(ql_packed_32, ql_shift);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let ql = ctx.and_u32(ql_shifted, mask_4bit);

                // Load high 2-bit value from qh (offset 128 + global_val_idx / 4)
                let qh_offset = ctx.mov_u64_imm(128);
                let qh_base = ctx.add_u64(sb_addr, qh_offset);
                let qh_byte_idx = ctx.div_u32(global_val_idx_full, 4);
                let qh_bit_pos = ctx.rem_u32(global_val_idx_full, 4);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_packed = ctx.ld_global_u8(qh_addr);
                let qh_packed_32 = ctx.cvt_u32_u8(qh_packed);
                let two = ctx.mov_u32_imm(2);
                let qh_shift = ctx.mul_u32_reg(qh_bit_pos, two);
                let qh_shifted = ctx.shr_u32(qh_packed_32, qh_shift);
                let mask_2bit = ctx.mov_u32_imm(0x3);
                let qh = ctx.and_u32(qh_shifted, mask_2bit);

                // Combine: quant = ql + 4 * qh - 32 (6-bit signed: -32 to 31)
                let qh_scaled = ctx.mul_u32_reg(qh, four);
                let ql_qh = ctx.add_u32_reg(ql, qh_scaled);
                // Convert to signed by subtracting 32
                let ql_qh_f32 = ctx.cvt_f32_u32(ql_qh);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);
                let quant_signed = ctx.sub_f32(ql_qh_f32, thirty_two_f32);

                // Dequantize: val = d x scale x quant
                let d_scale = ctx.mul_f32(d, scale_f32);
                let dequant = ctx.mul_f32(d_scale, quant_signed);

                // Load activation and accumulate
                let two_fifty_six = ctx.mov_u32_imm(256);
                let sb_k_offset = ctx.mul_u32_reg(sb_idx, two_fifty_six);
                let sub_k_offset = ctx.mul_u32_reg(sub_block_idx, sixteen_values);
                let k_offset = ctx.add_u32_reg(sb_k_offset, sub_k_offset);
                let k_offset_full = ctx.add_u32_reg(k_offset, lane);

                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset_full);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce (16 threads for Q6_K sub-blocks)
                let shuffled_8 = ctx.shfl_down_f32(prod, 8, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(prod_1, 4, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(prod_2, 2, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(prod_3, 1, 0xFFFF_FFFF);
                let sub_block_sum = ctx.add_f32(prod_3, shuffled_1);

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
