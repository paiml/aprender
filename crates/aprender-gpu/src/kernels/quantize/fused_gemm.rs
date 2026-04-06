//! Fused Q4_K GEMM kernel implementations
//!
//! Contains the PTX builder methods for both simplified and GGML super-block formats.

use super::{QuantizeKernel, Q4K_BLOCK_BYTES, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

impl QuantizeKernel {
    /// Build kernel for simplified Q4_K format (legacy, 32 values/block)
    pub(super) fn build_fused_gemm_simplified(&self) -> PtxKernel {
        // Q4_K GEMM with fused dequantization
        // Each warp processes one block of 32 weights
        let tile_size = self.tile_size;
        let block_size = self.block_size;

        // Shared memory for dequantized tile
        let smem_size = tile_size * tile_size * 4;

        PtxKernel::new("q4k_gemm_fused")
            .param(PtxType::U64, "a_ptr") // Input activations (f32)
            .param(PtxType::U64, "b_quant_ptr") // Quantized weights (Q4_K)
            .param(PtxType::U64, "c_ptr") // Output (f32)
            .param(PtxType::U32, "m") // Output rows
            .param(PtxType::U32, "n") // Output columns
            .param(PtxType::U32, "k") // Inner dimension
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

                // Thread's position within tile
                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                // Global output position
                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check - compute predicates for later store
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp global_row and global_col to valid range [0, m-1] and [0, n-1]
                // This ensures all memory accesses are valid even for out-of-bounds threads.
                // Out-of-bounds threads will compute redundant values but won't store them.
                // This is necessary because all threads in a warp must participate in
                // warp shuffle reductions (shfl.sync with mask 0xFFFFFFFF).
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator (all threads)
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate number of blocks in K dimension
                let block_size_reg = ctx.mov_u32_imm(block_size);
                let num_k_blocks = ctx.div_u32(k_param, block_size);

                // Loop over K blocks
                let k_block = ctx.mov_u32_imm(0);

                ctx.label("k_block_loop");
                let k_done = ctx.setp_ge_u32(k_block, num_k_blocks);
                ctx.branch_if(k_done, "k_block_done");

                // ===== Load and dequantize weight block =====
                // Weight layout: each row has (K/32) Q4_K blocks

                // Calculate block address for weight[clamped_col][k_block]
                // Use clamped_col to ensure valid memory access for all threads
                // Block address = b_quant_ptr + clamped_col * (K/32) * 18 + k_block * 18
                let blocks_per_row = num_k_blocks;
                let block_bytes = ctx.mov_u32_imm(Q4K_BLOCK_BYTES);
                let row_offset = ctx.mul_u32_reg(clamped_col, blocks_per_row);
                let block_offset = ctx.add_u32_reg(row_offset, k_block);
                let byte_offset = ctx.mul_wide_u32_reg(block_offset, block_bytes);
                let block_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load scale from block header (f16 at offset 0)
                // Simplified Q4K format: 2-byte f16 scale + 16 bytes data = 18 bytes
                let scale_addr = block_addr;
                let scale_f16 = ctx.ld_global_f16(scale_addr);
                let scale = ctx.cvt_f32_f16(scale_f16);

                // Load packed 4-bit values
                // Thread i loads values at position (i % 32) within block
                let lane = ctx.rem_u32(tid, block_size);
                let byte_idx = ctx.div_u32(lane, 2);
                let nibble_idx = ctx.rem_u32(lane, 2);

                // Data starts at offset 2 (after 2-byte f16 scale)
                let header_size = ctx.mov_u64_imm(2);
                let data_addr = ctx.add_u64(block_addr, header_size);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let packed_addr = ctx.add_u64(data_addr, byte_idx_64);
                let packed = ctx.ld_global_u8(packed_addr);

                // Extract 4-bit value (no branch - use shift/mask)
                let four = ctx.mov_u32_imm(4);
                let shift = ctx.mul_u32_reg(nibble_idx, four);
                let packed_32 = ctx.cvt_u32_u8(packed);
                let fifteen = ctx.mov_u32_imm(0xF);
                let shifted = ctx.shr_u32(packed_32, shift);
                let quant = ctx.and_u32(shifted, fifteen);

                // Fused dequantization: val = scale * quant
                // (simplified format has no min/bias term)
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let dequant = ctx.mul_f32(scale, quant_f32);

                // ===== Load activation value =====
                // A[clamped_row][k_block * 32 + lane]
                // Use clamped_row to ensure valid memory access for all threads
                let k_offset_base = ctx.mul_u32_reg(k_block, block_size_reg);
                let k_offset = ctx.add_u32_reg(k_offset_base, lane);

                // A address = a_ptr + clamped_row * K + k_offset
                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_offset_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_offset_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                // ===== Accumulate: acc += a_val * dequant =====
                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce for dot product
                let shuffled_16 = ctx.shfl_down_f32(prod, 16, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_16);

                let shuffled_8 = ctx.shfl_down_f32(prod_1, 8, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_8);

                let shuffled_4 = ctx.shfl_down_f32(prod_2, 4, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_4);

                let shuffled_2 = ctx.shfl_down_f32(prod_3, 2, 0xFFFF_FFFF);
                let prod_4 = ctx.add_f32(prod_3, shuffled_2);

                let shuffled_1 = ctx.shfl_down_f32(prod_4, 1, 0xFFFF_FFFF);
                let block_sum = ctx.add_f32(prod_4, shuffled_1);

                // Broadcast sum to all lanes (use shfl_idx, NOT shfl_down with 0!)
                // shfl_down(x, 0) is a no-op - it returns x unchanged
                // shfl_idx(x, 0) broadcasts lane 0's value to all lanes
                let broadcast_sum = ctx.shfl_idx_f32(block_sum, 0, 0xFFFF_FFFF);

                // Add to accumulator IN-PLACE (not shadowing!)
                // Previous: let acc = ctx.add_f32(acc, broadcast_sum); // WRONG: creates new reg
                ctx.add_f32_inplace(acc, broadcast_sum);

                // Increment K block counter IN-PLACE and loop back
                // Previous: let _k_next = ctx.add_u32(k_block, 1); // WRONG: discarded
                // Previous: ctx.branch("k_block_done"); // WRONG: exits loop
                ctx.add_u32_inplace(k_block, 1);
                ctx.branch("k_block_loop"); // CORRECT: loop back

                ctx.label("k_block_done");

                // ===== Store result =====
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                // C address = c_ptr + global_row * N + global_col
                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_offset_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_offset_bytes);

                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build kernel for real GGML Q4_K super-block format (GH-182 rewrite)
    ///
    /// Super-block layout (144 bytes for 256 values):
    /// - Offset 0-1: d (f16 super-block scale)
    /// - Offset 2-3: dmin (f16 super-block min)
    /// - Offset 4-15: scales (12 bytes, split format — see below)
    /// - Offset 16-143: qs (128 bytes, 256 × 4-bit values packed)
    ///
    /// Scale format (12 bytes):
    ///   bytes[0..3]: bits[5:0] = scale SB 0-3, bits[7:6] = high2 of scale SB 4-7
    ///   bytes[4..7]: bits[5:0] = min SB 0-3, bits[7:6] = high2 of min SB 4-7
    ///   bytes[8..11]: bits[3:0] = low4 of scale SB 4-7, bits[7:4] = low4 of min SB 4-7
    ///
    /// qs packing (sub-block pairs share 32 bytes):
    ///   Even SB (0,2,4,6): low nibble of qs[(i/2)*32 + val]
    ///   Odd SB (1,3,5,7): high nibble of qs[(i/2)*32 + val]
    ///
    /// Dequantization: val = d × sc_int × quant - dmin × mn_int
    ///
    /// Design: 1-thread-per-output-element, serial accumulation (no warp reduction).
    /// Grid: (ceil(N/blockDim.x), M) 2D. Each thread computes C[row][col].
    pub(super) fn build_fused_gemm_ggml(&self) -> PtxKernel {
        PtxKernel::new("q4k_gemm_ggml")
            .param(PtxType::U64, "a_ptr") // Input activations [M × K] (f32)
            .param(PtxType::U64, "b_quant_ptr") // Quantized weights [N × (K/256) × 144B]
            .param(PtxType::U64, "c_ptr") // Output [M × N] (f32)
            .param(PtxType::U32, "m") // Output rows
            .param(PtxType::U32, "n") // Output columns
            .param(PtxType::U32, "k") // Inner dimension
            .build(|ctx| {
                // Thread → output element mapping (2D grid)
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_quant_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // row = ctaid_y, col = ctaid_x * blockDim.x + tid
                let row = ctaid_y;
                let col = ctx.mul_u32_reg(ctaid_x, ntid_x);
                let col = ctx.add_u32_reg(col, tid);

                // Bounds check — exit early for OOB threads
                let row_oob = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(row_oob, "exit");
                let col_oob = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(col_oob, "exit");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks per row (K / 256)
                let num_sb = ctx.div_u32(k_param, Q4K_SUPER_BLOCK_SIZE);

                // ===== Outer loop: super-blocks =====
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Super-block address = b_quant_ptr + col * num_sb * 144 + sb_idx * 144
                let row_sb_offset = ctx.mul_u32_reg(col, num_sb);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let sb_byte_offset = ctx.mul_wide_u32(total_sb_offset, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, sb_byte_offset);

                // Load d (f16 at offset 0) and dmin (f16 at offset 2)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let c_2_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, c_2_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ===== Middle loop: 8 sub-blocks =====
                let sub_idx = ctx.mov_u32_imm(0);
                ctx.label("sub_loop");
                let c_8_u32 = ctx.mov_u32_imm(8);
                let sub_done = ctx.setp_ge_u32(sub_idx, c_8_u32);
                ctx.branch_if(sub_done, "sub_loop_done");

                // --- Scale/min extraction (correct GGML split format) ---
                // Scale index = sub_idx directly (blocks 0-7 map sequentially)
                let c_4_u32 = ctx.mov_u32_imm(4);
                let is_high = ctx.setp_ge_u32(sub_idx, c_4_u32);

                // Clamp i_hi = min(sub_idx - 4, 3) for safe high-path loads
                let i_hi_raw = ctx.sub_u32_reg(sub_idx, c_4_u32);
                let c_3_u32 = ctx.mov_u32_imm(3);
                let i_hi = ctx.min_u32(i_hi_raw, c_3_u32);

                let c_4_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, c_4_64);

                // -- Low path (SB 0-3): scale = scales[sub_idx] & 0x3F --
                let sub_idx_64 = ctx.cvt_u64_u32(sub_idx);
                let lo_sc_addr = ctx.add_u64(scales_base, sub_idx_64);
                let lo_sc_byte = ctx.ld_global_u8(lo_sc_addr);
                let lo_sc_32 = ctx.cvt_u32_u8(lo_sc_byte);
                let mask_6 = ctx.mov_u32_imm(0x3F);
                let lo_scale = ctx.and_u32(lo_sc_32, mask_6);

                // min = scales[4 + sub_idx] & 0x3F
                let lo_mn_base = ctx.add_u64(scales_base, c_4_64);
                let lo_mn_addr = ctx.add_u64(lo_mn_base, sub_idx_64);
                let lo_mn_byte = ctx.ld_global_u8(lo_mn_addr);
                let lo_mn_32 = ctx.cvt_u32_u8(lo_mn_byte);
                let lo_min = ctx.and_u32(lo_mn_32, mask_6);

                // -- High path (SB 4-7): split extraction --
                let i_hi_64 = ctx.cvt_u64_u32(i_hi);
                // combo = scales[8 + i_hi]
                let c_8_64 = ctx.mov_u64_imm(8);
                let combo_base = ctx.add_u64(scales_base, c_8_64);
                let combo_addr = ctx.add_u64(combo_base, i_hi_64);
                let combo_byte = ctx.ld_global_u8(combo_addr);
                let combo_32 = ctx.cvt_u32_u8(combo_byte);

                // sc_low4 = combo & 0x0F
                let mask_4 = ctx.mov_u32_imm(0x0F);
                let sc_low4 = ctx.and_u32(combo_32, mask_4);

                // sc_high2 = (scales[i_hi] >> 6) & 0x03
                let hi_sc_addr = ctx.add_u64(scales_base, i_hi_64);
                let hi_sc_byte = ctx.ld_global_u8(hi_sc_addr);
                let hi_sc_32 = ctx.cvt_u32_u8(hi_sc_byte);
                let c_6_u32 = ctx.mov_u32_imm(6);
                let sc_shifted = ctx.shr_u32(hi_sc_32, c_6_u32);
                let mask_2 = ctx.mov_u32_imm(0x03);
                let sc_high2 = ctx.and_u32(sc_shifted, mask_2);
                let sc_high_pos = ctx.shl_u32(sc_high2, c_4_u32);
                let hi_scale = ctx.or_u32(sc_low4, sc_high_pos);

                // mn_low4 = (combo >> 4) & 0x0F
                let mn_shifted = ctx.shr_u32(combo_32, c_4_u32);
                let mn_low4 = ctx.and_u32(mn_shifted, mask_4);

                // mn_high2 = (scales[4 + i_hi] >> 6) & 0x03
                let hi_mn_base = ctx.add_u64(scales_base, c_4_64);
                let hi_mn_addr = ctx.add_u64(hi_mn_base, i_hi_64);
                let hi_mn_byte = ctx.ld_global_u8(hi_mn_addr);
                let hi_mn_32 = ctx.cvt_u32_u8(hi_mn_byte);
                let mn_hi_shifted = ctx.shr_u32(hi_mn_32, c_6_u32);
                let mn_high2 = ctx.and_u32(mn_hi_shifted, mask_2);
                let mn_high_pos = ctx.shl_u32(mn_high2, c_4_u32);
                let hi_min = ctx.or_u32(mn_low4, mn_high_pos);

                // Select: low path for SB 0-3, high path for SB 4-7
                let scale_int = ctx.selp_u32(is_high, hi_scale, lo_scale);
                let min_int = ctx.selp_u32(is_high, hi_min, lo_min);

                // Convert to float (integer scales, NO normalization by 63)
                let scale_f32 = ctx.cvt_f32_u32(scale_int);
                let min_f32 = ctx.cvt_f32_u32(min_int);

                // Pre-compute d*scale and dmin*min for this sub-block
                let d_scale = ctx.mul_f32(d, scale_f32);
                let dmin_min = ctx.mul_f32(dmin, min_f32);

                // GH-182: qs byte + K offset mapping
                // GGML Q4K: each 64-element group uses 32 qs bytes (low+high nibbles)
                //   pair = sub_idx / 2 → which 32-byte qs chunk
                //   nibble_sel = sub_idx % 2 → low (0) or high (4) nibble shift
                // qs byte: qs_base + pair*32 + val_idx
                // K offset: sb*256 + pair*64 + nibble*32 + val_idx
                let pair = ctx.div_u32(sub_idx, 2);
                let nibble_sel = ctx.rem_u32(sub_idx, 2);
                let pair_byte_base = ctx.mul_u32(pair, 32); // for qs addressing
                let nibble_shift = ctx.mul_u32(nibble_sel, 4); // 0 or 4

                // K offset: sb_idx*256 + pair*64 + nibble*32
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let pair_k_base = ctx.mul_u32(pair, 64);
                let pair_k_offset = ctx.add_u32_reg(sb_k_base, pair_k_base);
                let nibble_k_offset = ctx.mul_u32(nibble_sel, 32);
                let base_k = ctx.add_u32_reg(pair_k_offset, nibble_k_offset);

                // ===== Inner loop: 32 values per sub-block =====
                let val_idx = ctx.mov_u32_imm(0);
                let c_32_u32 = ctx.mov_u32_imm(32);
                ctx.label("val_loop");
                let val_done = ctx.setp_ge_u32(val_idx, c_32_u32);
                ctx.branch_if(val_done, "val_loop_done");

                // Load qs byte: offset 16 + pair*32 + val_idx
                let qs_offset = ctx.add_u32_reg(pair_byte_base, val_idx);
                let qs_offset_16 = ctx.add_u32(qs_offset, 16);
                let qs_offset_64 = ctx.cvt_u64_u32(qs_offset_16);
                let qs_addr = ctx.add_u64(sb_addr, qs_offset_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                // Extract 4-bit value using nibble_shift (0=low, 4=high)
                let shifted = ctx.shr_u32(packed_32, nibble_shift);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let quant = ctx.and_u32(shifted, mask_4bit);
                let quant_f32 = ctx.cvt_f32_u32(quant);

                // Dequantize: val = d*scale*quant - dmin*min
                let weighted = ctx.mul_f32(d_scale, quant_f32);
                let dequant = ctx.sub_f32(weighted, dmin_min);

                // Load activation A[row][k_offset]
                let k_offset = ctx.add_u32_reg(base_k, val_idx);
                let a_row_base = ctx.mul_wide_u32_reg(row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset);
                let a_elem_offset = ctx.add_u64(a_row_base, k_offset_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);
                let a_val = ctx.ld_global_f32(a_addr);

                // FMA: acc += a_val * dequant
                ctx.fma_f32_inplace(acc, a_val, dequant);

                ctx.add_u32_inplace(val_idx, 1);
                ctx.branch("val_loop");
                ctx.label("val_loop_done");

                // Next sub-block
                ctx.add_u32_inplace(sub_idx, 1);
                ctx.branch("sub_loop");
                ctx.label("sub_loop_done");

                // Next super-block
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");
                ctx.label("sb_loop_done");

                // Store result C[row][col]
                let c_row_offset = ctx.mul_wide_u32_reg(row, n_param);
                let col_64 = ctx.cvt_u64_u32(col);
                let c_elem_offset = ctx.add_u64(c_row_offset, col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build tiled Q4_K GGML GEMM kernel (GH-182: weight reuse across M rows)
    ///
    /// Each thread computes `tile_m` output values for one column, loading
    /// weight super-blocks once and reusing across all tile rows. This reduces
    /// weight memory traffic by `tile_m`×.
    ///
    /// Grid: (ceil(N/blockDim.x), ceil(M/tile_m))
    /// Block: (blockDim.x,) — e.g., 128 threads
    ///
    /// For prefill (M=10-500), this closes the 3.56× bandwidth gap vs HGEMM
    /// by reading Q4K weights directly (0.5625 B/elem vs 2 B/elem for FP16).
    pub(super) fn build_fused_gemm_ggml_tiled(&self) -> PtxKernel {
        let tile_m = self.tile_m;

        PtxKernel::new("q4k_gemm_ggml_tiled")
            .param(PtxType::U64, "a_ptr") // Input activations [M × K] (f32)
            .param(PtxType::U64, "b_quant_ptr") // Quantized weights [N × (K/256) × 144B]
            .param(PtxType::U64, "c_ptr") // Output [M × N] (f32)
            .param(PtxType::U32, "m") // Output rows
            .param(PtxType::U32, "n") // Output columns
            .param(PtxType::U32, "k") // Inner dimension
            .build(|ctx| {
                // Thread → column mapping
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_quant_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // col = ctaid_x * blockDim.x + tid
                let col = ctx.mul_u32_reg(ctaid_x, ntid_x);
                let col = ctx.add_u32_reg(col, tid);

                // Bounds check column
                let col_oob = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(col_oob, "exit");

                // row_base = ctaid_y * tile_m
                let row_base = ctx.mul_u32(ctaid_y, tile_m);

                // Initialize tile_m accumulators and pre-compute row addresses
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);

                let mut accs = Vec::with_capacity(tile_m as usize);
                let mut a_row_addrs = Vec::with_capacity(tile_m as usize);

                for m_idx in 0..tile_m {
                    accs.push(ctx.mov_f32_imm(0.0));

                    // Compute row index, clamped to [0, M-1] for safe loads
                    let row = if m_idx == 0 { row_base } else { ctx.add_u32(row_base, m_idx) };
                    let clamped_row = ctx.min_u32(row, m_minus_1);

                    // Pre-compute A[row] base address = a_ptr + row * K * 4
                    let row_k_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                    let row_bytes = ctx.mul_u64(row_k_offset, 4);
                    a_row_addrs.push(ctx.add_u64(a_ptr, row_bytes));
                }

                // Number of super-blocks per row (K / 256)
                let num_sb = ctx.div_u32(k_param, Q4K_SUPER_BLOCK_SIZE);

                // ===== Outer loop: super-blocks =====
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Super-block address = b_quant_ptr + col * num_sb * 144 + sb_idx * 144
                let row_sb_offset = ctx.mul_u32_reg(col, num_sb);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let sb_byte_offset = ctx.mul_wide_u32(total_sb_offset, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, sb_byte_offset);

                // Load d (f16 at offset 0) and dmin (f16 at offset 2)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let c_2_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, c_2_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ===== Middle loop: 8 sub-blocks =====
                let sub_idx = ctx.mov_u32_imm(0);
                ctx.label("sub_loop");
                let c_8_u32 = ctx.mov_u32_imm(8);
                let sub_done = ctx.setp_ge_u32(sub_idx, c_8_u32);
                ctx.branch_if(sub_done, "sub_loop_done");

                // --- Scale/min extraction ---
                // Scale index = sub_idx (0-7 maps to GGML blocks 0-7 sequentially).
                // GGML Q4K: each pair of qs bytes shares low/high nibbles for two
                // consecutive 32-element blocks. sub_idx directly indexes scales.
                let c_4_u32 = ctx.mov_u32_imm(4);
                let is_high = ctx.setp_ge_u32(sub_idx, c_4_u32);

                let i_hi_raw = ctx.sub_u32_reg(sub_idx, c_4_u32);
                let c_3_u32 = ctx.mov_u32_imm(3);
                let i_hi = ctx.min_u32(i_hi_raw, c_3_u32);

                let c_4_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, c_4_64);

                // Low path (SB 0-3): scale = scales[sub_idx] & 0x3F
                let sub_idx_64 = ctx.cvt_u64_u32(sub_idx);
                let lo_sc_addr = ctx.add_u64(scales_base, sub_idx_64);
                let lo_sc_byte = ctx.ld_global_u8(lo_sc_addr);
                let lo_sc_32 = ctx.cvt_u32_u8(lo_sc_byte);
                let mask_6 = ctx.mov_u32_imm(0x3F);
                let lo_scale = ctx.and_u32(lo_sc_32, mask_6);

                let lo_mn_base = ctx.add_u64(scales_base, c_4_64);
                let lo_mn_addr = ctx.add_u64(lo_mn_base, sub_idx_64);
                let lo_mn_byte = ctx.ld_global_u8(lo_mn_addr);
                let lo_mn_32 = ctx.cvt_u32_u8(lo_mn_byte);
                let lo_min = ctx.and_u32(lo_mn_32, mask_6);

                // High path (SB 4-7): split extraction
                let i_hi_64 = ctx.cvt_u64_u32(i_hi);
                let c_8_64 = ctx.mov_u64_imm(8);
                let combo_base = ctx.add_u64(scales_base, c_8_64);
                let combo_addr = ctx.add_u64(combo_base, i_hi_64);
                let combo_byte = ctx.ld_global_u8(combo_addr);
                let combo_32 = ctx.cvt_u32_u8(combo_byte);

                let mask_4 = ctx.mov_u32_imm(0x0F);
                let sc_low4 = ctx.and_u32(combo_32, mask_4);

                let hi_sc_addr = ctx.add_u64(scales_base, i_hi_64);
                let hi_sc_byte = ctx.ld_global_u8(hi_sc_addr);
                let hi_sc_32 = ctx.cvt_u32_u8(hi_sc_byte);
                let c_6_u32 = ctx.mov_u32_imm(6);
                let sc_shifted = ctx.shr_u32(hi_sc_32, c_6_u32);
                let mask_2 = ctx.mov_u32_imm(0x03);
                let sc_high2 = ctx.and_u32(sc_shifted, mask_2);
                let sc_high_pos = ctx.shl_u32(sc_high2, c_4_u32);
                let hi_scale = ctx.or_u32(sc_low4, sc_high_pos);

                let mn_shifted = ctx.shr_u32(combo_32, c_4_u32);
                let mn_low4 = ctx.and_u32(mn_shifted, mask_4);

                let hi_mn_base = ctx.add_u64(scales_base, c_4_64);
                let hi_mn_addr = ctx.add_u64(hi_mn_base, i_hi_64);
                let hi_mn_byte = ctx.ld_global_u8(hi_mn_addr);
                let hi_mn_32 = ctx.cvt_u32_u8(hi_mn_byte);
                let mn_hi_shifted = ctx.shr_u32(hi_mn_32, c_6_u32);
                let mn_high2 = ctx.and_u32(mn_hi_shifted, mask_2);
                let mn_high_pos = ctx.shl_u32(mn_high2, c_4_u32);
                let hi_min = ctx.or_u32(mn_low4, mn_high_pos);

                // Select: low path for SB 0-3, high path for SB 4-7
                let scale_int = ctx.selp_u32(is_high, hi_scale, lo_scale);
                let min_int = ctx.selp_u32(is_high, hi_min, lo_min);

                let scale_f32 = ctx.cvt_f32_u32(scale_int);
                let min_f32 = ctx.cvt_f32_u32(min_int);

                let d_scale = ctx.mul_f32(d, scale_f32);
                let dmin_min = ctx.mul_f32(dmin, min_f32);

                // GH-182: qs byte + K offset mapping
                // GGML Q4K: each 64-element group uses 32 qs bytes (low+high nibbles)
                //   pair = sub_idx / 2 → which 32-byte qs chunk
                //   nibble_sel = sub_idx % 2 → low (0) or high (4) nibble shift
                // qs byte: qs_base + pair*32 + val_idx
                // K offset: sb*256 + pair*64 + nibble*32 + val_idx
                let pair = ctx.div_u32(sub_idx, 2);
                let nibble_sel = ctx.rem_u32(sub_idx, 2);
                let pair_byte_base = ctx.mul_u32(pair, 32); // for qs addressing
                let nibble_shift = ctx.mul_u32(nibble_sel, 4); // 0 or 4 bit shift

                // K offset base: sb_idx*256 + pair*64 + nibble*32
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let pair_k_base = ctx.mul_u32(pair, 64);
                let pair_k_offset = ctx.add_u32_reg(sb_k_base, pair_k_base);
                let nibble_k_offset = ctx.mul_u32(nibble_sel, 32);
                let base_k = ctx.add_u32_reg(pair_k_offset, nibble_k_offset);

                // ===== Inner loop: 32 values per sub-block =====
                let val_idx = ctx.mov_u32_imm(0);
                let c_32_u32 = ctx.mov_u32_imm(32);
                ctx.label("val_loop");
                let val_done = ctx.setp_ge_u32(val_idx, c_32_u32);
                ctx.branch_if(val_done, "val_loop_done");

                // Load qs byte and extract nibble (same as serial)
                let qs_offset = ctx.add_u32_reg(pair_byte_base, val_idx);
                let qs_offset_16 = ctx.add_u32(qs_offset, 16);
                let qs_offset_64 = ctx.cvt_u64_u32(qs_offset_16);
                let qs_addr = ctx.add_u64(sb_addr, qs_offset_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);

                let shifted = ctx.shr_u32(packed_32, nibble_shift);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let quant = ctx.and_u32(shifted, mask_4bit);
                let quant_f32 = ctx.cvt_f32_u32(quant);

                // Dequantize: val = d*scale*quant - dmin*min
                let weighted = ctx.mul_f32(d_scale, quant_f32);
                let dequant = ctx.sub_f32(weighted, dmin_min);

                // K offset for this value
                let k_offset = ctx.add_u32_reg(base_k, val_idx);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset);
                let k_bytes = ctx.mul_u64(k_offset_64, 4);

                // TILED: load activation from each of tile_m rows and FMA
                for m_idx in 0..tile_m as usize {
                    let a_addr = ctx.add_u64(a_row_addrs[m_idx], k_bytes);
                    let a_val = ctx.ld_global_f32(a_addr);
                    ctx.fma_f32_inplace(accs[m_idx], a_val, dequant);
                }

                ctx.add_u32_inplace(val_idx, 1);
                ctx.branch("val_loop");
                ctx.label("val_loop_done");

                ctx.add_u32_inplace(sub_idx, 1);
                ctx.branch("sub_loop");
                ctx.label("sub_loop_done");

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");
                ctx.label("sb_loop_done");

                // Store tile_m results (rows are consecutive, so once OOB all subsequent are too)
                for m_idx in 0..tile_m {
                    let row = if m_idx == 0 { row_base } else { ctx.add_u32(row_base, m_idx) };
                    let row_oob = ctx.setp_ge_u32(row, m_param);
                    ctx.branch_if(row_oob, "exit");

                    let c_row_offset = ctx.mul_wide_u32_reg(row, n_param);
                    let col_64 = ctx.cvt_u64_u32(col);
                    let c_elem_offset = ctx.add_u64(c_row_offset, col_64);
                    let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                    let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);
                    ctx.st_global_f32(c_addr, accs[m_idx as usize]);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}
