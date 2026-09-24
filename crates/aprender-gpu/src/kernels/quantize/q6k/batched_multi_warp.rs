//! #4234: [`MultiWarpQ6KGemvKernel`] for `m` activation vectors at once.
//!
//! `y[r][row] = W[row] · x[r]` for `r < m`, with `x` `[m][k]` and `y` `[m][n]`, both
//! row-major and contiguous. Every thread dequantizes its eight weights of a
//! super-block ONCE and applies them to all `m` activation vectors, so a weight row
//! is read from DRAM once per launch instead of once per vector.
//!
//! **Bitwise the single-vector kernel, per vector.** The dequantization below is the
//! M=1 kernel's, copied verbatim. For each `r` the arithmetic is the M=1 kernel's:
//! the same eight FMAs into a fresh partial in the same order, the same
//! `acc += partial`, the same shuffle-down tree and the same warp-ordered
//! shared-memory sum. Only the weight loads are shared, so a sequence's output does
//! not depend on which other sequences share its launch.
//!
//! [`MultiWarpQ6KGemvKernel`]: super::MultiWarpQ6KGemvKernel

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// See the module docs.
pub struct BatchedMwvQ6KGemvKernel {
    /// K dimension (input dimension; need not be a multiple of 256 — bounds-checked)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Warps per block — must equal the single-vector kernel's for bitwise parity.
    pub num_warps: u32,
    /// Activation vectors per launch, `1..=`[`Self::MAX_M`].
    pub m: u32,
}

impl BatchedMwvQ6KGemvKernel {
    /// The widest batch one launch carries; wider batches are tiled by the caller.
    pub const MAX_M: u32 = 8;

    /// A kernel for `m` vectors with `num_warps` warps per block.
    #[must_use]
    pub fn new(k: u32, n: u32, num_warps: u32, m: u32) -> Self {
        Self { k, n, num_warps, m }
    }
}

impl Kernel for BatchedMwvQ6KGemvKernel {
    fn name(&self) -> &str {
        "batched_mwv_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let m = self.m.clamp(1, Self::MAX_M);
        let smem_size = (num_warps * m * 4) as usize;

        PtxKernel::new("batched_mwv_q6k_gemv")
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

                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "bmwv_q6k_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // One accumulator and one activation base per vector.
                let accs: Vec<_> = (0..m).map(|_| ctx.mov_f32_imm(0.0)).collect();
                let x_bases: Vec<_> = (0..m)
                    .map(|r| {
                        let off = ctx.mul_wide_u32(k_dim, r * 4);
                        ctx.add_u64(x_ptr, off)
                    })
                    .collect();

                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                // The dequantized weight of each of the eight offsets, in order.
                let mut slots = Vec::with_capacity(8);
                ctx.label("bmwv_q6k_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "bmwv_q6k_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // ================================================================
                // Scale loading: lanes 0-15 each load one scale byte,
                // broadcast to all lanes via warp shuffle (PAR-066 pattern).
                // Q6K scales are i8 at bytes 192-207 of the super-block.
                // ================================================================
                let scales_base_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_base_offset);

                let lane_mod_16 = ctx.rem_u32(lane_id, 16);
                let lane_offset = ctx.cvt_u64_u32(lane_mod_16);
                let scale_addr = ctx.add_u64(scales_base, lane_offset);

                let my_scale_byte = ctx.mov_u32_imm(0);
                let sixteen_const = ctx.mov_u32_imm(16);
                let is_low_lane = ctx.setp_lt_u32(lane_id, sixteen_const);
                ctx.branch_if_not(is_low_lane, "bmwv_q6k_skip_scale_load");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("bmwv_q6k_skip_scale_load");

                // Broadcast all 16 scales via warp shuffle
                let mut scale_regs = Vec::with_capacity(16);
                for i in 0..16u32 {
                    scale_regs.push(ctx.shfl_idx_u32(my_scale_byte, i, 0xFFFF_FFFF));
                }

                // Convert i8 scales to signed f32: if >= 128, subtract 256
                let seven = ctx.mov_u32_imm(7);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);

                let mut scale_f32s = Vec::with_capacity(16);
                for &sr in &scale_regs {
                    let sign_bit = ctx.shr_u32(sr, seven);
                    let raw_f32 = ctx.cvt_f32_u32(sr);
                    let sign_f32 = ctx.cvt_f32_u32(sign_bit);
                    let correction = ctx.mul_f32(sign_f32, twofiftysix_f32);
                    let signed_f32 = ctx.sub_f32(raw_f32, correction);
                    scale_f32s.push(signed_f32);
                }

                // Precompute d * scale for all 16 sub-block scales
                let mut ds = Vec::with_capacity(16);
                for &sf in &scale_f32s {
                    ds.push(ctx.mul_f32(d, sf));
                }

                // ================================================================
                // Dequantize 256 values: 8 offsets × 32 lanes = 256 values per sb
                //
                // Contract: identical dequant formula to Q6KGemvKernel
                //   quant = (ql_nibble | (qh_2bits << 4)) - 32
                //   scale_idx = 8 * n_idx + is + 2 * group
                //   value = d * scale[scale_idx] * quant
                // ================================================================
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);

                // offset_params: (offset, n_idx, group, ds_even_idx, ds_odd_idx)
                // scale_idx = 8 * n_idx + is + 2 * group
                // is = 0 for lanes 0-15, 1 for lanes 16-31
                // ds_even = ds[8*n + 2*g], ds_odd = ds[8*n + 2*g + 1]
                let offset_params: [(u32, u32, u32, usize, usize); 8] = [
                    (0, 0, 0, 0, 1),     // n=0, g=0: scale_idx = 0 or 1
                    (32, 0, 1, 2, 3),    // n=0, g=1: scale_idx = 2 or 3
                    (64, 0, 2, 4, 5),    // n=0, g=2: scale_idx = 4 or 5
                    (96, 0, 3, 6, 7),    // n=0, g=3: scale_idx = 6 or 7
                    (128, 1, 0, 8, 9),   // n=1, g=0: scale_idx = 8 or 9
                    (160, 1, 1, 10, 11), // n=1, g=1: scale_idx = 10 or 11
                    (192, 1, 2, 12, 13), // n=1, g=2: scale_idx = 12 or 13
                    (224, 1, 3, 14, 15), // n=1, g=3: scale_idx = 14 or 15
                ];

                // Precompute lane_is: 0 for lanes 0-15, 1 for lanes 16-31
                let lane_is = ctx.div_u32(lane_id, 16);
                let lane_is_f32 = ctx.cvt_f32_u32(lane_is);

                for &(offset, n_idx_val, group_val, ds_even, ds_odd) in &offset_params {
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Select ds based on lane position (is=0 → ds_even, is=1 → ds_odd)
                    let ds_diff = ctx.sub_f32(ds[ds_odd], ds[ds_even]);
                    let ds_selected = ctx.fma_f32(lane_is_f32, ds_diff, ds[ds_even]);

                    // l = lane_id (position within 32-value group)
                    let l = lane_id;

                    let n_idx = ctx.mov_u32_imm(n_idx_val);
                    let group = ctx.mov_u32_imm(group_val);

                    // ql_byte_offset = 64 * n_idx + l + (32 * group_is_odd)
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one_32 = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one_32);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low if group < 2, high if group >= 2
                    let group_div_2 = ctx.shr_u32(group, one_32);
                    let four = ctx.mov_u32_imm(4);
                    let nibble_shift = ctx.mul_u32_reg(group_div_2, four);
                    let ql_shifted = ctx.shr_u32(ql_byte_32, nibble_shift);
                    let mask_0xf = ctx.mov_u32_imm(0xF);
                    let ql_nibble = ctx.and_u32(ql_shifted, mask_0xf);

                    // qh_byte_offset = 32 * n_idx + l
                    let n_idx_x32 = ctx.mul_u32_reg(n_idx, thirty_two);
                    let qh_byte_offset = ctx.add_u32_reg(n_idx_x32, l);

                    // Load qh byte (offset 128 + qh_byte_offset)
                    let qh_base_offset = ctx.mov_u64_imm(128);
                    let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                    let qh_byte_offset_64 = ctx.cvt_u64_u32(qh_byte_offset);
                    let qh_addr = ctx.add_u64(qh_base, qh_byte_offset_64);
                    let qh_byte = ctx.ld_global_u8(qh_addr);
                    let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);

                    // qh_bit_shift = 2 * group
                    let two = ctx.mov_u32_imm(2);
                    let qh_shift = ctx.mul_u32_reg(group, two);
                    let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                    let mask_0x3 = ctx.mov_u32_imm(0x3);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_0x3);

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = ds_selected * quant
                    let dequant = ctx.mul_f32(ds_selected, quant_signed);
                    slots.push((val_idx, dequant));
                }

                // Activation offsets and bounds predicates, shared by every vector.
                let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                let slots: Vec<_> = slots
                    .into_iter()
                    .map(|(val_idx, dequant)| {
                        let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                        let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                        let x_bytes = ctx.mul_u64(x_idx_64, 4);
                        let in_bounds = ctx.setp_lt_u32(x_idx, k_dim);
                        (x_bytes, in_bounds, dequant)
                    })
                    .collect();

                for (acc, &x_base) in accs.iter().zip(&x_bases) {
                    let pt = ctx.mov_f32_imm(0.0);
                    for &(x_bytes, in_bounds, dequant) in &slots {
                        let xa = ctx.add_u64(x_base, x_bytes);
                        let xv = ctx.ld_global_f32_predicated(xa, in_bounds, 0.0);
                        ctx.fma_f32_inplace(pt, xv, dequant);
                    }
                    ctx.add_f32_inplace(*acc, pt);
                }

                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("bmwv_q6k_sb_loop");

                ctx.label("bmwv_q6k_sb_end");

                // Phase 1: intra-warp reduction, per vector.
                for &acc in &accs {
                    for off in [16, 8, 4, 2, 1] {
                        let t = ctx.shfl_down_f32(acc, off, 0xFFFF_FFFF);
                        ctx.add_f32_inplace(acc, t);
                    }
                }

                // Phase 2: cross-warp reduction via shared memory; vector r's
                // warp w partial at [(r * num_warps + w) * 4].
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "bmwv_q6k_skip_sm");
                for (r, &acc) in (0u32..).zip(&accs) {
                    let slot = ctx.add_u32(warp_id, r * num_warps);
                    let f4 = ctx.mov_u32_imm(4);
                    let wo = ctx.mul_u32_reg(slot, f4);
                    let sa = ctx.cvt_u64_u32(wo);
                    ctx.st_shared_f32(sa, acc);
                }
                ctx.label("bmwv_q6k_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "bmwv_q6k_exit");

                let yo_row = ctx.mul_wide_u32(block_id, 4);
                for r in 0..m {
                    let fs = ctx.mov_f32_imm(0.0);
                    for w in 0..num_warps {
                        let wo = ctx.mov_u64_imm(u64::from((r * num_warps + w) * 4));
                        let pv = ctx.ld_shared_f32(wo);
                        ctx.add_f32_inplace(fs, pv);
                    }
                    let yo_vec = ctx.mul_wide_u32(n_dim, r * 4);
                    let yo = ctx.add_u64(yo_vec, yo_row);
                    let ya = ctx.add_u64(y_ptr, yo);
                    ctx.st_global_f32(ya, fs);
                }

                ctx.label("bmwv_q6k_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_mwv_q6k_emits_one_accumulator_set_per_vector() {
        let one = BatchedMwvQ6KGemvKernel::new(1024, 64, 4, 1).emit_ptx();
        let four = BatchedMwvQ6KGemvKernel::new(1024, 64, 4, 4).emit_ptx();
        assert!(one.contains(".entry batched_mwv_q6k_gemv"), "{one}");
        // The weight loads are shared: widening the batch adds activation loads, never
        // weight (u8) loads.
        let u8_loads = |p: &str| p.matches("ld.global.u8").count();
        assert_eq!(
            u8_loads(&one),
            u8_loads(&four),
            "weights must be loaded once"
        );
        let f32_loads = |p: &str| p.matches("ld.global.f32").count();
        assert_eq!(
            f32_loads(&four),
            4 * f32_loads(&one),
            "one activation load per vector"
        );
    }
}
