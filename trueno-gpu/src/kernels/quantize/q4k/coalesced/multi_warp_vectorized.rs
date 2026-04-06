use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Multi-warp Vectorized Q4_K GEMV kernel (PAR-082-V2)
///
/// Combines the best of both approaches:
/// - **VectorizedQ4K**: u32 coalesced loads (128 bytes/warp/transaction)
/// - **WideQ4K**: Multi-warp parallelism for memory latency hiding
///
/// 4 warps (128 threads) per block, matching llama.cpp `mul_mat_vec_q`.
/// Each warp strides across super-blocks, uses u32 coalesced loads,
/// then cross-warp reduction via shared memory.
///
/// # Performance Target
///
/// 50% BW utilization → ~128 tok/s on RTX 4090 for 7B Q4K (Ollama parity)
pub struct MultiWarpVectorizedQ4KGemvKernel {
    /// K dimension (input dimension, need not be multiple of 256 — bounds-checked)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 4, matching llama.cpp)
    pub num_warps: u32,
}

impl MultiWarpVectorizedQ4KGemvKernel {
    /// Create with 4 warps (128 threads), matching llama.cpp
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 4 }
    }
}

impl Kernel for MultiWarpVectorizedQ4KGemvKernel {
    fn name(&self) -> &str {
        "mwv_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("mwv_q4k_gemv")
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
                ctx.branch_if(oob, "mwv_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each warp starts at warp_id, strides by num_warps
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("mwv_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "mwv_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d and dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Scale loading: lane 0 loads 3 x u32, broadcasts
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let sc03r = ctx.mov_u32_imm(0);
                let sc47r = ctx.mov_u32_imm(0);
                let sc811r = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "mwv_skip_sc");
                ctx.ld_global_u32_into(sc03r, scales_base);
                let f64b = ctx.mov_u64_imm(4);
                let s4a = ctx.add_u64(scales_base, f64b);
                ctx.ld_global_u32_into(sc47r, s4a);
                let e64 = ctx.mov_u64_imm(8);
                let s8a = ctx.add_u64(scales_base, e64);
                ctx.ld_global_u32_into(sc811r, s8a);
                ctx.label("mwv_skip_sc");

                let sc03 = ctx.shfl_idx_u32(sc03r, 0, 0xFFFF_FFFF);
                let sc47 = ctx.shfl_idx_u32(sc47r, 0, 0xFFFF_FFFF);
                let sc811 = ctx.shfl_idx_u32(sc811r, 0, 0xFFFF_FFFF);

                let m8 = ctx.mov_u32_imm(0xFF);
                let sh8 = ctx.mov_u32_imm(8);
                let sh16 = ctx.mov_u32_imm(16);
                let sh24 = ctx.mov_u32_imm(24);

                let s0 = ctx.and_u32(sc03, m8);
                let t = ctx.shr_u32(sc03, sh8);
                let s1 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc03, sh16);
                let s2 = ctx.and_u32(t, m8);
                let s3 = ctx.shr_u32(sc03, sh24);

                let s4 = ctx.and_u32(sc47, m8);
                let t = ctx.shr_u32(sc47, sh8);
                let s5 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc47, sh16);
                let s6 = ctx.and_u32(t, m8);
                let s7 = ctx.shr_u32(sc47, sh24);

                let s8 = ctx.and_u32(sc811, m8);
                let t = ctx.shr_u32(sc811, sh8);
                let s9 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc811, sh16);
                let s10 = ctx.and_u32(t, m8);
                let s11 = ctx.shr_u32(sc811, sh24);

                let m6 = ctx.mov_u32_imm(0x3F);
                let m4 = ctx.mov_u32_imm(0x0F);
                let four_c = ctx.mov_u32_imm(4);
                let six_c = ctx.mov_u32_imm(6);

                // Blocks 0-3
                let sc0 = ctx.and_u32(s0, m6);
                let mn0 = ctx.and_u32(s4, m6);
                let sc1 = ctx.and_u32(s1, m6);
                let mn1 = ctx.and_u32(s5, m6);
                let sc2 = ctx.and_u32(s2, m6);
                let mn2 = ctx.and_u32(s6, m6);
                let sc3 = ctx.and_u32(s3, m6);
                let mn3 = ctx.and_u32(s7, m6);

                // Blocks 4-7
                let t = ctx.and_u32(s8, m4);
                let u = ctx.shr_u32(s0, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc4 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s8, four_c);
                let u = ctx.shr_u32(s4, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn4 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s9, m4);
                let u = ctx.shr_u32(s1, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc5 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s9, four_c);
                let u = ctx.shr_u32(s5, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn5 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s10, m4);
                let u = ctx.shr_u32(s2, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc6 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s10, four_c);
                let u = ctx.shr_u32(s6, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn6 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s11, m4);
                let u = ctx.shr_u32(s3, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc7 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s11, four_c);
                let u = ctx.shr_u32(s7, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn7 = ctx.or_u32(t, u);

                // d*scale and dmin*min
                let f0 = ctx.cvt_f32_u32(sc0);
                let g0 = ctx.cvt_f32_u32(mn0);
                let ds0 = ctx.mul_f32(d, f0);
                let dm0 = ctx.mul_f32(dmin, g0);
                let f1 = ctx.cvt_f32_u32(sc1);
                let g1 = ctx.cvt_f32_u32(mn1);
                let ds1 = ctx.mul_f32(d, f1);
                let dm1 = ctx.mul_f32(dmin, g1);
                let f2 = ctx.cvt_f32_u32(sc2);
                let g2 = ctx.cvt_f32_u32(mn2);
                let ds2 = ctx.mul_f32(d, f2);
                let dm2 = ctx.mul_f32(dmin, g2);
                let f3 = ctx.cvt_f32_u32(sc3);
                let g3 = ctx.cvt_f32_u32(mn3);
                let ds3 = ctx.mul_f32(d, f3);
                let dm3 = ctx.mul_f32(dmin, g3);
                let f4 = ctx.cvt_f32_u32(sc4);
                let g4 = ctx.cvt_f32_u32(mn4);
                let ds4 = ctx.mul_f32(d, f4);
                let dm4 = ctx.mul_f32(dmin, g4);
                let f5 = ctx.cvt_f32_u32(sc5);
                let g5 = ctx.cvt_f32_u32(mn5);
                let ds5 = ctx.mul_f32(d, f5);
                let dm5 = ctx.mul_f32(dmin, g5);
                let f6 = ctx.cvt_f32_u32(sc6);
                let g6 = ctx.cvt_f32_u32(mn6);
                let ds6 = ctx.mul_f32(d, f6);
                let dm6 = ctx.mul_f32(dmin, g6);
                let f7 = ctx.cvt_f32_u32(sc7);
                let g7 = ctx.cvt_f32_u32(mn7);
                let ds7 = ctx.mul_f32(d, f7);
                let dm7 = ctx.mul_f32(dmin, g7);

                // COALESCED u32 LOAD: 128 bytes per warp transaction
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);
                let four = ctx.mov_u32_imm(4);
                let tbo = ctx.mul_u32_reg(lane_id, four);
                let tbo64 = ctx.cvt_u64_u32(tbo);
                let qa = ctx.add_u64(qs_base, tbo64);
                let packed = ctx.ld_global_u32(qa);

                // Unpack 8 nibbles
                let nib0 = ctx.and_u32(packed, m4);
                let sh4 = ctx.mov_u32_imm(4);
                let nib1 = ctx.shr_u32(packed, sh4);
                let nib1 = ctx.and_u32(nib1, m4);
                let nib2 = ctx.shr_u32(packed, sh8);
                let nib2 = ctx.and_u32(nib2, m4);
                let s12 = ctx.mov_u32_imm(12);
                let nib3 = ctx.shr_u32(packed, s12);
                let nib3 = ctx.and_u32(nib3, m4);
                let nib4 = ctx.shr_u32(packed, sh16);
                let nib4 = ctx.and_u32(nib4, m4);
                let s20 = ctx.mov_u32_imm(20);
                let nib5 = ctx.shr_u32(packed, s20);
                let nib5 = ctx.and_u32(nib5, m4);
                let nib6 = ctx.shr_u32(packed, sh24);
                let nib6 = ctx.and_u32(nib6, m4);
                let s28 = ctx.mov_u32_imm(28);
                let nib7 = ctx.shr_u32(packed, s28);

                // Chunk and scale indices (deinterleaved)
                let three_c = ctx.mov_u32_imm(3);
                let ci = ctx.shr_u32(lane_id, three_c);
                let lsi = ctx.shl_u32(ci, one);
                let hsi = ctx.add_u32(lsi, 1);

                // Select low scale
                let dl = ds0;
                let ml = dm0;
                let p = ctx.setp_eq_u32(lsi, one);
                let dl = ctx.selp_f32(p, ds1, dl);
                let ml = ctx.selp_f32(p, dm1, ml);
                let two_u = ctx.mov_u32_imm(2);
                let p = ctx.setp_eq_u32(lsi, two_u);
                let dl = ctx.selp_f32(p, ds2, dl);
                let ml = ctx.selp_f32(p, dm2, ml);
                let three_u = ctx.mov_u32_imm(3);
                let p = ctx.setp_eq_u32(lsi, three_u);
                let dl = ctx.selp_f32(p, ds3, dl);
                let ml = ctx.selp_f32(p, dm3, ml);
                let p = ctx.setp_eq_u32(lsi, four);
                let dl = ctx.selp_f32(p, ds4, dl);
                let ml = ctx.selp_f32(p, dm4, ml);
                let five_u = ctx.mov_u32_imm(5);
                let p = ctx.setp_eq_u32(lsi, five_u);
                let dl = ctx.selp_f32(p, ds5, dl);
                let ml = ctx.selp_f32(p, dm5, ml);
                let six_u = ctx.mov_u32_imm(6);
                let p = ctx.setp_eq_u32(lsi, six_u);
                let dl = ctx.selp_f32(p, ds6, dl);
                let ml = ctx.selp_f32(p, dm6, ml);
                let seven_u = ctx.mov_u32_imm(7);
                let p = ctx.setp_eq_u32(lsi, seven_u);
                let dl = ctx.selp_f32(p, ds7, dl);
                let ml = ctx.selp_f32(p, dm7, ml);

                // Select high scale
                let dh = ds0;
                let mh = dm0;
                let p = ctx.setp_eq_u32(hsi, one);
                let dh = ctx.selp_f32(p, ds1, dh);
                let mh = ctx.selp_f32(p, dm1, mh);
                let p = ctx.setp_eq_u32(hsi, two_u);
                let dh = ctx.selp_f32(p, ds2, dh);
                let mh = ctx.selp_f32(p, dm2, mh);
                let p = ctx.setp_eq_u32(hsi, three_u);
                let dh = ctx.selp_f32(p, ds3, dh);
                let mh = ctx.selp_f32(p, dm3, mh);
                let p = ctx.setp_eq_u32(hsi, four);
                let dh = ctx.selp_f32(p, ds4, dh);
                let mh = ctx.selp_f32(p, dm4, mh);
                let p = ctx.setp_eq_u32(hsi, five_u);
                let dh = ctx.selp_f32(p, ds5, dh);
                let mh = ctx.selp_f32(p, dm5, mh);
                let p = ctx.setp_eq_u32(hsi, six_u);
                let dh = ctx.selp_f32(p, ds6, dh);
                let mh = ctx.selp_f32(p, dm6, mh);
                let p = ctx.setp_eq_u32(hsi, seven_u);
                let dh = ctx.selp_f32(p, ds7, dh);
                let mh = ctx.selp_f32(p, dm7, mh);

                // Nibbles to f32
                let n0f = ctx.cvt_f32_u32(nib0);
                let n1f = ctx.cvt_f32_u32(nib1);
                let n2f = ctx.cvt_f32_u32(nib2);
                let n3f = ctx.cvt_f32_u32(nib3);
                let n4f = ctx.cvt_f32_u32(nib4);
                let n5f = ctx.cvt_f32_u32(nib5);
                let n6f = ctx.cvt_f32_u32(nib6);
                let n7f = ctx.cvt_f32_u32(nib7);

                // Dequantize
                let dq0 = ctx.mul_f32(dl, n0f);
                let dq0 = ctx.sub_f32(dq0, ml);
                let dq1 = ctx.mul_f32(dh, n1f);
                let dq1 = ctx.sub_f32(dq1, mh);
                let dq2 = ctx.mul_f32(dl, n2f);
                let dq2 = ctx.sub_f32(dq2, ml);
                let dq3 = ctx.mul_f32(dh, n3f);
                let dq3 = ctx.sub_f32(dq3, mh);
                let dq4 = ctx.mul_f32(dl, n4f);
                let dq4 = ctx.sub_f32(dq4, ml);
                let dq5 = ctx.mul_f32(dh, n5f);
                let dq5 = ctx.sub_f32(dq5, mh);
                let dq6 = ctx.mul_f32(dl, n6f);
                let dq6 = ctx.sub_f32(dq6, ml);
                let dq7 = ctx.mul_f32(dh, n7f);
                let dq7 = ctx.sub_f32(dq7, mh);

                // Activation indices (deinterleaved)
                let skb = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let s64 = ctx.mov_u32_imm(64);
                let cb = ctx.mul_u32_reg(ci, s64);
                let cs = ctx.add_u32_reg(skb, cb);

                let sm = ctx.mov_u32_imm(7);
                let lic = ctx.and_u32(lane_id, sm);
                let bic = ctx.shl_u32(lic, two_u);

                let lb = ctx.add_u32_reg(cs, bic);
                let s32 = ctx.mov_u32_imm(32);
                let hb = ctx.add_u32_reg(cs, s32);
                let hb = ctx.add_u32_reg(hb, bic);

                let pt = ctx.mov_f32_imm(0.0);

                // GH-215 FIX: Bounds-check activation loads for non-256-aligned K.
                // When K is not a multiple of 256, the last super-block has
                // padding elements that would cause OOB reads into uninitialized
                // GPU memory, producing garbage output.

                // LOW nibbles
                let lb64 = ctx.cvt_u64_u32(lb);
                let xo = ctx.mul_u64(lb64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb0 = ctx.setp_lt_u32(lb, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb0, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq0);

                let v = ctx.add_u32(lb, 1);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb1 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb1, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq2);

                let v = ctx.add_u32(lb, 2);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb2 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb2, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq4);

                let v = ctx.add_u32(lb, 3);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb3 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb3, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq6);

                // HIGH nibbles
                let hb64 = ctx.cvt_u64_u32(hb);
                let xo = ctx.mul_u64(hb64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb0 = ctx.setp_lt_u32(hb, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb0, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq1);

                let v = ctx.add_u32(hb, 1);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb1 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb1, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq3);

                let v = ctx.add_u32(hb, 2);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb2 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb2, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq5);

                let v = ctx.add_u32(hb, 3);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb3 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb3, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq7);

                ctx.add_f32_inplace(acc, pt);

                // Stride by num_warps
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("mwv_sb_loop");

                ctx.label("mwv_sb_end");

                // Phase 1: Intra-warp reduction
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // Phase 2: Cross-warp reduction via shared memory
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "mwv_skip_sm");

                let f4 = ctx.mov_u32_imm(4);
                let wo = ctx.mul_u32_reg(warp_id, f4);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("mwv_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "mwv_exit");

                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                let yo = ctx.mul_wide_u32(block_id, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("mwv_exit");
                ctx.ret();
            })
    }
}
