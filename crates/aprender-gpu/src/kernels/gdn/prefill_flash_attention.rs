//! PMAT-3596: fused causal flash attention for Qwen3.5's 256-wide heads, prefill.
//!
//! The first batched prefill materialised the attention scores — cuBLAS `QKᵀ` into a
//! scratch, a causal softmax over it, cuBLAS `PV` — so its memory traffic grew as
//! `L²` and made long context attention-bound (9B, 4090: 1,749 tok/s at 20k, 576 at
//! 148k). This kernel never writes a score: the FA2 recurrence runs in registers.
//!
//! ## Precision (the cop's #3596 ruling: f16 INPUTS, f32 ACCUMULATION)
//!
//! `Q`, `K`, `V` and the probabilities `P` enter the tensor cores as f16; every
//! product is accumulated in f32 (`mma.sync.m16n8k16.row.col.f32.f16.f16.f32`), and
//! the online-softmax running max `m`, running sum `l` and the output accumulator `O`
//! are f32 throughout. The f32 cuBLAS path stays selectable for diagnosis.
//!
//! ## Shape
//!
//! One block per (16 query positions, KV head). Warp `w` owns query head
//! `kv * heads_per_kv + w` over those 16 positions, so every `K`/`V` tile staged in
//! shared memory serves all `heads_per_kv` query heads that read it (GQA reuse).
//!
//! Per warp: its `Q` tile (16 × 256) is staged through shared memory as f16 in two
//! 128-wide halves and held in registers as 16 `mma` A-fragments. The key loop walks
//! 16-key tiles `0 ..= (pos0 + last_row) / 16`; per tile it stages `K` and `V` as f16
//! (row stride padded to 264 halves, so `ldmatrix` rows fall in distinct banks), forms
//! `S = Q Kᵀ` (32 `mma`), masks keys past each row's position, updates `m`/`l` with
//! quad shuffles (a row lives in the four lanes of one quad in the accumulator
//! layout), rescales `O`, round-trips `P` through a per-warp shared tile into an
//! A-fragment and accumulates `O += P V` (32 `mma`). At the end `O / l` is written.
//!
//! Static shared memory: `2 × 16 × 264 × 2` (K, V) `+ heads_per_kv × (16 × 24 × 2 +
//! 16 × 136 × 2)` (P, Q staging) — 37,376 B for 4 heads per KV head (0.8B–9B) and
//! 47,616 B for 6 (27B), under the 48 KiB static limit.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// The head width this kernel is written for.
pub const FLASH_HEAD_DIM: u32 = 256;
/// Query positions per block (one mma row tile).
const BR: u32 = 16;
/// Keys per tile.
const BC: u32 = 16;
/// f16 elements per shared K/V row (256 + 8 of padding).
const KV_ROW_H: u32 = 264;
/// f16 elements per shared P row (16 + 8).
const P_ROW_H: u32 = 24;
/// f16 elements per shared Q-staging row (128 + 8).
const Q_ROW_H: u32 = 136;

const K_OFF: u32 = 0;
const V_OFF: u32 = BC * KV_ROW_H * 2;
const P_OFF: u32 = 2 * BC * KV_ROW_H * 2;
const P_TILE_BYTES: u32 = BR * P_ROW_H * 2;
const Q_TILE_BYTES: u32 = BR * Q_ROW_H * 2;

/// Fused causal flash-attention prefill, `head_dim = 256`, GQA.
#[derive(Debug, Clone, Copy)]
pub struct PrefillFlashAttention256Kernel {
    /// Query heads.
    pub num_heads: u32,
    /// KV heads (`num_heads` must be a multiple).
    pub num_kv_heads: u32,
}

impl PrefillFlashAttention256Kernel {
    /// Create the kernel.
    ///
    /// # Panics
    /// If `num_heads` is not a positive multiple of `num_kv_heads`, or the shared
    /// memory for that many heads per KV head exceeds the 48 KiB static limit.
    #[must_use]
    pub fn new(num_heads: u32, num_kv_heads: u32) -> Self {
        assert!(
            num_kv_heads > 0 && num_heads % num_kv_heads == 0,
            "num_heads {num_heads} must be a positive multiple of num_kv_heads {num_kv_heads}"
        );
        let k = Self {
            num_heads,
            num_kv_heads,
        };
        assert!(
            k.shared_bytes() <= 48 * 1024,
            "{} heads per KV head need {} B of static shared memory (> 48 KiB)",
            k.heads_per_kv(),
            k.shared_bytes()
        );
        k
    }

    /// Would [`Self::new`] accept these heads? (A multiple, and the per-KV-group warps'
    /// tiles within the 48 KiB static shared memory.)
    #[must_use]
    pub const fn fits(num_heads: u32, num_kv_heads: u32) -> bool {
        num_kv_heads > 0
            && num_heads % num_kv_heads == 0
            && (P_OFF + (num_heads / num_kv_heads) * (P_TILE_BYTES + Q_TILE_BYTES)) <= 48 * 1024
    }

    /// Query heads per KV head — one warp each.
    #[must_use]
    pub const fn heads_per_kv(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }

    /// Static shared memory the kernel declares.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        (P_OFF + self.heads_per_kv() * (P_TILE_BYTES + Q_TILE_BYTES)) as usize
    }

    /// Launch grid for `rows` query rows.
    #[must_use]
    pub const fn grid(&self, rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(BR), self.num_kv_heads, 1)
    }

    /// Launch block — one warp per query head of the KV group.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32 * self.heads_per_kv(), 1, 1)
    }
}

/// `(lane & !3) + k` for k in 0..4 — the lanes of this lane's quad.
fn quad_lanes(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    lane: VirtualReg,
) -> [VirtualReg; 4] {
    let mask = ctx.mov_u32_imm(!3u32);
    let base = ctx.and_u32(lane, mask);
    [
        base,
        ctx.add_u32(base, 1),
        ctx.add_u32(base, 2),
        ctx.add_u32(base, 3),
    ]
}

/// Max (or sum) of `v` over this lane's quad; every lane of the quad gets it.
fn quad_reduce(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    v: VirtualReg,
    lanes: &[VirtualReg; 4],
    max: bool,
) -> VirtualReg {
    let vals: Vec<VirtualReg> = lanes
        .iter()
        .map(|&l| ctx.shfl_idx_f32_reg(v, l, 0xFFFF_FFFF))
        .collect();
    let mut acc = vals[0];
    for &x in &vals[1..] {
        acc = if max {
            ctx.max_f32(acc, x)
        } else {
            ctx.add_f32(acc, x)
        };
    }
    acc
}

impl Kernel for PrefillFlashAttention256Kernel {
    fn name(&self) -> &str {
        "gdn_prefill_flash_attention_256"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let hpk = self.heads_per_kv();
        let nthreads = 32 * hpk;
        let q_stride = self.num_heads * FLASH_HEAD_DIM;
        let kv_stride = self.num_kv_heads * FLASH_HEAD_DIM;
        let q_off = P_OFF + hpk * P_TILE_BYTES;
        // exp(x * scale) == ex2(x * scale * log2 e); scale = 1/sqrt(256).
        let c_exp = std::f32::consts::LOG2_E / (FLASH_HEAD_DIM as f32).sqrt();

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [rows][num_heads * 256]
            .param(PtxType::U64, "k_ptr") // [>= pos0 + rows][num_kv_heads * 256]
            .param(PtxType::U64, "v_ptr") // [>= pos0 + rows][num_kv_heads * 256]
            .param(PtxType::U64, "out_ptr") // [rows][num_heads * 256]
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "pos0")
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let qblk = ctx.special_reg(PtxReg::CtaIdX);
                let kvh = ctx.special_reg(PtxReg::CtaIdY);
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let rows = ctx.load_param_u32("rows");
                let pos0 = ctx.load_param_u32("pos0");

                let lane_mask = ctx.mov_u32_imm(31);
                let lane = ctx.and_u32(tid, lane_mask);
                let five = ctx.mov_u32_imm(5);
                let warp = ctx.shr_u32(tid, five);
                let two = ctx.mov_u32_imm(2);
                let g = ctx.shr_u32(lane, two); // accumulator row (and row + 8)
                let three = ctx.mov_u32_imm(3);
                let t = ctx.and_u32(lane, three); // accumulator column pair
                let fifteen = ctx.mov_u32_imm(15);
                let lane16 = ctx.and_u32(lane, fifteen);
                let four = ctx.mov_u32_imm(4);
                let lane_hi = ctx.shr_u32(lane, four); // 0 or 1
                let kvh_heads = ctx.mul_u32(kvh, hpk);
                let head = ctx.add_u32_reg(kvh_heads, warp);
                let q0 = ctx.mul_u32(qblk, BR);
                let fz = ctx.mov_f32_imm(0.0);
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let quad = quad_lanes(ctx, lane);

                // ---- Q: stage each 128-wide half as f16, lift it into A-fragments.
                let q_warp_off = ctx.mul_u32(warp, Q_TILE_BYTES);
                let q_warp_base = ctx.add_u32(q_warp_off, q_off);
                let head_col = ctx.mul_u32(head, FLASH_HEAD_DIM);
                let mut qfrag: Vec<[VirtualReg; 4]> = Vec::with_capacity(16);
                for half in 0..2u32 {
                    ctx.bar_sync(0);
                    let it = ctx.mov_u32_imm(0);
                    let lbl = format!("fa_qstage_{half}");
                    let end = format!("fa_qstage_end_{half}");
                    ctx.label(&lbl);
                    let n = ctx.mov_u32_imm(BR * 128 / 32);
                    let more = ctx.setp_lt_u32(it, n);
                    ctx.branch_if_not(more, &end);
                    let it32 = ctx.mul_u32(it, 32);
                    let idx = ctx.add_u32_reg(it32, lane);
                    let seven = ctx.mov_u32_imm(7);
                    let r = ctx.shr_u32(idx, seven); // / 128
                    let m127 = ctx.mov_u32_imm(127);
                    let c = ctx.and_u32(idx, m127);
                    let qrow = ctx.add_u32_reg(q0, r);
                    let val = ctx.mov_f32_imm(0.0);
                    let in_rows = ctx.setp_lt_u32(qrow, rows);
                    let skip = format!("fa_qload_skip_{half}");
                    ctx.branch_if_not(in_rows, &skip);
                    let rowel = ctx.mul_u32(qrow, q_stride);
                    let col = ctx.add_u32(c, half * 128);
                    let colh = ctx.add_u32_reg(head_col, col);
                    let el = ctx.add_u32_reg(rowel, colh);
                    let off = ctx.mul_wide_u32(el, 4);
                    let addr = ctx.add_u64(q_ptr, off);
                    let x = ctx.ld_global_f32(addr);
                    ctx.mov_f32_reg(val, x);
                    ctx.label(&skip);
                    let h = ctx.cvt_f16_f32(val);
                    let srow = ctx.mul_u32(r, Q_ROW_H);
                    let sel = ctx.add_u32_reg(srow, c);
                    let sb = ctx.mul_u32(sel, 2);
                    let sa = ctx.add_u32_reg(q_warp_base, sb);
                    let sa64 = ctx.cvt_u64_u32(sa);
                    ctx.st_shared_f16(sa64, h);
                    ctx.add_u32_inplace(it, 1);
                    ctx.branch(&lbl);
                    ctx.label(&end);
                    ctx.bar_sync(0);
                    // A-fragment of k-step kk: lane reads row lane%16, cols kk*16 + (lane/16)*8.
                    let lrow = ctx.mul_u32(lane16, Q_ROW_H);
                    let lcol = ctx.mul_u32(lane_hi, 8);
                    let lel = ctx.add_u32_reg(lrow, lcol);
                    let lb = ctx.mul_u32(lel, 2);
                    let lbase = ctx.add_u32_reg(q_warp_base, lb);
                    for kk in 0..8u32 {
                        let a = ctx.add_u32(lbase, kk * 16 * 2);
                        qfrag.push(ctx.ldmatrix_x4(a));
                    }
                }

                // ---- Per-row running state (rows g and g + 8) and the O accumulator.
                let m = [ctx.mov_f32_imm(f32::NEG_INFINITY), ctx.mov_f32_imm(f32::NEG_INFINITY)];
                let l = [ctx.mov_f32_imm(0.0), ctx.mov_f32_imm(0.0)];
                let o: Vec<[VirtualReg; 4]> = (0..FLASH_HEAD_DIM / 8)
                    .map(|_| {
                        [
                            ctx.mov_f32_imm(0.0),
                            ctx.mov_f32_imm(0.0),
                            ctx.mov_f32_imm(0.0),
                            ctx.mov_f32_imm(0.0),
                        ]
                    })
                    .collect();

                // Key tiles 0 ..= (pos0 + last valid row) / 16.
                let r15 = ctx.add_u32(q0, BR - 1);
                let one = ctx.mov_u32_imm(1);
                let rows_m1 = ctx.sub_u32_reg(rows, one);
                let last_row = ctx.min_u32(r15, rows_m1);
                let last_pos = ctx.add_u32_reg(pos0, last_row);
                let last_tile = ctx.div_u32(last_pos, BC);
                let tiles = ctx.add_u32(last_tile, 1);
                let total_keys = ctx.add_u32_reg(pos0, rows);
                let kv_col = ctx.mul_u32(kvh, FLASH_HEAD_DIM);
                let p_warp_off = ctx.mul_u32(warp, P_TILE_BYTES);
                let p_base = ctx.add_u32(p_warp_off, P_OFF);
                // This lane's query rows, absolute positions.
                let row_lo_abs = {
                    let a = ctx.add_u32_reg(pos0, q0);
                    ctx.add_u32_reg(a, g)
                };
                let row_hi_abs = ctx.add_u32(row_lo_abs, 8);
                let c_exp_r = ctx.mov_f32_imm(c_exp);
                let finite_floor = ctx.mov_f32_imm(-1.0e30);

                let kb = ctx.mov_u32_imm(0);
                ctx.label("fa_key_loop");
                let more = ctx.setp_lt_u32(kb, tiles);
                ctx.branch_if_not(more, "fa_key_end");
                ctx.bar_sync(0);
                let j0 = ctx.mul_u32(kb, BC);

                // Stage K and V tiles as f16.
                for (src, dst, name) in [(k_ptr, K_OFF, "k"), (v_ptr, V_OFF, "v")] {
                    let it = ctx.mov_u32_imm(0);
                    let lbl = format!("fa_kv_{name}");
                    let end = format!("fa_kv_end_{name}");
                    ctx.label(&lbl);
                    // ceil: 4096 is not a multiple of 192 (6 warps, the 27B) — a floor
                    // left the tile's last 64 elements stale (measured: rel L∞ 0.218).
                    let n = ctx.mov_u32_imm((BC * FLASH_HEAD_DIM).div_ceil(nthreads));
                    let go = ctx.setp_lt_u32(it, n);
                    ctx.branch_if_not(go, &end);
                    let itn = ctx.mul_u32(it, nthreads);
                    let idx = ctx.add_u32_reg(itn, tid);
                    let tile_elems = ctx.mov_u32_imm(BC * FLASH_HEAD_DIM);
                    let in_tile = ctx.setp_lt_u32(idx, tile_elems);
                    let step = format!("fa_kv_step_{name}");
                    ctx.branch_if_not(in_tile, &step);
                    let eight = ctx.mov_u32_imm(8);
                    let r = ctx.shr_u32(idx, eight); // / 256
                    let m255 = ctx.mov_u32_imm(255);
                    let c = ctx.and_u32(idx, m255);
                    let j = ctx.add_u32_reg(j0, r);
                    let val = ctx.mov_f32_imm(0.0);
                    let exists = ctx.setp_lt_u32(j, total_keys);
                    let skip = format!("fa_kv_skip_{name}");
                    ctx.branch_if_not(exists, &skip);
                    let rowel = ctx.mul_u32(j, kv_stride);
                    let colh = ctx.add_u32_reg(kv_col, c);
                    let el = ctx.add_u32_reg(rowel, colh);
                    let off = ctx.mul_wide_u32(el, 4);
                    let addr = ctx.add_u64(src, off);
                    let x = ctx.ld_global_f32(addr);
                    ctx.mov_f32_reg(val, x);
                    ctx.label(&skip);
                    let h = ctx.cvt_f16_f32(val);
                    let srow = ctx.mul_u32(r, KV_ROW_H);
                    let sel = ctx.add_u32_reg(srow, c);
                    let sb = ctx.mul_u32(sel, 2);
                    let sa = ctx.add_u32(sb, dst);
                    let sa64 = ctx.cvt_u64_u32(sa);
                    ctx.st_shared_f16(sa64, h);
                    ctx.label(&step);
                    ctx.add_u32_inplace(it, 1);
                    ctx.branch(&lbl);
                    ctx.label(&end);
                }
                ctx.bar_sync(0);

                // S = Q Kᵀ over 16 k-steps; two n-tiles of 8 keys.
                let mut s0 = [fz, fz, fz, fz];
                let mut s1 = [fz, fz, fz, fz];
                {
                    // B from K ([key][d] row-major = col-major k×n): non-trans x4.
                    let seven = ctx.mov_u32_imm(7);
                    let n_lo = ctx.and_u32(lane, seven);
                    let n_hi = ctx.mul_u32(lane_hi, 8);
                    let n = ctx.add_u32_reg(n_lo, n_hi);
                    let three_s = ctx.mov_u32_imm(3);
                    let kbit = ctx.shr_u32(lane, three_s);
                    let kone = ctx.and_u32(kbit, one);
                    let koff = ctx.mul_u32(kone, 8);
                    let nrow = ctx.mul_u32(n, KV_ROW_H);
                    let el = ctx.add_u32_reg(nrow, koff);
                    let b = ctx.mul_u32(el, 2);
                    let kbase = ctx.add_u32(b, K_OFF);
                    for (kk, a) in qfrag.iter().enumerate() {
                        let addr = ctx.add_u32(kbase, kk as u32 * 16 * 2);
                        let bf = ctx.ldmatrix_x4(addr);
                        s0 = ctx.mma_sync_m16n8k16(a, &[bf[0], bf[1]], &s0);
                        s1 = ctx.mma_sync_m16n8k16(a, &[bf[2], bf[3]], &s1);
                    }
                }

                // Causal mask: key j0 + col is visible to row p iff key <= p.
                let t2 = ctx.mul_u32(t, 2);
                let mut sv = [s0, s1];
                for (nt, tile) in sv.iter_mut().enumerate() {
                    for e in 0..4usize {
                        let colbase = ctx.add_u32(t2, nt as u32 * 8 + (e as u32 & 1));
                        let key = ctx.add_u32_reg(j0, colbase);
                        let p_abs = if e < 2 { row_lo_abs } else { row_hi_abs };
                        let visible = ctx.setp_le_u32(key, p_abs);
                        tile[e] = ctx.selp_f32(visible, tile[e], neg_inf);
                    }
                }

                // Online softmax per row half (hr 0: row g, hr 1: row g + 8).
                let mut p16 = Vec::with_capacity(8);
                for hr in 0..2usize {
                    let vals = [sv[0][2 * hr], sv[0][2 * hr + 1], sv[1][2 * hr], sv[1][2 * hr + 1]];
                    let a = ctx.max_f32(vals[0], vals[1]);
                    let b = ctx.max_f32(vals[2], vals[3]);
                    let local = ctx.max_f32(a, b);
                    let rmax = quad_reduce(ctx, local, &quad, true);
                    let m_new = ctx.max_f32(m[hr], rmax);
                    let finite = ctx.setp_gt_f32(m_new, finite_floor);
                    let m_use = ctx.selp_f32(finite, m_new, fz);
                    let dm = ctx.sub_f32(m[hr], m_use);
                    let dms = ctx.mul_f32(dm, c_exp_r);
                    let alpha = ctx.ex2_f32(dms);
                    let mut psum = fz;
                    let mut ps = Vec::with_capacity(4);
                    for &v in &vals {
                        let d = ctx.sub_f32(v, m_use);
                        let ds = ctx.mul_f32(d, c_exp_r);
                        let pv = ctx.ex2_f32(ds);
                        psum = ctx.add_f32(psum, pv);
                        ps.push(pv);
                    }
                    let rsum = quad_reduce(ctx, psum, &quad, false);
                    ctx.mul_f32_inplace(l[hr], alpha);
                    ctx.add_f32_inplace(l[hr], rsum);
                    ctx.mov_f32_reg(m[hr], m_new);
                    for tile in &o {
                        ctx.mul_f32_inplace(tile[2 * hr], alpha);
                        ctx.mul_f32_inplace(tile[2 * hr + 1], alpha);
                    }
                    p16.push((hr, ps));
                }

                // P -> the warp's shared tile as f16: row g/g+8, cols nt*8 + 2t + {0,1}.
                for (hr, ps) in &p16 {
                    let row = ctx.add_u32(g, *hr as u32 * 8);
                    let prow = ctx.mul_u32(row, P_ROW_H);
                    for (i, &pv) in ps.iter().enumerate() {
                        // vals order: (nt0,e0),(nt0,e1),(nt1,e0),(nt1,e1)
                        let nt = (i / 2) as u32;
                        let e = (i % 2) as u32;
                        let col = ctx.add_u32(t2, nt * 8 + e);
                        let el = ctx.add_u32_reg(prow, col);
                        let b = ctx.mul_u32(el, 2);
                        let sa = ctx.add_u32_reg(p_base, b);
                        let sa64 = ctx.cvt_u64_u32(sa);
                        let h = ctx.cvt_f16_f32(pv);
                        ctx.st_shared_f16(sa64, h);
                    }
                }
                ctx.bar_sync(0);

                // O += P V: A = P (16×16) from the warp tile; B = V (k=16 keys × n=8 dims).
                {
                    let prow = ctx.mul_u32(lane16, P_ROW_H);
                    let pcol = ctx.mul_u32(lane_hi, 8);
                    let pel = ctx.add_u32_reg(prow, pcol);
                    let pb = ctx.mul_u32(pel, 2);
                    let paddr = ctx.add_u32_reg(p_base, pb);
                    let pfrag = ctx.ldmatrix_x4(paddr);
                    let vrow = ctx.mul_u32(lane16, KV_ROW_H);
                    let vb = ctx.mul_u32(vrow, 2);
                    let vbase = ctx.add_u32(vb, V_OFF);
                    for (nt, tile) in o.iter().enumerate() {
                        let addr = ctx.add_u32(vbase, nt as u32 * 8 * 2);
                        let bf = ctx.ldmatrix_x2_trans(addr);
                        ctx.mma_sync_m16n8k16_inplace(&pfrag, &bf, tile);
                    }
                }

                ctx.add_u32_inplace(kb, 1);
                ctx.branch("fa_key_loop");
                ctx.label("fa_key_end");

                // ---- O / l, written for the rows that exist.
                for hr in 0..2usize {
                    let row = ctx.add_u32(g, hr as u32 * 8);
                    let qrow = ctx.add_u32_reg(q0, row);
                    let skip = format!("fa_store_skip_{hr}");
                    let in_rows = ctx.setp_lt_u32(qrow, rows);
                    ctx.branch_if_not(in_rows, &skip);
                    let inv = ctx.rcp_f32(l[hr]);
                    let rowel = ctx.mul_u32(qrow, q_stride);
                    let rowh = ctx.add_u32_reg(rowel, head_col);
                    let colt = ctx.add_u32_reg(rowh, t2);
                    for (nt, tile) in o.iter().enumerate() {
                        for e in 0..2usize {
                            let el = ctx.add_u32(colt, nt as u32 * 8 + e as u32);
                            let off = ctx.mul_wide_u32(el, 4);
                            let addr = ctx.add_u64(out_ptr, off);
                            let v = ctx.mul_f32(tile[2 * hr + e], inv);
                            ctx.st_global_f32(addr, v);
                        }
                    }
                    ctx.label(&skip);
                }
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_flash_prefill_ptx_shape_and_shared_budget() {
        for (h, kv, bytes) in [(16u32, 4u32, 37_376usize), (24, 4, 47_616), (8, 2, 37_376)] {
            let k = PrefillFlashAttention256Kernel::new(h, kv);
            assert_eq!(k.shared_bytes(), bytes, "{h}/{kv}");
            let ptx = k.emit_ptx();
            assert!(
                ptx.contains(".entry gdn_prefill_flash_attention_256"),
                "{ptx}"
            );
            assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
            assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
            assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
        }
        assert_eq!(
            PrefillFlashAttention256Kernel::new(16, 4).grid(37),
            (3, 4, 1)
        );
        assert_eq!(
            PrefillFlashAttention256Kernel::new(24, 4).block(),
            (192, 1, 1)
        );
        assert!(PrefillFlashAttention256Kernel::fits(24, 4));
        assert!(!PrefillFlashAttention256Kernel::fits(32, 4));
        assert!(!PrefillFlashAttention256Kernel::fits(16, 3));
    }

    #[test]
    #[should_panic(expected = "static shared memory")]
    fn gdn_flash_prefill_refuses_what_does_not_fit_48k() {
        let _ = PrefillFlashAttention256Kernel::new(32, 4); // 8 heads per KV head
    }
}

/// Device parity: the kernel against an exact reference of what it computes (f16
/// inputs, f32/f64 math), and against the full-f32 reference within the f16 budget.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_flash_prefill_device_tests {
    use super::{PrefillFlashAttention256Kernel, FLASH_HEAD_DIM};
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::gdn::test_support::Lcg;
    use crate::kernels::Kernel;

    /// Round an f32 to the nearest f16 (ties to even), as `cvt.rn.f16.f32` does,
    /// within the normal and subnormal range the test data lives in.
    fn round_f16(x: f32) -> f32 {
        if x == 0.0 || !x.is_finite() {
            return x;
        }
        let exp = x.abs().log2().floor() as i32;
        let quantum = if exp < -14 {
            2f32.powi(-24)
        } else {
            2f32.powi(exp - 10)
        };
        (x / quantum).round_ties_even() * quantum
    }

    /// Causal GQA attention in f64, over (optionally f16-rounded) inputs.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rows: usize,
        pos0: usize,
        heads: usize,
        kv_heads: usize,
        f16_inputs: bool,
    ) -> Vec<f32> {
        let d = FLASH_HEAD_DIM as usize;
        let hpk = heads / kv_heads;
        let r = |x: &[f32]| -> Vec<f64> {
            x.iter()
                .map(|&v| f64::from(if f16_inputs { round_f16(v) } else { v }))
                .collect()
        };
        let (q, k, v) = (r(q), r(k), r(v));
        let mut out = vec![0.0f32; rows * heads * d];
        let mut acc = vec![0.0f64; d];
        for row in 0..rows {
            let p = pos0 + row;
            for h in 0..heads {
                let g = h / hpk;
                let qv = &q[(row * heads + h) * d..(row * heads + h + 1) * d];
                let scores: Vec<f64> = (0..=p)
                    .map(|j| {
                        let kv = &k[(j * kv_heads + g) * d..(j * kv_heads + g + 1) * d];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() / (d as f64).sqrt()
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f64 = w.iter().sum();
                acc.iter_mut().for_each(|a| *a = 0.0);
                for (j, &wj) in w.iter().enumerate() {
                    // The kernel feeds P to the tensor cores as f16 too.
                    let pw = if f16_inputs {
                        f64::from(round_f16(wj as f32))
                    } else {
                        wj
                    };
                    let vv = &v[(j * kv_heads + g) * d..(j * kv_heads + g + 1) * d];
                    acc.iter_mut().zip(vv).for_each(|(a, x)| *a += pw * x);
                }
                for (i, a) in acc.iter().enumerate() {
                    out[(row * heads + h) * d + i] = (a / sum) as f32;
                }
            }
        }
        out
    }

    fn run(heads: u32, kv_heads: u32, rows: usize, pos0: usize) {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("flash prefill: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let d = FLASH_HEAD_DIM as usize;
        let (h, kvh) = (heads as usize, kv_heads as usize);
        let total = pos0 + rows;
        let mut rng = Lcg::new(0x3596_0300 ^ (rows as u32) ^ ((pos0 as u32) << 8));
        // Magnitudes like the model's after per-head RMSNorm: O(1), with a spread
        // wide enough that the softmax is not uniform.
        let q = rng.vec(rows * h * d, 1.5);
        let k = rng.vec(total * kvh * d, 1.5);
        let v = rng.vec(total * kvh * d, 1.0);

        let kern = PrefillFlashAttention256Kernel::new(heads, kv_heads);
        let qb = GpuBuffer::from_host(&ctx, &q).expect("q");
        let kb = GpuBuffer::from_host(&ctx, &k).expect("k");
        let vb = GpuBuffer::from_host(&ctx, &v).expect("v");
        let ob = GpuBuffer::from_host(&ctx, &vec![f32::NAN; rows * h * d]).expect("o");
        let mut module =
            CudaModule::from_ptx(&ctx, &kern.emit_ptx_for_target("sm_80")).expect("module");
        let config = LaunchConfig {
            grid: kern.grid(rows as u32),
            block: kern.block(),
            shared_mem: 0,
        };
        let mut args = [
            qb.as_ptr(),
            kb.as_ptr(),
            vb.as_ptr(),
            ob.as_ptr(),
            rows as u64,
            pos0 as u64,
        ];
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: q/k/v/o are sized for rows × heads × 256 and (pos0 + rows) keys; the
        // two u32 scalars sit in the low half of their slots.
        unsafe {
            stream
                .launch_kernel(&mut module, kern.name(), &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; rows * h * d];
        ob.copy_to_host(&mut got).expect("o");

        let scale = |w: &[f32]| w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let exact16 = reference(&q, &k, &v, rows, pos0, h, kvh, true);
        let exact32 = reference(&q, &k, &v, rows, pos0, h, kvh, false);
        assert!(got.iter().all(|x| x.is_finite()), "a non-finite output");
        let linf = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        let e16 = linf(&got, &exact16) / scale(&exact16);
        let e32 = linf(&got, &exact32) / scale(&exact32);
        println!(
            "[3596] flash prefill heads {heads}/{kv_heads} rows {rows} pos0 {pos0}: rel L∞ vs f16-input \
             reference {e16:.3e}, vs full-f32 reference {e32:.3e}"
        );
        assert!(e16 <= 2e-3, "vs the f16-input reference: {e16}");
        assert!(e32 <= 1e-2, "vs the full-f32 reference: {e32}");
    }

    #[test]
    fn gdn_flash_prefill_matches_reference_9b_shape_with_prefix_and_tail() {
        // 9B: 16 query heads over 4 KV heads; 37 rows (a partial last block) after 29
        // cached positions (keys from an earlier chunk).
        run(16, 4, 37, 29);
    }

    #[test]
    fn gdn_flash_prefill_matches_reference_27b_shape() {
        run(24, 4, 20, 3);
    }

    #[test]
    fn gdn_flash_prefill_matches_reference_from_position_zero() {
        run(8, 2, 16, 0);
    }

    #[test]
    fn gdn_flash_prefill_matches_reference_over_many_key_tiles() {
        // 100 rows after 500 cached positions: 38 key tiles, so the running max and
        // sum are rescaled dozens of times per row.
        run(16, 4, 100, 500);
    }
}
