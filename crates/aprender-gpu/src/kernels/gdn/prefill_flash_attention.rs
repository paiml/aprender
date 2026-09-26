//! PMAT-3596: fused causal flash attention for Qwen3.5's 256-wide heads, prefill.
//!
//! The first batched prefill materialised the attention scores — cuBLAS `QKᵀ` into a
//! scratch, a causal softmax over it, cuBLAS `PV` — so its memory traffic grew as
//! `L²` and made long context attention-bound (the rungs are in
//! `docs/audits/impl-PMAT-3596-receipt.md`). This kernel never writes a score: the FA2
//! recurrence runs in registers.
//!
//! ## Precision (the cop's #3596 ruling: f16 INPUTS, f32 ACCUMULATION)
//!
//! `Q`, `K`, `V` and the probabilities `P` enter the tensor cores as f16; every
//! product is accumulated in f32 (`mma.sync.m16n8k16.row.col.f32.f16.f16.f32`), and
//! the online-softmax running max `m`, running sum `l` and the output accumulator `O`
//! are f32 throughout. The f32 cuBLAS path stays selectable for diagnosis.
//!
//! ## Shape (#4442 v2)
//!
//! `K` and `V` arrive **already f16** (the caller converts the cache once per call with
//! `cvt.rn`, the rounding v1 applied per element in-kernel). v1 staged them with one
//! scalar `ld.global.f32` + `cvt` + `st.shared` per element and ran at ~3 TFLOP/s on
//! GB10 — latency-bound on those loads (#4442 profile, `docs/findings`).
//!
//! One block per (`16 × RT` query positions, KV head), `heads_per_kv × RT` warps:
//! warp `w` owns query head `kv * heads_per_kv + w % heads_per_kv` over row tile
//! `w / heads_per_kv`. `RT = (8 / heads_per_kv).clamp(1, 4)`, so every `K`/`V` tile
//! staged in shared memory serves all the GQA heads *and* up to four row tiles.
//!
//! Per warp: its `Q` tile (16 × 256) is staged through shared memory as f16 in four
//! 64-wide quarters and held in registers as 16 `mma` A-fragments. The key loop walks
//! 16-key tiles through a two-stage `cp.async` (16 B) pipeline — tile `kb + 1` is in
//! flight while tile `kb` is consumed. Per tile a warp forms `S = Q Kᵀ` (32 `mma`),
//! masks keys past each row's position, updates `m`/`l` with quad shuffles, rescales
//! `O`, packs `P` straight from the accumulator registers into an A-fragment
//! (`cvt.rn.f16x2.f32` — the C and A fragment layouts coincide, so no shared round
//! trip and no block barrier) and accumulates `O += P V` (32 `mma`). A warp skips the
//! tiles past its own rows' causal extent (they would be fully masked: an exact
//! identity on `m`, `l` and `O`). At the end `O / l` is written.
//!
//! Tile order, masking and every rounding are v1's, so v2 is bit-identical to v1.
//!
//! Static shared memory: two stages of `K`+`V`, `2 × 2 × 16 × 264 × 2` = 33,792 B for
//! every head count; the `Q` staging (`warps × 16 × 72 × 2` ≤ 18,432 B) reuses it
//! before the pipeline starts.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// The head width this kernel is written for.
pub const FLASH_HEAD_DIM: u32 = 256;
/// Query positions per warp (one mma row tile).
const BR: u32 = 16;
/// Keys per tile.
const BC: u32 = 16;
/// f16 elements per shared K/V row (256 + 8 of padding: `ldmatrix` rows fall in
/// distinct banks, and 528 B keeps every 16 B `cp.async` destination aligned).
const KV_ROW_H: u32 = 264;
/// f16 elements per shared Q-staging row (64 + 8).
const Q_ROW_H: u32 = 72;
/// Most warps per block.
const MAX_WARPS: u32 = 8;

const KV_TILE_BYTES: u32 = BC * KV_ROW_H * 2;
const STAGE_BYTES: u32 = 2 * KV_TILE_BYTES;
const SHARED_BYTES: u32 = 2 * STAGE_BYTES;
const Q_TILE_BYTES: u32 = BR * Q_ROW_H * 2;
/// 16 B chunks in one stage (K and V, 16 rows × 32 chunks each).
const STAGE_CHUNKS: u32 = 2 * BC * (FLASH_HEAD_DIM / 8);

const _: () = assert!(MAX_WARPS * Q_TILE_BYTES <= SHARED_BYTES);
const _: () = assert!(SHARED_BYTES <= 48 * 1024);

/// Fused causal flash-attention prefill, `head_dim = 256`, GQA, f16 `K`/`V`.
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
    /// If `num_heads` is not a positive multiple of `num_kv_heads`, or there are more
    /// than 8 heads per KV head (one warp each, and the `Q` staging must fit the
    /// pipeline's shared memory).
    #[must_use]
    pub fn new(num_heads: u32, num_kv_heads: u32) -> Self {
        assert!(
            num_kv_heads > 0 && num_heads % num_kv_heads == 0,
            "num_heads {num_heads} must be a positive multiple of num_kv_heads {num_kv_heads}"
        );
        assert!(
            Self::fits(num_heads, num_kv_heads),
            "{} heads per KV head exceed the {MAX_WARPS} warps whose Q staging fits the \
             static shared memory",
            num_heads / num_kv_heads
        );
        Self {
            num_heads,
            num_kv_heads,
        }
    }

    /// Would [`Self::new`] accept these heads? (A multiple, at most 8 per KV head.)
    #[must_use]
    pub const fn fits(num_heads: u32, num_kv_heads: u32) -> bool {
        num_kv_heads > 0
            && num_heads % num_kv_heads == 0
            && num_heads / num_kv_heads >= 1
            && num_heads / num_kv_heads <= MAX_WARPS
    }

    /// Query heads per KV head.
    #[must_use]
    pub const fn heads_per_kv(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }

    /// Row tiles per block: `(8 / heads_per_kv).clamp(1, 4)`.
    #[must_use]
    pub const fn row_tiles(&self) -> u32 {
        let rt = MAX_WARPS / self.heads_per_kv();
        if rt < 1 {
            1
        } else if rt > 4 {
            4
        } else {
            rt
        }
    }

    /// Query rows per block.
    #[must_use]
    pub const fn rows_per_block(&self) -> u32 {
        BR * self.row_tiles()
    }

    /// Static shared memory the kernel declares.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        SHARED_BYTES as usize
    }

    /// Launch grid for `rows` query rows.
    #[must_use]
    pub const fn grid(&self, rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(self.rows_per_block()), self.num_kv_heads, 1)
    }

    /// Launch block — one warp per (query head of the KV group, row tile).
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32 * self.heads_per_kv() * self.row_tiles(), 1, 1)
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

/// Per-block constants the `cp.async` tile issue reads.
struct Issue {
    tid: VirtualReg,
    nthreads: u32,
    k_ptr: VirtualReg,
    v_ptr: VirtualReg,
    kv_stride: u32,
    kv_col: VirtualReg,
    last_key: VirtualReg,
}

/// Issue the 16 B `cp.async` copies of key tile `tile` (K then V, f16) into stage
/// `tile & 1`. Rows past the last key re-read the last key: they are masked, and
/// keep V finite. Does not commit.
fn issue_tile(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    is: &Issue,
    tile: VirtualReg,
    tag: &str,
) {
    let j0 = ctx.mul_u32(tile, BC);
    let one = ctx.mov_u32_imm(1);
    let parity = ctx.and_u32(tile, one);
    let stage = ctx.mul_u32(parity, STAGE_BYTES);
    let chunks = ctx.mov_u32_imm(STAGE_CHUNKS);
    let half_chunks = ctx.mov_u32_imm(STAGE_CHUNKS / 2);
    for i in 0..STAGE_CHUNKS.div_ceil(is.nthreads) {
        let idx = ctx.add_u32(is.tid, i * is.nthreads);
        let guarded = (i + 1) * is.nthreads > STAGE_CHUNKS;
        let skip = format!("fa_issue_skip_{tag}_{i}");
        if guarded {
            let in_stage = ctx.setp_lt_u32(idx, chunks);
            ctx.branch_if_not(in_stage, &skip);
        }
        let is_k = ctx.setp_lt_u32(idx, half_chunks);
        let m511 = ctx.mov_u32_imm(STAGE_CHUNKS / 2 - 1);
        let rem = ctx.and_u32(idx, m511);
        let five = ctx.mov_u32_imm(5);
        let r = ctx.shr_u32(rem, five); // row: 32 chunks per row
        let m31 = ctx.mov_u32_imm(31);
        let c = ctx.and_u32(rem, m31);
        let jr = ctx.add_u32_reg(j0, r);
        let j = ctx.min_u32(jr, is.last_key);
        let rowel = ctx.mul_u32(j, is.kv_stride);
        let c8 = ctx.mul_u32(c, 8);
        let colh = ctx.add_u32_reg(is.kv_col, c8);
        let el = ctx.add_u32_reg(rowel, colh);
        let off = ctx.mul_wide_u32(el, 2);
        let ka = ctx.add_u64(is.k_ptr, off);
        let va = ctx.add_u64(is.v_ptr, off);
        let gaddr = ctx.selp_u64(is_k, ka, va);
        // Shared: stage + (V ? KV_TILE : 0) + r * 528 + c * 16.
        let nine = ctx.mov_u32_imm(9);
        let which = ctx.shr_u32(idx, nine);
        let wb = ctx.mul_u32(which, KV_TILE_BYTES);
        let rb = ctx.mul_u32(r, KV_ROW_H * 2);
        let cb = ctx.mul_u32(c, 16);
        let a = ctx.add_u32_reg(stage, wb);
        let b = ctx.add_u32_reg(a, rb);
        let soff = ctx.add_u32_reg(b, cb);
        ctx.cp_async_global_to_shared(soff, gaddr, 16);
        if guarded {
            ctx.label(&skip);
        }
    }
}

impl Kernel for PrefillFlashAttention256Kernel {
    fn name(&self) -> &str {
        "gdn_prefill_flash_attention_256"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let hpk = self.heads_per_kv();
        let nthreads = self.block().0;
        let br_block = self.rows_per_block();
        let q_stride = self.num_heads * FLASH_HEAD_DIM;
        let kv_stride = self.num_kv_heads * FLASH_HEAD_DIM;
        // exp(x * scale) == ex2(x * scale * log2 e); scale = 1/sqrt(256).
        let c_exp = std::f32::consts::LOG2_E / (FLASH_HEAD_DIM as f32).sqrt();

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // f32 [rows][num_heads * 256]
            .param(PtxType::U64, "k_ptr") // f16 [>= pos0 + rows][num_kv_heads * 256]
            .param(PtxType::U64, "v_ptr") // f16 [>= pos0 + rows][num_kv_heads * 256]
            .param(PtxType::U64, "out_ptr") // f32 [rows][num_heads * 256]
            .param(PtxType::U64, "part_o_ptr") // f32 [splits][rows][num_heads * 256]
            .param(PtxType::U64, "part_ml_ptr") // f32 [splits][rows][num_heads][m, l]
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "pos0")
            .param(PtxType::U32, "splits")
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let qblk = ctx.special_reg(PtxReg::CtaIdX);
                let kvh = ctx.special_reg(PtxReg::CtaIdY);
                let split = ctx.special_reg(PtxReg::CtaIdZ);
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let rows = ctx.load_param_u32("rows");
                let pos0 = ctx.load_param_u32("pos0");
                let splits = ctx.load_param_u32("splits");

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
                let rtile = ctx.div_u32(warp, hpk);
                let rt_heads = ctx.mul_u32(rtile, hpk);
                let hw = ctx.sub_u32_reg(warp, rt_heads);
                let kvh_heads = ctx.mul_u32(kvh, hpk);
                let head = ctx.add_u32_reg(kvh_heads, hw);
                let qb0 = ctx.mul_u32(qblk, br_block);
                let rt_rows = ctx.mul_u32(rtile, BR);
                let q0 = ctx.add_u32_reg(qb0, rt_rows);
                let fz = ctx.mov_f32_imm(0.0);
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let quad = quad_lanes(ctx, lane);
                let one = ctx.mov_u32_imm(1);

                // ---- Q: stage each 64-wide quarter as f16, lift it into A-fragments.
                let q_warp_base = ctx.mul_u32(warp, Q_TILE_BYTES);
                let head_col = ctx.mul_u32(head, FLASH_HEAD_DIM);
                let mut qfrag: Vec<[VirtualReg; 4]> = Vec::with_capacity(16);
                for quarter in 0..4u32 {
                    ctx.bar_sync(0);
                    let it = ctx.mov_u32_imm(0);
                    let lbl = format!("fa_qstage_{quarter}");
                    let end = format!("fa_qstage_end_{quarter}");
                    ctx.label(&lbl);
                    let n = ctx.mov_u32_imm(BR * 64 / 32);
                    let more = ctx.setp_lt_u32(it, n);
                    ctx.branch_if_not(more, &end);
                    let it32 = ctx.mul_u32(it, 32);
                    let idx = ctx.add_u32_reg(it32, lane);
                    let six = ctx.mov_u32_imm(6);
                    let r = ctx.shr_u32(idx, six); // / 64
                    let m63 = ctx.mov_u32_imm(63);
                    let c = ctx.and_u32(idx, m63);
                    let qrow = ctx.add_u32_reg(q0, r);
                    let val = ctx.mov_f32_imm(0.0);
                    let in_rows = ctx.setp_lt_u32(qrow, rows);
                    let skip = format!("fa_qload_skip_{quarter}");
                    ctx.branch_if_not(in_rows, &skip);
                    let rowel = ctx.mul_u32(qrow, q_stride);
                    let col = ctx.add_u32(c, quarter * 64);
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
                    for kk in 0..4u32 {
                        let a = ctx.add_u32(lbase, kk * 16 * 2);
                        qfrag.push(ctx.ldmatrix_x4(a));
                    }
                }
                // Every warp's last ldmatrix is done before tile 0 lands on the staging.
                ctx.bar_sync(0);

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

                // Block: key tiles 0 ..= (pos0 + the block's last valid row) / 16.
                let rows_m1 = ctx.sub_u32_reg(rows, one);
                let blk_end = ctx.add_u32(qb0, br_block - 1);
                let blk_last = ctx.min_u32(blk_end, rows_m1);
                let blk_pos = ctx.add_u32_reg(pos0, blk_last);
                let blk_tile = ctx.div_u32(blk_pos, BC);
                let tiles = ctx.add_u32(blk_tile, 1);
                // Split-KV (#4484): split z walks key tiles [t_begin, t_end) of the
                // block's range. `splits == 1` is the whole range.
                let tps = {
                    let s_m1 = ctx.sub_u32_reg(splits, one);
                    let n = ctx.add_u32_reg(tiles, s_m1);
                    ctx.div_u32_reg(n, splits)
                };
                let t_begin = ctx.mul_u32_reg(split, tps);
                let t_end = {
                    let e = ctx.add_u32_reg(t_begin, tps);
                    ctx.min_u32(e, tiles)
                };
                // Warp: its own rows' extent; 0 if the row tile is past `rows`.
                let w_end = ctx.add_u32(q0, BR - 1);
                let w_last = ctx.min_u32(w_end, rows_m1);
                let w_pos = ctx.add_u32_reg(pos0, w_last);
                let w_tile = ctx.div_u32(w_pos, BC);
                let w_tiles1 = ctx.add_u32(w_tile, 1);
                let has_rows = ctx.setp_lt_u32(q0, rows);
                let zero = ctx.mov_u32_imm(0);
                let warp_tiles = ctx.selp_u32(has_rows, w_tiles1, zero);

                let total_keys = ctx.add_u32_reg(pos0, rows);
                let last_key = ctx.sub_u32_reg(total_keys, one);
                let kv_col = ctx.mul_u32(kvh, FLASH_HEAD_DIM);
                let issue = Issue {
                    tid,
                    nthreads,
                    k_ptr,
                    v_ptr,
                    kv_stride,
                    kv_col,
                    last_key,
                };
                // This lane's query rows, absolute positions.
                let row_lo_abs = {
                    let a = ctx.add_u32_reg(pos0, q0);
                    ctx.add_u32_reg(a, g)
                };
                let row_hi_abs = ctx.add_u32(row_lo_abs, 8);
                let c_exp_r = ctx.mov_f32_imm(c_exp);
                let finite_floor = ctx.mov_f32_imm(-1.0e30);
                let t2 = ctx.mul_u32(t, 2);

                // Lane parts of the ldmatrix addresses (stage offset added per tile).
                // K as B ([key][d] row-major = col-major k×n): non-trans x4.
                let k_lane = {
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
                    ctx.mul_u32(el, 2)
                };
                let v_lane = {
                    let vrow = ctx.mul_u32(lane16, KV_ROW_H);
                    let vb = ctx.mul_u32(vrow, 2);
                    ctx.add_u32(vb, KV_TILE_BYTES)
                };

                // Prologue: tile t_begin in flight (none for an empty split).
                let nonempty = ctx.setp_lt_u32(t_begin, t_end);
                ctx.branch_if_not(nonempty, "fa_pro_none");
                issue_tile(ctx, &issue, t_begin, "pro");
                ctx.label("fa_pro_none");
                ctx.cp_async_commit_group();

                let kb = ctx.mov_u32_imm(0);
                ctx.mov_u32_reg(kb, t_begin);
                ctx.label("fa_key_loop");
                let more = ctx.setp_lt_u32(kb, t_end);
                ctx.branch_if_not(more, "fa_key_end");
                // Tile kb + 1 in flight while kb is consumed. Always commit (an empty
                // group on the last tile) so `wait_group 1` means "tile kb landed".
                let nxt = ctx.add_u32(kb, 1);
                let has_next = ctx.setp_lt_u32(nxt, t_end);
                ctx.branch_if_not(has_next, "fa_issue_none");
                issue_tile(ctx, &issue, nxt, "loop");
                ctx.label("fa_issue_none");
                ctx.cp_async_commit_group();
                ctx.cp_async_wait_group(1);
                ctx.bar_sync(0);

                let computes = ctx.setp_lt_u32(kb, warp_tiles);
                ctx.branch_if_not(computes, "fa_tile_skip");
                let j0 = ctx.mul_u32(kb, BC);
                let parity = ctx.and_u32(kb, one);
                let stage = ctx.mul_u32(parity, STAGE_BYTES);

                // S = Q Kᵀ over 16 k-steps; two n-tiles of 8 keys.
                let mut s0 = [fz, fz, fz, fz];
                let mut s1 = [fz, fz, fz, fz];
                {
                    let kbase = ctx.add_u32_reg(stage, k_lane);
                    for (kk, a) in qfrag.iter().enumerate() {
                        let addr = ctx.add_u32(kbase, kk as u32 * 16 * 2);
                        let bf = ctx.ldmatrix_x4(addr);
                        s0 = ctx.mma_sync_m16n8k16(a, &[bf[0], bf[1]], &s0);
                        s1 = ctx.mma_sync_m16n8k16(a, &[bf[2], bf[3]], &s1);
                    }
                }

                // Causal mask: key j0 + col is visible to row p iff key <= p.
                let mut sv = [s0, s1];
                for (nt, tile) in sv.iter_mut().enumerate() {
                    for e in 0..4usize {
                        let colbase = ctx.add_u32(t2, nt as u32 * 8 + (e as u32 & 1));
                        let key = ctx.add_u32_reg(j0, colbase);
                        // Fragment elements 0,1 sit on row g, elements 2,3 on row g + 8.
                        let p_abs = [row_lo_abs, row_hi_abs][e / 2];
                        let visible = ctx.setp_le_u32(key, p_abs);
                        tile[e] = ctx.selp_f32(visible, tile[e], neg_inf);
                    }
                }

                // Online softmax per row half (hr 0: row g, hr 1: row g + 8).
                let mut ps: Vec<Vec<VirtualReg>> = Vec::with_capacity(2);
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
                    let mut row_p = Vec::with_capacity(4);
                    for &v in &vals {
                        let d = ctx.sub_f32(v, m_use);
                        let ds = ctx.mul_f32(d, c_exp_r);
                        let pv = ctx.ex2_f32(ds);
                        psum = ctx.add_f32(psum, pv);
                        row_p.push(pv);
                    }
                    let rsum = quad_reduce(ctx, psum, &quad, false);
                    ctx.mul_f32_inplace(l[hr], alpha);
                    ctx.add_f32_inplace(l[hr], rsum);
                    ctx.mov_f32_reg(m[hr], m_new);
                    for tile in &o {
                        ctx.mul_f32_inplace(tile[2 * hr], alpha);
                        ctx.mul_f32_inplace(tile[2 * hr + 1], alpha);
                    }
                    ps.push(row_p);
                }

                // P -> A-fragment in registers. The accumulator holds (row g|g+8,
                // cols 2t, 2t+1) of each 8-key n-tile; the A-fragment wants a0 = (g,
                // 2t..), a1 = (g+8, 2t..), a2 = (g, 8+2t..), a3 = (g+8, 8+2t..), the
                // lower column in the lower half.
                let pfrag = [
                    ctx.cvt_rn_f16x2_f32(ps[0][1], ps[0][0]),
                    ctx.cvt_rn_f16x2_f32(ps[1][1], ps[1][0]),
                    ctx.cvt_rn_f16x2_f32(ps[0][3], ps[0][2]),
                    ctx.cvt_rn_f16x2_f32(ps[1][3], ps[1][2]),
                ];

                // O += P V: B = V (k=16 keys × n=8 dims), transposed ldmatrix.
                {
                    let vbase = ctx.add_u32_reg(stage, v_lane);
                    for (nt, tile) in o.iter().enumerate() {
                        let addr = ctx.add_u32(vbase, nt as u32 * 8 * 2);
                        let bf = ctx.ldmatrix_x2_trans(addr);
                        ctx.mma_sync_m16n8k16_inplace(&pfrag, &bf, tile);
                    }
                }
                ctx.label("fa_tile_skip");
                // The stage just read is the one tile kb + 2 lands on.
                ctx.bar_sync(0);

                ctx.add_u32_inplace(kb, 1);
                ctx.branch("fa_key_loop");
                ctx.label("fa_key_end");

                // ---- One split: O / l, written for the rows that exist.
                let two_splits = ctx.mov_u32_imm(2);
                let whole = ctx.setp_lt_u32(splits, two_splits);
                ctx.branch_if_not(whole, "fa_store_part");
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
                ctx.branch("fa_done");

                // ---- Split-KV: unnormalised O and (m, l) per row, merged by
                // `PrefillFlashCombine256Kernel`.
                ctx.label("fa_store_part");
                let part_o = ctx.load_param_u64("part_o_ptr");
                let part_ml = ctx.load_param_u64("part_ml_ptr");
                let split_rows = ctx.mul_u32_reg(split, rows);
                let is_t0 = ctx.setp_lt_u32(t, one);
                for hr in 0..2usize {
                    let row = ctx.add_u32(g, hr as u32 * 8);
                    let qrow = ctx.add_u32_reg(q0, row);
                    let skip = format!("fa_part_skip_{hr}");
                    let in_rows = ctx.setp_lt_u32(qrow, rows);
                    ctx.branch_if_not(in_rows, &skip);
                    let prow = ctx.add_u32_reg(split_rows, qrow);
                    let rowel = ctx.mul_u32(prow, q_stride);
                    let rowh = ctx.add_u32_reg(rowel, head_col);
                    let colt = ctx.add_u32_reg(rowh, t2);
                    for (nt, tile) in o.iter().enumerate() {
                        for e in 0..2usize {
                            let el = ctx.add_u32(colt, nt as u32 * 8 + e as u32);
                            let off = ctx.mul_wide_u32(el, 4);
                            let addr = ctx.add_u64(part_o, off);
                            ctx.st_global_f32(addr, tile[2 * hr + e]);
                        }
                    }
                    let ml_skip = format!("fa_ml_skip_{hr}");
                    ctx.branch_if_not(is_t0, &ml_skip);
                    let mrow = ctx.mul_u32(prow, self.num_heads);
                    let mh = ctx.add_u32_reg(mrow, head);
                    let off = ctx.mul_wide_u32(mh, 8);
                    let addr = ctx.add_u64(part_ml, off);
                    ctx.st_global_f32(addr, m[hr]);
                    let four_b = ctx.mov_u64_imm(4);
                    let addr_l = ctx.add_u64(addr, four_b);
                    ctx.st_global_f32(addr_l, l[hr]);
                    ctx.label(&ml_skip);
                    ctx.label(&skip);
                }
                ctx.label("fa_done");
                ctx.ret();
            })
    }
}

/// Most key splits one flash launch takes (#4484); sizes the partial scratch.
pub const FLASH_MAX_SPLITS: u32 = 16;
/// Fewest 16-key tiles a split walks: below this the combine costs more than the
/// parallelism it buys.
const FLASH_MIN_TILES_PER_SPLIT: u32 = 16;

impl PrefillFlashAttention256Kernel {
    /// Key splits for `rows` query rows at `pos0`: enough blocks to cover
    /// `2 × sm_count` (one 8-warp block fills an SM's registers), never more than
    /// [`FLASH_MAX_SPLITS`] nor below [`FLASH_MIN_TILES_PER_SPLIT`] tiles per split.
    /// Short chunks at short context stay at 1: the unsplit, bit-stable path.
    #[must_use]
    pub fn splits_for(&self, rows: u32, pos0: u32, sm_count: u32) -> u32 {
        let (gx, gy, _) = self.grid(rows);
        let base = (gx * gy).max(1);
        let want = (2 * sm_count.max(1)).div_ceil(base);
        let tiles = (pos0 + rows).div_ceil(BC);
        let by_tiles = (tiles / FLASH_MIN_TILES_PER_SPLIT).max(1);
        want.min(by_tiles).clamp(1, FLASH_MAX_SPLITS)
    }

    /// f32 floats of partial `O` and of partial `(m, l)` for `splits` splits.
    #[must_use]
    pub fn partial_floats(&self, rows: u32, splits: u32) -> (usize, usize) {
        let per = splits as usize * rows as usize;
        (
            per * (self.num_heads * FLASH_HEAD_DIM) as usize,
            per * self.num_heads as usize * 2,
        )
    }
}

/// Merges the split-KV partials of [`PrefillFlashAttention256Kernel`] (#4484):
/// `O = Σ_s O_s·2^((m_s − m*)·c) / Σ_s l_s·2^((m_s − m*)·c)`, `m* = max_s m_s`. A
/// split that saw no key for a row carries `m_s = −∞, l_s = 0, O_s = 0` and weighs 0;
/// split 0 always holds key 0, so `m*` is finite.
///
/// One block per (row, head), one thread per output dimension.
#[derive(Debug, Clone, Copy)]
pub struct PrefillFlashCombine256Kernel {
    num_heads: u32,
}

impl PrefillFlashCombine256Kernel {
    #[must_use]
    pub fn new(num_heads: u32) -> Self {
        Self { num_heads }
    }

    #[must_use]
    pub fn grid(&self, rows: u32) -> (u32, u32, u32) {
        (rows, self.num_heads, 1)
    }

    #[must_use]
    pub fn block(&self) -> (u32, u32, u32) {
        (FLASH_HEAD_DIM, 1, 1)
    }
}

impl Kernel for PrefillFlashCombine256Kernel {
    fn name(&self) -> &str {
        "gdn_prefill_flash_combine_256"
    }

    fn build_ptx(&self) -> PtxKernel {
        let q_stride = self.num_heads * FLASH_HEAD_DIM;
        let num_heads = self.num_heads;
        let c_exp = std::f32::consts::LOG2_E / (FLASH_HEAD_DIM as f32).sqrt();
        PtxKernel::new(self.name())
            .param(PtxType::U64, "part_o_ptr")
            .param(PtxType::U64, "part_ml_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "splits")
            .build(|ctx| {
                let d = ctx.special_reg(PtxReg::TidX);
                let row = ctx.special_reg(PtxReg::CtaIdX);
                let head = ctx.special_reg(PtxReg::CtaIdY);
                let part_o = ctx.load_param_u64("part_o_ptr");
                let part_ml = ctx.load_param_u64("part_ml_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let rows = ctx.load_param_u32("rows");
                let splits = ctx.load_param_u32("splits");
                let c_exp_r = ctx.mov_f32_imm(c_exp);
                let head_col = ctx.mul_u32(head, FLASH_HEAD_DIM);
                let col = ctx.add_u32_reg(head_col, d);

                // ml index of split s: ((s * rows + row) * H + head) * 2.
                let ml_addr = |ctx: &mut crate::ptx::builder::KernelBuilder<'_>, s: VirtualReg| {
                    let sr = ctx.mul_u32_reg(s, rows);
                    let pr = ctx.add_u32_reg(sr, row);
                    let ph = ctx.mul_u32(pr, num_heads);
                    let idx = ctx.add_u32_reg(ph, head);
                    let off = ctx.mul_wide_u32(idx, 8);
                    (ctx.add_u64(part_ml, off), pr)
                };

                let m_star = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let s = ctx.mov_u32_imm(0);
                ctx.label("fc_max_loop");
                let more = ctx.setp_lt_u32(s, splits);
                ctx.branch_if_not(more, "fc_max_end");
                let (a, _) = ml_addr(ctx, s);
                let ms = ctx.ld_global_f32(a);
                ctx.max_f32_inplace(m_star, ms);
                ctx.add_u32_inplace(s, 1);
                ctx.branch("fc_max_loop");
                ctx.label("fc_max_end");

                let acc = ctx.mov_f32_imm(0.0);
                let lsum = ctx.mov_f32_imm(0.0);
                let s2 = ctx.mov_u32_imm(0);
                ctx.label("fc_sum_loop");
                let more2 = ctx.setp_lt_u32(s2, splits);
                ctx.branch_if_not(more2, "fc_sum_end");
                let (a, pr) = ml_addr(ctx, s2);
                let ms = ctx.ld_global_f32(a);
                let four_b = ctx.mov_u64_imm(4);
                let al = ctx.add_u64(a, four_b);
                let ls = ctx.ld_global_f32(al);
                let dm = ctx.sub_f32(ms, m_star);
                let dms = ctx.mul_f32(dm, c_exp_r);
                let w = ctx.ex2_f32(dms);
                ctx.fma_f32_inplace(lsum, ls, w);
                let rowel = ctx.mul_u32(pr, q_stride);
                let el = ctx.add_u32_reg(rowel, col);
                let off = ctx.mul_wide_u32(el, 4);
                let oa = ctx.add_u64(part_o, off);
                let os = ctx.ld_global_f32(oa);
                ctx.fma_f32_inplace(acc, os, w);
                ctx.add_u32_inplace(s2, 1);
                ctx.branch("fc_sum_loop");
                ctx.label("fc_sum_end");

                let inv = ctx.rcp_f32(lsum);
                let v = ctx.mul_f32(acc, inv);
                let orow = ctx.mul_u32(row, q_stride);
                let oel = ctx.add_u32_reg(orow, col);
                let ooff = ctx.mul_wide_u32(oel, 4);
                let oaddr = ctx.add_u64(out_ptr, ooff);
                ctx.st_global_f32(oaddr, v);
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_flash_prefill_ptx_shape_and_shared_budget() {
        // (heads, kv, block threads, rows per block)
        for (h, kv, threads, rpb) in [
            (16u32, 4u32, 256u32, 32u32), // 0.8B–9B: 4 heads per KV, 2 row tiles
            (24, 4, 192, 16),             // 27B: 6 heads per KV, 1 row tile
            (8, 2, 256, 32),
            (4, 4, 128, 64), // MHA: 4 row tiles
            (32, 4, 256, 16),
        ] {
            let k = PrefillFlashAttention256Kernel::new(h, kv);
            assert_eq!(k.shared_bytes(), 33_792, "{h}/{kv}");
            assert_eq!(k.block(), (threads, 1, 1), "{h}/{kv}");
            assert_eq!(k.rows_per_block(), rpb, "{h}/{kv}");
            let ptx = k.emit_ptx();
            assert!(
                ptx.contains(".entry gdn_prefill_flash_attention_256"),
                "{ptx}"
            );
            assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
            assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
            assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
            // #4442 v2: K/V arrive by 16 B cp.async, P never touches shared memory.
            assert!(ptx.contains("cp.async.ca.shared.global"), "{h}/{kv}");
            assert!(ptx.contains("cp.async.wait_group 1"), "{h}/{kv}");
            assert_eq!(ptx.matches("cvt.rn.f16x2.f32").count(), 4, "{h}/{kv}");
            // The only shared stores left are the Q staging (4 quarters).
            assert_eq!(ptx.matches("st.shared.b16").count(), 4, "{h}/{kv}");
        }
        assert_eq!(
            PrefillFlashAttention256Kernel::new(16, 4).grid(37),
            (2, 4, 1)
        );
        // #4484 split-KV sizing on a 128-SM 4090, Qwen3.5-4B (64 base blocks).
        let k4 = PrefillFlashAttention256Kernel::new(16, 4);
        assert_eq!(
            k4.splits_for(512, 0, 128),
            2,
            "512 keys: 32 tiles, 2 splits"
        );
        assert_eq!(k4.splits_for(512, 30_000, 128), 4, "2*128 / 64 blocks");
        assert_eq!(k4.splits_for(16, 0, 128), 1, "one tile: unsplit");
        assert_eq!(
            k4.splits_for(16, 30_000, 128),
            16,
            "capped at FLASH_MAX_SPLITS"
        );
        assert_eq!(
            k4.partial_floats(512, 4),
            (4 * 512 * 4096, 4 * 512 * 16 * 2)
        );
        let comb = PrefillFlashCombine256Kernel::new(16).emit_ptx();
        assert!(
            comb.contains(".entry gdn_prefill_flash_combine_256"),
            "{comb}"
        );
        assert!(PrefillFlashAttention256Kernel::fits(24, 4));
        assert!(PrefillFlashAttention256Kernel::fits(32, 4));
        assert!(!PrefillFlashAttention256Kernel::fits(36, 4));
        assert!(!PrefillFlashAttention256Kernel::fits(16, 3));
    }

    #[test]
    #[should_panic(expected = "static shared memory")]
    fn gdn_flash_prefill_refuses_more_than_8_heads_per_kv() {
        let _ = PrefillFlashAttention256Kernel::new(36, 4); // 9 heads per KV head
    }
}

/// Device parity: the kernel against an exact reference of what it computes (f16
/// inputs, f32/f64 math), and against the full-f32 reference within the f16 budget.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_flash_prefill_device_tests {
    use super::{PrefillFlashAttention256Kernel, PrefillFlashCombine256Kernel, FLASH_HEAD_DIM};
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

    /// The IEEE binary16 bits of `round_f16(x)` — what `cvt.rn.f16.f32` stores.
    fn f16_bits(x: f32) -> u16 {
        let r = round_f16(x);
        let sign = if r.is_sign_negative() { 0x8000u16 } else { 0 };
        let a = r.abs();
        if a == 0.0 {
            return sign;
        }
        if a < 2f32.powi(-14) {
            return sign | (a / 2f32.powi(-24)) as u16;
        }
        let e = a.log2().floor() as i32;
        let mant = ((a / 2f32.powi(e) - 1.0) * 1024.0) as u16;
        sign | (((e + 15) as u16) << 10) | mant
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
        run_split(heads, kv_heads, rows, pos0, 1);
    }

    /// `splits > 1` runs the split-KV path (#4484): partials, then the combine kernel.
    fn run_split(heads: u32, kv_heads: u32, rows: usize, pos0: usize, splits: u32) {
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
        // K/V enter as f16, as the executor's one-pass conversion leaves them.
        let k16: Vec<u16> = k.iter().map(|&x| f16_bits(x)).collect();
        let v16: Vec<u16> = v.iter().map(|&x| f16_bits(x)).collect();
        let kb = GpuBuffer::from_host(&ctx, &k16).expect("k");
        let vb = GpuBuffer::from_host(&ctx, &v16).expect("v");
        let ob = GpuBuffer::from_host(&ctx, &vec![f32::NAN; rows * h * d]).expect("o");
        let mut module =
            CudaModule::from_ptx(&ctx, &kern.emit_ptx_for_target("sm_80")).expect("module");
        let (gx, gy, _) = kern.grid(rows as u32);
        let config = LaunchConfig {
            grid: (gx, gy, splits),
            block: kern.block(),
            shared_mem: 0,
        };
        let (po, pml) = kern.partial_floats(rows as u32, splits);
        let pob = GpuBuffer::from_host(&ctx, &vec![f32::NAN; po]).expect("part o");
        let pmb = GpuBuffer::from_host(&ctx, &vec![f32::NAN; pml]).expect("part ml");
        let mut args = [
            qb.as_ptr(),
            kb.as_ptr(),
            vb.as_ptr(),
            ob.as_ptr(),
            pob.as_ptr(),
            pmb.as_ptr(),
            rows as u64,
            pos0 as u64,
            u64::from(splits),
        ];
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: q/o are f32 rows × heads × 256, k/v f16 × (pos0 + rows) keys, the
        // partials are sized by `partial_floats`; the u32 scalars sit in the low half
        // of their slots.
        unsafe {
            stream
                .launch_kernel(&mut module, kern.name(), &config, &mut raw)
                .expect("launch");
        }
        if splits > 1 {
            let comb = PrefillFlashCombine256Kernel::new(heads);
            let mut cmod =
                CudaModule::from_ptx(&ctx, &comb.emit_ptx_for_target("sm_80")).expect("module");
            let cconfig = LaunchConfig {
                grid: comb.grid(rows as u32),
                block: comb.block(),
                shared_mem: 0,
            };
            let mut cargs = [
                pob.as_ptr(),
                pmb.as_ptr(),
                ob.as_ptr(),
                rows as u64,
                u64::from(splits),
            ];
            let mut craw: Vec<*mut std::ffi::c_void> = cargs
                .iter_mut()
                .map(|a| std::ptr::from_mut(a).cast())
                .collect();
            // SAFETY: the partials the launch above wrote, and `o` as before.
            unsafe {
                stream
                    .launch_kernel(&mut cmod, comb.name(), &cconfig, &mut craw)
                    .expect("combine launch");
            }
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
            "[3596] flash prefill heads {heads}/{kv_heads} rows {rows} pos0 {pos0} splits {splits}: rel L∞ vs f16-input \
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

    #[test]
    fn gdn_flash_prefill_matches_reference_mha_four_row_tiles() {
        // One head per KV head: 4 row tiles per block, 70 rows (the last block's
        // later row tiles have no rows and still take part in every barrier).
        run(4, 4, 70, 10);
    }

    #[test]
    fn gdn_flash_prefill_split_kv_matches_reference() {
        // #4484: 4 splits over 38 key tiles.
        run_split(16, 4, 100, 500, 4);
    }

    #[test]
    fn gdn_flash_prefill_split_kv_with_empty_splits() {
        // Block 0 has 4 key tiles, so splits 4..8 walk nothing and must weigh 0;
        // the partial last block still merges.
        run_split(16, 4, 37, 29, 8);
        run_split(4, 4, 70, 10, 3);
    }

    #[test]
    fn f16_bits_matches_known_encodings() {
        for (x, bits) in [
            (1.0f32, 0x3C00u16),
            (-2.0, 0xC000),
            (0.5, 0x3800),
            (65504.0, 0x7BFF),
            (2f32.powi(-24), 0x0001),
            (1.0 + 1.0 / 1024.0, 0x3C01),
        ] {
            assert_eq!(f16_bits(x), bits, "{x}");
        }
    }
}
