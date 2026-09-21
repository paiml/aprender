//! PMAT-3725 / aprender#3725: split-K (flash-decoding) single-query decode attention
//! for `head_dim = 256`, over an f32 **or f16** KV cache.
//!
//! The same computation as [`super::DecodeAttention256Kernel`] — the attention half of
//! `forward_attention` in the CPU reference (forward_qwen35.rs) — laid out so that a
//! long context is spread over the whole device instead of `num_heads` blocks:
//!
//! ```text
//! for kv_h in 0..num_kv_heads, for split s:           <- kernel A, one block each
//!     for every query head h of kv_h's GQA group:
//!         m_s, l_s, acc_s = online softmax over positions [s*chunk, min((s+1)*chunk, L))
//! for h in 0..num_heads:                               <- kernel B, one block each
//!     M   = max_s m_s
//!     out = sum_s exp(m_s - M) * acc_s  /  sum_s exp(m_s - M) * l_s
//! ```
//!
//! ## Why the old kernel is slow at long context (aprender#3725)
//!
//! `DecodeAttention256Kernel` launches one block per query head (16 on the 9B), each
//! walking every cached position alone; thread `t` scores position `t` by reading its
//! whole row one float at a time, so neighbouring threads load addresses one KV row
//! (4 KiB on the 9B) apart; and each KV head is read once per query head of its group.
//!
//! ## Shape
//!
//! Kernel A ([`DecodeAttentionSplitKKernel`]): grid `(num_kv_heads, n_splits)`, block
//! 256 = 8 warps. A block serves **all** `group_size` query heads of one KV head, so
//! K/V are read once per token. Warp `w` takes positions `start + w, start + w + 8, …`
//! of its split; lane `l` holds row elements `l, l + 32, …`, so each load instruction
//! reads one contiguous span (128 B of f32 or 64 B of f16). A warp keeps one online
//! softmax per query head in registers; the 8 warps are merged through shared memory
//! and the block writes one un-normalised partial `(m, l, acc[head_dim])` per
//! (query head, split).
//!
//! Kernel B ([`DecodeAttentionSplitKReduceKernel`]): grid `num_heads`, block 256, one
//! thread per output element, merging the `n_splits` partials with the log-sum-exp
//! rescale above.
//!
//! `seq_len`, `chunk` and `n_splits` are runtime parameters (see [`SplitKPlan`]), so
//! one compiled module pair serves the whole decode.
//!
//! ## Numerics
//!
//! All arithmetic is fp32, whatever the storage; f16 storage is widened with
//! `cvt.f32.f16` on load. The score divides by `sqrt(head_dim)` as the CPU does.
//! `exp` is `ex2.approx` after a `log2(e)` scale ([`super::emit_exp_f32`]). Summation
//! order differs from the CPU's position-ascending loop, so parity is a tolerance,
//! never bitwise — the CPU twin [`decode_attention_splitk_cpu`] reproduces the kernel's
//! own split / warp / merge order, and the device tests hold both kernels to it.

use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{KernelBuilder, PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// Threads per block, both kernels.
const BLOCK: u32 = 256;
/// Warps per kernel-A block — the stride between one warp's positions.
pub const SPLITK_WARPS: u32 = BLOCK / 32;
/// A split never holds fewer positions than this (unless the whole context is
/// shorter): below it the per-block merge costs more than the positions it covers.
pub const SPLITK_MIN_CHUNK: u32 = 64;
/// Splits per KV head the planner aims for when the context is long enough.
pub const SPLITK_DEFAULT_TARGET_SPLITS: u32 = 256;

/// How the KV cache stores one element. The layout is identical either way:
/// `[max_len][num_kv_heads * head_dim]`, row-major, one row per position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvStorage {
    /// `f32`, 4 bytes per element.
    F32,
    /// IEEE half, 2 bytes per element; widened to f32 on load.
    F16,
}

impl KvStorage {
    /// Bytes per stored element.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    /// Short tag used in kernel names and cache keys.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// The split of `seq_len` positions into `n_splits` chunks of `chunk` positions.
///
/// Every split holds at least one position (`(n_splits - 1) * chunk < seq_len`), so
/// no partial is ever an empty softmax, and `n_splits <= target_splits` always.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKPlan {
    /// Positions per split (the last may hold fewer).
    pub chunk: u32,
    /// Number of splits — kernel A's grid `y`, and kernel B's loop bound.
    pub n_splits: u32,
}

impl SplitKPlan {
    /// Plan `seq_len` positions: `chunk = max(min_chunk, ceil(seq_len / target))`.
    ///
    /// # Panics
    /// If `seq_len`, `target_splits` or `min_chunk` is zero.
    #[must_use]
    pub fn new(seq_len: u32, target_splits: u32, min_chunk: u32) -> Self {
        assert!(seq_len > 0, "an empty KV cache has no softmax to split");
        assert!(target_splits > 0 && min_chunk > 0, "degenerate split plan");
        let chunk = seq_len.div_ceil(target_splits).max(min_chunk);
        Self {
            chunk,
            n_splits: seq_len.div_ceil(chunk),
        }
    }

    /// The default plan: [`SPLITK_DEFAULT_TARGET_SPLITS`], [`SPLITK_MIN_CHUNK`].
    #[must_use]
    pub fn for_seq_len(seq_len: u32) -> Self {
        Self::new(seq_len, SPLITK_DEFAULT_TARGET_SPLITS, SPLITK_MIN_CHUNK)
    }
}

/// Floats in the `partial_acc` scratch for `max_splits` splits:
/// `[num_heads][max_splits][head_dim]`.
#[must_use]
pub const fn splitk_partial_acc_len(num_heads: u32, head_dim: u32, max_splits: u32) -> usize {
    num_heads as usize * max_splits as usize * head_dim as usize
}

/// Floats in the `partial_ml` scratch for `max_splits` splits:
/// `[num_heads][max_splits][2]` (running max `m`, then the sum `l`).
#[must_use]
pub const fn splitk_partial_ml_len(num_heads: u32, max_splits: u32) -> usize {
    num_heads as usize * max_splits as usize * 2
}

fn check_shape(num_heads: u32, num_kv_heads: u32, head_dim: u32) {
    assert!(
        head_dim > 0 && head_dim <= BLOCK && head_dim % 32 == 0,
        "head_dim {head_dim} must be a multiple of 32 and at most {BLOCK}: a lane owns \
         head_dim / 32 elements and the reduce kernel one thread per element"
    );
    assert!(
        num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}"
    );
}

/// Kernel A: per-(KV head, split) online softmax over a chunk of positions, writing
/// one un-normalised partial per (query head, split).
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttentionSplitKKernel {
    /// Query heads (`config.num_heads`).
    pub num_heads: u32,
    /// Key/value heads; divides `num_heads`.
    pub num_kv_heads: u32,
    /// Width of one head; a multiple of 32, at most 256.
    pub head_dim: u32,
    /// How the cache stores K and V.
    pub kv: KvStorage,
}

impl DecodeAttentionSplitKKernel {
    /// # Panics
    /// If `head_dim` is not a multiple of 32 in `32..=256`, or `num_kv_heads` does not
    /// divide `num_heads` — both would silently produce wrong output.
    #[must_use]
    pub fn new(num_heads: u32, num_kv_heads: u32, head_dim: u32, kv: KvStorage) -> Self {
        check_shape(num_heads, num_kv_heads, head_dim);
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            kv,
        }
    }

    /// GQA group size — the query heads one block serves.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }

    /// Static shared memory: `[WARPS][2]` running `(m, l)` then `[WARPS][head_dim]` acc.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        (SPLITK_WARPS * 2 * 4 + SPLITK_WARPS * self.head_dim * 4) as usize
    }

    /// Launch grid for a plan: one block per (KV head, split).
    #[must_use]
    pub const fn grid(&self, plan: SplitKPlan) -> (u32, u32, u32) {
        (self.num_kv_heads, plan.n_splits, 1)
    }

    /// Launch block — 256 threads.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (BLOCK, 1, 1)
    }
}

/// Load one stored KV element at `addr` as f32.
fn emit_load_kv(ctx: &mut KernelBuilder<'_>, kv: KvStorage, addr: VirtualReg) -> VirtualReg {
    match kv {
        KvStorage::F32 => ctx.ld_global_f32(addr),
        KvStorage::F16 => {
            let h = ctx.ld_global_f16(addr);
            ctx.cvt_f32_f16(h)
        }
    }
}

/// Warp sum, broadcast: a `shfl.down` tree into lane 0, then `shfl.idx` from lane 0,
/// so every lane ends with the same full sum.
///
/// Not a `shfl.bfly` butterfly: the builder's `shfl_xor_f32` emits the opcode as
/// `shflbfly`, which ptxas rejects (found here; no kernel had ever loaded it).
fn emit_warp_sum(ctx: &mut KernelBuilder<'_>, val: VirtualReg) -> VirtualReg {
    let mut v = val;
    for offset in [16u32, 8, 4, 2, 1] {
        let other = ctx.shfl_down_f32(v, offset, 0xFFFF_FFFF);
        v = ctx.add_f32(v, other);
    }
    ctx.shfl_idx_f32(v, 0, 0xFFFF_FFFF)
}

impl Kernel for DecodeAttentionSplitKKernel {
    fn name(&self) -> &str {
        match self.kv {
            KvStorage::F32 => "gdn_decode_attention_splitk_f32",
            KvStorage::F16 => "gdn_decode_attention_splitk_f16",
        }
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let per_lane = head_dim / 32;
        let group = self.group_size();
        let bytes = self.kv.bytes();
        let row_stride_bytes = self.num_kv_heads * head_dim * bytes;
        let sqrt_head_dim = (head_dim as f32).sqrt();
        let acc_base = SPLITK_WARPS * 2 * 4;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_heads * head_dim] f32
            .param(PtxType::U64, "k_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "v_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "partial_acc_ptr") // [num_heads][n_splits][head_dim]
            .param(PtxType::U64, "partial_ml_ptr") // [num_heads][n_splits][2]
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "chunk")
            .param(PtxType::U32, "n_splits")
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let kv_h = ctx.special_reg(PtxReg::CtaIdX);
                let split = ctx.special_reg(PtxReg::CtaIdY);
                let lane = ctx.and_u32_imm(tid, 31);
                let warp = ctx.shr_u32_imm(tid, 5);

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_cache_ptr");
                let v_ptr = ctx.load_param_u64("v_cache_ptr");
                let pacc_ptr = ctx.load_param_u64("partial_acc_ptr");
                let pml_ptr = ctx.load_param_u64("partial_ml_ptr");
                let seq_len = ctx.load_param_u32("seq_len");
                let chunk = ctx.load_param_u32("chunk");
                let n_splits = ctx.load_param_u32("n_splits");

                // [start, end) of this split. The planner never launches an empty one,
                // but a split past the end must still not read past the cache.
                let start = ctx.mul_u32_reg(split, chunk);
                let has_work = ctx.setp_lt_u32(start, seq_len);
                ctx.branch_if_not(has_work, "splitk_exit");
                let remaining = ctx.sub_u32_reg(seq_len, start);
                let span = ctx.min_u32(remaining, chunk);
                let end = ctx.add_u32_reg(start, span);

                // This lane's first element in the KV head's slice of a row.
                let kv_head_off = ctx.mul_wide_u32(kv_h, head_dim * bytes);
                let lane_off = ctx.mul_wide_u32(lane, bytes);
                let k_head = ctx.add_u64(k_ptr, kv_head_off);
                let v_head = ctx.add_u64(v_ptr, kv_head_off);
                let k_lane = ctx.add_u64(k_head, lane_off);
                let v_lane = ctx.add_u64(v_head, lane_off);
                let elem_stride: Vec<VirtualReg> = (0..per_lane)
                    .map(|e| ctx.mov_u64_imm(u64::from(e * 32 * bytes)))
                    .collect();

                // q for every query head of the group, in the same lane layout.
                let first_head = ctx.mul_u32(kv_h, group);
                let q_lane_off = ctx.mul_wide_u32(lane, 4);
                let mut q_regs: Vec<Vec<VirtualReg>> = Vec::with_capacity(group as usize);
                for g in 0..group {
                    let h = ctx.add_u32(first_head, g);
                    let head_off = ctx.mul_wide_u32(h, head_dim * 4);
                    let q_head = ctx.add_u64(q_ptr, head_off);
                    let q_lane = ctx.add_u64(q_head, q_lane_off);
                    let mut row = Vec::with_capacity(per_lane as usize);
                    for e in 0..per_lane {
                        let off = ctx.mov_u64_imm(u64::from(e * 32 * 4));
                        let addr = ctx.add_u64(q_lane, off);
                        row.push(ctx.ld_global_f32(addr));
                    }
                    q_regs.push(row);
                }

                // Running online-softmax state per query head, in registers.
                let mut m_regs = Vec::with_capacity(group as usize);
                let mut l_regs = Vec::with_capacity(group as usize);
                let mut acc_regs: Vec<Vec<VirtualReg>> = Vec::with_capacity(group as usize);
                for _ in 0..group {
                    m_regs.push(ctx.mov_f32_imm(f32::NEG_INFINITY));
                    l_regs.push(ctx.mov_f32_imm(0.0));
                    acc_regs.push((0..per_lane).map(|_| ctx.mov_f32_imm(0.0)).collect());
                }
                let sqrt_hd = ctx.mov_f32_imm(sqrt_head_dim);

                // ---- positions start + warp, + WARPS, ... (warp-uniform loop)
                let p = ctx.add_u32_reg(start, warp);
                ctx.label("splitk_pos_loop");
                let pos_go = ctx.setp_lt_u32(p, end);
                ctx.branch_if_not(pos_go, "splitk_pos_end");
                let row_off = ctx.mul_wide_u32(p, row_stride_bytes);
                let k_row = ctx.add_u64(k_lane, row_off);
                let v_row = ctx.add_u64(v_lane, row_off);
                let mut k_vals = Vec::with_capacity(per_lane as usize);
                let mut v_vals = Vec::with_capacity(per_lane as usize);
                for &stride in &elem_stride {
                    let ka = ctx.add_u64(k_row, stride);
                    k_vals.push(emit_load_kv(ctx, self.kv, ka));
                }
                for &stride in &elem_stride {
                    let va = ctx.add_u64(v_row, stride);
                    v_vals.push(emit_load_kv(ctx, self.kv, va));
                }

                for g in 0..group as usize {
                    // score = dot(q_h, k_p) / sqrt(head_dim)
                    let partial = ctx.mul_f32(q_regs[g][0], k_vals[0]);
                    for e in 1..per_lane as usize {
                        ctx.fma_f32_inplace(partial, q_regs[g][e], k_vals[e]);
                    }
                    let dot = emit_warp_sum(ctx, partial);
                    let score = ctx.div_f32(dot, sqrt_hd);

                    // online softmax: m' = max(m, s); l = l*e^(m-m') + e^(s-m')
                    let new_m = ctx.max_f32(m_regs[g], score);
                    let dm = ctx.sub_f32(m_regs[g], new_m);
                    let correction = emit_exp_f32(ctx, dm);
                    let ds = ctx.sub_f32(score, new_m);
                    let weight = emit_exp_f32(ctx, ds);
                    ctx.mul_f32_inplace(l_regs[g], correction);
                    ctx.add_f32_inplace(l_regs[g], weight);
                    for e in 0..per_lane as usize {
                        ctx.mul_f32_inplace(acc_regs[g][e], correction);
                        ctx.fma_f32_inplace(acc_regs[g][e], weight, v_vals[e]);
                    }
                    ctx.mov_f32_reg(m_regs[g], new_m);
                }

                ctx.add_u32_inplace(p, SPLITK_WARPS);
                ctx.branch("splitk_pos_loop");
                ctx.label("splitk_pos_end");

                // ---- merge the 8 warps, one query head at a time.
                let zero = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane, zero);
                let warp_ml = ctx.mul_u32(warp, 8);
                let warp_ml64 = ctx.cvt_u64_u32(warp_ml);
                let four64 = ctx.mov_u64_imm(4);
                let warp_l64 = ctx.add_u64(warp_ml64, four64);
                let warp_acc = ctx.mul_u32(warp, head_dim * 4);
                let lane_acc = ctx.mul_u32(lane, 4);
                let warp_lane_acc = ctx.add_u32_reg(warp_acc, lane_acc);
                let warp_lane_acc = ctx.add_u32(warp_lane_acc, acc_base);
                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let in_head = ctx.setp_lt_u32(tid, head_dim_r);
                let tid_acc = ctx.mul_u32(tid, 4);
                let tid_acc = ctx.add_u32(tid_acc, acc_base);
                let tid_bytes = ctx.mul_wide_u32(tid, 4);

                for g in 0..group as usize {
                    let skip_ml = format!("splitk_skip_ml_{g}");
                    ctx.branch_if_not(is_lane0, &skip_ml);
                    ctx.st_shared_f32(warp_ml64, m_regs[g]);
                    ctx.st_shared_f32(warp_l64, l_regs[g]);
                    ctx.label(&skip_ml);
                    for e in 0..per_lane {
                        let slot = ctx.add_u32(warp_lane_acc, e * 32 * 4);
                        let slot64 = ctx.cvt_u64_u32(slot);
                        ctx.st_shared_f32(slot64, acc_regs[g][e as usize]);
                    }
                    ctx.bar_sync(0);

                    // Every thread folds the WARPS maxima itself (no broadcast slot).
                    let mut warp_m = Vec::with_capacity(SPLITK_WARPS as usize);
                    let mut block_m: Option<VirtualReg> = None;
                    for w in 0..SPLITK_WARPS {
                        let off = ctx.mov_u64_imm(u64::from(w * 8));
                        let mw = ctx.ld_shared_f32(off);
                        block_m = Some(match block_m {
                            None => mw,
                            Some(a) => ctx.max_f32(a, mw),
                        });
                        warp_m.push(mw);
                    }
                    let block_m = block_m.expect("WARPS > 0");

                    // A warp that saw no position has m = -inf, l = 0, acc = 0, so its
                    // scale e^(-inf - M) is 0 — M itself is finite, because warp 0
                    // always holds the split's first position.
                    let skip_out = format!("splitk_skip_out_{g}");
                    ctx.branch_if_not(in_head, &skip_out);
                    let block_l = ctx.mov_f32_imm(0.0);
                    let block_acc = ctx.mov_f32_imm(0.0);
                    for (w, &mw) in warp_m.iter().enumerate() {
                        let w = w as u32;
                        let dm = ctx.sub_f32(mw, block_m);
                        let scale = emit_exp_f32(ctx, dm);
                        let l_off = ctx.mov_u64_imm(u64::from(w * 8 + 4));
                        let lw = ctx.ld_shared_f32(l_off);
                        ctx.fma_f32_inplace(block_l, lw, scale);
                        let a_off = ctx.add_u32(tid_acc, w * head_dim * 4);
                        let a_off64 = ctx.cvt_u64_u32(a_off);
                        let aw = ctx.ld_shared_f32(a_off64);
                        ctx.fma_f32_inplace(block_acc, aw, scale);
                    }

                    // partial_acc[h][split][tid], partial_ml[h][split] = (m, l)
                    let h = ctx.add_u32(first_head, g as u32);
                    let h_split = ctx.mul_u32_reg(h, n_splits);
                    let slot = ctx.add_u32_reg(h_split, split);
                    let acc_row = ctx.mul_wide_u32(slot, head_dim * 4);
                    let acc_row_ptr = ctx.add_u64(pacc_ptr, acc_row);
                    let acc_addr = ctx.add_u64(acc_row_ptr, tid_bytes);
                    ctx.st_global_f32(acc_addr, block_acc);
                    let is_tid0 = ctx.setp_eq_u32(tid, zero);
                    ctx.branch_if_not(is_tid0, &skip_out);
                    let ml_off = ctx.mul_wide_u32(slot, 8);
                    let ml_addr = ctx.add_u64(pml_ptr, ml_off);
                    ctx.st_global_f32(ml_addr, block_m);
                    let l_addr = ctx.add_u64(ml_addr, four64);
                    ctx.st_global_f32(l_addr, block_l);
                    ctx.label(&skip_out);
                    // The next head overwrites the shared rows.
                    ctx.bar_sync(0);
                }

                ctx.label("splitk_exit");
                ctx.ret();
            })
    }
}

/// Kernel B: merge `n_splits` partials per query head into the attention output.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttentionSplitKReduceKernel {
    /// Query heads.
    pub num_heads: u32,
    /// Width of one head; at most 256 (one thread per output element).
    pub head_dim: u32,
}

impl DecodeAttentionSplitKReduceKernel {
    /// # Panics
    /// If `head_dim` exceeds the 256-thread block or is not a multiple of 32.
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32) -> Self {
        check_shape(num_heads, 1, head_dim);
        Self {
            num_heads,
            head_dim,
        }
    }

    /// Launch grid — one block per query head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads, 1, 1)
    }

    /// Launch block — 256 threads.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (BLOCK, 1, 1)
    }
}

impl Kernel for DecodeAttentionSplitKReduceKernel {
    fn name(&self) -> &str {
        "gdn_decode_attention_splitk_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        PtxKernel::new(self.name())
            .param(PtxType::U64, "partial_acc_ptr") // [num_heads][n_splits][head_dim]
            .param(PtxType::U64, "partial_ml_ptr") // [num_heads][n_splits][2]
            .param(PtxType::U64, "out_ptr") // [num_heads * head_dim]
            .param(PtxType::U32, "n_splits")
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let pacc_ptr = ctx.load_param_u64("partial_acc_ptr");
                let pml_ptr = ctx.load_param_u64("partial_ml_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let n_splits = ctx.load_param_u32("n_splits");

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let in_head = ctx.setp_lt_u32(tid, head_dim_r);
                ctx.branch_if_not(in_head, "splitk_reduce_exit");

                let h_first = ctx.mul_u32_reg(h, n_splits);
                let ml_head_off = ctx.mul_wide_u32(h_first, 8);
                let ml_head = ctx.add_u64(pml_ptr, ml_head_off);
                let acc_head_off = ctx.mul_wide_u32(h_first, head_dim * 4);
                let acc_head = ctx.add_u64(pacc_ptr, acc_head_off);
                let tid_bytes = ctx.mul_wide_u32(tid, 4);
                let acc_lane = ctx.add_u64(acc_head, tid_bytes);
                let four64 = ctx.mov_u64_imm(4);

                // M = max_s m_s
                let big_m = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let s = ctx.mov_u32_imm(0);
                ctx.label("splitk_reduce_max_loop");
                let max_go = ctx.setp_lt_u32(s, n_splits);
                ctx.branch_if_not(max_go, "splitk_reduce_max_end");
                let m_off = ctx.mul_wide_u32(s, 8);
                let m_addr = ctx.add_u64(ml_head, m_off);
                let ms = ctx.ld_global_f32(m_addr);
                ctx.max_f32_inplace(big_m, ms);
                ctx.add_u32_inplace(s, 1);
                ctx.branch("splitk_reduce_max_loop");
                ctx.label("splitk_reduce_max_end");

                // num = sum_s e^(m_s - M) acc_s, den = sum_s e^(m_s - M) l_s
                let num = ctx.mov_f32_imm(0.0);
                let den = ctx.mov_f32_imm(0.0);
                let t = ctx.mov_u32_imm(0);
                ctx.label("splitk_reduce_sum_loop");
                let sum_go = ctx.setp_lt_u32(t, n_splits);
                ctx.branch_if_not(sum_go, "splitk_reduce_sum_end");
                let ml_off = ctx.mul_wide_u32(t, 8);
                let m_addr = ctx.add_u64(ml_head, ml_off);
                let mt = ctx.ld_global_f32(m_addr);
                let l_addr = ctx.add_u64(m_addr, four64);
                let lt = ctx.ld_global_f32(l_addr);
                let dm = ctx.sub_f32(mt, big_m);
                let scale = emit_exp_f32(ctx, dm);
                let a_off = ctx.mul_wide_u32(t, head_dim * 4);
                let a_addr = ctx.add_u64(acc_lane, a_off);
                let at = ctx.ld_global_f32(a_addr);
                ctx.fma_f32_inplace(num, at, scale);
                ctx.fma_f32_inplace(den, lt, scale);
                ctx.add_u32_inplace(t, 1);
                ctx.branch("splitk_reduce_sum_loop");
                ctx.label("splitk_reduce_sum_end");

                let result = ctx.div_f32(num, den);
                let out_head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let out_head = ctx.add_u64(out_ptr, out_head_off);
                let out_addr = ctx.add_u64(out_head, tid_bytes);
                ctx.st_global_f32(out_addr, result);

                ctx.label("splitk_reduce_exit");
                ctx.ret();
            })
    }
}

/// One online-softmax partial: running max `m`, running sum `l`, and the
/// un-normalised accumulator — what a warp holds in registers and what kernel A
/// writes per (query head, split).
#[derive(Debug, Clone)]
struct Partial {
    m: f32,
    l: f32,
    acc: Vec<f32>,
}

impl Partial {
    fn empty(head_dim: usize) -> Self {
        Self {
            m: f32::NEG_INFINITY,
            l: 0.0,
            acc: vec![0.0; head_dim],
        }
    }

    /// Fold one scored position in — kernel A's per-position update.
    fn push(&mut self, score: f32, v: &[f32]) {
        let new_m = self.m.max(score);
        let correction = (self.m - new_m).exp();
        let weight = (score - new_m).exp();
        self.l = self.l * correction + weight;
        for (a, x) in self.acc.iter_mut().zip(v) {
            *a = *a * correction + weight * x;
        }
        self.m = new_m;
    }

    /// The log-sum-exp merge, parts in order — kernel A's merge of its 8 warps and
    /// kernel B's merge of the splits are this same arithmetic.
    fn merge(parts: &[Self], head_dim: usize) -> Self {
        let m = parts.iter().fold(f32::NEG_INFINITY, |m, p| m.max(p.m));
        let mut merged = Self {
            m,
            l: 0.0,
            acc: vec![0.0; head_dim],
        };
        for part in parts {
            let scale = (part.m - m).exp();
            merged.l += part.l * scale;
            for (a, x) in merged.acc.iter_mut().zip(&part.acc) {
                *a += x * scale;
            }
        }
        merged
    }
}

/// The partial one warp computes: query head `q_h` over `positions` of KV head `kv_h`.
fn warp_partial(
    q_h: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    row: usize,
    kv_h: usize,
    positions: impl Iterator<Item = usize>,
) -> Partial {
    let head_dim = q_h.len();
    let sqrt_hd = (head_dim as f32).sqrt();
    let mut part = Partial::empty(head_dim);
    for p in positions {
        let base = p * row + kv_h * head_dim;
        let dot: f32 = q_h
            .iter()
            .zip(&k_cache[base..base + head_dim])
            .map(|(a, b)| a * b)
            .sum();
        part.push(dot / sqrt_hd, &v_cache[base..base + head_dim]);
    }
    part
}

/// CPU twin of the kernel pair: the same split, warp and merge order, in f32.
///
/// `k_cache` and `v_cache` are `[>= seq_len][num_kv_heads * head_dim]` f32 — for an f16
/// cache pass the widened values. Uses `f32::exp` where the device uses
/// `ex2.approx`, so the twin and the device agree to a tolerance, not bitwise.
///
/// # Panics
/// On the same shapes [`DecodeAttentionSplitKKernel::new`] refuses, or if a slice is
/// shorter than the shapes require.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_splitk_cpu(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    plan: SplitKPlan,
) -> Vec<f32> {
    check_shape(num_heads as u32, num_kv_heads as u32, head_dim as u32);
    assert!(seq_len > 0, "an empty KV cache has no softmax");
    let row = num_kv_heads * head_dim;
    assert!(q.len() >= num_heads * head_dim, "q too short");
    assert!(
        k_cache.len() >= seq_len * row && v_cache.len() >= seq_len * row,
        "cache too short"
    );
    let group = num_heads / num_kv_heads;
    let warps = SPLITK_WARPS as usize;
    let chunk = plan.chunk as usize;
    let n_splits = plan.n_splits as usize;

    let mut out = vec![0.0f32; num_heads * head_dim];
    for (h, out_h) in out.chunks_exact_mut(head_dim).enumerate() {
        let kv_h = h / group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        // kernel A: one partial per split, itself the merge of its 8 warps
        let splits: Vec<Partial> = (0..n_splits)
            .map(|s| {
                let start = (s * chunk).min(seq_len);
                let end = (start + chunk).min(seq_len);
                let per_warp: Vec<Partial> = (0..warps)
                    .map(|w| {
                        let positions = (start + w..end).step_by(warps);
                        warp_partial(q_h, k_cache, v_cache, row, kv_h, positions)
                    })
                    .collect();
                Partial::merge(&per_warp, head_dim)
            })
            .collect();
        // kernel B: merge the splits, then normalise
        let merged = Partial::merge(&splits, head_dim);
        for (o, a) in out_h.iter_mut().zip(&merged.acc) {
            *o = a / merged.l;
        }
    }
    out
}

/// f64 reference: exact softmax attention, positions ascending — the arithmetic of
/// `forward_attention` without fp32 rounding. The ground truth both f32 paths
/// (the old kernel and this one) are measured against.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_reference_f64(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f64> {
    let row = num_kv_heads * head_dim;
    let group = num_heads / num_kv_heads;
    let sqrt_hd = (head_dim as f64).sqrt();
    let mut out = vec![0.0f64; num_heads * head_dim];
    for h in 0..num_heads {
        let kv_h = h / group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let scores: Vec<f64> = (0..seq_len)
            .map(|p| {
                let k_p = &k_cache[p * row + kv_h * head_dim..p * row + (kv_h + 1) * head_dim];
                q_h.iter()
                    .zip(k_p)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
                    / sqrt_hd
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for (p, w) in weights.iter().enumerate() {
            let v_p = &v_cache[p * row + kv_h * head_dim..p * row + (kv_h + 1) * head_dim];
            for (o, v) in out_h.iter_mut().zip(v_p) {
                *o += w / sum * f64::from(*v);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitk_plan_never_launches_an_empty_split() {
        for seq_len in [
            1u32, 2, 63, 64, 65, 127, 4096, 4097, 20_000, 148_000, 262_144,
        ] {
            let plan = SplitKPlan::for_seq_len(seq_len);
            assert!(plan.n_splits >= 1);
            assert!(
                plan.n_splits <= SPLITK_DEFAULT_TARGET_SPLITS,
                "{seq_len}: {plan:?}"
            );
            assert!(plan.chunk >= SPLITK_MIN_CHUNK, "{seq_len}: {plan:?}");
            // the last split starts inside the context, and the splits cover it
            assert!(
                (plan.n_splits - 1) * plan.chunk < seq_len,
                "{seq_len}: {plan:?}"
            );
            assert!(u64::from(plan.n_splits) * u64::from(plan.chunk) >= u64::from(seq_len));
        }
        assert_eq!(
            SplitKPlan::for_seq_len(262_144),
            SplitKPlan {
                chunk: 1024,
                n_splits: 256
            }
        );
        assert_eq!(
            SplitKPlan::for_seq_len(4096),
            SplitKPlan {
                chunk: 64,
                n_splits: 64
            }
        );
        assert_eq!(
            SplitKPlan::for_seq_len(5),
            SplitKPlan {
                chunk: 64,
                n_splits: 1
            }
        );
    }

    #[test]
    fn splitk_kernel_ptx_shape() {
        for kv in [KvStorage::F32, KvStorage::F16] {
            let kernel = DecodeAttentionSplitKKernel::new(16, 4, 256, kv);
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(&format!(".entry {}", kernel.name())), "{ptx}");
            assert!(
                ptx.contains("shfl.sync.down.b32"),
                "the dot is a warp reduction:\n{ptx}"
            );
            assert!(
                ptx.contains("shfl.sync.idx.b32"),
                "and a lane-0 broadcast:\n{ptx}"
            );
            assert!(
                ptx.contains("%ctaid.y"),
                "the split index is grid y:\n{ptx}"
            );
            match kv {
                KvStorage::F32 => assert!(!ptx.contains("cvt.f32.f16"), "{ptx}"),
                KvStorage::F16 => {
                    assert!(ptx.contains("ld.global.b16"), "f16 KV loads b16:\n{ptx}");
                    assert!(ptx.contains("cvt.f32.f16"), "f16 KV widens to f32:\n{ptx}");
                }
            }
            assert_eq!(kernel.group_size(), 4);
            assert_eq!(kernel.shared_bytes(), 8 * 2 * 4 + 8 * 256 * 4);
            let plan = SplitKPlan::for_seq_len(20_000);
            assert_eq!(kernel.grid(plan), (4, plan.n_splits, 1));
        }
        let reduce = DecodeAttentionSplitKReduceKernel::new(16, 256);
        let ptx = reduce.emit_ptx();
        assert!(
            ptx.contains(".entry gdn_decode_attention_splitk_reduce"),
            "{ptx}"
        );
        assert!(ptx.contains("ex2.approx.f32"), "{ptx}");
        assert_eq!(reduce.grid(), (16, 1, 1));
    }

    #[test]
    fn splitk_refuses_shapes_it_would_truncate() {
        assert!(
            std::panic::catch_unwind(|| DecodeAttentionSplitKKernel::new(
                16,
                4,
                512,
                KvStorage::F32
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| DecodeAttentionSplitKKernel::new(
                16,
                4,
                200,
                KvStorage::F32
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| DecodeAttentionSplitKKernel::new(
                16,
                5,
                256,
                KvStorage::F16
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| DecodeAttentionSplitKReduceKernel::new(16, 288)).is_err()
        );
    }

    /// The twin is a different association order from the f64 reference, never a
    /// different computation: at every plan, including one that forces many splits
    /// over a short context, it lands on the reference.
    #[test]
    fn splitk_cpu_twin_matches_f64_reference() {
        let (num_heads, num_kv_heads, head_dim) = (8usize, 2usize, 256usize);
        let mut rng = crate::kernels::gdn::test_support::Lcg::new(0x3725_0001);
        let seq_len = 203usize;
        let row = num_kv_heads * head_dim;
        let q = rng.vec(num_heads * head_dim, 0.5);
        let k = rng.vec(seq_len * row, 0.5);
        let v = rng.vec(seq_len * row, 1.0);
        let want =
            decode_attention_reference_f64(&q, &k, &v, num_heads, num_kv_heads, head_dim, seq_len);
        for plan in [
            SplitKPlan::for_seq_len(seq_len as u32),
            SplitKPlan::new(seq_len as u32, 64, 3),
            SplitKPlan::new(seq_len as u32, 1, 1),
        ] {
            let got = decode_attention_splitk_cpu(
                &q,
                &k,
                &v,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                plan,
            );
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(*g) - w).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-5, "{plan:?}: worst |twin - f64| = {worst}");
        }
    }

    /// A mutated twin (the split merge drops the e^(m_s - M) rescale) must be caught
    /// by the comparison above — the test is not vacuous at this fixture.
    #[test]
    fn splitk_cpu_twin_comparison_catches_a_missing_rescale() {
        let (num_heads, num_kv_heads, head_dim) = (8usize, 2usize, 256usize);
        let mut rng = crate::kernels::gdn::test_support::Lcg::new(0x3725_0002);
        let seq_len = 203usize;
        let row = num_kv_heads * head_dim;
        let q = rng.vec(num_heads * head_dim, 4.0);
        let k = rng.vec(seq_len * row, 4.0);
        let v = rng.vec(seq_len * row, 1.0);
        let want =
            decode_attention_reference_f64(&q, &k, &v, num_heads, num_kv_heads, head_dim, seq_len);
        // Unnormalised mean of per-split softmax outputs: what a merge without the
        // log-sum-exp rescale computes.
        let plan = SplitKPlan::new(seq_len as u32, 8, 1);
        let mut wrong = vec![0.0f64; num_heads * head_dim];
        for s in 0..plan.n_splits as usize {
            let start = s * plan.chunk as usize;
            let end = (start + plan.chunk as usize).min(seq_len);
            let sub_k: Vec<f32> = k[start * row..end * row].to_vec();
            let sub_v: Vec<f32> = v[start * row..end * row].to_vec();
            let part = decode_attention_reference_f64(
                &q,
                &sub_k,
                &sub_v,
                num_heads,
                num_kv_heads,
                head_dim,
                end - start,
            );
            for (w, p) in wrong.iter_mut().zip(part) {
                *w += p / f64::from(plan.n_splits);
            }
        }
        let worst = wrong
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst > 1e-3,
            "the fixture cannot tell a missing rescale apart: {worst}"
        );
    }
}

/// Device parity: the split-K pair against its CPU twin, the f64 reference, and the
/// kernel it replaces ([`super::DecodeAttention256Kernel`]).
#[cfg(test)]
#[cfg(feature = "cuda")]
mod device_tests {
    use super::*;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg};
    use crate::kernels::gdn::DecodeAttention256Kernel;

    const HEAD_DIM: usize = 256;

    /// f32 -> IEEE half bits for `|x| < 65504`, round to nearest even. Which rounding
    /// does not matter to the tests — the references read the SAME bits widened by
    /// [`widen`] — only that the bits are a valid half.
    fn f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let a = x.abs();
        if a < 6.103_515_6e-5 {
            // subnormal half: a multiple of 2^-24 (1024 rounds up to the least normal)
            return sign | (a * 16_777_216.0).round() as u16;
        }
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = b & 0x7f_ffff;
        let mut h = ((exp as u32) << 10) | (mant >> 13);
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
            h += 1;
        }
        sign | h as u16
    }

    /// IEEE half bits -> f32, exactly (what `cvt.f32.f16` computes).
    fn widen(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&b| {
                let sign = if b & 0x8000 == 0 { 1.0f32 } else { -1.0 };
                let e = i32::from((b >> 10) & 0x1f);
                let m = f32::from(b & 0x3ff);
                if e == 0 {
                    sign * m * 2f32.powi(-24)
                } else {
                    sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
                }
            })
            .collect()
    }

    #[test]
    fn f16_codec_round_trips_known_values() {
        for (x, bits) in [
            (1.0f32, 0x3c00u16),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (0.0, 0),
            (6.103_515_6e-5, 0x0400),
        ] {
            assert_eq!(f16_bits(x), bits, "{x}");
            assert_eq!(widen(&[bits])[0], x, "{bits:#x}");
        }
        // 1 + 2^-11 is the tie between 1.0 and the next half: nearest even is 1.0.
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(widen(&[0x0001])[0], 2f32.powi(-24));
    }

    struct Fixture {
        num_heads: usize,
        num_kv_heads: usize,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
    }

    /// Rows differ in scale so a kernel reading the wrong position or KV head lands
    /// somewhere else; q is scaled so the softmax is neither flat nor one-hot.
    fn fixture(num_heads: usize, num_kv_heads: usize, seq_len: usize, seed: u32) -> Fixture {
        let mut rng = Lcg::new(seed);
        let q = rng.vec(num_heads * HEAD_DIM, 0.3);
        let row = num_kv_heads * HEAD_DIM;
        let mut k = Vec::with_capacity(seq_len * row);
        let mut v = Vec::with_capacity(seq_len * row);
        for p in 0..seq_len {
            let scale = 0.5 + 0.5 * ((p % 97) as f32 / 97.0);
            for _ in 0..row {
                k.push(rng.next_scaled(scale));
            }
            for _ in 0..row {
                v.push(rng.next_scaled(scale));
            }
        }
        Fixture {
            num_heads,
            num_kv_heads,
            q,
            k,
            v,
        }
    }

    /// One split-K run (kernel A then kernel B) over device K/V pointers.
    #[allow(clippy::too_many_arguments)]
    fn run_splitk(
        ctx: &CudaContext,
        stream: &CudaStream,
        f: &Fixture,
        kv: KvStorage,
        k_ptr: u64,
        v_ptr: u64,
        seq_len: usize,
        plan: SplitKPlan,
    ) -> Vec<f32> {
        let (nh, nkv) = (f.num_heads as u32, f.num_kv_heads as u32);
        let a = DecodeAttentionSplitKKernel::new(nh, nkv, HEAD_DIM as u32, kv);
        let b = DecodeAttentionSplitKReduceKernel::new(nh, HEAD_DIM as u32);
        let q_buf = GpuBuffer::from_host(ctx, &f.q).expect("q");
        let pacc = GpuBuffer::<f32>::new(
            ctx,
            splitk_partial_acc_len(nh, HEAD_DIM as u32, plan.n_splits),
        )
        .expect("pacc");
        let pml =
            GpuBuffer::<f32>::new(ctx, splitk_partial_ml_len(nh, plan.n_splits)).expect("pml");
        let out = GpuBuffer::<f32>::new(ctx, f.num_heads * HEAD_DIM).expect("out");
        let mut args_a = [
            q_buf.as_ptr(),
            k_ptr,
            v_ptr,
            pacc.as_ptr(),
            pml.as_ptr(),
            seq_len as u64,
            u64::from(plan.chunk),
            u64::from(plan.n_splits),
        ];
        run_kernel(ctx, stream, &a, a.grid(plan), a.block(), &mut args_a);
        let mut args_b = [
            pacc.as_ptr(),
            pml.as_ptr(),
            out.as_ptr(),
            u64::from(plan.n_splits),
        ];
        run_kernel(ctx, stream, &b, b.grid(), b.block(), &mut args_b);
        let mut got = vec![0.0f32; f.num_heads * HEAD_DIM];
        out.copy_to_host(&mut got).expect("download");
        got
    }

    /// The kernel this row replaces, on the same f32 cache.
    fn run_old(
        ctx: &CudaContext,
        stream: &CudaStream,
        f: &Fixture,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        seq_len: usize,
    ) -> Vec<f32> {
        let kernel = DecodeAttention256Kernel::new(
            f.num_heads as u32,
            f.num_kv_heads as u32,
            HEAD_DIM as u32,
        );
        let q_buf = GpuBuffer::from_host(ctx, &f.q).expect("q");
        let out = GpuBuffer::<f32>::new(ctx, f.num_heads * HEAD_DIM).expect("out");
        let mut args = [
            q_buf.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_ptr(),
            seq_len as u64,
        ];
        run_kernel(
            ctx,
            stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );
        let mut got = vec![0.0f32; f.num_heads * HEAD_DIM];
        out.copy_to_host(&mut got).expect("download");
        got
    }

    fn to_f32(x: &[f64]) -> Vec<f32> {
        x.iter().map(|v| *v as f32).collect()
    }

    #[test]
    fn splitk_f32_matches_twin_reference_and_old_kernel() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("splitk f32: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // 0.8B/2B, 4B/9B, 27B geometries.
        for (nh, nkv) in [(8usize, 2usize), (16, 4), (24, 4)] {
            let max_len = 8192usize;
            let f = fixture(nh, nkv, max_len, 0x3725_0100 + nh as u32);
            let k = GpuBuffer::from_host(&ctx, &f.k).expect("k");
            let v = GpuBuffer::from_host(&ctx, &f.v).expect("v");
            for seq_len in [1usize, 5, 37, 64, 65, 520, 4096, 4097, 8192] {
                let plan = SplitKPlan::for_seq_len(seq_len as u32);
                let got = run_splitk(
                    &ctx,
                    &stream,
                    &f,
                    KvStorage::F32,
                    k.as_ptr(),
                    v.as_ptr(),
                    seq_len,
                    plan,
                );
                let what = format!("{nh}/{nkv} seq_len {seq_len} {plan:?}");
                let twin =
                    decode_attention_splitk_cpu(&f.q, &f.k, &f.v, nh, nkv, HEAD_DIM, seq_len, plan);
                assert_close(
                    &got,
                    &twin,
                    1e-4,
                    &format!("split-K vs its CPU twin, {what}"),
                );
                let reference = to_f32(&decode_attention_reference_f64(
                    &f.q, &f.k, &f.v, nh, nkv, HEAD_DIM, seq_len,
                ));
                assert_close(
                    &got,
                    &reference,
                    1e-4,
                    &format!("split-K vs f64 reference, {what}"),
                );
                let old = run_old(&ctx, &stream, &f, &k, &v, seq_len);
                assert_close(
                    &got,
                    &old,
                    1e-4,
                    &format!("split-K vs DecodeAttention256Kernel, {what}"),
                );
            }
        }
    }

    /// Many tiny splits over a short context: the split merge (kernel B) and the warp
    /// merge (a split that is not a multiple of 8 positions) do all the work.
    #[test]
    fn splitk_forced_many_splits_matches_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("splitk many splits: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (nh, nkv, seq_len) = (16usize, 4usize, 203usize);
        let f = fixture(nh, nkv, seq_len, 0x3725_0200);
        let k = GpuBuffer::from_host(&ctx, &f.k).expect("k");
        let v = GpuBuffer::from_host(&ctx, &f.v).expect("v");
        let reference = to_f32(&decode_attention_reference_f64(
            &f.q, &f.k, &f.v, nh, nkv, HEAD_DIM, seq_len,
        ));
        for plan in [
            SplitKPlan::new(203, 64, 3),
            SplitKPlan::new(203, 29, 1),
            SplitKPlan::new(203, 1, 1),
        ] {
            let got = run_splitk(
                &ctx,
                &stream,
                &f,
                KvStorage::F32,
                k.as_ptr(),
                v.as_ptr(),
                seq_len,
                plan,
            );
            assert_close(&got, &reference, 1e-4, &format!("forced {plan:?}"));
        }
    }

    /// f16 storage: the split-K kernel reading f16 K/V, against the old kernel and the
    /// f64 reference reading the SAME values widened to f32 — this isolates the read
    /// path from the storage rounding (f16-vs-f32 storage parity is #3596's rung).
    #[test]
    fn splitk_f16_read_matches_widened_f32() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("splitk f16: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        for (nh, nkv) in [(8usize, 2usize), (16, 4), (24, 4)] {
            let max_len = 4097usize;
            let f = fixture(nh, nkv, max_len, 0x3725_0300 + nh as u32);
            let k_bits: Vec<u16> = f.k.iter().map(|x| f16_bits(*x)).collect();
            let v_bits: Vec<u16> = f.v.iter().map(|x| f16_bits(*x)).collect();
            let widened = Fixture {
                num_heads: nh,
                num_kv_heads: nkv,
                q: f.q.clone(),
                k: widen(&k_bits),
                v: widen(&v_bits),
            };
            let k16 = GpuBuffer::from_host(&ctx, &k_bits).expect("k16");
            let v16 = GpuBuffer::from_host(&ctx, &v_bits).expect("v16");
            let k32 = GpuBuffer::from_host(&ctx, &widened.k).expect("k32");
            let v32 = GpuBuffer::from_host(&ctx, &widened.v).expect("v32");
            for seq_len in [1usize, 37, 520, 4097] {
                let plan = SplitKPlan::for_seq_len(seq_len as u32);
                let got = run_splitk(
                    &ctx,
                    &stream,
                    &f,
                    KvStorage::F16,
                    k16.as_ptr(),
                    v16.as_ptr(),
                    seq_len,
                    plan,
                );
                let what = format!("f16 {nh}/{nkv} seq_len {seq_len}");
                let reference = to_f32(&decode_attention_reference_f64(
                    &widened.q, &widened.k, &widened.v, nh, nkv, HEAD_DIM, seq_len,
                ));
                assert_close(
                    &got,
                    &reference,
                    1e-4,
                    &format!("{what} vs f64 on widened values"),
                );
                let old = run_old(&ctx, &stream, &widened, &k32, &v32, seq_len);
                assert_close(
                    &got,
                    &old,
                    1e-4,
                    &format!("{what} vs DecodeAttention256Kernel on widened values"),
                );
            }
        }
    }

    #[test]
    fn splitk_reads_the_right_kv_head() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("splitk kv head: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // 4 KV heads whose V is the constant kv_h + 1 at every position: whatever the
        // softmax, head h's output must be its own KV head's constant.
        let (nh, nkv, seq_len) = (16usize, 4usize, 300usize);
        let row = nkv * HEAD_DIM;
        let mut f = fixture(nh, nkv, seq_len, 0x3725_0400);
        for p in 0..seq_len {
            for kv_h in 0..nkv {
                for i in 0..HEAD_DIM {
                    f.v[p * row + kv_h * HEAD_DIM + i] = (kv_h + 1) as f32;
                }
            }
        }
        let k = GpuBuffer::from_host(&ctx, &f.k).expect("k");
        let v = GpuBuffer::from_host(&ctx, &f.v).expect("v");
        let got = run_splitk(
            &ctx,
            &stream,
            &f,
            KvStorage::F32,
            k.as_ptr(),
            v.as_ptr(),
            seq_len,
            SplitKPlan::new(300, 7, 1),
        );
        for h in 0..nh {
            let expected = (h / (nh / nkv) + 1) as f32;
            for i in 0..HEAD_DIM {
                let g = got[h * HEAD_DIM + i];
                assert!(
                    (g - expected).abs() < 1e-4,
                    "head {h} element {i} = {g}, its KV head is {expected}"
                );
            }
        }
    }
}
