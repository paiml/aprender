//! PMAT-3477 / aprender#3090: single-query decode attention for `head_dim = 256`.
//!
//! The attention half of `forward_attention` in the CPU reference
//! (forward_qwen35.rs:1052), over a resident KV cache:
//!
//! ```text
//! group_size = num_heads / num_kv_heads
//! for h in 0..num_heads:
//!     kv_h = h / group_size
//!     for p in 0..seq_len:
//!         scores[p] = dot(q_h, k_cache[p][kv_h]) / sqrt(head_dim)
//!     softmax(scores)                       // max-subtracted, fp32
//!     out_h = sum_p scores[p] * v_cache[p][kv_h]
//! ```
//!
//! ## Why a new kernel
//!
//! Every attention kernel under `kernels/attention/` hard-codes `head_dim <= 128`
//! (four floats per lane times a 32-lane warp: `incremental.rs:210`,
//! `flash_decoding/chunk_kernel.rs:90`) and **silently truncates** a 256-wide head.
//! Qwen3.5's gated full-attention layers are `head_dim = 256`, so they cannot use
//! them at all.
//!
//! ## Shape
//!
//! Grid `(num_heads, 1, 1)`, block `(256, 1, 1)`. `k_cache` and `v_cache` are
//! `[max_len][num_kv_heads * head_dim]` with row stride `num_kv_heads * head_dim`,
//! exactly as `Qwen35State::kv_cache` stores them; `seq_len` (the number of valid
//! positions, i.e. `position + 1`) is a kernel parameter, so one compiled module
//! serves the whole decode. `head_dim` must be at most the 256-thread block.
//!
//! ## Positions per pass
//!
//! The scores of a pass live in shared memory, so a pass covers at most
//! [`DEFAULT_MAX_POSITIONS_PER_PASS`] = 4096 positions (16 KiB of the 48 KiB static
//! shared budget). A longer context is processed in several passes with the running
//! max/sum rescaling of flash decoding — mathematically the same softmax, and the
//! pass loop is exercised by a test that sets the cap to 8 with `seq_len = 37`.

use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{KernelBuilder, PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// Threads per block.
pub(super) const BLOCK: u32 = 256;
/// Warps per block.
pub(super) const WARPS: u32 = BLOCK / 32;
/// Positions whose scores fit in shared memory in one pass (16 KiB).
pub const DEFAULT_MAX_POSITIONS_PER_PASS: u32 = 4096;

/// Which reduction a [`emit_block_reduce`] call performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceOp {
    Max,
    Sum,
}

/// Single-query decode attention over a resident KV cache, `head_dim` up to 256.
///
/// For Qwen3.5-0.8B: `num_heads = 16`, `num_kv_heads = 2` or `4`, `head_dim = 256`.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttention256Kernel {
    /// Query heads (`config.num_heads`).
    pub num_heads: u32,
    /// Key/value heads (`config.num_kv_heads`); `num_heads` must be a multiple of it.
    pub num_kv_heads: u32,
    /// Width of one head (256 for Qwen3.5); must be `<= 256`.
    pub head_dim: u32,
    /// Positions scored per pass — the shared scores buffer holds this many floats.
    pub max_positions_per_pass: u32,
    /// #4233: read `position` from a device `u32` (`pos_ptr`) and attend over
    /// `position + 1` entries, so a captured CUDA graph replays at any position.
    pub indirect: bool,
}

impl DecodeAttention256Kernel {
    /// Create the kernel with the default 4096-position pass.
    ///
    /// # Panics
    /// If `head_dim` exceeds the 256-thread block or `num_kv_heads` does not divide
    /// `num_heads` — both would silently produce wrong output.
    #[must_use]
    pub fn new(num_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        assert!(
            head_dim <= BLOCK,
            "head_dim {head_dim} exceeds the {BLOCK}-thread block; one thread owns one output element"
        );
        assert!(
            head_dim % 4 == 0,
            "head_dim {head_dim} must be a multiple of 4: the score dot reads 16-byte vectors"
        );
        assert!(
            num_kv_heads > 0 && num_heads % num_kv_heads == 0,
            "num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}"
        );
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_positions_per_pass: DEFAULT_MAX_POSITIONS_PER_PASS,
            indirect: false,
        }
    }

    /// #4233: the graph-capturable variant — `seq_len = *pos_ptr + 1`, read on
    /// the device, so the launch arguments are identical at every token.
    #[must_use]
    pub const fn indirect(mut self) -> Self {
        self.indirect = true;
        self
    }

    /// Override the positions-per-pass cap (the shared scores buffer size).
    #[must_use]
    pub const fn with_max_positions_per_pass(mut self, cap: u32) -> Self {
        self.max_positions_per_pass = cap;
        self
    }

    /// GQA group size — how many query heads share one KV head.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }

    /// Static shared memory: the scores buffer plus two reduction scratch rows.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        (self.max_positions_per_pass * 4 + WARPS * 4 * 2) as usize
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

/// Block-wide max or sum of `val`, broadcast to every thread.
///
/// Warp shuffles, then one f32 per warp in shared memory at `scratch_off`, then every
/// thread folds the `WARPS` partials itself — which avoids a second barrier and a
/// broadcast slot. `label_prefix` must be unique per call site: PTX labels are flat.
fn emit_block_reduce(
    ctx: &mut KernelBuilder<'_>,
    val: VirtualReg,
    lane: VirtualReg,
    warp: VirtualReg,
    scratch_off: u32,
    op: ReduceOp,
    label_prefix: &str,
) -> VirtualReg {
    let mut v = val;
    for offset in [16u32, 8, 4, 2, 1] {
        let shuffled = ctx.shfl_down_f32(v, offset, 0xFFFF_FFFF);
        v = match op {
            ReduceOp::Max => ctx.max_f32(v, shuffled),
            ReduceOp::Sum => ctx.add_f32(v, shuffled),
        };
    }

    let zero = ctx.mov_u32_imm(0);
    let is_lane0 = ctx.setp_eq_u32(lane, zero);
    let skip = format!("{label_prefix}_skip_partial_store");
    ctx.branch_if_not(is_lane0, &skip);
    let warp_bytes = ctx.mul_u32(warp, 4);
    let slot = ctx.add_u32(warp_bytes, scratch_off);
    let slot64 = ctx.cvt_u64_u32(slot);
    ctx.st_shared_f32(slot64, v);
    ctx.label(&skip);
    ctx.bar_sync(0);

    let mut acc: Option<VirtualReg> = None;
    for w in 0..WARPS {
        let off = ctx.mov_u32_imm(scratch_off + w * 4);
        let off64 = ctx.cvt_u64_u32(off);
        let partial = ctx.ld_shared_f32(off64);
        acc = Some(match acc {
            None => partial,
            Some(a) => match op {
                ReduceOp::Max => ctx.max_f32(a, partial),
                ReduceOp::Sum => ctx.add_f32(a, partial),
            },
        });
    }
    acc.expect("WARPS > 0")
}

/// Compile-time shape of the pass loop.
#[derive(Debug, Clone, Copy)]
pub(super) struct PassShape {
    /// Width of one head; one thread owns one output element.
    pub head_dim: u32,
    /// Positions scored per pass (the shared scores buffer).
    pub cap: u32,
    /// Bytes between consecutive positions of the cache.
    pub row_stride_bytes: u32,
    /// Score with a warp per position (coalesced, not bit-identical to the CPU)
    /// instead of a thread per position. Needs `head_dim % 128 == 0`.
    pub warp_dot: bool,
}

/// Registers the scoring phase reads.
struct ScoreRegs {
    tid: VirtualReg,
    lane: VirtualReg,
    warp: VirtualReg,
    q_base: VirtualReg,
    k_head: VirtualReg,
    chunk: VirtualReg,
    n_c: VirtualReg,
    sqrt_hd: VirtualReg,
    head_dim_r: VirtualReg,
    four: VirtualReg,
}

/// Scores with one thread per position, summed in ascending element order: the
/// CPU's float sequence, bit for bit.
fn emit_scores_thread(ctx: &mut KernelBuilder<'_>, s: PassShape, r: &ScoreRegs) -> VirtualReg {
    let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
    let pp = ctx.add_u32(r.tid, 0);
    ctx.label("gdn_attn_score_loop");
    let score_go = ctx.setp_lt_u32(pp, r.n_c);
    ctx.branch_if_not(score_go, "gdn_attn_score_end");
    let p = ctx.add_u32_reg(r.chunk, pp);
    let row_off = ctx.mul_wide_u32(p, s.row_stride_bytes);
    let k_row = ctx.add_u64(r.k_head, row_off);

    let dot = ctx.mov_f32_imm(0.0);
    let i = ctx.mov_u32_imm(0);
    ctx.label("gdn_attn_dot_loop");
    let dot_go = ctx.setp_lt_u32(i, r.head_dim_r);
    ctx.branch_if_not(dot_go, "gdn_attn_dot_end");
    let elem_off = ctx.mul_wide_u32_reg(i, r.four);
    let q_addr = ctx.add_u64(r.q_base, elem_off);
    let k_addr = ctx.add_u64(k_row, elem_off);
    // Four elements per 16-byte load, still summed one at a time in ascending
    // `i` — the same float sequence as scalar loads.
    let q4 = ctx.ld_global_f32_v4(q_addr);
    let k4 = ctx.ld_global_f32_v4(k_addr);
    for (q_val, k_val) in q4.into_iter().zip(k4) {
        // mul then add, like the CPU's `dot += q_h[i] * k_p[i]` — Rust does
        // not contract into an fma, so neither do we.
        let prod = ctx.mul_f32(q_val, k_val);
        ctx.add_f32_inplace(dot, prod);
    }
    ctx.add_u32_inplace(i, 4);
    ctx.branch("gdn_attn_dot_loop");
    ctx.label("gdn_attn_dot_end");

    let score = ctx.div_f32(dot, r.sqrt_hd);
    let score_slot = ctx.mul_u32(pp, 4);
    let score_addr = ctx.cvt_u64_u32(score_slot);
    ctx.st_shared_f32(score_addr, score);
    ctx.max_f32_inplace(local_max, score);
    ctx.add_u32_inplace(pp, BLOCK);
    ctx.branch("gdn_attn_score_loop");
    ctx.label("gdn_attn_score_end");
    local_max
}

/// Scores with one warp per position: each lane reads 16-byte pieces of the K
/// row, so a warp's load is one contiguous 512-byte run, and a shuffle tree sums
/// the lanes. A different summation order from the CPU's, so not bit-identical
/// (aprender#4273: a thread walking its own row read the cache at 36–42% of
/// copy bandwidth).
fn emit_scores_warp(ctx: &mut KernelBuilder<'_>, s: PassShape, r: &ScoreRegs) -> VirtualReg {
    let loads = s.head_dim / 128;
    let lane_off = ctx.mul_wide_u32(r.lane, 16);
    let q_lane = ctx.add_u64(r.q_base, lane_off);
    let stride = ctx.mov_u64_imm(512);
    let mut q_addrs = vec![q_lane];
    for _ in 1..loads {
        let prev = *q_addrs.last().expect("one load");
        q_addrs.push(ctx.add_u64(prev, stride));
    }
    let q: Vec<[VirtualReg; 4]> = q_addrs
        .into_iter()
        .map(|a| ctx.ld_global_f32_v4(a))
        .collect();
    let one = ctx.mov_u32_imm(1);

    let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
    let pp = ctx.add_u32(r.warp, 0);
    ctx.label("gdn_attn_wscore_loop");
    let score_go = ctx.setp_lt_u32(pp, r.n_c);
    ctx.branch_if_not(score_go, "gdn_attn_wscore_end");
    let p = ctx.add_u32_reg(r.chunk, pp);
    let row_off = ctx.mul_wide_u32(p, s.row_stride_bytes);
    let k_row = ctx.add_u64(r.k_head, row_off);
    let mut k_addr = ctx.add_u64(k_row, lane_off);
    let mut k = Vec::with_capacity(loads as usize);
    for j in 0..loads {
        if j > 0 {
            k_addr = ctx.add_u64(k_addr, stride);
        }
        k.push(ctx.ld_global_f32_v4(k_addr));
    }
    let dot = ctx.mov_f32_imm(0.0);
    for (q4, k4) in q.iter().zip(&k) {
        for (&q_val, &k_val) in q4.iter().zip(k4) {
            let prod = ctx.mul_f32(q_val, k_val);
            ctx.add_f32_inplace(dot, prod);
        }
    }
    for offset in [16, 8, 4, 2, 1] {
        let other = ctx.shfl_down_f32(dot, offset, 0xFFFF_FFFF);
        ctx.add_f32_inplace(dot, other);
    }
    let lead = ctx.setp_lt_u32(r.lane, one);
    ctx.branch_if_not(lead, "gdn_attn_wscore_next");
    let score = ctx.div_f32(dot, r.sqrt_hd);
    let score_slot = ctx.mul_u32(pp, 4);
    let score_addr = ctx.cvt_u64_u32(score_slot);
    ctx.st_shared_f32(score_addr, score);
    ctx.max_f32_inplace(local_max, score);
    ctx.label("gdn_attn_wscore_next");
    ctx.add_u32_inplace(pp, WARPS);
    ctx.branch("gdn_attn_wscore_loop");
    ctx.label("gdn_attn_wscore_end");
    local_max
}

/// Registers the pass loop reads from its caller.
pub(super) struct PassRegs {
    pub tid: VirtualReg,
    pub lane: VirtualReg,
    pub warp: VirtualReg,
    /// This head's `q`.
    pub q_base: VirtualReg,
    /// Position 0 of this head's KV head in the K cache.
    pub k_head: VirtualReg,
    /// Position 0 of this head's KV head in the V cache.
    pub v_head: VirtualReg,
    /// First position, inclusive.
    pub begin: VirtualReg,
    /// Last position, exclusive; `begin < end` is the caller's guarantee.
    pub end: VirtualReg,
}

/// The softmax state after the last pass: the output element `tid` is `acc / sum`.
pub(super) struct PassState {
    pub running_max: VirtualReg,
    pub running_sum: VirtualReg,
    pub acc: VirtualReg,
    /// `tid * 4`, the byte offset of this thread's output element.
    pub out_elem_off: VirtualReg,
    pub head_dim_r: VirtualReg,
}

/// Attention over positions `begin..end`, in passes of `cap`, with the running
/// max/sum rescaling of flash decoding. The unsplit kernel runs it over
/// `0..seq_len`; each block of the split kernel over its own slice (aprender#4273).
pub(super) fn emit_passes(ctx: &mut KernelBuilder<'_>, s: PassShape, r: PassRegs) -> PassState {
    let PassRegs {
        tid,
        lane,
        warp,
        q_base,
        k_head,
        v_head,
        begin,
        end,
    } = r;
    // The CPU divides by sqrt(head_dim); the same division, not a reciprocal.
    let sqrt_head_dim = (s.head_dim as f32).sqrt();
    let scratch_max = s.cap * 4;
    let scratch_sum = s.cap * 4 + WARPS * 4;
    let row_stride_bytes = s.row_stride_bytes;

    let head_dim_r = ctx.mov_u32_imm(s.head_dim);
    let cap_r = ctx.mov_u32_imm(s.cap);
    let sqrt_hd = ctx.mov_f32_imm(sqrt_head_dim);
    let four = ctx.mov_u32_imm(4);
    let out_elem_off = ctx.mul_wide_u32_reg(tid, four);

    // Running softmax state, carried across passes (flash decoding).
    let running_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
    let running_sum = ctx.mov_f32_imm(0.0);
    let acc = ctx.mov_f32_imm(0.0);
    let chunk = ctx.add_u32(begin, 0);

    ctx.label("gdn_attn_chunk_loop");
    let more_chunks = ctx.setp_lt_u32(chunk, end);
    ctx.branch_if_not(more_chunks, "gdn_attn_chunk_end");
    // n_c = min(cap, end - chunk)
    let remaining = ctx.sub_u32_reg(end, chunk);
    let n_c = ctx.min_u32(remaining, cap_r);

    // ---- Phase 1: scores for this pass.
    let score_regs = ScoreRegs {
        tid,
        lane,
        warp,
        q_base,
        k_head,
        chunk,
        n_c,
        sqrt_hd,
        head_dim_r,
        four,
    };
    let local_max = if s.warp_dot {
        emit_scores_warp(ctx, s, &score_regs)
    } else {
        emit_scores_thread(ctx, s, &score_regs)
    };
    ctx.bar_sync(0);

    // ---- Phase 2: softmax over the pass, folded into the running state.
    let chunk_max = emit_block_reduce(
        ctx,
        local_max,
        lane,
        warp,
        scratch_max,
        ReduceOp::Max,
        "gdn_attn_max",
    );
    let new_max = ctx.max_f32(running_max, chunk_max);
    // correction = exp(old_max - new_max); on the first pass old_max is
    // -inf, so this is 0 and the (still zero) accumulators are unaffected.
    let max_delta = ctx.sub_f32(running_max, new_max);
    let correction = emit_exp_f32(ctx, max_delta);
    ctx.mov_f32_reg(running_max, new_max);

    let local_sum = ctx.mov_f32_imm(0.0);
    let wp = ctx.add_u32(tid, 0);
    ctx.label("gdn_attn_weight_loop");
    let weight_go = ctx.setp_lt_u32(wp, n_c);
    ctx.branch_if_not(weight_go, "gdn_attn_weight_end");
    let w_slot = ctx.mul_u32(wp, 4);
    let w_addr = ctx.cvt_u64_u32(w_slot);
    let raw = ctx.ld_shared_f32(w_addr);
    let shifted = ctx.sub_f32(raw, running_max);
    let weight = emit_exp_f32(ctx, shifted);
    ctx.st_shared_f32(w_addr, weight);
    ctx.add_f32_inplace(local_sum, weight);
    ctx.add_u32_inplace(wp, BLOCK);
    ctx.branch("gdn_attn_weight_loop");
    ctx.label("gdn_attn_weight_end");
    ctx.bar_sync(0);

    let chunk_sum = emit_block_reduce(
        ctx,
        local_sum,
        lane,
        warp,
        scratch_sum,
        ReduceOp::Sum,
        "gdn_attn_sum",
    );
    ctx.mul_f32_inplace(running_sum, correction);
    ctx.add_f32_inplace(running_sum, chunk_sum);
    ctx.mul_f32_inplace(acc, correction);

    // ---- Phase 3: thread `tid` accumulates output element `tid`,
    // positions ascending, exactly the CPU's accumulation order.
    let in_head = ctx.setp_lt_u32(tid, head_dim_r);
    ctx.branch_if_not(in_head, "gdn_attn_value_skip");
    let vp = ctx.mov_u32_imm(0);
    // Four positions per trip: their loads issue together, then the four
    // products are added in ascending order, so the sum is the same as one
    // position per trip (aprender#4273: one load in flight per thread left
    // the value phase latency-bound).
    let n_c4 = ctx.and_u32_imm(n_c, !3);
    let row_stride_r = ctx.mov_u32_imm(row_stride_bytes);
    let row_stride64 = ctx.cvt_u64_u32(row_stride_r);
    ctx.label("gdn_attn_value4_loop");
    let value4_go = ctx.setp_lt_u32(vp, n_c4);
    ctx.branch_if_not(value4_go, "gdn_attn_value4_end");
    let w_slot = ctx.mul_u32(vp, 4);
    let w_slot64 = ctx.cvt_u64_u32(w_slot);
    let vpos = ctx.add_u32_reg(chunk, vp);
    let v_row_off = ctx.mul_wide_u32(vpos, row_stride_bytes);
    let v_row = ctx.add_u64(v_head, v_row_off);
    let v_addr0 = ctx.add_u64(v_row, out_elem_off);
    let v_addr1 = ctx.add_u64(v_addr0, row_stride64);
    let v_addr2 = ctx.add_u64(v_addr1, row_stride64);
    let v_addr3 = ctx.add_u64(v_addr2, row_stride64);
    let four64 = ctx.cvt_u64_u32(four);
    let w_addr1 = ctx.add_u64(w_slot64, four64);
    let w_addr2 = ctx.add_u64(w_addr1, four64);
    let w_addr3 = ctx.add_u64(w_addr2, four64);
    let w4 = [w_slot64, w_addr1, w_addr2, w_addr3].map(|a| ctx.ld_shared_f32(a));
    let v0 = ctx.ld_global_f32(v_addr0);
    let v1 = ctx.ld_global_f32(v_addr1);
    let v2 = ctx.ld_global_f32(v_addr2);
    let v3 = ctx.ld_global_f32(v_addr3);
    for (w_val, v_val) in w4.into_iter().zip([v0, v1, v2, v3]) {
        let contrib = ctx.mul_f32(w_val, v_val);
        ctx.add_f32_inplace(acc, contrib);
    }
    ctx.add_u32_inplace(vp, 4);
    ctx.branch("gdn_attn_value4_loop");
    ctx.label("gdn_attn_value4_end");
    ctx.label("gdn_attn_value_loop");
    let value_go = ctx.setp_lt_u32(vp, n_c);
    ctx.branch_if_not(value_go, "gdn_attn_value_end");
    let v_slot = ctx.mul_u32(vp, 4);
    let v_slot64 = ctx.cvt_u64_u32(v_slot);
    let w_val = ctx.ld_shared_f32(v_slot64);
    let vpos = ctx.add_u32_reg(chunk, vp);
    let v_row_off = ctx.mul_wide_u32(vpos, row_stride_bytes);
    let v_row = ctx.add_u64(v_head, v_row_off);
    let v_addr = ctx.add_u64(v_row, out_elem_off);
    let v_val = ctx.ld_global_f32(v_addr);
    let contrib = ctx.mul_f32(w_val, v_val);
    ctx.add_f32_inplace(acc, contrib);
    ctx.add_u32_inplace(vp, 1);
    ctx.branch("gdn_attn_value_loop");
    ctx.label("gdn_attn_value_end");
    ctx.label("gdn_attn_value_skip");

    // The next pass overwrites the scores buffer.
    ctx.bar_sync(0);
    ctx.add_u32_reg_inplace(chunk, cap_r);
    ctx.branch("gdn_attn_chunk_loop");
    ctx.label("gdn_attn_chunk_end");

    PassState {
        running_max,
        running_sum,
        acc,
        out_elem_off,
        head_dim_r,
    }
}

impl Kernel for DecodeAttention256Kernel {
    fn name(&self) -> &str {
        if self.indirect {
            "gdn_decode_attention_indirect"
        } else {
            "gdn_decode_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let group_size = self.group_size();
        let shape = PassShape {
            head_dim,
            cap: self.max_positions_per_pass,
            row_stride_bytes: self.num_kv_heads * head_dim * 4,
            warp_dot: false,
        };
        let indirect = self.indirect;

        let kernel = PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "k_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "v_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "out_ptr"); // [num_heads * head_dim]
        let kernel = if indirect {
            kernel.param(PtxType::U64, "pos_ptr") // device u32 position; seq_len = it + 1
        } else {
            kernel.param(PtxType::U32, "seq_len") // valid positions, = position + 1
        };
        kernel.shared_memory(self.shared_bytes()).build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let h = ctx.special_reg(PtxReg::CtaIdX);
            let lane = ctx.and_u32_imm(tid, 31);
            let warp = ctx.shr_u32_imm(tid, 5);

            let q_ptr = ctx.load_param_u64("q_ptr");
            let k_ptr = ctx.load_param_u64("k_cache_ptr");
            let v_ptr = ctx.load_param_u64("v_cache_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");
            let seq_len = if indirect {
                let pos_ptr = ctx.load_param_u64("pos_ptr");
                let position = ctx.ld_global_u32(pos_ptr);
                ctx.add_u32(position, 1)
            } else {
                ctx.load_param_u32("seq_len")
            };

            // An empty cache has no softmax: leave the output alone.
            let zero_u32 = ctx.mov_u32_imm(0);
            let has_positions = ctx.setp_lt_u32(zero_u32, seq_len);
            ctx.branch_if_not(has_positions, "gdn_attn_exit");

            // kv head of this query head, and its byte offset inside a cache row.
            let kv_h = ctx.div_u32(h, group_size);
            let kv_off = ctx.mul_wide_u32(kv_h, head_dim * 4);
            let head_off = ctx.mul_wide_u32(h, head_dim * 4);
            let q_base = ctx.add_u64(q_ptr, head_off);
            let out_base = ctx.add_u64(out_ptr, head_off);
            let k_head = ctx.add_u64(k_ptr, kv_off);
            let v_head = ctx.add_u64(v_ptr, kv_off);

            let st = emit_passes(
                ctx,
                shape,
                PassRegs {
                    tid,
                    lane,
                    warp,
                    q_base,
                    k_head,
                    v_head,
                    begin: zero_u32,
                    end: seq_len,
                },
            );

            // out = sum_p exp(s_p - max) * v_p / sum_p exp(s_p - max)
            let in_head_out = ctx.setp_lt_u32(tid, st.head_dim_r);
            ctx.branch_if_not(in_head_out, "gdn_attn_exit");
            let result = ctx.div_f32(st.acc, st.running_sum);
            let out_addr = ctx.add_u64(out_base, st.out_elem_off);
            ctx.st_global_f32(out_addr, result);

            ctx.label("gdn_attn_exit");
            ctx.ret();
        })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_decode_attention_ptx_shape() {
        let kernel = DecodeAttention256Kernel::new(16, 2, 256);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_decode_attention"), "{ptx}");
        // The 256-wide dot is a loop over head_dim, not four floats per lane: the
        // truncation defect in kernels/attention is exactly a fixed lane budget.
        assert!(ptx.contains("gdn_attn_dot_loop"), "{ptx}");
        assert!(
            ptx.contains("bar.sync"),
            "a block-wide softmax needs barriers:\n{ptx}"
        );
        assert!(ptx.contains("shfl.sync.down.b32"), "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
        assert_eq!(kernel.block(), (256, 1, 1));
        assert_eq!(kernel.group_size(), 8);
        // 4096 scores + 8 warp maxes + 8 warp sums.
        assert_eq!(kernel.shared_bytes(), 4096 * 4 + 64);
    }

    #[test]
    fn gdn_decode_attention_head_dim_over_block_is_refused() {
        // A kernel that quietly handled only the first 256 elements would be the
        // very defect this ticket exists to fix.
        let refused = std::panic::catch_unwind(|| DecodeAttention256Kernel::new(16, 2, 512));
        assert!(refused.is_err(), "head_dim 512 must not be accepted");
        let refused_gqa = std::panic::catch_unwind(|| DecodeAttention256Kernel::new(16, 5, 256));
        assert!(refused_gqa.is_err(), "num_kv_heads must divide num_heads");
    }
}

/// Device parity against a verbatim port of `forward_attention`'s attention block.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_decode_attention_device_tests {
    use super::DecodeAttention256Kernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg};

    /// `config.num_heads` of the Qwen3.5-0.8B attention layers.
    const NUM_HEADS: usize = 16;
    /// `attn_q_norm.len()`.
    const HEAD_DIM: usize = 256;
    /// Positions the test cache holds.
    const MAX_LEN: usize = 64;

    /// Verbatim port of `aprender-serve`'s `gguf::ops::softmax` (ops.rs:348).
    fn softmax(logits: &mut [f32]) {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in logits.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        let inv_sum = 1.0 / sum;
        for x in logits.iter_mut() {
            *x *= inv_sum;
        }
    }

    /// Verbatim port of the attention block of `forward_attention`
    /// (forward_qwen35.rs:1052), with `position = seq_len - 1`.
    fn decode_attention(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let position = seq_len - 1;
        let group_size = num_heads / num_kv_heads;
        let mut attn_out_in = vec![0.0; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];

            let mut scores = vec![0.0; position + 1];
            for (p, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0;
                let k_p = &k_cache[p * (num_kv_heads * head_dim) + kv_h * head_dim
                    ..p * (num_kv_heads * head_dim) + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    dot += q_h[i] * k_p[i];
                }
                *score = dot / (head_dim as f32).sqrt();
            }
            softmax(&mut scores);

            let out_h = &mut attn_out_in[h * head_dim..(h + 1) * head_dim];
            for (p, &w) in scores.iter().enumerate() {
                let v_p = &v_cache[p * (num_kv_heads * head_dim) + kv_h * head_dim
                    ..p * (num_kv_heads * head_dim) + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    out_h[i] += w * v_p[i];
                }
            }
        }
        attn_out_in
    }

    /// One device run of `kernel` for `seq_len`, returned as a host vector.
    fn run(
        ctx: &CudaContext,
        stream: &CudaStream,
        kernel: &DecodeAttention256Kernel,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let q_buf = GpuBuffer::from_host(ctx, q).expect("q");
        let k_buf = GpuBuffer::from_host(ctx, k_cache).expect("k");
        let v_buf = GpuBuffer::from_host(ctx, v_cache).expect("v");
        let out_buf = GpuBuffer::<f32>::new(ctx, NUM_HEADS * HEAD_DIM).expect("out");
        let mut args = [
            q_buf.as_ptr(),
            k_buf.as_ptr(),
            v_buf.as_ptr(),
            out_buf.as_ptr(),
            seq_len as u64,
        ];
        run_kernel(
            ctx,
            stream,
            kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );
        let mut got = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        out_buf.copy_to_host(&mut got).expect("download");
        got
    }

    /// q, and a KV cache whose rows differ in scale so that a kernel reading the
    /// wrong position or the wrong KV head lands somewhere else.
    fn fixture(num_kv_heads: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Lcg::new(seed);
        let q = rng.vec(NUM_HEADS * HEAD_DIM, 0.1);
        let row = num_kv_heads * HEAD_DIM;
        let mut k = Vec::with_capacity(MAX_LEN * row);
        let mut v = Vec::with_capacity(MAX_LEN * row);
        for p in 0..MAX_LEN {
            let scale = 0.05f32.mul_add(p as f32, 0.5);
            for _ in 0..row {
                k.push(rng.next_scaled(scale));
            }
            for _ in 0..row {
                v.push(rng.next_scaled(scale));
            }
        }
        (q, k, v)
    }

    #[test]
    fn gdn_decode_attention_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        for num_kv_heads in [2usize, 4] {
            let (q, k, v) = fixture(num_kv_heads, 0x3477_0030 + num_kv_heads as u32);
            let kernel = DecodeAttention256Kernel::new(
                NUM_HEADS as u32,
                num_kv_heads as u32,
                HEAD_DIM as u32,
            );
            for seq_len in [1usize, 5, 37] {
                let want = decode_attention(&q, &k, &v, seq_len, NUM_HEADS, num_kv_heads, HEAD_DIM);
                let got = run(&ctx, &stream, &kernel, &q, &k, &v, seq_len);
                assert_close(
                    &got,
                    &want,
                    1e-3,
                    &format!("decode attention, num_kv_heads {num_kv_heads}, seq_len {seq_len}"),
                );
            }
        }
    }

    #[test]
    fn gdn_decode_attention_multi_pass_matches_single_pass() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention multi-pass: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let num_kv_heads = 2usize;
        let (q, k, v) = fixture(num_kv_heads, 0x3477_0033);
        let seq_len = 37;
        let want = decode_attention(&q, &k, &v, seq_len, NUM_HEADS, num_kv_heads, HEAD_DIM);

        // A cap of 8 forces 5 passes over the same 37 positions, so the running
        // max/sum rescaling is what is being measured, not the single-pass path.
        let kernel =
            DecodeAttention256Kernel::new(NUM_HEADS as u32, num_kv_heads as u32, HEAD_DIM as u32)
                .with_max_positions_per_pass(8);
        assert_eq!(kernel.shared_bytes(), 8 * 4 + 64);
        let got = run(&ctx, &stream, &kernel, &q, &k, &v, seq_len);
        assert_close(&got, &want, 1e-3, "decode attention across 5 passes");
    }

    #[test]
    fn gdn_decode_attention_reads_the_right_kv_head() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention kv head: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        // Two KV heads, one all +1 and one all -1 in V: with a single position the
        // softmax weight is exactly 1, so out_h must be the sign of its own KV head.
        let num_kv_heads = 2usize;
        let row = num_kv_heads * HEAD_DIM;
        let mut rng = Lcg::new(0x3477_0034);
        let q = rng.vec(NUM_HEADS * HEAD_DIM, 0.1);
        let k = vec![0.01f32; MAX_LEN * row];
        let mut v = vec![0.0f32; MAX_LEN * row];
        for p in 0..MAX_LEN {
            for i in 0..HEAD_DIM {
                v[p * row + i] = 1.0;
                v[p * row + HEAD_DIM + i] = -1.0;
            }
        }
        let kernel =
            DecodeAttention256Kernel::new(NUM_HEADS as u32, num_kv_heads as u32, HEAD_DIM as u32);
        let got = run(&ctx, &stream, &kernel, &q, &k, &v, 1);
        for h in 0..NUM_HEADS {
            let expected = if h < NUM_HEADS / 2 { 1.0 } else { -1.0 };
            for i in 0..HEAD_DIM {
                assert!(
                    (got[h * HEAD_DIM + i] - expected).abs() < 1e-5,
                    "head {h} element {i} = {} but its KV head is {expected}",
                    got[h * HEAD_DIM + i]
                );
            }
        }
    }

    /// #4233: the graph variant reads `position` from the device and must produce
    /// the eager kernel's output bit for bit — including across the multi-pass
    /// boundary, where `seq_len` drives the chunk loop.
    #[test]
    fn gdn_decode_attention_indirect_is_bit_identical_to_direct() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention indirect: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let num_kv_heads = 2usize;
        let (q, k, v) = fixture(num_kv_heads, 0x4233_0002);
        for cap in [None, Some(8u32)] {
            let mut direct = DecodeAttention256Kernel::new(
                NUM_HEADS as u32,
                num_kv_heads as u32,
                HEAD_DIM as u32,
            );
            if let Some(cap) = cap {
                direct = direct.with_max_positions_per_pass(cap);
            }
            let indirect = direct.indirect();
            for seq_len in [1usize, 5, 37] {
                let want = run(&ctx, &stream, &direct, &q, &k, &v, seq_len);

                let q_buf = GpuBuffer::from_host(&ctx, &q).expect("q");
                let k_buf = GpuBuffer::from_host(&ctx, &k).expect("k");
                let v_buf = GpuBuffer::from_host(&ctx, &v).expect("v");
                let out_buf = GpuBuffer::<f32>::new(&ctx, NUM_HEADS * HEAD_DIM).expect("out");
                let pos = u32::try_from(seq_len - 1).expect("pos");
                let pos_buf = GpuBuffer::from_host(&ctx, &[pos]).expect("pos");
                let mut args = [
                    q_buf.as_ptr(),
                    k_buf.as_ptr(),
                    v_buf.as_ptr(),
                    out_buf.as_ptr(),
                    pos_buf.as_ptr(),
                ];
                run_kernel(
                    &ctx,
                    &stream,
                    &indirect,
                    indirect.grid(),
                    indirect.block(),
                    &mut args,
                );
                let mut got = vec![0.0f32; NUM_HEADS * HEAD_DIM];
                out_buf.copy_to_host(&mut got).expect("download");
                assert_eq!(
                    got, want,
                    "indirect vs direct, cap {cap:?}, seq_len {seq_len}"
                );
            }
        }
    }
}
