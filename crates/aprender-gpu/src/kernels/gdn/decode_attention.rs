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
const BLOCK: u32 = 256;
/// Warps per block.
const WARPS: u32 = BLOCK / 32;
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
            num_kv_heads > 0 && num_heads % num_kv_heads == 0,
            "num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}"
        );
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_positions_per_pass: DEFAULT_MAX_POSITIONS_PER_PASS,
        }
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

impl Kernel for DecodeAttention256Kernel {
    fn name(&self) -> &str {
        "gdn_decode_attention"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let group_size = self.group_size();
        let cap = self.max_positions_per_pass;
        let row_stride_bytes = self.num_kv_heads * head_dim * 4;
        // The CPU divides by sqrt(head_dim); the same division, not a reciprocal.
        let sqrt_head_dim = (head_dim as f32).sqrt();
        let scratch_max = cap * 4;
        let scratch_sum = cap * 4 + WARPS * 4;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "k_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "v_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "out_ptr") // [num_heads * head_dim]
            .param(PtxType::U32, "seq_len") // valid positions, = position + 1
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let lane = ctx.and_u32_imm(tid, 31);
                let warp = ctx.shr_u32_imm(tid, 5);

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_cache_ptr");
                let v_ptr = ctx.load_param_u64("v_cache_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_len = ctx.load_param_u32("seq_len");

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

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let cap_r = ctx.mov_u32_imm(cap);
                let sqrt_hd = ctx.mov_f32_imm(sqrt_head_dim);
                let four = ctx.mov_u32_imm(4);
                let out_elem_off = ctx.mul_wide_u32_reg(tid, four);

                // Running softmax state, carried across passes (flash decoding).
                let running_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let running_sum = ctx.mov_f32_imm(0.0);
                let acc = ctx.mov_f32_imm(0.0);
                let chunk = ctx.mov_u32_imm(0);

                ctx.label("gdn_attn_chunk_loop");
                let more_chunks = ctx.setp_lt_u32(chunk, seq_len);
                ctx.branch_if_not(more_chunks, "gdn_attn_chunk_end");
                // n_c = min(cap, seq_len - chunk)
                let remaining = ctx.sub_u32_reg(seq_len, chunk);
                let n_c = ctx.min_u32(remaining, cap_r);

                // ---- Phase 1: scores for this pass, one thread per few positions.
                let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let pp = ctx.add_u32(tid, 0);
                ctx.label("gdn_attn_score_loop");
                let score_go = ctx.setp_lt_u32(pp, n_c);
                ctx.branch_if_not(score_go, "gdn_attn_score_end");
                let p = ctx.add_u32_reg(chunk, pp);
                let row_off = ctx.mul_wide_u32(p, row_stride_bytes);
                let k_row = ctx.add_u64(k_head, row_off);

                let dot = ctx.mov_f32_imm(0.0);
                let i = ctx.mov_u32_imm(0);
                ctx.label("gdn_attn_dot_loop");
                let dot_go = ctx.setp_lt_u32(i, head_dim_r);
                ctx.branch_if_not(dot_go, "gdn_attn_dot_end");
                let elem_off = ctx.mul_wide_u32_reg(i, four);
                let q_addr = ctx.add_u64(q_base, elem_off);
                let k_addr = ctx.add_u64(k_row, elem_off);
                let q_val = ctx.ld_global_f32(q_addr);
                let k_val = ctx.ld_global_f32(k_addr);
                // mul then add, like the CPU's `dot += q_h[i] * k_p[i]` — Rust does
                // not contract into an fma, so neither do we.
                let prod = ctx.mul_f32(q_val, k_val);
                ctx.add_f32_inplace(dot, prod);
                ctx.add_u32_inplace(i, 1);
                ctx.branch("gdn_attn_dot_loop");
                ctx.label("gdn_attn_dot_end");

                let score = ctx.div_f32(dot, sqrt_hd);
                let score_slot = ctx.mul_u32(pp, 4);
                let score_addr = ctx.cvt_u64_u32(score_slot);
                ctx.st_shared_f32(score_addr, score);
                ctx.max_f32_inplace(local_max, score);
                ctx.add_u32_inplace(pp, BLOCK);
                ctx.branch("gdn_attn_score_loop");
                ctx.label("gdn_attn_score_end");
                ctx.bar_sync(0);

                // ---- Phase 2: softmax over the pass, folded into the running state.
                let chunk_max =
                    emit_block_reduce(ctx, local_max, lane, warp, scratch_max, ReduceOp::Max, "gdn_attn_max");
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

                let chunk_sum =
                    emit_block_reduce(ctx, local_sum, lane, warp, scratch_sum, ReduceOp::Sum, "gdn_attn_sum");
                ctx.mul_f32_inplace(running_sum, correction);
                ctx.add_f32_inplace(running_sum, chunk_sum);
                ctx.mul_f32_inplace(acc, correction);

                // ---- Phase 3: thread `tid` accumulates output element `tid`,
                // positions ascending, exactly the CPU's accumulation order.
                let in_head = ctx.setp_lt_u32(tid, head_dim_r);
                ctx.branch_if_not(in_head, "gdn_attn_value_skip");
                let vp = ctx.mov_u32_imm(0);
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

                // out = sum_p exp(s_p - max) * v_p / sum_p exp(s_p - max)
                let in_head_out = ctx.setp_lt_u32(tid, head_dim_r);
                ctx.branch_if_not(in_head_out, "gdn_attn_exit");
                let result = ctx.div_f32(acc, running_sum);
                let out_addr = ctx.add_u64(out_base, out_elem_off);
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
}
