//! Batched Incremental Attention Kernel (PAR-118)

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// PAR-118: Batched Incremental Attention for M sequences in parallel
///
/// Processes M independent sequences in a single kernel launch, reducing
/// kernel launch overhead from 3M to 3 per layer (batched KV scatter + batched attention).
///
/// Grid: (num_heads, batch_size, 1)
/// Block: (32, 1, 1) - one warp per head
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - output: [M, num_heads, head_dim] - contiguous output
/// - seq_lens: [M] - array of M sequence lengths (indirect mode)
#[derive(Debug, Clone)]
pub struct BatchedIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl BatchedIncrementalAttentionKernel {
    /// Create a new batched incremental attention kernel
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        batch_size: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

impl Kernel for BatchedIncrementalAttentionKernel {
    fn name(&self) -> &str {
        "batched_incremental_attention"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch) pair
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("batched_incremental_attention")
            .param(PtxType::U64, "q_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr") // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr") // [M] array of V cache pointers
            .param(PtxType::U64, "out_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr") // [M] array of sequence lengths
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");

                // Load seq_len for this batch element
                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q/output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (same as IncrementalAttentionKernel)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over sequence positions
                let pos = ctx.mov_u32_imm(0);
                ctx.label("batched_seq_loop");
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "batched_seq_loop_end");

                // Load K[pos] and compute Q·K dot product
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce - use full warp mask for all 32 threads
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // PAR-118-FIX: Broadcast reduced dot product from lane 0 to all lanes.
                // After shfl_down reduction, only lane 0 has the correct sum.
                // All lanes need the score for softmax and V accumulation.
                let dot = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update (Milakov & Gimelshein 2018).
                //
                // PERF-050 / FALSIFY-CB-008: copy max_score into a NEW register before the
                // in-place max. `let old_max = max_score;` binds the SAME VirtualReg, so
                // `max_f32_inplace` below clobbered it too and the rescale factor emitted as
                // `sub.f32 %f27, %f8, %f8;` -> `ex2(0)` -> correction == 1.0 for every
                // position. The running max still tracked correctly and `exp_score` was still
                // right, so nothing overflowed or NaN'd; what broke is that `sum_exp` and the
                // `out*` accumulators were never brought onto the new max's scale. Every term
                // accumulated before a max increase stays weighted by exp(old_max - new_max)
                // too much, which silently over-weights early KV positions by an unbounded
                // factor. The output is a plausible-magnitude but wrong attention vector, and
                // 28 layers of it is the `!!!!` / `strarstrar...` garbage in aprender#2753.
                //
                // The identical hazard is documented at the fixed sibling
                // flash_decoding/chunk_kernel.rs; incremental.rs (the M=1 decode kernel that
                // the fast path uses, and the reason m=1 looked healthy) sidesteps it by
                // computing `new_max` into a fresh register instead of updating in place.
                let old_max = ctx.mov_f32_imm(0.0);
                ctx.mov_f32_reg(old_max, max_score);
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("batched_seq_loop");

                ctx.label("batched_seq_loop_end");

                // Normalize output
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);
                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "batched_skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("batched_skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "batched_skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("batched_skip_store1");

                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "batched_skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("batched_skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "batched_skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("batched_skip_store3");

                ctx.ret();
            })
    }
}

/// FALSIFY-CB-008 (`contracts/continuous-batching-v1.yaml`), executed rather than described.
///
/// The contract's rule is "No frozen slots — all M slots produce distinct tokens per decode
/// step (not constant)" and its `test:` field named a `BATCHED_DECODE_TRACE` log nobody ever
/// read. aprender#2753 is that rule failing: every slot served from a batch emitted one token
/// to the `max_tokens` cap. The mechanism turned out to be one line of this kernel, so the
/// obligation is discharged here, at the defect, in a check that needs no GPU: PTX generation
/// is pure string building, so this runs anywhere the `cuda` feature compiles.
///
/// WHAT IS ASSERTED. The online softmax (Milakov & Gimelshein 2018) must rescale the running
/// `sum_exp` and output accumulators by `exp(old_max - new_max)` whenever the running max
/// grows. That requires the OLD max to survive the in-place `max.f32` that computes the new
/// one. `let old_max = max_score;` in a PTX builder does not copy a value, it binds the same
/// VirtualReg — so the emitted correction was
///
/// ```text
///     max.f32 %f8, %f8, %f23;     // running max updated in place
///     sub.f32 %f27, %f8, %f8;     // "old_max - new_max" — the SAME register
///     ex2.approx.f32 %f29, %f28;  // correction == exp2(0) == 1.0, always
/// ```
///
/// Nothing overflows and nothing is NaN, which is why this survived: the running max is still
/// right and `exp_score` is still right. Only the rescale is missing, so every term accumulated
/// before a max increase keeps a weight that is too large by an unbounded factor. Twenty-eight
/// layers of subtly-wrong attention is the `!!!!` / `strarstrar…` output in #2753.
#[cfg(test)]
mod cb008_online_softmax_rescale {
    use super::BatchedIncrementalAttentionKernel;
    use crate::kernels::attention::paged::flash_decoding::FlashDecodingChunkKernel;
    use crate::kernels::Kernel;

    /// One parsed `op.f32 dst, a, b;` line.
    fn ternary(line: &str, op: &str) -> Option<(String, String, String)> {
        let line = line.trim().trim_end_matches(';');
        let rest = line.strip_prefix(op)?.trim();
        let mut parts = rest.split(',').map(str::trim);
        let dst = parts.next()?.to_string();
        let a = parts.next()?.to_string();
        let b = parts.next()?.to_string();
        if parts.next().is_some() {
            return None;
        }
        Some((dst, a, b))
    }

    /// The property, stated over emitted PTX.
    ///
    /// Returns `Err` when the shape this test reasons about is absent — a check that silently
    /// finds nothing to check is the failure mode this repo keeps hitting, so "not found" is a
    /// failure, never a pass.
    fn rescale_reads_a_saved_max(ptx: &str) -> Result<(), String> {
        // 1. The in-place running-max update: `max.f32 %fM, %fM, %fS;` (dst == first source).
        let running_max = ptx
            .lines()
            .filter_map(|l| ternary(l, "max.f32"))
            .find(|(dst, a, _)| dst == a)
            .map(|(dst, _, _)| dst)
            .ok_or_else(|| {
                "no in-place `max.f32 %fM, %fM, %fS;` found — this kernel does not have the \
                 online-softmax shape this test asserts about, so the assertion is vacuous"
                    .to_string()
            })?;

        // 2. Every `ex2.approx.f32` argument, so we can tell the correction from exp_score.
        let ex2_args: Vec<String> = ptx
            .lines()
            .filter_map(|l| {
                let l = l.trim().trim_end_matches(';');
                let rest = l.strip_prefix("ex2.approx.f32")?.trim();
                rest.split(',').nth(1).map(|s| s.trim().to_string())
            })
            .collect();
        if ex2_args.is_empty() {
            return Err("no `ex2.approx.f32` found — no exponential, so no online softmax".into());
        }

        // 3. `mul.f32 %fX, %fD, %flog2e;` feeding one of those ex2 args, whose %fD came from a
        //    `sub.f32 %fD, %fA, %fM` against the running max. That sub is the rescale term.
        let subs: Vec<(String, String, String)> =
            ptx.lines().filter_map(|l| ternary(l, "sub.f32")).collect();
        let muls: Vec<(String, String, String)> =
            ptx.lines().filter_map(|l| ternary(l, "mul.f32")).collect();

        let mut checked = 0usize;
        for (sub_dst, sub_a, sub_b) in &subs {
            if sub_b != &running_max {
                continue; // not `something - new_max`
            }
            let feeds_ex2 = muls
                .iter()
                .any(|(mul_dst, mul_a, _)| mul_a == sub_dst && ex2_args.contains(mul_dst));
            if !feeds_ex2 {
                continue;
            }
            checked += 1;
            assert_ne!(
                sub_a, sub_b,
                "FALSIFY-CB-008: online-softmax rescale computes `{sub_a} - {sub_b}`, i.e. the \
                 running max minus ITSELF, so the correction is exp2(0) == 1.0 for every KV \
                 position and `sum_exp`/`out` are never brought onto the new max's scale. \
                 `let old_max = max_score;` binds the same VirtualReg; copy it into a fresh one \
                 with `mov_f32_imm` + `mov_f32_reg` first (see flash_decoding/chunk_kernel.rs). \
                 This is aprender#2753: batched CUDA decode emitting a constant token to the cap."
            );
        }
        if checked == 0 {
            return Err(format!(
                "found the running max ({running_max}) but no `sub.f32 _, _, {running_max}` \
                 feeding an ex2 — the rescale term was not located, so nothing was asserted"
            ));
        }
        Ok(())
    }

    /// The load-bearing case: the kernel that #2753 was traced to.
    #[test]
    fn batched_incremental_attention_rescales_online_softmax() {
        // Qwen2.5-Coder-1.5B on the RTX 4090 where #2753 was reproduced: 2048 ctx, head_dim
        // 128, 12 query heads, 2 KV heads (GQA), and a 4-slot batch.
        let kernel = BatchedIncrementalAttentionKernel::new(2048, 128, 12, 2, 4);
        let ptx = kernel.emit_ptx_for_target("sm_89");
        rescale_reads_a_saved_max(&ptx).expect("PTX shape");
    }

    /// Discrimination case. This sibling kernel already carries the fix AND the comment
    /// explaining the hazard, so it must stay GREEN: a checker that is RED on everything
    /// proves nothing about the kernel it was written for.
    #[test]
    fn flash_decoding_chunk_kernel_stays_green() {
        let kernel = FlashDecodingChunkKernel::new(2048, 128, 12, 2, 4);
        let ptx = kernel.emit_ptx_for_target("sm_89");
        rescale_reads_a_saved_max(&ptx).expect("PTX shape");
    }

    /// Positive control: the checker must be ABLE to fire. Without this, a future change to
    /// how instructions are spelled would make `rescale_reads_a_saved_max` match nothing and
    /// the two tests above would pass while asserting nothing.
    #[test]
    fn checker_rejects_a_self_subtraction() {
        let poisoned = "\
            max.f32 %f8, %f8, %f23;\n\
            sub.f32 %f24, %f23, %f8;\n\
            mul.f32 %f25, %f24, %f10;\n\
            ex2.approx.f32 %f26, %f25;\n\
            sub.f32 %f27, %f8, %f8;\n\
            mul.f32 %f28, %f27, %f10;\n\
            ex2.approx.f32 %f29, %f28;\n";
        let caught = std::panic::catch_unwind(|| rescale_reads_a_saved_max(poisoned));
        assert!(
            caught.is_err(),
            "the checker did not fire on PTX that literally contains \
             `sub.f32 %f27, %f8, %f8;` — it cannot detect the defect it exists for"
        );
    }

    /// And it must NOT fire on the repaired form of that same PTX.
    #[test]
    fn checker_accepts_a_saved_max() {
        let repaired = "\
            mov.f32 %f24, %f8;\n\
            max.f32 %f8, %f8, %f23;\n\
            sub.f32 %f25, %f23, %f8;\n\
            mul.f32 %f26, %f25, %f10;\n\
            ex2.approx.f32 %f27, %f26;\n\
            sub.f32 %f28, %f24, %f8;\n\
            mul.f32 %f29, %f28, %f10;\n\
            ex2.approx.f32 %f30, %f29;\n";
        rescale_reads_a_saved_max(repaired).expect("repaired PTX must pass");
    }
}
