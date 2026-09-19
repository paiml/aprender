//! PMAT-3477: the gated delta-rule recurrence, one token.
//!
//! `delta_rule_recurrence_gqa` in the CPU reference, per **value** head `h` with the
//! state `S_h` stored transposed (`s_h[j * Dk + i] == S[i][j]`, so memory row `j` is
//! column `j` of `S`, `i` runs over `head_k_dim` and `j` over `head_v_dim`):
//!
//! ```text
//! kh             = h % num_k_heads          (the key/query head this value head reads)
//! 1. s_h         *= exp(gate[h])
//! 2. delta[j]     = (v[j] - sum_i s_h[j*Dk+i] * k[kh*Dk+i]) * beta[h]
//! 3. s_h[j*Dk+i] += k[kh*Dk+i] * delta[j]
//! 4. out[j]       = (sum_i s_h[j*Dk+i] * q[kh*Dk+i]) * Dk^-0.5
//! ```
//!
//! `num_v_heads` may exceed `num_k_heads` (Qwen3.5-4B/9B: 32 against 16; 27B: 48
//! against 16). The mapping is `h % num_k_heads`, **tiled**, not `h / ratio`: the GGUF
//! conversion (`_LinearAttentionVReorderBase`) already permutes the value heads out of
//! HF's grouped order so that llama.cpp can expand q/k with `ggml_repeat_4d`. The CPU
//! reference (`gguf/inference/forward/forward_qwen35.rs`) carries the citations and the
//! measurement; this kernel only mirrors it.
//!
//! Grid: `(num_v_heads, 1, 1)`, Block: `(head_v_dim, 1, 1)`. Thread `j` owns memory
//! row `j` of `S_h` **exclusively**: every one of steps 1–4 touches only row `j` for
//! output `j`, so the whole recurrence runs with no barrier and no cross-thread
//! dependency. Steps 1 and 2 are fused into one ascending pass over `i` and steps 3
//! and 4 into a second, which keeps the fp32 accumulation order identical to the CPU
//! loop (`i = 0..head_k_dim`) — the per-layer L∞ ≤ 1e-3 parity contract depends on it.
//! Two value heads that share a key head still touch disjoint state and output, so the
//! grouped case needs no more synchronisation than the symmetric one.

use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Gated delta-rule recurrence for a single token.
///
/// For Qwen3.5-0.8B and -2B: `num_k_heads = num_v_heads = 16`,
/// `head_k_dim = head_v_dim = 128`. For -4B and -9B: `num_k_heads = 16`,
/// `num_v_heads = 32`. For -27B: `num_k_heads = 16`, `num_v_heads = 48`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleRecurrenceKernel {
    /// Number of key/query heads — `q` and `k` are `num_k_heads * head_k_dim` long.
    pub num_k_heads: u32,
    /// Key/query head width `Dk`, the state's row length and the recurrence's scale.
    pub head_k_dim: u32,
    /// Number of value heads — one block each.
    pub num_v_heads: u32,
    /// Value head width `Dv`, the state's row count and the output width per head.
    pub head_v_dim: u32,
}

impl DeltaRuleRecurrenceKernel {
    /// Create the kernel.
    ///
    /// `num_v_heads` is expected to be a positive multiple of `num_k_heads`, exactly as
    /// the CPU reference asserts; a `num_k_heads` of zero is clamped to one so the PTX
    /// never emits a `rem.u32` by zero.
    #[must_use]
    pub const fn new(num_k_heads: u32, head_k_dim: u32, num_v_heads: u32, head_v_dim: u32) -> Self {
        Self {
            num_k_heads: if num_k_heads == 0 { 1 } else { num_k_heads },
            head_k_dim,
            num_v_heads,
            head_v_dim,
        }
    }

    /// Launch grid — one block per value head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_v_heads, 1, 1)
    }

    /// Launch block — one thread per state row.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (self.head_v_dim, 1, 1)
    }
}

impl Kernel for DeltaRuleRecurrenceKernel {
    fn name(&self) -> &str {
        "gdn_delta_rule_recurrence"
    }

    fn build_ptx(&self) -> PtxKernel {
        let dk = self.head_k_dim;
        let dv = self.head_v_dim;
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        // The CPU computes `1.0 / (head_k_dim as f32).sqrt()` once; the same value is
        // folded in here as an immediate so no rsqrt approximation enters the output.
        let scale = 1.0 / (dk as f32).sqrt();

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_k_heads * Dk]
            .param(PtxType::U64, "k_ptr") // [num_k_heads * Dk]
            .param(PtxType::U64, "v_ptr") // [num_v_heads * Dv]
            .param(PtxType::U64, "beta_ptr") // [num_v_heads]
            .param(PtxType::U64, "gate_ptr") // [num_v_heads] (dt)
            .param(PtxType::U64, "state_ptr") // [num_v_heads * Dv * Dk], updated in place
            .param(PtxType::U64, "output_ptr") // [num_v_heads * Dv]
            .shared_memory(0)
            .build(|ctx| {
                let j = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);

                let dv_r = ctx.mov_u32_imm(dv);
                let row_in_bounds = ctx.setp_lt_u32(j, dv_r);
                ctx.branch_if_not(row_in_bounds, "gdn_dr_exit");

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let four = ctx.mov_u32_imm(4);
                let dk_r = ctx.mov_u32_imm(dk);
                let dk_bytes = ctx.mov_u32_imm(dk * 4);
                let dv_bytes = ctx.mov_u32_imm(dv * 4);

                // q/k head base: (h % num_k_heads) * Dk * 4. With nk == nv the block
                // index IS the key head, so the `rem` is not emitted at all and the
                // symmetric PTX is unchanged.
                let kh = if nk == nv { h } else { ctx.rem_u32(h, nk) };
                let qk_head_off = ctx.mul_wide_u32_reg(kh, dk_bytes);
                let q_base = ctx.add_u64(q_ptr, qk_head_off);
                let k_base = ctx.add_u64(k_ptr, qk_head_off);

                // v/out head base: h * Dv * 4
                let v_head_off = ctx.mul_wide_u32_reg(h, dv_bytes);
                let v_base = ctx.add_u64(v_ptr, v_head_off);
                let out_base = ctx.add_u64(output_ptr, v_head_off);

                // state row base: (h * Dv * Dk + j * Dk) * 4
                let state_head_bytes = ctx.mov_u32_imm(dv * dk * 4);
                let state_head_off = ctx.mul_wide_u32_reg(h, state_head_bytes);
                let state_head = ctx.add_u64(state_ptr, state_head_off);
                let row_off = ctx.mul_wide_u32_reg(j, dk_bytes);
                let s_row = ctx.add_u64(state_head, row_off);

                // Per-head scalars, read once.
                let scalar_off = ctx.mul_wide_u32_reg(h, four);
                let beta_addr = ctx.add_u64(beta_ptr, scalar_off);
                let gate_addr = ctx.add_u64(gate_ptr, scalar_off);
                let beta = ctx.ld_global_f32(beta_addr);
                let gate = ctx.ld_global_f32(gate_addr);
                let exp_gate = emit_exp_f32(ctx, gate);

                // Steps 1 + 2: decay row j in place and dot it with k, i ascending.
                let sum = ctx.mov_f32_imm(0.0);
                let i = ctx.mov_u32_imm(0);
                ctx.label("gdn_dr_decay_loop");
                let go = ctx.setp_lt_u32(i, dk_r);
                ctx.branch_if_not(go, "gdn_dr_decay_end");
                let off = ctx.mul_wide_u32_reg(i, four);
                let s_addr = ctx.add_u64(s_row, off);
                let k_addr = ctx.add_u64(k_base, off);
                let s_val = ctx.ld_global_f32(s_addr);
                let s_scaled = ctx.mul_f32(s_val, exp_gate);
                ctx.st_global_f32(s_addr, s_scaled);
                let k_val = ctx.ld_global_f32(k_addr);
                let prod = ctx.mul_f32(s_scaled, k_val);
                ctx.add_f32_inplace(sum, prod);
                ctx.add_u32_inplace(i, 1);
                ctx.branch("gdn_dr_decay_loop");
                ctx.label("gdn_dr_decay_end");

                // delta[j] = (v[j] - sum) * beta
                let v_off = ctx.mul_wide_u32_reg(j, four);
                let v_addr = ctx.add_u64(v_base, v_off);
                let v_j = ctx.ld_global_f32(v_addr);
                let diff = ctx.sub_f32(v_j, sum);
                let delta = ctx.mul_f32(diff, beta);

                // Steps 3 + 4: update row j and dot it with q, i ascending.
                let out_sum = ctx.mov_f32_imm(0.0);
                let i2 = ctx.mov_u32_imm(0);
                ctx.label("gdn_dr_update_loop");
                let go2 = ctx.setp_lt_u32(i2, dk_r);
                ctx.branch_if_not(go2, "gdn_dr_update_end");
                let off2 = ctx.mul_wide_u32_reg(i2, four);
                let s_addr2 = ctx.add_u64(s_row, off2);
                let k_addr2 = ctx.add_u64(k_base, off2);
                let q_addr2 = ctx.add_u64(q_base, off2);
                let s_old = ctx.ld_global_f32(s_addr2);
                let k_val2 = ctx.ld_global_f32(k_addr2);
                let upd = ctx.mul_f32(k_val2, delta);
                let s_new = ctx.add_f32(s_old, upd);
                ctx.st_global_f32(s_addr2, s_new);
                let q_val = ctx.ld_global_f32(q_addr2);
                let prod2 = ctx.mul_f32(s_new, q_val);
                ctx.add_f32_inplace(out_sum, prod2);
                ctx.add_u32_inplace(i2, 1);
                ctx.branch("gdn_dr_update_loop");
                ctx.label("gdn_dr_update_end");

                let scale_r = ctx.mov_f32_imm(scale);
                let result = ctx.mul_f32(out_sum, scale_r);
                let out_addr = ctx.add_u64(out_base, v_off);
                ctx.st_global_f32(out_addr, result);

                ctx.label("gdn_dr_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_delta_rule_ptx_shape() {
        let kernel = DeltaRuleRecurrenceKernel::new(16, 128, 16, 128);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_delta_rule_recurrence"), "{ptx}");
        // exp(gate) is the only transcendental in the recurrence.
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 1, "{ptx}");
        // No barrier: thread j owns state row j exclusively.
        assert!(!ptx.contains("bar.sync"), "{ptx}");
        // Dk^-0.5 is a host-computed immediate, not an rsqrt.
        assert!(!ptx.contains("rsqrt"), "{ptx}");
        // Symmetric heads: the block index IS the key head, so no modulo is emitted.
        assert!(!ptx.contains("rem.u32"), "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
        assert_eq!(kernel.block(), (128, 1, 1));
    }

    /// Qwen3.5-4B/9B (`nk = 16`, `nv = 32`) and -27B (`nk = 16`, `nv = 48`): the block
    /// count follows the VALUE heads and the key head is `h % num_k_heads`.
    #[test]
    fn gdn_delta_rule_ptx_grouped_heads_emit_the_tiled_modulo() {
        for (nk, nv) in [(16u32, 32u32), (16, 48)] {
            let kernel = DeltaRuleRecurrenceKernel::new(nk, 128, nv, 128);
            let ptx = kernel.emit_ptx();
            assert_eq!(kernel.grid(), (nv, 1, 1), "one block per VALUE head");
            assert_eq!(kernel.block(), (128, 1, 1));
            assert_eq!(
                ptx.matches("rem.u32").count(),
                1,
                "the key head must be h % {nk}: {ptx}"
            );
            assert!(
                ptx.contains(&format!("{nk};")) || ptx.contains(&format!("{nk} ")),
                "{ptx}"
            );
        }
    }

    /// A rectangular state (`Dk != Dv`) sizes the row length from the KEY dim and the
    /// row count from the VALUE dim.
    #[test]
    fn gdn_delta_rule_rectangular_state_uses_dk_for_the_row() {
        let kernel = DeltaRuleRecurrenceKernel::new(2, 8, 4, 6);
        assert_eq!(kernel.grid(), (4, 1, 1));
        assert_eq!(kernel.block(), (6, 1, 1), "one thread per state ROW (Dv)");
        let ptx = kernel.emit_ptx();
        // The per-head state block is Dv * Dk * 4 = 6 * 8 * 4 = 192 bytes.
        assert!(ptx.contains("192"), "{ptx}");
    }

    /// `num_k_heads = 0` would be a `rem.u32` by zero; it is clamped instead.
    #[test]
    fn gdn_delta_rule_zero_key_heads_is_clamped() {
        assert_eq!(DeltaRuleRecurrenceKernel::new(0, 8, 4, 8).num_k_heads, 1);
    }
}

/// Device parity against a verbatim port of `delta_rule_recurrence`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_delta_rule_device_tests {
    use super::DeltaRuleRecurrenceKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg, HEAD_DIM, NUM_V_HEADS};

    /// Verbatim port of `aprender-serve`'s `delta_rule_recurrence_gqa`
    /// (forward_qwen35.rs:216) — the specification for this kernel. `h % num_k_heads`
    /// is the reference's own head mapping, not a restatement of the kernel's.
    #[allow(clippy::too_many_arguments)]
    fn delta_rule_recurrence_gqa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        gate: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        num_k_heads: usize,
        head_k_dim: usize,
        num_v_heads: usize,
        head_v_dim: usize,
    ) {
        assert_eq!(q.len(), num_k_heads * head_k_dim);
        assert_eq!(k.len(), num_k_heads * head_k_dim);
        assert_eq!(v.len(), num_v_heads * head_v_dim);
        assert_eq!(state.len(), num_v_heads * head_v_dim * head_k_dim);

        let scale = 1.0 / (head_k_dim as f32).sqrt();

        for h in 0..num_v_heads {
            let kh = h % num_k_heads;
            let q_h = &q[kh * head_k_dim..(kh + 1) * head_k_dim];
            let k_h = &k[kh * head_k_dim..(kh + 1) * head_k_dim];
            let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
            let beta_val = beta[h];
            let gate_val = gate[h];

            let state_stride = head_v_dim * head_k_dim;
            let state_offset = h * state_stride;
            let s_h = &mut state[state_offset..state_offset + state_stride];

            let exp_gate = gate_val.exp();
            for s in s_h.iter_mut() {
                *s *= exp_gate;
            }

            let mut delta = vec![0.0; head_v_dim];
            for j in 0..head_v_dim {
                let row_j = &s_h[j * head_k_dim..(j + 1) * head_k_dim];
                let mut sum = 0.0;
                for i in 0..head_k_dim {
                    sum += row_j[i] * k_h[i];
                }
                delta[j] = (v_h[j] - sum) * beta_val;
            }

            for j in 0..head_v_dim {
                let row_j = &mut s_h[j * head_k_dim..(j + 1) * head_k_dim];
                let d_j = delta[j];
                for i in 0..head_k_dim {
                    row_j[i] += k_h[i] * d_j;
                }
            }

            for j in 0..head_v_dim {
                let row_j = &s_h[j * head_k_dim..(j + 1) * head_k_dim];
                let mut sum = 0.0;
                for i in 0..head_k_dim {
                    sum += row_j[i] * q_h[i];
                }
                output[h * head_v_dim + j] = sum * scale;
            }
        }
    }

    /// q and k arrive L2-normalised per head, so reproduce that here.
    fn l2_norm_per_head(x: &mut [f32], head_dim: usize) {
        for head in x.chunks_exact_mut(head_dim) {
            let sq: f32 = head.iter().map(|v| v * v).sum();
            let s = 1.0 / (sq + 1e-6).sqrt();
            for v in head.iter_mut() {
                *v *= s;
            }
        }
    }

    /// Per-element budget, **relative to the reference's own scale** (that is what
    /// [`assert_close`] applies it as — an absolute tolerance on outputs that live
    /// near 5e-3 is vacuous, PMAT-3477).
    ///
    /// MEASURED on this box (RTX 4090, sm_89), worst over all four shapes below and
    /// all three steps, output and state alike: **2.2e-7** relative
    /// (`nk 2 dk 8 nv 4 dv 8`, 1.49e-8 against a 6.84e-2 scale). The kernel keeps the
    /// CPU's fp32 accumulation order, so the whole residual is one ulp of the
    /// accumulator; 1e-5 is ~45x that and still two thousand times tighter than the
    /// 1e-2 a mis-indexed head mapping produces.
    const TOL: f32 = 1e-5;

    /// Three decode steps of the kernel against the host port, at any head shape.
    ///
    /// Returns `false` when there is no CUDA device (so the caller can say SKIPPED).
    fn three_steps_match_the_reference(
        num_k_heads: usize,
        head_k_dim: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        seed: u32,
    ) -> bool {
        let Ok(ctx) = CudaContext::new(0) else {
            return false;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let qk_n = num_k_heads * head_k_dim;
        let n = num_v_heads * head_v_dim;
        let mut rng = Lcg::new(seed);
        let mut host_state = rng.vec(num_v_heads * head_v_dim * head_k_dim, 0.05);
        let state_buf = GpuBuffer::from_host(&ctx, &host_state).expect("state");
        let out_buf = GpuBuffer::<f32>::new(&ctx, n).expect("out");
        let kernel = DeltaRuleRecurrenceKernel::new(
            num_k_heads as u32,
            head_k_dim as u32,
            num_v_heads as u32,
            head_v_dim as u32,
        );

        for step in 0..3 {
            let mut q = rng.vec(qk_n, 1.0);
            let mut k = rng.vec(qk_n, 1.0);
            l2_norm_per_head(&mut q, head_k_dim);
            l2_norm_per_head(&mut k, head_k_dim);
            let v = rng.vec(n, 1.0);
            let beta: Vec<f32> = (0..num_v_heads).map(|_| 0.5 + 0.25 * rng.next()).collect();
            // dt = softplus(...) * ssm_a, with ssm_a negative: the state decays.
            let gate: Vec<f32> = (0..num_v_heads)
                .map(|_| -0.1 - 0.3 * rng.next().abs())
                .collect();

            let mut want = vec![0.0f32; n];
            delta_rule_recurrence_gqa(
                &q,
                &k,
                &v,
                &beta,
                &gate,
                &mut host_state,
                &mut want,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
            );

            let q_buf = GpuBuffer::from_host(&ctx, &q).expect("q");
            let k_buf = GpuBuffer::from_host(&ctx, &k).expect("k");
            let v_buf = GpuBuffer::from_host(&ctx, &v).expect("v");
            let beta_buf = GpuBuffer::from_host(&ctx, &beta).expect("beta");
            let gate_buf = GpuBuffer::from_host(&ctx, &gate).expect("gate");
            let mut args = [
                q_buf.as_ptr(),
                k_buf.as_ptr(),
                v_buf.as_ptr(),
                beta_buf.as_ptr(),
                gate_buf.as_ptr(),
                state_buf.as_ptr(),
                out_buf.as_ptr(),
            ];
            run_kernel(
                &ctx,
                &stream,
                &kernel,
                kernel.grid(),
                kernel.block(),
                &mut args,
            );

            let shape =
                format!("nk {num_k_heads} dk {head_k_dim} nv {num_v_heads} dv {head_v_dim}");
            let mut got = vec![0.0f32; n];
            out_buf.copy_to_host(&mut got).expect("download output");
            assert_close(
                &got,
                &want,
                TOL,
                &format!("delta-rule output, step {step} ({shape})"),
            );

            let mut got_state = vec![0.0f32; host_state.len()];
            state_buf
                .copy_to_host(&mut got_state)
                .expect("download state");
            assert_close(
                &got_state,
                &host_state,
                TOL,
                &format!("delta-rule state after step {step} ({shape})"),
            );

            // The fixture must actually move the state, or the carry-over is untested.
            let moved = got_state.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            assert!(moved > 1e-3, "state is inert at step {step} ({shape})");
        }
        true
    }

    /// Qwen3.5-0.8B / -2B: one key head per value head, square state. Unchanged.
    #[test]
    fn gdn_delta_rule_recurrence_matches_cpu_reference_three_steps() {
        if !three_steps_match_the_reference(
            NUM_V_HEADS,
            HEAD_DIM,
            NUM_V_HEADS,
            HEAD_DIM,
            0x3477_0004,
        ) {
            println!("gdn_delta_rule_recurrence: no CUDA device — SKIPPED.");
        }
    }

    /// PMAT-3477 (#3346/#3510): two value heads per key head, as Qwen3.5-4B/9B have.
    ///
    /// This is the case the kernel refused before: `q` and `k` are HALF as long as `v`,
    /// and value heads 0/2 and 1/3 must read key heads 0 and 1 — `h % num_k_heads`.
    /// A kernel that kept the old one-stride-for-everything indexing reads `q`/`k` past
    /// their allocation for `h >= num_k_heads`, so this fails loudly rather than
    /// silently.
    #[test]
    fn gdn_delta_rule_recurrence_grouped_key_heads_match_cpu_reference() {
        if !three_steps_match_the_reference(2, 8, 4, 8, 0x3477_0032) {
            println!("gdn_delta_rule_recurrence (grouped): no CUDA device — SKIPPED.");
        }
    }

    /// The same, with three value heads per key head — `num_v_heads / num_k_heads` is
    /// not hard-coded to 2 anywhere.
    #[test]
    fn gdn_delta_rule_recurrence_ratio_three_matches_cpu_reference() {
        if !three_steps_match_the_reference(2, 8, 6, 8, 0x3477_0033) {
            println!("gdn_delta_rule_recurrence (ratio 3): no CUDA device — SKIPPED.");
        }
    }

    /// Qwen3.5-27B's own head counts — `nk = 16`, `nv = 48`, ratio 3 — at a small head
    /// width so the test costs nothing.
    ///
    /// The 27B file is 16 GB and there is no host here that can hold it alongside a
    /// CPU reference run, so this is the only place its head shape is exercised at all.
    /// The two smaller ratio tests above use `nk = 2`, which cannot tell `h % nk` from
    /// `h & (nk - 1)` or from a ratio hard-coded to 2; 16 and 48 pin both.
    #[test]
    fn gdn_delta_rule_recurrence_27b_head_counts_match_cpu_reference() {
        if !three_steps_match_the_reference(16, 8, 48, 8, 0x3477_0048) {
            println!("gdn_delta_rule_recurrence (27B heads): no CUDA device — SKIPPED.");
        }
    }

    /// A rectangular state (`Dk = 8`, `Dv = 6`): the state row length follows the KEY
    /// dim, the row count and the output follow the VALUE dim, and the scale is
    /// `Dk^-0.5`. No Qwen3.5 file is rectangular today; this is what keeps the two dims
    /// from silently collapsing back into one.
    #[test]
    fn gdn_delta_rule_recurrence_rectangular_state_matches_cpu_reference() {
        if !three_steps_match_the_reference(2, 8, 4, 6, 0x3477_0034) {
            println!("gdn_delta_rule_recurrence (rectangular): no CUDA device — SKIPPED.");
        }
    }
}
