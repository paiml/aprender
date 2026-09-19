//! PMAT-3477: the gated delta-rule recurrence, one token.
//!
//! `delta_rule_recurrence` in the CPU reference, per value head `h` with the state
//! `S_h` stored transposed (`s_h[j * D + i] == S[i][j]`, so memory row `j` is column
//! `j` of `S`):
//!
//! ```text
//! 1. s_h        *= exp(gate[h])
//! 2. delta[j]    = (v[j] - sum_i s_h[j*D+i] * k[i]) * beta[h]
//! 3. s_h[j*D+i] += k[i] * delta[j]
//! 4. out[j]      = (sum_i s_h[j*D+i] * q[i]) * D^-0.5
//! ```
//!
//! Grid: `(num_v_heads, 1, 1)`, Block: `(head_v_dim, 1, 1)`. Thread `j` owns memory
//! row `j` of `S_h` **exclusively**: every one of steps 1–4 touches only row `j` for
//! output `j`, so the whole recurrence runs with no barrier and no cross-thread
//! dependency. Steps 1 and 2 are fused into one ascending pass over `i` and steps 3
//! and 4 into a second, which keeps the fp32 accumulation order identical to the CPU
//! loop (`i = 0..D`) — the per-layer L∞ ≤ 1e-3 parity contract depends on it.

use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Gated delta-rule recurrence for a single token.
///
/// For Qwen3.5-0.8B: `num_v_heads = 16`, `head_v_dim = 128`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleRecurrenceKernel {
    /// Number of value heads.
    pub num_v_heads: u32,
    /// Value head width `D` (also the state's row and column count).
    pub head_v_dim: u32,
}

impl DeltaRuleRecurrenceKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_v_heads: u32, head_v_dim: u32) -> Self {
        Self {
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
        let d = self.head_v_dim;
        // The CPU computes `1.0 / (head_v_dim as f32).sqrt()` once; the same value is
        // folded in here as an immediate so no rsqrt approximation enters the output.
        let scale = 1.0 / (d as f32).sqrt();

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_v_heads * D]
            .param(PtxType::U64, "k_ptr") // [num_v_heads * D]
            .param(PtxType::U64, "v_ptr") // [num_v_heads * D]
            .param(PtxType::U64, "beta_ptr") // [num_v_heads]
            .param(PtxType::U64, "gate_ptr") // [num_v_heads] (dt)
            .param(PtxType::U64, "state_ptr") // [num_v_heads * D * D], updated in place
            .param(PtxType::U64, "output_ptr") // [num_v_heads * D]
            .shared_memory(0)
            .build(|ctx| {
                let j = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);

                let d_r = ctx.mov_u32_imm(d);
                let row_in_bounds = ctx.setp_lt_u32(j, d_r);
                ctx.branch_if_not(row_in_bounds, "gdn_dr_exit");

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let four = ctx.mov_u32_imm(4);
                let d_bytes = ctx.mov_u32_imm(d * 4);

                // q/k/v/out head base: h * D * 4
                let head_off = ctx.mul_wide_u32_reg(h, d_bytes);
                let q_base = ctx.add_u64(q_ptr, head_off);
                let k_base = ctx.add_u64(k_ptr, head_off);
                let v_base = ctx.add_u64(v_ptr, head_off);
                let out_base = ctx.add_u64(output_ptr, head_off);

                // state row base: (h * D * D + j * D) * 4
                let state_head_bytes = ctx.mov_u32_imm(d * d * 4);
                let state_head_off = ctx.mul_wide_u32_reg(h, state_head_bytes);
                let state_head = ctx.add_u64(state_ptr, state_head_off);
                let row_off = ctx.mul_wide_u32_reg(j, d_bytes);
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
                let go = ctx.setp_lt_u32(i, d_r);
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
                let go2 = ctx.setp_lt_u32(i2, d_r);
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
        let kernel = DeltaRuleRecurrenceKernel::new(16, 128);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_delta_rule_recurrence"), "{ptx}");
        // exp(gate) is the only transcendental in the recurrence.
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 1, "{ptx}");
        // No barrier: thread j owns state row j exclusively.
        assert!(!ptx.contains("bar.sync"), "{ptx}");
        // D^-0.5 is a host-computed immediate, not an rsqrt.
        assert!(!ptx.contains("rsqrt"), "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
        assert_eq!(kernel.block(), (128, 1, 1));
    }
}

/// Device parity against a verbatim port of `delta_rule_recurrence`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_delta_rule_device_tests {
    use super::DeltaRuleRecurrenceKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg, HEAD_DIM, NUM_V_HEADS};

    /// Verbatim port of `aprender-serve`'s `delta_rule_recurrence`
    /// (forward_qwen35.rs:121).
    #[allow(clippy::too_many_arguments)]
    fn delta_rule_recurrence(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        gate: &[f32],
        state: &mut [f32],
        output: &mut [f32],
        num_v_heads: usize,
        head_v_dim: usize,
    ) {
        let scale = 1.0 / (head_v_dim as f32).sqrt();

        for h in 0..num_v_heads {
            let q_h = &q[h * head_v_dim..(h + 1) * head_v_dim];
            let k_h = &k[h * head_v_dim..(h + 1) * head_v_dim];
            let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
            let beta_val = beta[h];
            let gate_val = gate[h];

            let state_offset = h * head_v_dim * head_v_dim;
            let s_h = &mut state[state_offset..state_offset + head_v_dim * head_v_dim];

            let exp_gate = gate_val.exp();
            for s in s_h.iter_mut() {
                *s *= exp_gate;
            }

            let mut delta = vec![0.0; head_v_dim];
            for j in 0..head_v_dim {
                let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
                let mut sum = 0.0;
                for i in 0..head_v_dim {
                    sum += row_j[i] * k_h[i];
                }
                delta[j] = (v_h[j] - sum) * beta_val;
            }

            for j in 0..head_v_dim {
                let row_j = &mut s_h[j * head_v_dim..(j + 1) * head_v_dim];
                let d_j = delta[j];
                for i in 0..head_v_dim {
                    row_j[i] += k_h[i] * d_j;
                }
            }

            for j in 0..head_v_dim {
                let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
                let mut sum = 0.0;
                for i in 0..head_v_dim {
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

    #[test]
    fn gdn_delta_rule_recurrence_matches_cpu_reference_three_steps() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_delta_rule_recurrence: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let n = NUM_V_HEADS * HEAD_DIM;
        let mut rng = Lcg::new(0x3477_0004);
        let mut host_state = rng.vec(NUM_V_HEADS * HEAD_DIM * HEAD_DIM, 0.05);
        let state_buf = GpuBuffer::from_host(&ctx, &host_state).expect("state");
        let out_buf = GpuBuffer::<f32>::new(&ctx, n).expect("out");
        let kernel = DeltaRuleRecurrenceKernel::new(NUM_V_HEADS as u32, HEAD_DIM as u32);

        for step in 0..3 {
            let mut q = rng.vec(n, 1.0);
            let mut k = rng.vec(n, 1.0);
            l2_norm_per_head(&mut q, HEAD_DIM);
            l2_norm_per_head(&mut k, HEAD_DIM);
            let v = rng.vec(n, 1.0);
            let beta: Vec<f32> = (0..NUM_V_HEADS).map(|_| 0.5 + 0.25 * rng.next()).collect();
            // dt = softplus(...) * ssm_a, with ssm_a negative: the state decays.
            let gate: Vec<f32> = (0..NUM_V_HEADS)
                .map(|_| -0.1 - 0.3 * rng.next().abs())
                .collect();

            let mut want = vec![0.0f32; n];
            delta_rule_recurrence(
                &q,
                &k,
                &v,
                &beta,
                &gate,
                &mut host_state,
                &mut want,
                NUM_V_HEADS,
                HEAD_DIM,
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

            let mut got = vec![0.0f32; n];
            out_buf.copy_to_host(&mut got).expect("download output");
            assert_close(
                &got,
                &want,
                1e-3,
                &format!("delta-rule output, step {step}"),
            );

            let mut got_state = vec![0.0f32; host_state.len()];
            state_buf
                .copy_to_host(&mut got_state)
                .expect("download state");
            assert_close(
                &got_state,
                &host_state,
                1e-3,
                &format!("delta-rule state after step {step}"),
            );

            // The fixture must actually move the state, or the carry-over is untested.
            let moved = got_state.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            assert!(moved > 1e-3, "state is inert at step {step}");
        }
    }
}
