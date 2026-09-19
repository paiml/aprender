//! PMAT-3477: gated RMSNorm — the Gated `DeltaNet` output norm.
//!
//! `gated_rmsnorm` in the CPU reference. The **input** is normalised, the gate is
//! not:
//!
//! ```text
//! for each head chunk of head_dim:
//!     rms_scale = 1 / sqrt(sum(input^2) / head_dim + eps)
//!     out[i]    = (input[i] * rms_scale * weight[i]) * silu(gate[i])
//! ```
//!
//! `weight` is `ssm_norm` and is `[head_dim]` — shared across heads, no head stride.
//! `gate` is `attn_gate . x` and *is* indexed per head.
//!
//! Grid: `(num_heads, 1, 1)`, Block: `(32, 1, 1)` — one warp per head.

use crate::kernels::gdn::emit_silu_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Gated RMSNorm over `num_heads` chunks of `head_dim`.
///
/// For Qwen3.5-0.8B: `head_dim = 128` (`head_v_dim`), `num_heads = 16`, `eps = 1e-6`.
#[derive(Debug, Clone, Copy)]
pub struct GatedRmsNormKernel {
    /// Elements per head (`head_v_dim`).
    pub head_dim: u32,
    /// Number of heads (`num_v_heads`).
    pub num_heads: u32,
    /// Epsilon, added to the mean square.
    pub eps: f32,
}

impl GatedRmsNormKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(head_dim: u32, num_heads: u32, eps: f32) -> Self {
        Self {
            head_dim,
            num_heads,
            eps,
        }
    }

    /// Launch grid — one block per head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads, 1, 1)
    }

    /// Launch block — one warp.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for GatedRmsNormKernel {
    fn name(&self) -> &str {
        "gdn_gated_rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let eps = self.eps;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "gate_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "weight_ptr") // [head_dim], shared across heads
            .param(PtxType::U64, "output_ptr") // [num_heads * head_dim]
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);

                let input_ptr = ctx.load_param_u64("input_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let weight_ptr = ctx.load_param_u64("weight_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let four = ctx.mov_u32_imm(4);
                let head_elems = ctx.mul_u32_reg(head_idx, head_dim_r);
                let head_bytes = ctx.mul_wide_u32_reg(head_elems, four);
                let in_base = ctx.add_u64(input_ptr, head_bytes);
                let gate_base = ctx.add_u64(gate_ptr, head_bytes);
                let out_base = ctx.add_u64(output_ptr, head_bytes);

                // Pass 1: sum of squares of the INPUT (the gate is not normalised).
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);
                ctx.label("gdn_grms_sum_loop");
                let i = ctx.add_u32_reg(idx, tid);
                let go = ctx.setp_lt_u32(i, head_dim_r);
                ctx.branch_if_not(go, "gdn_grms_sum_end");
                let off = ctx.mul_wide_u32_reg(i, four);
                let addr = ctx.add_u64(in_base, off);
                let val = ctx.ld_global_f32(addr);
                ctx.fma_f32_inplace(sq_sum, val, val);
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("gdn_grms_sum_loop");
                ctx.label("gdn_grms_sum_end");

                let s16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s16);
                let s8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s8);
                let s4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s4);
                let s2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s2);
                let s1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, s1);
                let total = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                // rms_scale = 1 / sqrt(sum / head_dim + eps)
                let head_dim_f = ctx.cvt_f32_u32(head_dim_r);
                let mean_sq = ctx.div_f32(total, head_dim_f);
                let eps_r = ctx.mov_f32_imm(eps);
                let denom = ctx.add_f32(mean_sq, eps_r);
                let rms_scale = ctx.rsqrt_f32(denom);

                // Pass 2: out[i] = input[i] * rms_scale * weight[i] * silu(gate[i])
                let idx2 = ctx.mov_u32_imm(0);
                ctx.label("gdn_grms_out_loop");
                let i2 = ctx.add_u32_reg(idx2, tid);
                let go2 = ctx.setp_lt_u32(i2, head_dim_r);
                ctx.branch_if_not(go2, "gdn_grms_exit");
                let off2 = ctx.mul_wide_u32_reg(i2, four);
                let in_addr = ctx.add_u64(in_base, off2);
                let gate_addr = ctx.add_u64(gate_base, off2);
                // weight is [head_dim] — indexed by position within the head only.
                let w_addr = ctx.add_u64(weight_ptr, off2);
                let out_addr = ctx.add_u64(out_base, off2);
                let x = ctx.ld_global_f32(in_addr);
                let w = ctx.ld_global_f32(w_addr);
                let g = ctx.ld_global_f32(gate_addr);
                let normed = ctx.mul_f32(x, rms_scale);
                let norm_v = ctx.mul_f32(normed, w);
                let gate_act = emit_silu_f32(ctx, g);
                let result = ctx.mul_f32(norm_v, gate_act);
                ctx.st_global_f32(out_addr, result);
                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("gdn_grms_out_loop");

                ctx.label("gdn_grms_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_gated_rmsnorm_ptx_normalises_the_input_only() {
        let kernel = GatedRmsNormKernel::new(128, 16, 1e-6);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_gated_rmsnorm"), "{ptx}");
        // Exactly one squared-accumulate site: the gate must not enter the norm.
        assert_eq!(ptx.matches("fma.rn.f32").count(), 1, "{ptx}");
        // RMSNorm divides by head_dim before the rsqrt; plus silu's division.
        assert_eq!(ptx.matches("div.rn.f32").count(), 2, "{ptx}");
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 1, "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
    }
}

/// Device parity against a verbatim port of `gated_rmsnorm`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_gated_rmsnorm_device_tests {
    use super::GatedRmsNormKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{
        assert_close, run_kernel, Lcg, EPS, HEAD_DIM, NUM_V_HEADS,
    };

    /// Verbatim port of `silu` (forward_qwen35.rs:7).
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Verbatim port of `gated_rmsnorm` (forward_qwen35.rs:52).
    fn gated_rmsnorm(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        eps: f32,
        head_v_dim: usize,
        output: &mut [f32],
    ) {
        for (chunk_in, (chunk_gate, chunk_out)) in input.chunks_exact(head_v_dim).zip(
            gate.chunks_exact(head_v_dim)
                .zip(output.chunks_exact_mut(head_v_dim)),
        ) {
            let mut sq_sum = 0.0;
            for &v in chunk_in {
                sq_sum += v * v;
            }
            let rms_scale = 1.0 / ((sq_sum / head_v_dim as f32) + eps).sqrt();

            for i in 0..head_v_dim {
                let norm_v = chunk_in[i] * rms_scale * weight[i];
                chunk_out[i] = norm_v * silu(chunk_gate[i]);
            }
        }
    }

    #[test]
    fn gdn_gated_rmsnorm_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_gated_rmsnorm: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let n = NUM_V_HEADS * HEAD_DIM;
        let mut rng = Lcg::new(0x3477_0005);
        // Per-head magnitudes differ, so a norm taken over the whole vector instead
        // of per head lands somewhere else.
        let mut input = Vec::with_capacity(n);
        for h in 0..NUM_V_HEADS {
            let s = 0.2 * (h as f32 + 1.0);
            for _ in 0..HEAD_DIM {
                input.push(rng.next_scaled(s));
            }
        }
        let gate = rng.vec(n, 2.0);
        let weight = rng.vec(HEAD_DIM, 1.0);

        let mut want = vec![0.0f32; n];
        gated_rmsnorm(&input, &gate, &weight, EPS, HEAD_DIM, &mut want);

        let in_buf = GpuBuffer::from_host(&ctx, &input).expect("input");
        let gate_buf = GpuBuffer::from_host(&ctx, &gate).expect("gate");
        let w_buf = GpuBuffer::from_host(&ctx, &weight).expect("weight");
        let out_buf = GpuBuffer::<f32>::new(&ctx, n).expect("out");

        let kernel = GatedRmsNormKernel::new(HEAD_DIM as u32, NUM_V_HEADS as u32, EPS);
        let mut args = [
            in_buf.as_ptr(),
            gate_buf.as_ptr(),
            w_buf.as_ptr(),
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
        out_buf.copy_to_host(&mut got).expect("download");
        assert_close(&got, &want, 1e-3, "gated RMSNorm");
    }
}
