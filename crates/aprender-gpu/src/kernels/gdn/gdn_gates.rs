//! PMAT-3477: the Gated `DeltaNet` per-head gates.
//!
//! `forward_deltanet` computes, for each of the `num_v_heads` heads:
//!
//! ```text
//! dt[h]   = softplus(alpha_raw[h] + dt_bias[h]) * a[h]
//! beta[h] = sigmoid(beta_raw[h])
//! ```
//!
//! `alpha_raw` is `ssm_alpha . x`, `beta_raw` is `ssm_beta . x`, `dt_bias` is
//! `ssm_dt.bias` and `a` is `ssm_a`. `dt` is the gate the delta-rule recurrence
//! exponentiates; `beta` is its update rate.
//!
//! Grid: `(ceil(num_heads / 256), 1, 1)`, Block: `(256, 1, 1)` — one thread per head
//! (16 heads for Qwen3.5-0.8B, so a single partly-idle block: this runs once per layer
//! per token and is latency-, not occupancy-bound).

use crate::kernels::gdn::{emit_sigmoid_f32, emit_softplus_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Per-head `dt` and `beta` gates for one Gated `DeltaNet` layer.
#[derive(Debug, Clone, Copy)]
pub struct GdnGatesKernel {
    /// Number of value heads (`num_v_heads`, 16 for Qwen3.5-0.8B).
    pub num_heads: u32,
}

impl GdnGatesKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_heads: u32) -> Self {
        Self { num_heads }
    }

    /// Launch grid.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads.div_ceil(ELEMENTWISE_BLOCK), 1, 1)
    }

    /// Launch block.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for GdnGatesKernel {
    fn name(&self) -> &str {
        "gdn_gates"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_heads = self.num_heads;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "alpha_ptr") // [num_heads] ssm_alpha . x
            .param(PtxType::U64, "dt_bias_ptr") // [num_heads] ssm_dt.bias
            .param(PtxType::U64, "a_ptr") // [num_heads] ssm_a
            .param(PtxType::U64, "beta_raw_ptr") // [num_heads] ssm_beta . x
            .param(PtxType::U64, "dt_ptr") // [num_heads] out
            .param(PtxType::U64, "beta_ptr") // [num_heads] out
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let h = ctx.mad_lo_u32(cta, block, tid);

                let n = ctx.mov_u32_imm(num_heads);
                let in_bounds = ctx.setp_lt_u32(h, n);
                ctx.branch_if_not(in_bounds, "gdn_gates_exit");

                let alpha_ptr = ctx.load_param_u64("alpha_ptr");
                let dt_bias_ptr = ctx.load_param_u64("dt_bias_ptr");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let beta_raw_ptr = ctx.load_param_u64("beta_raw_ptr");
                let dt_ptr = ctx.load_param_u64("dt_ptr");
                let beta_ptr = ctx.load_param_u64("beta_ptr");

                let four = ctx.mov_u32_imm(4);
                let off = ctx.mul_wide_u32_reg(h, four);

                // dt[h] = softplus(alpha[h] + dt_bias[h]) * a[h]
                let alpha_addr = ctx.add_u64(alpha_ptr, off);
                let bias_addr = ctx.add_u64(dt_bias_ptr, off);
                let a_addr = ctx.add_u64(a_ptr, off);
                let alpha = ctx.ld_global_f32(alpha_addr);
                let bias = ctx.ld_global_f32(bias_addr);
                let a = ctx.ld_global_f32(a_addr);
                let pre = ctx.add_f32(alpha, bias);
                let sp = emit_softplus_f32(ctx, pre, "gdn_gates_dt");
                let dt = ctx.mul_f32(sp, a);
                let dt_addr = ctx.add_u64(dt_ptr, off);
                ctx.st_global_f32(dt_addr, dt);

                // beta[h] = sigmoid(beta_raw[h])
                let beta_raw_addr = ctx.add_u64(beta_raw_ptr, off);
                let beta_raw = ctx.ld_global_f32(beta_raw_addr);
                let beta = emit_sigmoid_f32(ctx, beta_raw);
                let beta_addr = ctx.add_u64(beta_ptr, off);
                ctx.st_global_f32(beta_addr, beta);

                ctx.label("gdn_gates_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_gates_ptx_has_softplus_branch_and_sigmoid() {
        let ptx = GdnGatesKernel::new(16).emit_ptx();
        assert!(ptx.contains(".entry gdn_gates"), "{ptx}");
        assert!(ptx.contains("gdn_gates_dt_softplus_big:"), "{ptx}");
        // one ex2 for softplus, one for the sigmoid
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 2, "{ptx}");
        assert_eq!(ptx.matches("lg2.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 2, "{ptx}");
    }
}

/// Device parity against a verbatim port of `forward_deltanet`'s gate arithmetic.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_gates_device_tests {
    use super::GdnGatesKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg, NUM_V_HEADS};

    /// Verbatim port of `fn_softplus` (forward_qwen35.rs:864, == `softplus` at :12).
    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn gdn_gates_match_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_gates: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let mut rng = Lcg::new(0x3477_0003);
        let mut alpha = rng.vec(NUM_V_HEADS, 3.0);
        let dt_bias = rng.vec(NUM_V_HEADS, 1.0);
        // ssm_a is negative in the real model (the state decays); keep that sign.
        let a: Vec<f32> = (0..NUM_V_HEADS).map(|_| -rng.next().abs() - 0.05).collect();
        let beta_raw = rng.vec(NUM_V_HEADS, 4.0);
        // Exercise BOTH arms of the softplus branch: head 0 takes the x > 20 shortcut.
        alpha[0] = 40.0;
        alpha[1] = -30.0;

        let want_dt: Vec<f32> = (0..NUM_V_HEADS)
            .map(|h| softplus(alpha[h] + dt_bias[h]) * a[h])
            .collect();
        let want_beta: Vec<f32> = beta_raw.iter().map(|&b| sigmoid(b)).collect();

        let alpha_buf = GpuBuffer::from_host(&ctx, &alpha).expect("alpha");
        let bias_buf = GpuBuffer::from_host(&ctx, &dt_bias).expect("dt_bias");
        let a_buf = GpuBuffer::from_host(&ctx, &a).expect("a");
        let beta_raw_buf = GpuBuffer::from_host(&ctx, &beta_raw).expect("beta_raw");
        let dt_buf = GpuBuffer::<f32>::new(&ctx, NUM_V_HEADS).expect("dt");
        let beta_buf = GpuBuffer::<f32>::new(&ctx, NUM_V_HEADS).expect("beta");

        let kernel = GdnGatesKernel::new(NUM_V_HEADS as u32);
        let mut args = [
            alpha_buf.as_ptr(),
            bias_buf.as_ptr(),
            a_buf.as_ptr(),
            beta_raw_buf.as_ptr(),
            dt_buf.as_ptr(),
            beta_buf.as_ptr(),
        ];
        run_kernel(
            &ctx,
            &stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );

        let mut got_dt = vec![0.0f32; NUM_V_HEADS];
        let mut got_beta = vec![0.0f32; NUM_V_HEADS];
        dt_buf.copy_to_host(&mut got_dt).expect("download dt");
        beta_buf.copy_to_host(&mut got_beta).expect("download beta");

        assert_close(&got_dt, &want_dt, 1e-3, "dt = softplus(alpha + bias) * a");
        assert_close(&got_beta, &want_beta, 1e-3, "beta = sigmoid(beta_raw)");
        assert!(
            want_dt[0].is_finite() && (got_dt[0] - want_dt[0]).abs() <= 1e-3,
            "the x > 20 shortcut arm of softplus must be taken for head 0"
        );
    }
}
