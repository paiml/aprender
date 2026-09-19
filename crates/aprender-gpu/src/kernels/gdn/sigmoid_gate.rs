//! PMAT-3477: the output sigmoid gate of Qwen3.5's full-attention layers.
//!
//! `apply_sigmoid_gate` in the CPU reference: `x[i] *= sigmoid(gate[i])`, applied to
//! the attention output before the output projection (llama.cpp's `attn_gated`).
//!
//! Grid: `(ceil(n / 256), 1, 1)`, Block: `(256, 1, 1)`.

use crate::kernels::gdn::{emit_sigmoid_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Elementwise `x *= sigmoid(gate)`, in place.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidGateKernel {
    /// Number of elements.
    pub n: u32,
}

impl SigmoidGateKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }

    /// Launch grid.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.n.div_ceil(ELEMENTWISE_BLOCK), 1, 1)
    }

    /// Launch block.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for SigmoidGateKernel {
    fn name(&self) -> &str {
        "gdn_sigmoid_gate"
    }

    fn build_ptx(&self) -> PtxKernel {
        let n = self.n;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "x_ptr") // [n], scaled in place
            .param(PtxType::U64, "gate_ptr") // [n]
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let i = ctx.mad_lo_u32(cta, block, tid);

                let n_r = ctx.mov_u32_imm(n);
                let go = ctx.setp_lt_u32(i, n_r);
                ctx.branch_if_not(go, "gdn_sigmoid_gate_exit");

                let x_ptr = ctx.load_param_u64("x_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let four = ctx.mov_u32_imm(4);
                let off = ctx.mul_wide_u32_reg(i, four);
                let x_addr = ctx.add_u64(x_ptr, off);
                let gate_addr = ctx.add_u64(gate_ptr, off);

                let x = ctx.ld_global_f32(x_addr);
                let g = ctx.ld_global_f32(gate_addr);
                let s = emit_sigmoid_f32(ctx, g);
                let result = ctx.mul_f32(x, s);
                ctx.st_global_f32(x_addr, result);

                ctx.label("gdn_sigmoid_gate_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_sigmoid_gate_ptx_shape() {
        let kernel = SigmoidGateKernel::new(2048);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_sigmoid_gate"), "{ptx}");
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 1, "{ptx}");
        assert_eq!(kernel.grid(), (8, 1, 1));
    }
}

/// Device parity against a verbatim port of `apply_sigmoid_gate`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_sigmoid_gate_device_tests {
    use super::SigmoidGateKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg, HEAD_DIM, NUM_K_HEADS};

    /// Verbatim port of `apply_sigmoid_gate` (forward_qwen35.rs:44).
    fn apply_sigmoid_gate(x: &mut [f32], gate: &[f32]) {
        for (o, g) in x.iter_mut().zip(gate) {
            *o *= 1.0 / (1.0 + (-*g).exp());
        }
    }

    #[test]
    fn gdn_sigmoid_gate_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_sigmoid_gate: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        // A full-attention layer's attn_output width for Qwen3.5-0.8B.
        let n = HEAD_DIM * NUM_K_HEADS;
        let mut rng = Lcg::new(0x3477_0006);
        let host = rng.vec(n, 2.0);
        // Gate values spanning the saturating tails as well as the linear region.
        let gate: Vec<f32> = (0..n)
            .map(|i| rng.next_scaled(6.0) + (i % 3) as f32 - 1.0)
            .collect();

        let mut want = host.clone();
        apply_sigmoid_gate(&mut want, &gate);

        let x_buf = GpuBuffer::from_host(&ctx, &host).expect("x");
        let gate_buf = GpuBuffer::from_host(&ctx, &gate).expect("gate");
        let kernel = SigmoidGateKernel::new(n as u32);
        let mut args = [x_buf.as_ptr(), gate_buf.as_ptr()];
        run_kernel(
            &ctx,
            &stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );

        let mut got = vec![0.0f32; n];
        x_buf.copy_to_host(&mut got).expect("download");
        assert_close(&got, &want, 1e-3, "x *= sigmoid(gate)");
    }
}
