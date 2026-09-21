//! PMAT-3477: fused causal depthwise conv1d + SiLU for Gated `DeltaNet`.
//!
//! One decode step of `causal_conv1d` followed by the SiLU loop that
//! `forward_deltanet` applies to its output:
//!
//! ```text
//! for c in 0..channels:
//!     sum = sum_{k<K-1} state[c][k] * weight[c][k] + input[c] * weight[c][K-1]
//!     state[c][0..K-2] = state[c][1..K-1]      // shift the window left
//!     state[c][K-2]    = input[c]              // append the new sample
//!     output[c]        = silu(sum)
//! ```
//!
//! State layout is `[channels][kernel_size - 1]`, weight layout `[channels][kernel_size]`,
//! both exactly as the CPU reference indexes them.
//!
//! Grid: `(ceil(channels / 256), 1, 1)`, Block: `(256, 1, 1)` — one thread per channel.
//! Each channel is owned by exactly one thread, so the in-place window shift needs no
//! synchronisation.

use crate::kernels::gdn::{emit_silu_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused causal conv1d (single time step) + SiLU.
///
/// For Qwen3.5-0.8B: `channels = 6144` (`conv_dim`), `kernel_size = 4`.
#[derive(Debug, Clone, Copy)]
pub struct CausalConv1dSiluKernel {
    /// Number of convolution channels (`conv_dim`).
    pub channels: u32,
    /// Convolution kernel width (`conv_kernel`, 4 for Qwen3.5).
    pub kernel_size: u32,
}

impl CausalConv1dSiluKernel {
    /// Create the kernel for `channels` channels and a `kernel_size`-wide window.
    ///
    /// # Panics
    /// If `kernel_size` is zero — the CPU reference indexes `weight[c * K + K - 1]`.
    #[must_use]
    pub fn new(channels: u32, kernel_size: u32) -> Self {
        assert!(kernel_size >= 1, "kernel_size must be >= 1");
        Self {
            channels,
            kernel_size,
        }
    }

    /// Launch grid for `channels` channels.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.channels.div_ceil(ELEMENTWISE_BLOCK), 1, 1)
    }

    /// Launch block (one thread per channel).
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for CausalConv1dSiluKernel {
    fn name(&self) -> &str {
        "gdn_causal_conv1d_silu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let channels = self.channels;
        let k = self.kernel_size;
        let state_len = k - 1;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_ptr") // [channels]
            .param(PtxType::U64, "state_ptr") // [channels][kernel_size - 1], updated in place
            .param(PtxType::U64, "weight_ptr") // [channels][kernel_size]
            .param(PtxType::U64, "output_ptr") // [channels]
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let chan = ctx.mad_lo_u32(cta, block, tid);

                let channels_r = ctx.mov_u32_imm(channels);
                let in_bounds = ctx.setp_lt_u32(chan, channels_r);
                ctx.branch_if_not(in_bounds, "gdn_conv_exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let weight_ptr = ctx.load_param_u64("weight_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let four = ctx.mov_u32_imm(4);
                let chan_bytes = ctx.mul_wide_u32_reg(chan, four);

                // input[c] — read before the window is rewritten.
                let in_addr = ctx.add_u64(input_ptr, chan_bytes);
                let x = ctx.ld_global_f32(in_addr);

                // state row base: c * (K - 1) * 4, weight row base: c * K * 4
                let s_stride = ctx.mov_u32_imm(state_len * 4);
                let s_off = ctx.mul_wide_u32_reg(chan, s_stride);
                let s_base = ctx.add_u64(state_ptr, s_off);
                let w_stride = ctx.mov_u32_imm(k * 4);
                let w_off = ctx.mul_wide_u32_reg(chan, w_stride);
                let w_base = ctx.add_u64(weight_ptr, w_off);

                // Load the whole window first: the shift below overwrites it, and the
                // CPU reference reads every past sample before it moves any of them.
                let state_addrs: Vec<_> = (0..state_len)
                    .map(|i| {
                        let off = ctx.mov_u64_imm(u64::from(i * 4));
                        ctx.add_u64(s_base, off)
                    })
                    .collect();
                let state_vals: Vec<_> = state_addrs
                    .iter()
                    .map(|&addr| ctx.ld_global_f32(addr))
                    .collect();

                // sum = sum_k state[k] * w[k] + x * w[K-1], in the CPU's order and with
                // the CPU's separate multiply and add (no FMA contraction).
                let sum = ctx.mov_f32_imm(0.0);
                for i in 0..state_len {
                    let off = ctx.mov_u64_imm(u64::from(i * 4));
                    let w_addr = ctx.add_u64(w_base, off);
                    let w_val = ctx.ld_global_f32(w_addr);
                    let prod = ctx.mul_f32(state_vals[i as usize], w_val);
                    ctx.add_f32_inplace(sum, prod);
                }
                let last_off = ctx.mov_u64_imm(u64::from((k - 1) * 4));
                let w_last_addr = ctx.add_u64(w_base, last_off);
                let w_last = ctx.ld_global_f32(w_last_addr);
                let prod_last = ctx.mul_f32(x, w_last);
                ctx.add_f32_inplace(sum, prod_last);

                // Shift the window left and append the new sample.
                for i in 0..state_len.saturating_sub(1) {
                    ctx.st_global_f32(state_addrs[i as usize], state_vals[i as usize + 1]);
                }
                if state_len >= 1 {
                    ctx.st_global_f32(state_addrs[state_len as usize - 1], x);
                }

                let activated = emit_silu_f32(ctx, sum);
                let out_addr = ctx.add_u64(output_ptr, chan_bytes);
                ctx.st_global_f32(out_addr, activated);

                ctx.label("gdn_conv_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_conv1d_ptx_shape() {
        let kernel = CausalConv1dSiluKernel::new(6144, 4);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_causal_conv1d_silu"), "{ptx}");
        // 3 window loads + 4 weight loads + 1 input load
        assert_eq!(ptx.matches("ld.global.f32").count(), 8, "{ptx}");
        // 2 shifted window stores + 1 appended sample + 1 output
        assert_eq!(ptx.matches("st.global.f32").count(), 4, "{ptx}");
        assert_eq!(kernel.grid(), (24, 1, 1));
        assert_eq!(kernel.block(), (256, 1, 1));
    }
}

/// Device parity against a verbatim port of `causal_conv1d` + the SiLU loop.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_conv1d_device_tests {
    use super::CausalConv1dSiluKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{
        assert_close, run_kernel, Lcg, CONV_CHANNELS, CONV_KERNEL,
    };

    /// Verbatim port of `aprender-serve`'s `causal_conv1d` (forward_qwen35.rs:85).
    fn causal_conv1d(
        input: &[f32],
        state: &mut [f32],
        weight: &[f32],
        kernel_size: usize,
        channels: usize,
        output: &mut [f32],
    ) {
        for c in 0..channels {
            let mut sum = 0.0;
            let s_offset = c * (kernel_size - 1);
            let w_offset = c * kernel_size;

            for k in 0..(kernel_size - 1) {
                sum += state[s_offset + k] * weight[w_offset + k];
            }
            sum += input[c] * weight[w_offset + kernel_size - 1];

            for k in 0..(kernel_size - 2) {
                state[s_offset + k] = state[s_offset + k + 1];
            }
            if kernel_size > 1 {
                state[s_offset + kernel_size - 2] = input[c];
            }

            output[c] = sum;
        }
    }

    /// The SiLU loop `forward_deltanet` applies to the conv output, once.
    fn silu_in_place(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    fn reference(
        input: &[f32],
        state: &mut [f32],
        weight: &[f32],
        kernel_size: usize,
        channels: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; channels];
        causal_conv1d(input, state, weight, kernel_size, channels, &mut out);
        silu_in_place(&mut out);
        out
    }

    #[test]
    fn gdn_causal_conv1d_silu_matches_cpu_reference_three_steps() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!(
                "gdn_causal_conv1d_silu: no CUDA device — SKIPPED. gdn_conv1d_ptx_shape \
                 still guards the codegen."
            );
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let mut rng = Lcg::new(0x3477_0001);
        let weight = rng.vec(CONV_CHANNELS * CONV_KERNEL, 0.5);
        let mut host_state = rng.vec(CONV_CHANNELS * (CONV_KERNEL - 1), 1.0);
        let steps: Vec<Vec<f32>> = (0..3).map(|_| rng.vec(CONV_CHANNELS, 1.0)).collect();

        let weight_buf = GpuBuffer::from_host(&ctx, &weight).expect("weight");
        let state_buf = GpuBuffer::from_host(&ctx, &host_state).expect("state");
        let out_buf = GpuBuffer::<f32>::new(&ctx, CONV_CHANNELS).expect("out");

        let kernel = CausalConv1dSiluKernel::new(CONV_CHANNELS as u32, CONV_KERNEL as u32);

        // Three sequential steps: the window must carry over on the device exactly as
        // it does on the host, or step 2 and 3 diverge even if step 1 matched.
        for (step, input) in steps.iter().enumerate() {
            let want = reference(input, &mut host_state, &weight, CONV_KERNEL, CONV_CHANNELS);

            let in_buf = GpuBuffer::from_host(&ctx, input).expect("input");
            let mut args = [
                in_buf.as_ptr(),
                state_buf.as_ptr(),
                weight_buf.as_ptr(),
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

            let mut got = vec![0.0f32; CONV_CHANNELS];
            out_buf.copy_to_host(&mut got).expect("download output");
            assert_close(
                &got,
                &want,
                1e-3,
                &format!("conv1d+silu output, step {step}"),
            );

            let mut got_state = vec![0.0f32; CONV_CHANNELS * (CONV_KERNEL - 1)];
            state_buf
                .copy_to_host(&mut got_state)
                .expect("download state");
            assert_close(
                &got_state,
                &host_state,
                1e-6,
                &format!("conv1d window state after step {step}"),
            );
        }

        // Keep the buffers alive to the end of the test.
        drop(out_buf);
        drop(state_buf);
    }
}
