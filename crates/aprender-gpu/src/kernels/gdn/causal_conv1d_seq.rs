//! PMAT-3596: fused causal depthwise conv1d + SiLU over a whole prefill chunk.
//!
//! [`CausalConv1dSiluKernel`](super::CausalConv1dSiluKernel) runs ONE time step per
//! launch. This kernel runs `t_count` consecutive steps in one launch: each thread
//! owns one channel, keeps that channel's `kernel_size - 1` window and `kernel_size`
//! weights in registers, and walks the chunk's rows in order:
//!
//! ```text
//! for t in 0..T:                                   // one thread per channel c
//!     x      = input[t][c]
//!     sum    = sum_{k<K-1} window[k] * w[c][k] + x * w[c][K-1]
//!     window = window[1..] ++ [x]
//!     output[t][c] = silu(sum)
//! state[c] = window                                  // once, at the end
//! ```
//!
//! Per step the arithmetic is the per-token kernel's, op for op and in its order
//! (the multiplies and adds of the window sum, then the same `emit_silu_f32`), so one
//! launch over `T` rows equals `T` per-token launches exactly; the device test asserts
//! equality, not a tolerance. The window lives in registers between steps, so the
//! state is read once and written once instead of `T` times.
//!
//! Grid: `(ceil(channels / 256), 1, 1)`, Block: `(256, 1, 1)`. A channel is owned by
//! one thread for the whole chunk, so there is no synchronisation.

use crate::kernels::gdn::{emit_silu_f32, ELEMENTWISE_BLOCK};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused causal conv1d + SiLU over `t_count` rows.
///
/// For Qwen3.5-9B: `channels = 8192` (`conv_dim`), `kernel_size = 4`, and both row
/// strides are `conv_dim` (the `attn_qkv` GEMM output and the conv output are
/// `[T][conv_dim]`).
#[derive(Debug, Clone, Copy)]
pub struct CausalConv1dSiluSeqKernel {
    /// Number of convolution channels (`conv_dim`).
    pub channels: u32,
    /// Convolution kernel width (`conv_kernel`, 4 for Qwen3.5).
    pub kernel_size: u32,
    /// Floats between row `t` and row `t + 1` of the input.
    pub in_row_stride: u32,
    /// Floats between row `t` and row `t + 1` of the output.
    pub out_row_stride: u32,
}

impl CausalConv1dSiluSeqKernel {
    /// Create the kernel.
    ///
    /// # Panics
    /// If `kernel_size` is zero, or a row stride is narrower than `channels`.
    #[must_use]
    pub fn new(channels: u32, kernel_size: u32, in_row_stride: u32, out_row_stride: u32) -> Self {
        assert!(kernel_size >= 1, "kernel_size must be >= 1");
        assert!(
            in_row_stride >= channels && out_row_stride >= channels,
            "row strides ({in_row_stride}, {out_row_stride}) must cover {channels} channels"
        );
        Self {
            channels,
            kernel_size,
            in_row_stride,
            out_row_stride,
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

impl Kernel for CausalConv1dSiluSeqKernel {
    fn name(&self) -> &str {
        "gdn_causal_conv1d_silu_seq"
    }

    fn build_ptx(&self) -> PtxKernel {
        let channels = self.channels;
        let k = self.kernel_size;
        let state_len = k - 1;
        let in_stride = self.in_row_stride;
        let out_stride = self.out_row_stride;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_ptr") // [T][in_row_stride]
            .param(PtxType::U64, "state_ptr") // [channels][kernel_size - 1], updated in place
            .param(PtxType::U64, "weight_ptr") // [channels][kernel_size]
            .param(PtxType::U64, "output_ptr") // [T][out_row_stride]
            .param(PtxType::U32, "t_count")
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let chan = ctx.mad_lo_u32(cta, block, tid);

                let channels_r = ctx.mov_u32_imm(channels);
                let in_bounds = ctx.setp_lt_u32(chan, channels_r);
                ctx.branch_if_not(in_bounds, "gdn_conv_seq_exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let state_ptr = ctx.load_param_u64("state_ptr");
                let weight_ptr = ctx.load_param_u64("weight_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let t_count = ctx.load_param_u32("t_count");

                let four = ctx.mov_u32_imm(4);

                // The window and the weights, into registers once.
                let s_off = ctx.mul_wide_u32(chan, state_len * 4);
                let s_base = ctx.add_u64(state_ptr, s_off);
                let w_off = ctx.mul_wide_u32(chan, k * 4);
                let w_base = ctx.add_u64(weight_ptr, w_off);
                let state_addrs: Vec<_> = (0..state_len)
                    .map(|i| {
                        let off = ctx.mov_u64_imm(u64::from(i * 4));
                        ctx.add_u64(s_base, off)
                    })
                    .collect();
                let window: Vec<_> = state_addrs.iter().map(|&a| ctx.ld_global_f32(a)).collect();
                let weights: Vec<_> = (0..k)
                    .map(|i| {
                        let off = ctx.mov_u64_imm(u64::from(i * 4));
                        let addr = ctx.add_u64(w_base, off);
                        ctx.ld_global_f32(addr)
                    })
                    .collect();

                let t = ctx.mov_u32_imm(0);
                ctx.label("gdn_conv_seq_loop");
                let more = ctx.setp_lt_u32(t, t_count);
                ctx.branch_if_not(more, "gdn_conv_seq_end");

                let in_row = ctx.mul_u32(t, in_stride);
                let in_elem = ctx.add_u32_reg(in_row, chan);
                let in_off = ctx.mul_wide_u32_reg(in_elem, four);
                let in_addr = ctx.add_u64(input_ptr, in_off);
                let x = ctx.ld_global_f32(in_addr);

                // sum = sum_k window[k] * w[k] + x * w[K-1], the per-token order.
                let sum = ctx.mov_f32_imm(0.0);
                for i in 0..state_len as usize {
                    let prod = ctx.mul_f32(window[i], weights[i]);
                    ctx.add_f32_inplace(sum, prod);
                }
                let prod_last = ctx.mul_f32(x, weights[state_len as usize]);
                ctx.add_f32_inplace(sum, prod_last);

                // Shift the window left and append x — in registers.
                for i in 0..state_len.saturating_sub(1) as usize {
                    ctx.mov_f32_reg(window[i], window[i + 1]);
                }
                if state_len >= 1 {
                    ctx.mov_f32_reg(window[state_len as usize - 1], x);
                }

                let activated = emit_silu_f32(ctx, sum);
                let out_row = ctx.mul_u32(t, out_stride);
                let out_elem = ctx.add_u32_reg(out_row, chan);
                let out_off = ctx.mul_wide_u32_reg(out_elem, four);
                let out_addr = ctx.add_u64(output_ptr, out_off);
                ctx.st_global_f32(out_addr, activated);

                ctx.add_u32_inplace(t, 1);
                ctx.branch("gdn_conv_seq_loop");
                ctx.label("gdn_conv_seq_end");

                // The window, back to the state once.
                for (&addr, &w) in state_addrs.iter().zip(&window) {
                    ctx.st_global_f32(addr, w);
                }

                ctx.label("gdn_conv_seq_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_conv1d_seq_ptx_shape() {
        let kernel = CausalConv1dSiluSeqKernel::new(8192, 4, 8192, 8192);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_causal_conv1d_silu_seq"), "{ptx}");
        // 3 window + 4 weight loads once, and 1 input load per step.
        assert_eq!(ptx.matches("ld.global.f32").count(), 3 + 4 + 1, "{ptx}");
        // 1 output store per step, and the 3-float window once at the end.
        assert_eq!(ptx.matches("st.global.f32").count(), 1 + 3, "{ptx}");
        assert!(
            !ptx.contains("fma."),
            "the per-token kernel emits no fma: {ptx}"
        );
        assert_eq!(kernel.grid(), (32, 1, 1));
    }

    #[test]
    #[should_panic(expected = "must cover")]
    fn gdn_conv1d_seq_refuses_a_stride_narrower_than_the_channels() {
        let _ = CausalConv1dSiluSeqKernel::new(8192, 4, 4096, 8192);
    }
}

/// Device proof: one launch over `T` rows equals `T` per-token launches, exactly.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_conv1d_seq_device_tests {
    use super::CausalConv1dSiluSeqKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{run_kernel, Lcg, CONV_CHANNELS, CONV_KERNEL};
    use crate::kernels::gdn::CausalConv1dSiluKernel;

    fn seq_equals_per_token(channels: usize, tokens: usize, stride: usize) {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_conv1d_seq: no CUDA device — SKIPPED (the PTX tests still run)");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let mut rng = Lcg::new(0x3596_0002 ^ tokens as u32);
        let weight = rng.vec(channels * CONV_KERNEL, 0.5);
        let state0 = rng.vec(channels * (CONV_KERNEL - 1), 1.0);
        let rows = rng.vec(tokens * stride, 1.0);

        // --- per-token: T launches on contiguous single rows ---
        let per_token = CausalConv1dSiluKernel::new(channels as u32, CONV_KERNEL as u32);
        let weight_buf = GpuBuffer::from_host(&ctx, &weight).expect("weight");
        let state_a = GpuBuffer::from_host(&ctx, &state0).expect("state a");
        let mut want = vec![0.0f32; tokens * stride];
        for t in 0..tokens {
            let input =
                GpuBuffer::from_host(&ctx, &rows[t * stride..t * stride + channels]).expect("row");
            let out = GpuBuffer::<f32>::new(&ctx, channels).expect("out");
            let mut args = [
                input.as_ptr(),
                state_a.as_ptr(),
                weight_buf.as_ptr(),
                out.as_ptr(),
            ];
            run_kernel(
                &ctx,
                &stream,
                &per_token,
                per_token.grid(),
                per_token.block(),
                &mut args,
            );
            out.copy_to_host(&mut want[t * stride..t * stride + channels])
                .expect("download");
        }
        let mut want_state = vec![0.0f32; state0.len()];
        state_a.copy_to_host(&mut want_state).expect("state a");

        // --- sequence: one launch over the strided rows ---
        let seq = CausalConv1dSiluSeqKernel::new(
            channels as u32,
            CONV_KERNEL as u32,
            stride as u32,
            stride as u32,
        );
        let rows_buf = GpuBuffer::from_host(&ctx, &rows).expect("rows");
        let state_b = GpuBuffer::from_host(&ctx, &state0).expect("state b");
        let out_buf = GpuBuffer::from_host(&ctx, &vec![0.0f32; tokens * stride]).expect("out");
        let mut args = [
            rows_buf.as_ptr(),
            state_b.as_ptr(),
            weight_buf.as_ptr(),
            out_buf.as_ptr(),
            tokens as u64,
        ];
        run_kernel(&ctx, &stream, &seq, seq.grid(), seq.block(), &mut args);
        let mut got = vec![0.0f32; tokens * stride];
        out_buf.copy_to_host(&mut got).expect("out");
        let mut got_state = vec![0.0f32; state0.len()];
        state_b.copy_to_host(&mut got_state).expect("state b");

        assert!(
            want.iter().any(|v| v.abs() > 1e-3),
            "per-token output is ~zero"
        );
        for t in 0..tokens {
            for c in 0..channels {
                let (g, w) = (got[t * stride + c], want[t * stride + c]);
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "conv1d seq row {t} channel {c}: seq {g:e} vs per-token {w:e} — must be \
                     bitwise-identical"
                );
            }
            // The padding past `channels` in a strided row is never written.
            assert!(got[t * stride + channels..(t + 1) * stride]
                .iter()
                .all(|v| *v == 0.0));
        }
        assert_eq!(
            got_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "conv1d seq final window differs from the per-token window"
        );
    }

    #[test]
    fn gdn_conv1d_seq_is_bitwise_per_token_contiguous() {
        seq_equals_per_token(CONV_CHANNELS, 23, CONV_CHANNELS);
    }

    #[test]
    fn gdn_conv1d_seq_is_bitwise_per_token_strided() {
        // A row stride wider than the channel count, as a sub-view would have.
        seq_equals_per_token(1000, 7, 1024);
    }

    #[test]
    fn gdn_conv1d_seq_one_row_is_one_step() {
        seq_equals_per_token(CONV_CHANNELS, 1, CONV_CHANNELS);
    }
}
