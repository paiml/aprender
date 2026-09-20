//! PMAT-3477 / aprender#3090: de-interleave Qwen3.5's joint Q|gate projection.
//!
//! `forward_attention` in the CPU reference reads `attn_q` as one GEMV of width
//! `num_heads * 2 * head_dim`, laid out **per head** as `[q(head_dim) | gate(head_dim)]`,
//! and splits it with two `copy_from_slice`s:
//!
//! ```text
//! q   [h * head_dim .. +head_dim] = q_full[h * head_dim * 2                .. +head_dim]
//! gate[h * head_dim .. +head_dim] = q_full[h * head_dim * 2 + head_dim     .. +head_dim]
//! ```
//!
//! Grid: `(num_heads, 1, 1)`, Block: `(256, 1, 1)` with a strided loop, so the same
//! kernel serves any `head_dim`. Pure data movement — no arithmetic, so the device
//! result must be **bit-identical** to the host split.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Threads per block for the split.
const BLOCK: u32 = 256;

/// Split an interleaved `[q | gate]` attention projection into two contiguous buffers.
///
/// For Qwen3.5-0.8B: `num_heads = 16`, `head_dim = 256`, so the source is
/// `16 * 2 * 256 = 8192` wide and each destination is `4096` wide.
#[derive(Debug, Clone, Copy)]
pub struct SplitInterleavedKernel {
    /// Number of query heads (`config.num_heads`).
    pub num_heads: u32,
    /// Width of one head (`attn_q_norm.len()`, 256 for Qwen3.5).
    pub head_dim: u32,
}

impl SplitInterleavedKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_heads: u32, head_dim: u32) -> Self {
        Self {
            num_heads,
            head_dim,
        }
    }

    /// Launch grid — one block per head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads, 1, 1)
    }

    /// Launch block — 256 threads striding over `head_dim`.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (BLOCK, 1, 1)
    }
}

impl Kernel for SplitInterleavedKernel {
    fn name(&self) -> &str {
        "gdn_split_interleaved_q_gate"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "src_ptr") // [num_heads * 2 * head_dim]
            .param(PtxType::U64, "q_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "gate_ptr") // [num_heads * head_dim]
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);

                let src_ptr = ctx.load_param_u64("src_ptr");
                let q_ptr = ctx.load_param_u64("q_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");

                // src head base: h * 2 * head_dim * 4 bytes; dst head base: h * head_dim * 4.
                let src_head_off = ctx.mul_wide_u32(h, head_dim * 2 * 4);
                let dst_head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let src_head = ctx.add_u64(src_ptr, src_head_off);
                let q_head = ctx.add_u64(q_ptr, dst_head_off);
                let gate_head = ctx.add_u64(gate_ptr, dst_head_off);

                // The gate half starts head_dim floats into the head's source block.
                let gate_src_delta = ctx.mov_u64_imm(u64::from(head_dim) * 4);
                let gate_src_head = ctx.add_u64(src_head, gate_src_delta);

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let i = ctx.add_u32(tid, 0); // copy tid into a mutable counter
                ctx.label("gdn_split_loop");
                let go = ctx.setp_lt_u32(i, head_dim_r);
                ctx.branch_if_not(go, "gdn_split_exit");

                let off = ctx.mul_wide_u32(i, 4);
                let q_src = ctx.add_u64(src_head, off);
                let g_src = ctx.add_u64(gate_src_head, off);
                let q_dst = ctx.add_u64(q_head, off);
                let g_dst = ctx.add_u64(gate_head, off);
                let q_val = ctx.ld_global_f32(q_src);
                let g_val = ctx.ld_global_f32(g_src);
                ctx.st_global_f32(q_dst, q_val);
                ctx.st_global_f32(g_dst, g_val);

                ctx.add_u32_inplace(i, BLOCK);
                ctx.branch("gdn_split_loop");

                ctx.label("gdn_split_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_split_interleaved_ptx_is_pure_data_movement() {
        let kernel = SplitInterleavedKernel::new(16, 256);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_split_interleaved_q_gate"), "{ptx}");
        // A de-interleave is loads and stores only: any float arithmetic here would
        // mean the kernel is transforming what it is supposed to copy.
        for op in ["add.f32", "mul.f32", "fma.rn.f32", "ex2.approx.f32"] {
            assert!(!ptx.contains(op), "unexpected {op} in a pure copy:\n{ptx}");
        }
        // Two loads and two stores per iteration: q and gate.
        assert_eq!(ptx.matches("ld.global.f32").count(), 2, "{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 2, "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
        assert_eq!(kernel.block(), (256, 1, 1));
    }
}

/// Device parity against a verbatim port of `forward_attention`'s split.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_split_interleave_device_tests {
    use super::SplitInterleavedKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg};

    /// `num_heads` of the Qwen3.5-0.8B attention layers (`config.num_heads`).
    const NUM_HEADS: usize = 16;
    /// `head_dim` of the Qwen3.5-0.8B attention layers (`attn_q_norm.len()`).
    const HEAD_DIM: usize = 256;

    /// Verbatim port of the `q` / `gate` split in `forward_attention`
    /// (forward_qwen35.rs:1008).
    fn split_q_gate(q_full: &[f32], num_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0; num_heads * head_dim];
        let mut gate = vec![0.0; num_heads * head_dim];
        for h in 0..num_heads {
            let offset_q_full = h * head_dim * 2;
            let offset_q = h * head_dim;
            q[offset_q..offset_q + head_dim]
                .copy_from_slice(&q_full[offset_q_full..offset_q_full + head_dim]);
            gate[offset_q..offset_q + head_dim]
                .copy_from_slice(&q_full[offset_q_full + head_dim..offset_q_full + head_dim * 2]);
        }
        (q, gate)
    }

    #[test]
    fn gdn_split_interleaved_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_split_interleaved: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let mut rng = Lcg::new(0x3477_0010);
        let src = rng.vec(NUM_HEADS * 2 * HEAD_DIM, 1.0);
        let (want_q, want_gate) = split_q_gate(&src, NUM_HEADS, HEAD_DIM);

        let src_buf = GpuBuffer::from_host(&ctx, &src).expect("src");
        let q_buf = GpuBuffer::<f32>::new(&ctx, NUM_HEADS * HEAD_DIM).expect("q");
        let gate_buf = GpuBuffer::<f32>::new(&ctx, NUM_HEADS * HEAD_DIM).expect("gate");
        let kernel = SplitInterleavedKernel::new(NUM_HEADS as u32, HEAD_DIM as u32);
        let mut args = [src_buf.as_ptr(), q_buf.as_ptr(), gate_buf.as_ptr()];
        run_kernel(
            &ctx,
            &stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );

        let mut got_q = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        let mut got_gate = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        q_buf.copy_to_host(&mut got_q).expect("download q");
        gate_buf.copy_to_host(&mut got_gate).expect("download gate");

        // tol 0: a copy that is off by anything at all is a wrong copy.
        assert_close(&got_q, &want_q, 0.0, "split q");
        assert_close(&got_gate, &want_gate, 0.0, "split gate");

        // A kernel that ignored the per-head stride (e.g. treating the source as
        // [all q | all gate]) would still be bit-exact on head 0 alone.
        assert!(
            want_q[HEAD_DIM..] != want_gate[HEAD_DIM..],
            "fixture must distinguish the halves beyond head 0"
        );
    }
}
