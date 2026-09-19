//! PMAT-3477: per-head L2 normalisation for Gated `DeltaNet` q/k.
//!
//! `l2_norm_per_head` in the CPU reference: each `head_dim`-wide head is scaled on
//! its own by `1 / sqrt(sum(x_h^2) + eps)`. The epsilon is added to the *sum of
//! squares*, not to a mean — this is llama.cpp's `build_gdn_l2_norm`, not RMSNorm.
//! One norm over all heads would scale every head by the other heads' magnitudes.
//!
//! Grid: `(num_heads, 1, 1)`, Block: `(32, 1, 1)` — one warp per head, in place.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Per-head L2 normalisation, in place.
///
/// For Qwen3.5-0.8B: `head_dim = 128`, `num_heads = 16` (q and k alike), `eps = 1e-6`.
#[derive(Debug, Clone, Copy)]
pub struct PerHeadL2NormKernel {
    /// Elements per head (`head_k_dim`).
    pub head_dim: u32,
    /// Number of heads (`num_k_heads`).
    pub num_heads: u32,
    /// Epsilon, added to the sum of squares.
    pub eps: f32,
}

impl PerHeadL2NormKernel {
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

impl Kernel for PerHeadL2NormKernel {
    fn name(&self) -> &str {
        "gdn_per_head_l2_norm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let eps = self.eps;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "x_ptr") // [num_heads * head_dim], normalised in place
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");

                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let four = ctx.mov_u32_imm(4);
                let head_elems = ctx.mul_u32_reg(head_idx, head_dim_r);
                let head_bytes = ctx.mul_wide_u32_reg(head_elems, four);
                let head_base = ctx.add_u64(x_ptr, head_bytes);

                // Pass 1: sum of squares over this head, warp-strided.
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);
                ctx.label("gdn_l2_sum_loop");
                let i = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(i, head_dim_r);
                ctx.branch_if_not(in_bounds, "gdn_l2_sum_end");
                let off = ctx.mul_wide_u32_reg(i, four);
                let addr = ctx.add_u64(head_base, off);
                let val = ctx.ld_global_f32(addr);
                ctx.fma_f32_inplace(sq_sum, val, val);
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("gdn_l2_sum_loop");
                ctx.label("gdn_l2_sum_end");

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

                // scale = 1 / sqrt(sum + eps) — eps on the SUM, as l2_norm does.
                let eps_r = ctx.mov_f32_imm(eps);
                let denom = ctx.add_f32(total, eps_r);
                let scale = ctx.rsqrt_f32(denom);

                // Pass 2: scale in place.
                let idx2 = ctx.mov_u32_imm(0);
                ctx.label("gdn_l2_scale_loop");
                let i2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(i2, head_dim_r);
                ctx.branch_if_not(in_bounds2, "gdn_l2_exit");
                let off2 = ctx.mul_wide_u32_reg(i2, four);
                let addr2 = ctx.add_u64(head_base, off2);
                let v = ctx.ld_global_f32(addr2);
                let scaled = ctx.mul_f32(v, scale);
                ctx.st_global_f32(addr2, scaled);
                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("gdn_l2_scale_loop");

                ctx.label("gdn_l2_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_l2_norm_ptx_divides_by_no_count() {
        let ptx = PerHeadL2NormKernel::new(128, 16, 1e-6).emit_ptx();
        assert!(ptx.contains(".entry gdn_per_head_l2_norm"), "{ptx}");
        // L2 norm, not RMSNorm: no division by head_dim before the rsqrt.
        assert!(
            !ptx.contains("div.rn.f32"),
            "eps goes on the sum of squares; a mean divide would be RMSNorm:\n{ptx}"
        );
        assert!(ptx.contains("rsqrt.approx.f32"), "{ptx}");
    }
}

/// Device parity against a verbatim port of `l2_norm_per_head`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_l2_norm_device_tests {
    use super::PerHeadL2NormKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{
        assert_close, run_kernel, Lcg, EPS, HEAD_DIM, NUM_K_HEADS,
    };

    /// Verbatim port of `aprender-serve`'s `l2_norm` (forward_qwen35.rs:21).
    fn l2_norm(x: &mut [f32], eps: f32) {
        let mut sq_sum = 0.0;
        for &v in x.iter() {
            sq_sum += v * v;
        }
        let scale = 1.0 / (sq_sum + eps).sqrt();
        for v in x.iter_mut() {
            *v *= scale;
        }
    }

    /// Verbatim port of `l2_norm_per_head` (forward_qwen35.rs:36).
    fn l2_norm_per_head(x: &mut [f32], head_dim: usize, eps: f32) {
        for head in x.chunks_exact_mut(head_dim) {
            l2_norm(head, eps);
        }
    }

    #[test]
    fn gdn_per_head_l2_norm_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_per_head_l2_norm: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");

        let mut rng = Lcg::new(0x3477_0002);
        // Heads deliberately differ in magnitude: a single norm over the whole
        // vector (the defect this kernel exists to avoid) lands elsewhere.
        let mut host: Vec<f32> = Vec::with_capacity(HEAD_DIM * NUM_K_HEADS);
        for h in 0..NUM_K_HEADS {
            let scale = 0.25 * (h as f32 + 1.0);
            for _ in 0..HEAD_DIM {
                host.push(rng.next_scaled(scale));
            }
        }
        let mut want = host.clone();
        l2_norm_per_head(&mut want, HEAD_DIM, EPS);

        let buf = GpuBuffer::from_host(&ctx, &host).expect("x");
        let kernel = PerHeadL2NormKernel::new(HEAD_DIM as u32, NUM_K_HEADS as u32, EPS);
        let mut args = [buf.as_ptr()];
        run_kernel(
            &ctx,
            &stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args,
        );

        let mut got = vec![0.0f32; host.len()];
        buf.copy_to_host(&mut got).expect("download");
        assert_close(&got, &want, 1e-4, "per-head L2 norm");

        // Each head must be unit length on its own.
        for h in 0..NUM_K_HEADS {
            let norm: f32 = got[h * HEAD_DIM..(h + 1) * HEAD_DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "head {h} has L2 norm {norm}, not 1 — a global norm would leave the \
                 heads at different lengths"
            );
        }
    }
}
