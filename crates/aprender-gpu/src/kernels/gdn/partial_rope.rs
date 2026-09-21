//! PMAT-3477 / aprender#3090: partial NEOX RoPE for the Qwen3.5 attention heads.
//!
//! `apply_partial_neox_rope` in the CPU reference (forward_qwen35.rs:191), which is
//! llama.cpp's `ggml_rope_multi` for text input:
//!
//! ```text
//! half = n_rot / 2;  theta_scale = freq_base^(-2/n_rot)
//! for h in heads:
//!     theta = pos
//!     for j in 0..half:
//!         (sin, cos) = theta.sin_cos()
//!         (a, b) = (x[h*head_dim + j], x[h*head_dim + j + half])
//!         x[h*head_dim + j]        = a * cos - b * sin
//!         x[h*head_dim + j + half] = a * sin + b * cos
//!         theta *= theta_scale
//! ```
//!
//! Dimensions `n_rot..head_dim` are **not** rotated. The existing
//! `elementwise/rope/neox.rs` rotates the whole head, which is the defect this kernel
//! exists to avoid: for Qwen3.5 `n_rot = 2 * sum(rope.dimension_sections) = 128` of a
//! 256-wide head, so a full rotation corrupts the upper half.
//!
//! Grid: `(num_heads, 1, 1)`, Block: `(half, 1, 1)` — one thread per (head, pair), in
//! place.
//!
//! ## Numerics: why `theta_scale`, and not `freq_base`, is the parameter
//!
//! Thread `j` reproduces the CPU's *iterative* `theta` by multiplying `theta_scale`
//! into `pos` exactly `j` times, in the CPU's order. That is bit-identical to the
//! reference **only if `theta_scale` itself is bit-identical**, and it has to be:
//! the error in `theta` is amplified by `j * theta_j`, which peaks at 2556 for
//! `pos = 1000, freq_base = 1e4`. So a single ulp (6e-8 relative) in `theta_scale`
//! is already 1.5e-4 rad of `theta` — more than the entire 1.4e-4 tolerance. This
//! was measured, not assumed: computing `freq_base^(-2/n_rot)` on the device from
//! `lg2.approx` + a polynomial `exp2` (the best the ISA offers, ~1e-7 relative)
//! missed the CPU by 2.3e-4 at that position.
//!
//! `freq_base^(-2/n_rot)` is one scalar per layer, and the CPU computes it once with
//! `f32::powf`. So the kernel takes that scalar; [`PartialNeoxRopeKernel::theta_scale`]
//! computes it the CPU's way, and callers must use it rather than rolling their own.
//!
//! What remains is `sin`/`cos`. `sin.approx.f32`/`cos.approx.f32` are accurate to
//! ~2^-21 **only for arguments in `[-pi, pi]`**, and `theta` reaches `pos`
//! (thousands). A Cody-Waite reduction with a two-word `2*pi` (`6.28125` plus the
//! remainder — the head is exact for any `n < 2^16`, i.e. positions below ~411k)
//! brings the argument into range with ~3e-8 rad of error before the hardware sees
//! it, leaving ~2e-7 in the rotated values.

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// High word of `2*pi`: exactly representable with 8 significant bits, so `n * HI` is
/// exact for every `n < 2^16`.
const TWO_PI_HI: f32 = 6.281_25;
/// Low word of `2*pi`: `2*pi - TWO_PI_HI`, rounded to f32.
const TWO_PI_LO: f32 = 0.001_935_307_2;

/// Partial NEOX RoPE over the first `n_rot` dimensions of each head, in place.
///
/// For Qwen3.5-0.8B: `num_heads = 16` (or `num_kv_heads` when rotating K),
/// `head_dim = 256`, `n_rot = 128`.
#[derive(Debug, Clone, Copy)]
pub struct PartialNeoxRopeKernel {
    /// Number of heads in the buffer (`num_heads` for Q, `num_kv_heads` for K).
    pub num_heads: u32,
    /// Width of one head (256 for Qwen3.5).
    pub head_dim: u32,
    /// Rotated prefix of each head: `2 * sum(rope.dimension_sections)` (128).
    pub n_rot: u32,
}

impl PartialNeoxRopeKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(num_heads: u32, head_dim: u32, n_rot: u32) -> Self {
        Self {
            num_heads,
            head_dim,
            n_rot,
        }
    }

    /// Number of rotated pairs per head (`n_rot / 2`).
    #[must_use]
    pub const fn half(&self) -> u32 {
        self.n_rot / 2
    }

    /// The `theta_scale` kernel argument for a given `freq_base` (`config.rope_theta`).
    ///
    /// `freq_base.powf(-2.0 / n_rot as f32)`, the CPU reference's expression
    /// character for character. It must be *this* value: see the module docs — one
    /// ulp here is more than the whole parity budget at position 1000.
    #[must_use]
    pub fn theta_scale(&self, freq_base: f32) -> f32 {
        freq_base.powf(-2.0 / self.n_rot as f32)
    }

    /// Launch grid — one block per head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads, 1, 1)
    }

    /// Launch block — one thread per rotated pair.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (self.half(), 1, 1)
    }
}

/// `(sin(theta), cos(theta))` with a Cody-Waite reduction of `theta` into `[-pi, pi]`
/// before the hardware approximations, which are only accurate in that range.
pub(super) fn emit_sin_cos(
    ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
    theta: VirtualReg,
) -> (VirtualReg, VirtualReg) {
    let inv_two_pi = ctx.mov_f32_imm(1.0 / std::f32::consts::TAU);
    let half = ctx.mov_f32_imm(0.5);
    let scaled = ctx.mul_f32(theta, inv_two_pi);
    let biased = ctx.add_f32(scaled, half);
    let n = ctx.floor_f32(biased);

    let neg_hi = ctx.mov_f32_imm(-TWO_PI_HI);
    let neg_lo = ctx.mov_f32_imm(-TWO_PI_LO);
    let t1 = ctx.fma_f32(n, neg_hi, theta);
    let reduced = ctx.fma_f32(n, neg_lo, t1);

    let sin = ctx.sin_f32(reduced);
    let cos = ctx.cos_f32(reduced);
    (sin, cos)
}

impl Kernel for PartialNeoxRopeKernel {
    fn name(&self) -> &str {
        "gdn_partial_neox_rope"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let half = self.half();

        PtxKernel::new(self.name())
            .param(PtxType::U64, "x_ptr") // [num_heads * head_dim], rotated in place
            .param(PtxType::U32, "position")
            .param(PtxType::F32, "theta_scale") // freq_base^(-2/n_rot), from the host
            .shared_memory(0)
            .build(|ctx| {
                let j = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);

                let half_r = ctx.mov_u32_imm(half);
                let in_bounds = ctx.setp_lt_u32(j, half_r);
                ctx.branch_if_not(in_bounds, "gdn_rope_exit");

                let x_ptr = ctx.load_param_u64("x_ptr");
                let position = ctx.load_param_u32("position");
                let theta_scale = ctx.load_param_f32("theta_scale");

                // theta = pos * theta_scale^j, by j multiplications in the CPU's order.
                let theta = ctx.cvt_f32_u32(position);
                let step = ctx.mov_u32_imm(0);
                ctx.label("gdn_rope_theta_loop");
                let more = ctx.setp_lt_u32(step, j);
                ctx.branch_if_not(more, "gdn_rope_theta_end");
                ctx.mul_f32_inplace(theta, theta_scale);
                ctx.add_u32_inplace(step, 1);
                ctx.branch("gdn_rope_theta_loop");
                ctx.label("gdn_rope_theta_end");

                let (sin, cos) = emit_sin_cos(ctx, theta);

                // a at h*head_dim + j, b at h*head_dim + j + half.
                let head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let head_base = ctx.add_u64(x_ptr, head_off);
                let a_off = ctx.mul_wide_u32(j, 4);
                let a_addr = ctx.add_u64(head_base, a_off);
                let b_delta = ctx.mov_u64_imm(u64::from(half) * 4);
                let b_addr = ctx.add_u64(a_addr, b_delta);
                let a = ctx.ld_global_f32(a_addr);
                let b = ctx.ld_global_f32(b_addr);

                // Exactly the CPU's two expressions: separate mul/mul/sub and
                // mul/mul/add, not fma, so the rounding matches term for term.
                let a_cos = ctx.mul_f32(a, cos);
                let b_sin = ctx.mul_f32(b, sin);
                let rot_a = ctx.sub_f32(a_cos, b_sin);
                let a_sin = ctx.mul_f32(a, sin);
                let b_cos = ctx.mul_f32(b, cos);
                let rot_b = ctx.add_f32(a_sin, b_cos);
                ctx.st_global_f32(a_addr, rot_a);
                ctx.st_global_f32(b_addr, rot_b);

                ctx.label("gdn_rope_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_partial_rope_ptx_reduces_before_the_hardware_sine() {
        let kernel = PartialNeoxRopeKernel::new(16, 256, 128);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_partial_neox_rope"), "{ptx}");
        assert!(ptx.contains("sin.approx.f32"), "{ptx}");
        assert!(ptx.contains("cos.approx.f32"), "{ptx}");
        // Cody-Waite: the reduction is two fma steps against a split 2*pi, so a
        // theta of a few thousand never reaches sin.approx unreduced.
        assert!(
            ptx.contains("cvt.rmi.f32.f32"),
            "range reduction floor:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            2,
            "two-word reduction, and nothing else:\n{ptx}"
        );
        // theta_scale arrives from the host: no device pow, which cannot be made
        // sub-ulp here and an ulp is the whole budget (see the module docs).
        assert!(!ptx.contains("lg2.approx.f32"), "{ptx}");
        assert!(ptx.contains("theta_scale"), "{ptx}");
        assert_eq!(kernel.grid(), (16, 1, 1));
        assert_eq!(kernel.block(), (64, 1, 1));
    }

    #[test]
    fn gdn_partial_rope_theta_scale_is_the_cpu_expression() {
        let kernel = PartialNeoxRopeKernel::new(16, 256, 128);
        for freq_base in [10_000.0f32, 1_000_000.0f32] {
            assert_eq!(
                kernel.theta_scale(freq_base).to_bits(),
                freq_base.powf(-2.0 / 128.0f32).to_bits(),
                "theta_scale must be bit-identical to the CPU reference"
            );
        }
    }

    #[test]
    fn gdn_partial_rope_two_pi_split_is_exact() {
        // The reduction is only sound if TWO_PI_HI is exact with few enough bits and
        // the two words really sum to 2*pi.
        let sum = f64::from(TWO_PI_HI) + f64::from(TWO_PI_LO);
        assert!(
            (sum - std::f64::consts::TAU).abs() < 1e-10,
            "hi+lo must reconstruct 2*pi, got {sum}"
        );
        // n * TWO_PI_HI must be exact for n up to 2^16: 6.28125 = 201/32.
        assert!((f64::from(TWO_PI_HI) * 32.0 - 201.0).abs() < f64::EPSILON);
    }
}

/// Device parity against a verbatim port of `apply_partial_neox_rope`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_partial_rope_device_tests {
    use super::PartialNeoxRopeKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{assert_close, run_kernel, Lcg};

    /// `config.num_heads` of the Qwen3.5-0.8B attention layers.
    const NUM_HEADS: usize = 16;
    /// `attn_q_norm.len()`.
    const HEAD_DIM: usize = 256;
    /// `2 * sum(rope.dimension_sections)` = `2 * (16 + 24 + 24 + 0)`.
    const N_ROT: usize = 128;

    /// Verbatim port of `aprender-serve`'s `apply_partial_neox_rope`
    /// (forward_qwen35.rs:191).
    fn apply_partial_neox_rope(
        x: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        n_rot: usize,
        pos: usize,
        freq_base: f32,
    ) {
        let half = n_rot / 2;
        let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
        for h in 0..num_heads {
            let base = h * head_dim;
            let mut theta = pos as f32;
            for j in 0..half {
                let (sin, cos) = theta.sin_cos();
                let (a, b) = (x[base + j], x[base + j + half]);
                x[base + j] = a * cos - b * sin;
                x[base + j + half] = a * sin + b * cos;
                theta *= theta_scale;
            }
        }
    }

    #[test]
    fn gdn_partial_neox_rope_matches_cpu_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_partial_neox_rope: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let kernel = PartialNeoxRopeKernel::new(NUM_HEADS as u32, HEAD_DIM as u32, N_ROT as u32);

        let mut rng = Lcg::new(0x3477_0020);
        let host = rng.vec(NUM_HEADS * HEAD_DIM, 1.0);

        for freq_base in [10_000.0f32, 1_000_000.0f32] {
            for pos in [0usize, 1, 17, 1000] {
                let mut want = host.clone();
                apply_partial_neox_rope(&mut want, NUM_HEADS, HEAD_DIM, N_ROT, pos, freq_base);

                let buf = GpuBuffer::from_host(&ctx, &host).expect("x");
                // The f32 parameter travels in the low half of its argument slot.
                let mut args = [
                    buf.as_ptr(),
                    u64::try_from(pos).expect("pos"),
                    u64::from(kernel.theta_scale(freq_base).to_bits()),
                ];
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
                assert_close(
                    &got,
                    &want,
                    1e-4,
                    &format!("partial NEOX RoPE, pos {pos}, freq_base {freq_base}"),
                );

                // The tail past n_rot must be untouched — the whole point of the
                // "partial" in partial NEOX RoPE.
                for h in 0..NUM_HEADS {
                    let base = h * HEAD_DIM;
                    assert_eq!(
                        &got[base + N_ROT..base + HEAD_DIM],
                        &host[base + N_ROT..base + HEAD_DIM],
                        "head {h} was rotated past n_rot at pos {pos}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdn_partial_neox_rope_at_position_zero_is_the_identity() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_partial_neox_rope identity: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let kernel = PartialNeoxRopeKernel::new(NUM_HEADS as u32, HEAD_DIM as u32, N_ROT as u32);

        let mut rng = Lcg::new(0x3477_0021);
        let host = rng.vec(NUM_HEADS * HEAD_DIM, 1.0);
        let buf = GpuBuffer::from_host(&ctx, &host).expect("x");
        let scale = u64::from(kernel.theta_scale(1_000_000.0).to_bits());
        let mut args = [buf.as_ptr(), 0u64, scale];
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
        // theta = 0 for every pair, so sin = 0, cos = 1: bit-identical passthrough.
        assert_eq!(got, host, "position 0 must not move a single element");

        // A kernel that did nothing at all would also pass the assertion above, so
        // the same buffer is rotated at a position that must move it.
        let moving = GpuBuffer::from_host(&ctx, &host).expect("x");
        let mut args2 = [moving.as_ptr(), 3u64, scale];
        run_kernel(
            &ctx,
            &stream,
            &kernel,
            kernel.grid(),
            kernel.block(),
            &mut args2,
        );
        let mut moved = vec![0.0f32; host.len()];
        moving.copy_to_host(&mut moved).expect("download");
        assert!(
            moved != host,
            "position 3 must rotate the first n_rot dimensions"
        );
    }
}
