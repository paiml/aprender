//! PMAT-3477 / aprender#3090: device kernels for Qwen3.5's Gated `DeltaNet` block.
//!
//! The CPU reference in `aprender-serve`
//! (`gguf/inference/forward/forward_qwen35.rs`) is the specification; every kernel
//! here reproduces one of its functions element for element, and every test in this
//! module compares against a *verbatim port* of that function (copied, not imported —
//! `aprender-gpu` does not depend on `aprender-serve`).
//!
//! | kernel | CPU function it implements |
//! |--------|----------------------------|
//! | [`CausalConv1dSiluKernel`] | `causal_conv1d` + the `silu` loop that immediately follows it |
//! | [`PerHeadL2NormKernel`] | `l2_norm_per_head` |
//! | [`GdnGatesKernel`] | `dt[h] = softplus(alpha[h] + dt_bias[h]) * a[h]`, `beta[h] = sigmoid(beta_raw[h])` |
//! | [`DeltaRuleRecurrenceKernel`] | `delta_rule_recurrence` |
//! | [`GatedRmsNormKernel`] | `gated_rmsnorm` |
//! | [`SigmoidGateKernel`] | `apply_sigmoid_gate` |
//!
//! Qwen3.5 interleaves those Gated `DeltaNet` layers with *gated full-attention*
//! layers, whose 256-wide heads no attention kernel in this crate could serve
//! (`kernels/attention` truncates at `head_dim = 128`). Their three kernels live here
//! too, against the same CPU reference (`forward_attention`, forward_qwen35.rs:985):
//!
//! | kernel | CPU function it implements |
//! |--------|----------------------------|
//! | [`SplitInterleavedKernel`] | the `q` / `gate` de-interleave of the joint `attn_q` projection |
//! | [`PartialNeoxRopeKernel`] | `apply_partial_neox_rope` |
//! | [`DecodeAttention256Kernel`] | the scores / `softmax` / value accumulation block |
//! | [`DecodeAttentionSplitKKernel`] + [`DecodeAttentionSplitKReduceKernel`] | the same block, split-K over positions, f32 or f16 KV (#3725) |
//!
//! ## Shapes (Qwen3.5-0.8B)
//!
//! `head_k_dim = head_v_dim = 128`, `num_k_heads = 16`, `num_v_heads = 16`,
//! `conv_kernel = 4`, so the conv channel count is
//! `head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads = 6144` (`3 * inner_size`),
//! exactly as `forward_deltanet` computes `conv_dim`.
//!
//! ## Numerics
//!
//! `exp` is `ex2.approx.f32` after a `log2(e)` scale and `ln` is `lg2.approx.f32`
//! after a `ln(2)` scale; both carry ~2 ulp of error, which is why every kernel that
//! goes through them (conv+SiLU, gates, recurrence, gated RMSNorm, sigmoid gate) is
//! checked at 1e-3 instead of 1e-4. Reduction order inside a head is kept ascending
//! (`i = 0..D`) to match the CPU loop, so the per-layer L∞ ≤ 1e-3 parity contract for
//! the Gated `DeltaNet` block holds against the reference implementation.

mod causal_conv1d;
mod decode_attention;
mod decode_attention_splitk;
mod delta_rule;
mod gated_rmsnorm;
mod gdn_gates;
mod l2_norm;
mod partial_rope;
mod sigmoid_gate;
mod split_interleave;

#[cfg(test)]
mod test_support;

pub use causal_conv1d::CausalConv1dSiluKernel;
pub use decode_attention::{DecodeAttention256Kernel, DEFAULT_MAX_POSITIONS_PER_PASS};
pub use decode_attention_splitk::{
    decode_attention_reference_f64, decode_attention_splitk_cpu, splitk_partial_acc_len,
    splitk_partial_ml_len, DecodeAttentionSplitKKernel, DecodeAttentionSplitKReduceKernel,
    KvStorage, SplitKPlan, SPLITK_DEFAULT_TARGET_SPLITS, SPLITK_MIN_CHUNK, SPLITK_WARPS,
};
pub use delta_rule::DeltaRuleRecurrenceKernel;
pub use gated_rmsnorm::GatedRmsNormKernel;
pub use gdn_gates::GdnGatesKernel;
pub use l2_norm::PerHeadL2NormKernel;
pub use partial_rope::PartialNeoxRopeKernel;
pub use sigmoid_gate::SigmoidGateKernel;
pub use split_interleave::SplitInterleavedKernel;

use crate::ptx::builder::{KernelBuilder, PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::VirtualReg;

/// Threads per block for the elementwise kernels in this module.
pub(crate) const ELEMENTWISE_BLOCK: u32 = 256;

/// `exp(x)` as `ex2.approx.f32(x * log2(e))`.
///
/// ~2 ulp. The CPU reference uses `f32::exp`; the difference is well inside the
/// 1e-3 tolerance the Gated `DeltaNet` parity contract allows.
pub(crate) fn emit_exp_f32(ctx: &mut KernelBuilder<'_>, x: VirtualReg) -> VirtualReg {
    let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
    let scaled = ctx.mul_f32(x, log2e);
    ctx.ex2_f32(scaled)
}

/// `sigmoid(x) = 1 / (1 + exp(-x))` — the CPU expression, operation for operation.
pub(crate) fn emit_sigmoid_f32(ctx: &mut KernelBuilder<'_>, x: VirtualReg) -> VirtualReg {
    let neg = ctx.neg_f32(x);
    let e = emit_exp_f32(ctx, neg);
    let one = ctx.mov_f32_imm(1.0);
    let den = ctx.add_f32(one, e);
    ctx.div_f32(one, den)
}

/// `silu(x) = x / (1 + exp(-x))` — the CPU expression, operation for operation.
pub(crate) fn emit_silu_f32(ctx: &mut KernelBuilder<'_>, x: VirtualReg) -> VirtualReg {
    let neg = ctx.neg_f32(x);
    let e = emit_exp_f32(ctx, neg);
    let one = ctx.mov_f32_imm(1.0);
    let den = ctx.add_f32(one, e);
    ctx.div_f32(x, den)
}

/// `softplus(x) = if x > 20 { x } else { ln(1 + exp(x)) }`, branch and threshold
/// exactly as the CPU reference (`forward_qwen35.rs`'s `fn_softplus`).
///
/// `ln(y)` is `lg2.approx.f32(y) * ln(2)`. `label_prefix` must be unique within the
/// kernel: the builder emits flat PTX labels, so two softplus sites in one kernel
/// would otherwise collide.
pub(crate) fn emit_softplus_f32(
    ctx: &mut KernelBuilder<'_>,
    x: VirtualReg,
    label_prefix: &str,
) -> VirtualReg {
    let big_label = format!("{label_prefix}_softplus_big");
    let end_label = format!("{label_prefix}_softplus_end");

    let out = ctx.mov_f32_imm(0.0);
    let threshold = ctx.mov_f32_imm(20.0);
    let is_big = ctx.setp_gt_f32(x, threshold);
    ctx.branch_if(is_big, &big_label);

    // ln(1 + exp(x))
    let e = emit_exp_f32(ctx, x);
    let one = ctx.mov_f32_imm(1.0);
    let arg = ctx.add_f32(one, e);
    let log2_arg = ctx.lg2_f32(arg);
    let ln2 = ctx.mov_f32_imm(std::f32::consts::LN_2);
    let small = ctx.mul_f32(log2_arg, ln2);
    ctx.mov_f32_reg(out, small);
    ctx.branch(&end_label);

    ctx.label(&big_label);
    ctx.mov_f32_reg(out, x);

    ctx.label(&end_label);
    out
}

#[cfg(test)]
mod helper_ptx_tests {
    use super::*;
    use crate::kernels::Kernel;

    /// A throwaway kernel that exercises every helper, so the PTX shape of the
    /// transcendentals is asserted even on a host with no CUDA device.
    struct HelperProbe;

    impl Kernel for HelperProbe {
        fn name(&self) -> &str {
            "gdn_helper_probe"
        }

        fn build_ptx(&self) -> crate::ptx::PtxKernel {
            crate::ptx::PtxKernel::new(self.name())
                .param(crate::ptx::PtxType::U64, "x_ptr")
                .shared_memory(0)
                .build(|ctx| {
                    let ptr = ctx.load_param_u64("x_ptr");
                    let x = ctx.ld_global_f32(ptr);
                    let s = emit_silu_f32(ctx, x);
                    let g = emit_sigmoid_f32(ctx, s);
                    let p = emit_softplus_f32(ctx, g, "probe");
                    ctx.st_global_f32(ptr, p);
                    ctx.ret();
                })
        }
    }

    #[test]
    fn gdn_helpers_emit_ex2_and_lg2() {
        let ptx = HelperProbe.emit_ptx();
        assert!(
            ptx.contains("ex2.approx.f32"),
            "exp must lower to ex2.approx.f32:\n{ptx}"
        );
        assert!(
            ptx.contains("lg2.approx.f32"),
            "ln must lower to lg2.approx.f32:\n{ptx}"
        );
        // softplus branches on x > 20 and both arms write the same register.
        assert!(ptx.contains("probe_softplus_big:"), "{ptx}");
        assert!(ptx.contains("probe_softplus_end:"), "{ptx}");
        assert!(ptx.contains("setp.gt.f32"), "{ptx}");
    }
}
