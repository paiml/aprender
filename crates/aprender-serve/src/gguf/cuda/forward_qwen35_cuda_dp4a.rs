//! Per-projection DP4A arming for [`super::Qwen35CudaModel`] (#3513).
//!
//! The model pins the float Q4_K/Q6_K GEMV for the whole architecture because
//! the DP4A kernels' int8 activation quantization compounds through the Gated
//! `DeltaNet` recurrence (see `with_max_seq_len`). That pin is all-or-nothing;
//! this mask says WHICH projections may take the DP4A kernel, so the ones that
//! drive the divergence can be told apart from the ones that do not.

use crate::cuda::gpu_profile::{GpuProfile, Q4kVariant, Q6kVariant};

/// A set of projection groups, one bit each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dp4aGroups(u8);

impl Dp4aGroups {
    /// No group: every GEMV is float.
    pub const NONE: Self = Self(0);
    /// `DeltaNet` `attn_qkv` — the input of the causal conv and the recurrence.
    pub const GDN_QKV: Self = Self(1);
    /// `DeltaNet` `ssm_alpha` / `ssm_beta` — the decay and write gates.
    pub const GDN_GATES: Self = Self(1 << 1);
    /// `DeltaNet` `attn_gate` — the output gate of the gated rmsnorm.
    pub const GDN_OUT_GATE: Self = Self(1 << 2);
    /// `DeltaNet` `ssm_out` — after the recurrence, into the residual.
    pub const GDN_SSM_OUT: Self = Self(1 << 3);
    /// Full-attention `attn_q` / `attn_k` / `attn_v` / `attn_output`.
    pub const ATTN: Self = Self(1 << 4);
    /// The `SwiGLU` FFN of every layer, both kinds.
    pub const FFN: Self = Self(1 << 5);
    /// The final `lm_head`.
    pub const LM_HEAD: Self = Self(1 << 6);
    /// Every group.
    pub const ALL: Self = Self(0x7f);

    /// Each single group with a stable name, in dispatch order.
    pub const EACH: [(&'static str, Self); 7] = [
        ("gdn_qkv", Self::GDN_QKV),
        ("gdn_gates", Self::GDN_GATES),
        ("gdn_out_gate", Self::GDN_OUT_GATE),
        ("gdn_ssm_out", Self::GDN_SSM_OUT),
        ("attn", Self::ATTN),
        ("ffn", Self::FFN),
        ("lm_head", Self::LM_HEAD),
    ];

    /// Whether every bit of `other` is in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Every group except those in `other`.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// Set `profile` for the next GEMV of `group`: the DP4A pair when `mask`
/// arms the group, the float pair otherwise. `None` leaves the profile alone,
/// which is the production path — the profile the model pinned at build time
/// (or a caller's own override) decides every dispatch.
pub(super) fn arm(profile: &mut GpuProfile, mask: Option<Dp4aGroups>, group: Dp4aGroups) {
    let Some(mask) = mask else { return };
    if mask.contains(group) {
        profile.q4k = Q4kVariant::HwDp4a;
        profile.q6k = Q6kVariant::HwDp4a;
    } else {
        profile.q4k = Q4kVariant::Mwv;
        profile.q6k = Q6kVariant::Mwv;
    }
}
