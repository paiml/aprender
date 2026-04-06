// Per-architecture required weight roles.
//
// FALLBACK for crates.io builds where architecture-requirements-v1.yaml
// is not available. This is a snapshot of the generated code — keep in sync.
//
// UCBD §4 / GH-279: Compile-time enforcement that every loader
// provides all tensors required by the target architecture.

use crate::gguf::ArchConstraints;

/// Weight roles that may be required for a transformer layer.
/// Each architecture requires a subset of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    /// K projection bias (Qwen2, Phi)
    AttnKBias,
    /// Per-head K RMSNorm gamma (Qwen3)
    AttnKNorm,
    /// Pre-attention layer normalization gamma
    AttnNorm,
    /// Q projection bias (Qwen2, Phi)
    AttnQBias,
    /// Per-head Q RMSNorm gamma (Qwen3)
    AttnQNorm,
    /// V projection bias (Qwen2, Phi)
    AttnVBias,
    /// FFN down projection
    FfnDown,
    /// Pre-FFN layer normalization gamma
    FfnNorm,
    /// FFN gate projection (SwiGLU/GatedMLP)
    FfnGate,
    /// Key projection weights
    KProj,
    /// Output projection weights
    OProj,
    /// Query projection weights
    QProj,
    /// FFN up projection
    FfnUp,
    /// Value projection weights
    VProj,
}

impl WeightRole {
    /// Returns the field name prefix as it appears in `IndexedLayerWeights`.
    #[must_use]
    pub const fn field_name(&self) -> &'static str {
        match self {
            Self::AttnKBias => "attn_k_bias",
            Self::AttnKNorm => "attn_k_norm",
            Self::AttnNorm => "attn_norm",
            Self::AttnQBias => "attn_q_bias",
            Self::AttnQNorm => "attn_q_norm",
            Self::AttnVBias => "attn_v_bias",
            Self::FfnDown => "ffn_down",
            Self::FfnNorm => "ffn_norm",
            Self::FfnGate => "ffn_gate",
            Self::KProj => "attn_k (k_proj)",
            Self::OProj => "attn_output (o_proj)",
            Self::QProj => "attn_q (q_proj)",
            Self::FfnUp => "ffn_up",
            Self::VProj => "attn_v (v_proj)",
        }
    }
}

/// Roles for constraint cell: no_qk_norm_no_bias (has_qk_norm=false, has_bias=false).
const ROLES_NO_QK_NORM_NO_BIAS: &[WeightRole] = &[
    WeightRole::AttnNorm,
    WeightRole::FfnNorm,
    WeightRole::QProj,
    WeightRole::KProj,
    WeightRole::VProj,
    WeightRole::OProj,
    WeightRole::FfnGate,
    WeightRole::FfnUp,
    WeightRole::FfnDown,
];

const _: () = assert!(ROLES_NO_QK_NORM_NO_BIAS.len() == 9, "YAML declares 9 roles");

/// Roles for constraint cell: no_qk_norm_bias (has_qk_norm=false, has_bias=true).
const ROLES_NO_QK_NORM_BIAS: &[WeightRole] = &[
    WeightRole::AttnNorm,
    WeightRole::FfnNorm,
    WeightRole::QProj,
    WeightRole::KProj,
    WeightRole::VProj,
    WeightRole::OProj,
    WeightRole::FfnGate,
    WeightRole::FfnUp,
    WeightRole::FfnDown,
    WeightRole::AttnQBias,
    WeightRole::AttnKBias,
    WeightRole::AttnVBias,
];

const _: () = assert!(ROLES_NO_QK_NORM_BIAS.len() == 12, "YAML declares 12 roles");

/// Roles for constraint cell: qk_norm_no_bias (has_qk_norm=true, has_bias=false).
const ROLES_QK_NORM_NO_BIAS: &[WeightRole] = &[
    WeightRole::AttnNorm,
    WeightRole::FfnNorm,
    WeightRole::QProj,
    WeightRole::KProj,
    WeightRole::VProj,
    WeightRole::OProj,
    WeightRole::FfnGate,
    WeightRole::FfnUp,
    WeightRole::FfnDown,
    WeightRole::AttnQNorm,
    WeightRole::AttnKNorm,
];

const _: () = assert!(ROLES_QK_NORM_NO_BIAS.len() == 11, "YAML declares 11 roles");

/// Roles for constraint cell: qk_norm_and_bias (has_qk_norm=true, has_bias=true).
const ROLES_QK_NORM_AND_BIAS: &[WeightRole] = &[
    WeightRole::AttnNorm,
    WeightRole::FfnNorm,
    WeightRole::QProj,
    WeightRole::KProj,
    WeightRole::VProj,
    WeightRole::OProj,
    WeightRole::FfnGate,
    WeightRole::FfnUp,
    WeightRole::FfnDown,
    WeightRole::AttnQNorm,
    WeightRole::AttnKNorm,
    WeightRole::AttnQBias,
    WeightRole::AttnKBias,
    WeightRole::AttnVBias,
];

const _: () = assert!(ROLES_QK_NORM_AND_BIAS.len() == 14, "YAML declares 14 roles");

/// Returns the required weight roles for a given architecture.
#[must_use]
pub fn required_roles(arch: &ArchConstraints) -> &'static [WeightRole] {
    contract_pre_constraint_matrix_exhaustiveness!();
    match (arch.has_qk_norm, arch.has_bias) {
        (false, false) => ROLES_NO_QK_NORM_NO_BIAS,
        (false, true) => ROLES_NO_QK_NORM_BIAS,
        (true, false) => ROLES_QK_NORM_NO_BIAS,
        (true, true) => ROLES_QK_NORM_AND_BIAS,
    }
}
