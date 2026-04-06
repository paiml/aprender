//! GH-280: Kernel capability gate — contract-driven GPU admission control.
//!
//! Models declare required operations via [`ArchConstraints`]; GPU backends
//! declare supported operations. Mismatch = refuse at load time (not garbage
//! at inference time).
//!
//! # Architecture
//!
//! ```text
//! ArchConstraints ──► required_ops() ──► HashSet<RequiredOp>
//!                                              │
//!                          gpu_supported_ops() ─┤
//!                                              │
//!                         check_capability() ──► Ok(()) or Err(missing)
//! ```

use std::collections::HashSet;

use crate::gguf::{ArchConstraints, MlpType, NormType, PositionalEncoding};

/// An operation required by a model architecture for correct inference.
///
/// Each variant maps to a concrete GPU kernel or kernel feature.
/// If the GPU backend lacks the kernel, inference will produce garbage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequiredOp {
    /// Rotary Position Embedding
    RoPE,
    /// Grouped-Query Attention (num_kv_heads < num_heads)
    GQA,
    /// Multi-Head Attention (num_kv_heads == num_heads)
    MHA,
    /// SwiGLU feed-forward: gate ⊙ SiLU(up) → down
    SwiGLU,
    /// GELU MLP: up → GELU → down
    GeluMlp,
    /// RMS Normalization
    RMSNorm,
    /// Layer Normalization (with bias)
    LayerNorm,
    /// Bias addition in attention/FFN projections
    BiasAdd,
    /// Per-head QK RMSNorm (Qwen3)
    QkNorm,
    /// Learned absolute position embeddings (GPT-2, BERT)
    AbsolutePos,
    /// Causal attention mask
    CausalMask,
}

impl std::fmt::Display for RequiredOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoPE => write!(f, "RoPE"),
            Self::GQA => write!(f, "GQA"),
            Self::MHA => write!(f, "MHA"),
            Self::SwiGLU => write!(f, "SwiGLU"),
            Self::GeluMlp => write!(f, "GeluMlp"),
            Self::RMSNorm => write!(f, "RMSNorm"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::BiasAdd => write!(f, "BiasAdd"),
            Self::QkNorm => write!(f, "QkNorm"),
            Self::AbsolutePos => write!(f, "AbsolutePos"),
            Self::CausalMask => write!(f, "CausalMask"),
        }
    }
}

/// Derive the set of required operations from architecture constraints.
///
/// Each field of [`ArchConstraints`] maps to one or more [`RequiredOp`]s.
#[must_use]
pub fn required_ops(constraints: &ArchConstraints) -> HashSet<RequiredOp> {
    let mut ops = HashSet::new();

    // Positional encoding
    match constraints.positional_encoding {
        PositionalEncoding::Rope => {
            ops.insert(RequiredOp::RoPE);
        },
        PositionalEncoding::Absolute => {
            ops.insert(RequiredOp::AbsolutePos);
        },
        PositionalEncoding::Alibi => {
            ops.insert(RequiredOp::AbsolutePos); // ALiBi adds bias to attention scores
        },
        PositionalEncoding::Relative => {}, // T5: handled by attention layer
        PositionalEncoding::None => {},
    }

    // Normalization
    match constraints.norm_type {
        NormType::RmsNorm => {
            ops.insert(RequiredOp::RMSNorm);
        },
        NormType::LayerNorm => {
            ops.insert(RequiredOp::LayerNorm);
        },
    }

    // MLP type
    match constraints.mlp_type {
        MlpType::SwiGlu | MlpType::GatedMlp => {
            ops.insert(RequiredOp::SwiGLU);
        },
        MlpType::GeluMlp => {
            ops.insert(RequiredOp::GeluMlp);
        },
    }

    // Bias
    if constraints.has_bias {
        ops.insert(RequiredOp::BiasAdd);
    }

    // QK norm (Qwen3)
    if constraints.has_qk_norm {
        ops.insert(RequiredOp::QkNorm);
    }

    // All transformer architectures need causal masking
    ops.insert(RequiredOp::CausalMask);

    ops
}

/// Operations currently supported by the GPU (CUDA) backend.
///
/// This is a compile-time constant. When a new kernel is added to trueno,
/// add the corresponding [`RequiredOp`] here.
#[must_use]
pub fn gpu_supported_ops() -> HashSet<RequiredOp> {
    let mut ops = HashSet::new();
    ops.insert(RequiredOp::RoPE);
    ops.insert(RequiredOp::GQA);
    ops.insert(RequiredOp::MHA);
    ops.insert(RequiredOp::SwiGLU);
    ops.insert(RequiredOp::RMSNorm);
    ops.insert(RequiredOp::BiasAdd);
    ops.insert(RequiredOp::CausalMask);
    ops.insert(RequiredOp::QkNorm); // GH-280: trueno PerHeadRmsNormKernel
                                    // NOT supported yet:
                                    // - GeluMlp (GPU uses SwiGLU path; GELU MLP models fall back to CPU)
                                    // - LayerNorm (GPU uses RMSNorm path; LayerNorm models fall back to CPU)
                                    // - AbsolutePos (GPU uses RoPE; absolute-pos models fall back to CPU)
    ops
}

/// Check whether the GPU backend supports all operations required by a model.
///
/// # Returns
///
/// - `Ok(())` if all required ops are supported
/// - `Err(missing)` with the set of unsupported operations
pub fn check_capability<S: std::hash::BuildHasher>(
    required: &HashSet<RequiredOp, S>,
    supported: &HashSet<RequiredOp, S>,
) -> std::result::Result<(), Vec<RequiredOp>> {
    let missing: Vec<RequiredOp> = required.difference(supported).copied().collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}

/// Format a capability mismatch error for human display.
#[must_use]
pub fn format_mismatch(architecture: &str, missing: &[RequiredOp]) -> String {
    let ops: Vec<String> = missing.iter().map(ToString::to_string).collect();
    format!(
        "GPU capability mismatch for '{}': missing kernel support for [{}]. \
         Model will use CPU inference. To add GPU support, implement the missing \
         kernels in trueno.",
        architecture,
        ops.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_all_supported() {
        let constraints = ArchConstraints::from_architecture("llama");
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();
        assert!(check_capability(&required, &supported).is_ok());
    }

    #[test]
    fn test_qwen2_all_supported() {
        let constraints = ArchConstraints::from_architecture("qwen2");
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();
        assert!(check_capability(&required, &supported).is_ok());
    }

    #[test]
    fn test_qwen3_all_supported() {
        // GH-280: Qwen3 GPU inference now supported (PerHeadRmsNormKernel)
        let constraints = ArchConstraints::from_architecture("qwen3");
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();
        assert!(check_capability(&required, &supported).is_ok());
    }

    #[test]
    fn test_gpt2_missing_ops() {
        let constraints = ArchConstraints::from_architecture("gpt2");
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();
        let result = check_capability(&required, &supported);
        assert!(result.is_err());
        let missing = result.unwrap_err();
        // GPT-2 needs LayerNorm, GeluMlp, AbsolutePos — none in GPU
        assert!(missing.contains(&RequiredOp::LayerNorm));
        assert!(missing.contains(&RequiredOp::GeluMlp));
        assert!(missing.contains(&RequiredOp::AbsolutePos));
    }

    #[test]
    fn test_mistral_all_supported() {
        let constraints = ArchConstraints::from_architecture("mistral");
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();
        assert!(check_capability(&required, &supported).is_ok());
    }

    #[test]
    fn test_required_op_display() {
        assert_eq!(RequiredOp::QkNorm.to_string(), "QkNorm");
        assert_eq!(RequiredOp::RoPE.to_string(), "RoPE");
        assert_eq!(RequiredOp::SwiGLU.to_string(), "SwiGLU");
    }

    #[test]
    fn test_format_mismatch_message() {
        let msg = format_mismatch("qwen3", &[RequiredOp::QkNorm]);
        assert!(msg.contains("qwen3"));
        assert!(msg.contains("QkNorm"));
        assert!(msg.contains("CPU inference"));
    }

    #[test]
    fn test_empty_required_always_passes() {
        let required = HashSet::new();
        let supported = gpu_supported_ops();
        assert!(check_capability(&required, &supported).is_ok());
    }

    #[test]
    fn test_check_capability_returns_all_missing() {
        let mut required = HashSet::new();
        required.insert(RequiredOp::QkNorm); // now supported (GH-280)
        required.insert(RequiredOp::LayerNorm);
        required.insert(RequiredOp::GeluMlp);
        let supported = gpu_supported_ops();
        let result = check_capability(&required, &supported);
        assert!(result.is_err());
        let missing = result.unwrap_err();
        // QkNorm is now supported, only LayerNorm and GeluMlp are missing
        assert_eq!(missing.len(), 2);
    }
}
