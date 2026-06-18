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
    /// PMAT-824: tanh attention-logit + final-logit softcapping (Gemma2/Gemma3).
    ///
    /// Gemma2/Gemma3 clamp attention scores (`50.0`) and final logits (`30.0`)
    /// through `softcap * tanh(x / softcap)`. The CUDA `forward_gpu_resident`
    /// path applies NEITHER, so a model that needs softcapping but is run on the
    /// uncapped GPU forward produces silently-wrong logits. GPU does NOT support
    /// this op, so a model requiring it is routed to CPU at the capability layer.
    AttnFinalSoftcap,
    /// PMAT-824: per-layer post-attention + post-FFN RMSNorms (Gemma2/Gemma3).
    ///
    /// Gemma2/Gemma3 use FOUR norms per block (input, post-attn, pre-FFN,
    /// post-FFN) versus the LLaMA-style TWO (input, pre-FFN). The CUDA forward
    /// applies only the two LLaMA-style norms, so the extra post-attn/post-FFN
    /// normalization is dropped on GPU → wrong residual stream. GPU does NOT
    /// support this op; a model requiring it is routed to CPU at the capability
    /// layer.
    PostAttnFfnNorm,
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
            Self::AttnFinalSoftcap => write!(f, "AttnFinalSoftcap"),
            Self::PostAttnFfnNorm => write!(f, "PostAttnFfnNorm"),
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

/// PMAT-824: Whether `arch` denotes a Gemma architecture that needs tanh
/// attention/final-logit softcapping AND per-layer post-attn/post-FFN RMSNorms.
///
/// `ArchConstraints` alone CANNOT distinguish these families: the
/// arch-constraints contract maps `gemma`, `gemma2`, and `gemma3` to the SAME
/// constraint row via aliases (same norm/activation/mlp_type), so
/// [`required_ops`] sees identical ops for all three. The softcapping /
/// 4-norm-per-block behavior is a property of the **version** (gemma2/gemma3),
/// detectable only from the raw architecture string (or GGUF metadata). Gemma
/// **v1** (`gemma`) has NO softcapping and only two norms, so it is excluded.
///
/// Matching is case-insensitive and prefix-based on the digit suffix: any
/// `gemma2*` / `gemma3*` name (incl. `Gemma2ForCausalLM`, `gemma3n`) is caught;
/// bare `gemma` / `gemmaforcausallm` is NOT.
#[must_use]
pub fn arch_needs_softcap_postnorm(arch: &str) -> bool {
    let lower = arch.to_ascii_lowercase();
    if !lower.starts_with("gemma") {
        return false;
    }
    // Strip the "gemma" prefix and inspect the first remaining char. Gemma v1 is
    // bare "gemma" / "gemmaforcausallm" (next char is 'f' or end) — NOT softcap.
    // gemma2 / gemma3 (and gemma3n) have a digit immediately after "gemma".
    let suffix = &lower["gemma".len()..];
    matches!(suffix.chars().next(), Some('2' | '3'))
}

/// PMAT-824: Required ops for a concrete model, distinguishing version-specific
/// behaviors that [`required_ops`] (constraints-only) cannot see.
///
/// This is the model-aware capability entry point used by the GPU admission
/// gate. It returns [`required_ops`]`(constraints)` PLUS any version-gated ops
/// derived from the raw `arch` string — currently the Gemma2/Gemma3
/// [`RequiredOp::AttnFinalSoftcap`] and [`RequiredOp::PostAttnFfnNorm`] that the
/// CUDA forward does not implement. Because [`gpu_supported_ops`] omits those,
/// a Gemma2/Gemma3 model is refused GPU residency at the CAPABILITY layer
/// (LOUD, at LOAD) instead of only being caught later by the runtime cosine
/// parity gate.
#[must_use]
pub fn required_ops_for_model(constraints: &ArchConstraints, arch: &str) -> HashSet<RequiredOp> {
    let mut ops = required_ops(constraints);
    if arch_needs_softcap_postnorm(arch) {
        ops.insert(RequiredOp::AttnFinalSoftcap);
        ops.insert(RequiredOp::PostAttnFfnNorm);
    }
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
                                    // NOT supported yet (models requiring these fall back to CPU):
                                    // - GeluMlp (GPU uses SwiGLU path; GELU MLP models fall back to CPU)
                                    // - LayerNorm (GPU uses RMSNorm path; LayerNorm models fall back to CPU)
                                    // - AbsolutePos (GPU uses RoPE; absolute-pos models fall back to CPU)
                                    // - AttnFinalSoftcap (PMAT-824: CUDA forward_gpu_resident applies NO
                                    //   tanh attn/final-logit softcapping → Gemma2/Gemma3 fall back to CPU)
                                    // - PostAttnFfnNorm (PMAT-824: CUDA forward applies only the 2 LLaMA-style
                                    //   norms, not the 4-per-block Gemma2/Gemma3 norms → fall back to CPU)
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

    // ========================================================================
    // PMAT-824: Gemma2/Gemma3 softcap + post-attn/post-FFN norm capability gate
    // ========================================================================

    /// PMAT-824 FALSIFIER (RED pre-fix / GREEN post-fix). Pre-fix, the GPU gate
    /// used `required_ops(constraints)` which maps Gemma2/Gemma3 → the same alias
    /// row as Gemma v1 (GatedMlp→SwiGLU, RMSNorm, RoPE — all GPU-supported), so
    /// `check_capability` returned Ok ⇒ the capability layer said "GPU-OK" for a
    /// model whose softcap/post-norms the CUDA forward does NOT implement, leaving
    /// only the runtime cosine parity gate as the safety net. Post-fix, the
    /// model-aware op set adds the unsupported softcap/post-norm ops ⇒ Err.
    #[test]
    fn test_gemma2_routed_to_cpu_at_capability_layer() {
        let constraints = ArchConstraints::from_architecture("gemma2");
        let supported = gpu_supported_ops();

        // The old constraints-only path would (wrongly) pass for gemma2.
        let constraints_only = required_ops(&constraints);
        assert!(
            check_capability(&constraints_only, &supported).is_ok(),
            "documents the gap: constraints alone read gemma2 as GPU-supported"
        );

        // The model-aware path catches it: gemma2 needs softcap + post-norms.
        let model_aware = required_ops_for_model(&constraints, "gemma2");
        let result = check_capability(&model_aware, &supported);
        assert!(
            result.is_err(),
            "gemma2 must be refused GPU residency at the capability layer"
        );
        let missing = result.unwrap_err();
        assert!(missing.contains(&RequiredOp::AttnFinalSoftcap));
        assert!(missing.contains(&RequiredOp::PostAttnFfnNorm));
    }

    #[test]
    fn test_gemma3_routed_to_cpu_at_capability_layer() {
        let constraints = ArchConstraints::from_architecture("gemma3");
        let supported = gpu_supported_ops();
        let model_aware = required_ops_for_model(&constraints, "gemma3");
        let result = check_capability(&model_aware, &supported);
        assert!(result.is_err(), "gemma3 must be refused GPU residency");
        let missing = result.unwrap_err();
        assert!(missing.contains(&RequiredOp::AttnFinalSoftcap));
        assert!(missing.contains(&RequiredOp::PostAttnFfnNorm));
    }

    /// No-regression half of the falsifier: a non-softcap GPU-coherent model
    /// (Qwen2.5-coder = qwen2) is UNAFFECTED — the new check is gated strictly to
    /// softcap/post-norm models, so Qwen2 still runs on GPU.
    #[test]
    fn test_qwen2_still_gpu_supported_after_gate() {
        let constraints = ArchConstraints::from_architecture("qwen2");
        let supported = gpu_supported_ops();
        let model_aware = required_ops_for_model(&constraints, "qwen2");
        assert!(
            check_capability(&model_aware, &supported).is_ok(),
            "qwen2 (non-softcap) must stay GPU-supported"
        );
        // The model-aware set must be IDENTICAL to the constraints-only set for
        // non-gemma2/3 archs (no spurious softcap ops added).
        assert_eq!(model_aware, required_ops(&constraints));
    }

    #[test]
    fn test_llama_still_gpu_supported_after_gate() {
        let constraints = ArchConstraints::from_architecture("llama");
        let supported = gpu_supported_ops();
        let model_aware = required_ops_for_model(&constraints, "llama");
        assert!(check_capability(&model_aware, &supported).is_ok());
        assert_eq!(model_aware, required_ops(&constraints));
    }

    /// Gemma **v1** has no softcapping and only 2 norms; it must NOT be swept up
    /// by the gate (it has its own CPU-support story via PMAT-809). The arch-name
    /// detector excludes bare `gemma`.
    #[test]
    fn test_gemma_v1_not_flagged_as_softcap() {
        assert!(!arch_needs_softcap_postnorm("gemma"));
        assert!(!arch_needs_softcap_postnorm("Gemma"));
        assert!(!arch_needs_softcap_postnorm("GemmaForCausalLM"));
    }

    #[test]
    fn test_arch_needs_softcap_postnorm_detection() {
        // Gemma2/Gemma3 family (incl. HF class names + gemma3n) → true.
        assert!(arch_needs_softcap_postnorm("gemma2"));
        assert!(arch_needs_softcap_postnorm("gemma3"));
        assert!(arch_needs_softcap_postnorm("Gemma2ForCausalLM"));
        assert!(arch_needs_softcap_postnorm("Gemma3ForCausalLM"));
        assert!(arch_needs_softcap_postnorm("gemma3n"));
        assert!(arch_needs_softcap_postnorm("GEMMA2"));
        // Non-gemma and gemma-v1 → false.
        assert!(!arch_needs_softcap_postnorm("gemma"));
        assert!(!arch_needs_softcap_postnorm("llama"));
        assert!(!arch_needs_softcap_postnorm("qwen2"));
        assert!(!arch_needs_softcap_postnorm("qwen3"));
        assert!(!arch_needs_softcap_postnorm("mistral"));
        assert!(!arch_needs_softcap_postnorm(""));
        // Defensive: a hypothetical "gem" / "gemini" must not match.
        assert!(!arch_needs_softcap_postnorm("gem"));
        assert!(!arch_needs_softcap_postnorm("gemini"));
    }

    #[test]
    fn test_softcap_ops_not_gpu_supported() {
        let supported = gpu_supported_ops();
        assert!(!supported.contains(&RequiredOp::AttnFinalSoftcap));
        assert!(!supported.contains(&RequiredOp::PostAttnFfnNorm));
    }

    #[test]
    fn test_new_required_op_display() {
        assert_eq!(RequiredOp::AttnFinalSoftcap.to_string(), "AttnFinalSoftcap");
        assert_eq!(RequiredOp::PostAttnFfnNorm.to_string(), "PostAttnFfnNorm");
    }
}
