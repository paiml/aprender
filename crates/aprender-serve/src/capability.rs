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

/// Every `RequiredOp` variant, in declaration order.
///
/// An enum you cannot enumerate cannot be checked for exhaustiveness, and the
/// capability facts are exactly the kind that go stale one variant at a time.
/// `contracts/apr-model-capability-v1.yaml` is compared against this list, so a
/// variant added here without a contract row fails a test rather than becoming a
/// silent hole — the op-shaped form of #3850's "an unknown quant is decoded as
/// Q4_K".
pub const ALL_REQUIRED_OPS: [RequiredOp; 13] = [
    RequiredOp::RoPE,
    RequiredOp::GQA,
    RequiredOp::MHA,
    RequiredOp::SwiGLU,
    RequiredOp::GeluMlp,
    RequiredOp::RMSNorm,
    RequiredOp::LayerNorm,
    RequiredOp::BiasAdd,
    RequiredOp::QkNorm,
    RequiredOp::AbsolutePos,
    RequiredOp::CausalMask,
    RequiredOp::AttnFinalSoftcap,
    RequiredOp::PostAttnFfnNorm,
];

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

/// Why this build's CUDA path has no forward for `architecture`, or `None` when
/// it has one.
///
/// #3817. This is not a capability *mismatch* — no kernel is missing and no
/// arithmetic fails. The path simply was not built: `qwen3_moe` dispatches to
/// `infer::qwen3_moe_generate`, which is CPU-only and returns `used_gpu = false`
/// unconditionally (`infer/inference_result.rs`, "CPU-only path; GPU MoE wiring
/// is M32d follow-up"). A caller that forced the GPU must be told that **before**
/// a load, by name, instead of receiving a CPU generation and an exit code.
///
/// The dispatch site calls this function too, so the refusal and the routing
/// cannot drift apart: if a GPU MoE forward lands (#3714), deleting the arm here
/// is what turns the refusal off, and the dispatch follows.
///
/// The argument is a raw `general.architecture`; it is normalised here, so every
/// spelling the loader accepts (`qwen3moe`, `qwen3_moe`, `Qwen3MoeForCausalLM`,
/// `Qwen3CoderForCausalLM`, …) resolves to the same answer.
#[must_use]
pub fn no_cuda_forward_reason(architecture: &str) -> Option<String> {
    match crate::tensor_names::normalize_architecture(architecture) {
        "qwen3_moe" => Some(format!(
            "this build has no CUDA forward for architecture '{architecture}' (canonical \
             'qwen3_moe'): the mixture-of-experts GPU path is #3714 and lands in 0.70.0. 0.69.1 \
             runs this architecture on the CPU only. Re-run without --gpu to use the CPU path \
             deliberately, or use a dense Q4_K model on the GPU. This is a refusal, not a \
             fallback: nothing was loaded and nothing was generated."
        )),
        _ => None,
    }
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

    /// #3477 (operator ruling 2026-09-19): the capability gate must ADMIT the
    /// Qwen3.5 hybrid now that `Qwen35CudaModel` gives it a GPU forward (#3090).
    ///
    /// `ArchConstraints` declares no Gated DeltaNet operation — the struct has
    /// no field for one, and the qwen3_5 row is an ordinary
    /// RoPE/RMSNorm/SwiGLU/no-QK-norm row — so `required_ops` derives only ops
    /// `gpu_supported_ops` already contains. That is why `check_gpu_capability`
    /// admits this architecture without a new `RequiredOp`: the discriminator
    /// for "can the GPU run the hybrid layers" is the hybrid forward's own
    /// existence (`gguf::hybrid_forward_handles`), not this op set. This test
    /// pins the admission so a future op added to the qwen3_5 row cannot
    /// silently re-refuse it.
    #[test]
    fn qwen35_hybrid_is_admitted_by_the_gpu_capability_gate() {
        for arch in ["qwen35", "qwen3_5", "qwen3.5"] {
            let constraints = ArchConstraints::from_architecture(arch);
            let required = required_ops_for_model(&constraints, arch);
            let supported = gpu_supported_ops();
            assert!(
                check_capability(&required, &supported).is_ok(),
                "{arch}: the GPU forward exists (#3090) — the capability gate must not refuse it, \
                 missing {:?}",
                check_capability(&required, &supported)
            );
            assert!(
                crate::gguf::hybrid_forward_handles("qwen35"),
                "the runtime-dispatch predicate is what decides the hybrid layers"
            );
        }
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

    // ---- #3817: the architectures with no CUDA forward ---------------------

    /// Every spelling the loader accepts must reach the same refusal — a user
    /// who names the model one way and the file another gets one answer.
    #[test]
    fn every_qwen3moe_spelling_has_no_cuda_forward() {
        for spelling in [
            "qwen3moe",
            "qwen3_moe",
            "Qwen3MoeForCausalLM",
            "Qwen3MoEForCausalLM",
            "Qwen3CoderForCausalLM",
            "qwen3_5_moe",
        ] {
            let reason = no_cuda_forward_reason(spelling).unwrap_or_else(|| {
                panic!("{spelling} normalises to qwen3_moe and has no GPU forward")
            });
            assert!(
                reason.contains(spelling),
                "the refusal names what the user gave: {reason}"
            );
            assert!(
                reason.contains("qwen3_moe"),
                "and the canonical name: {reason}"
            );
            assert!(reason.contains("#3714"), "and the ticket: {reason}");
            assert!(reason.contains("0.70.0"), "and when it lands: {reason}");
            assert!(
                reason.contains("refusal, not a fallback"),
                "and that nothing ran: {reason}"
            );
        }
    }

    /// The architectures that DO have a CUDA forward must not be refused. This
    /// is the other half of the case table done_when 4 asks for: one dense
    /// Q4_K family, the hybrid, and a non-qwen model.
    #[test]
    fn architectures_with_a_cuda_forward_are_not_refused() {
        for arch in [
            "qwen2", "qwen2.5", "qwen3", "qwen35", "llama", "mistral", "phi2", "gemma", "gpt2",
        ] {
            assert!(
                no_cuda_forward_reason(arch).is_none(),
                "{arch} has a GPU forward and must not be refused"
            );
        }
    }

    /// An unknown architecture is not refused by this predicate: it is not the
    /// list of what we support, it is the list of what we know we did not build.
    /// Refusing the unknown here would be a claim we cannot back.
    #[test]
    fn an_unknown_architecture_is_not_refused_by_this_predicate() {
        assert!(no_cuda_forward_reason("some-new-arch").is_none());
        assert!(no_cuda_forward_reason("").is_none());
    }
}

/// #3077 (alfredodeza): the user-facing GPU-vs-CPU support table, DERIVED.
///
/// His complaint is precise: *"There is no user-facing documentation that says which
/// model architectures get real GPU inference vs. silent CPU fallback. The only source
/// of truth is a compile-time capability gate that a user would have to read Rust to
/// discover."* So the answer cannot be prose — prose drifts from the gate, and then the
/// doc is a third source of truth that is wrong.
///
/// This module RENDERS the table from [`gpu_supported_ops`],
/// [`required_ops_for_model`] and [`no_cuda_forward_reason`] — the functions the runtime
/// actually calls — and a test asserts the committed `docs/GPU-SUPPORT.md` is byte-equal
/// to what they produce. Add a kernel and the doc goes stale loudly, in CI, on the same
/// commit. That is the operator's standing rule: tests DERIVED from the interface
/// surface, never a hand-maintained list.
#[cfg(test)]
mod gpu_support_doc {
    use super::*;

    /// The architectures a user actually brings. Adding one here regenerates the doc;
    /// there is no canonical enum to iterate, so this list IS the declared surface and
    /// the test below is what stops it drifting from the gate.
    const ARCHES: &[(&str, &str)] = &[
        ("llama", "Llama 2/3, TinyLlama, CodeLlama"),
        ("mistral", "Mistral, Mixtral (dense)"),
        ("qwen2", "Qwen2, Qwen2.5 (incl. Coder)"),
        ("qwen3", "Qwen3 dense"),
        ("qwen35", "Qwen3.5 hybrid (Gated DeltaNet)"),
        ("qwen3_moe", "Qwen3 / Qwen3.5 MoE (A3B)"),
        ("gemma2", "Gemma 2"),
        ("gemma3", "Gemma 3"),
        ("phi2", "Phi-2"),
        ("phi3", "Phi-3"),
        ("gpt2", "GPT-2"),
    ];

    /// The GPU quantisation whitelist, as `dtype.rs` enforces it.
    const QUANTS_GPU: &[(&str, bool, &str)] = &[
        ("F32", true, "ggml type 0"),
        ("F16", false, "ggml type 1 — #3846/#3850"),
        ("Q4_0", true, "ggml type 2"),
        ("Q4_1", true, "ggml type 3"),
        ("Q5_0", true, "ggml type 6"),
        ("Q8_0", true, "ggml type 8"),
        ("Q4_K", true, "ggml type 12 — the release matrix's quant"),
        ("Q5_K", true, "ggml type 13"),
        ("Q6_K", true, "ggml type 14"),
        (
            "Q5_1 / Q8_1 / Q2_K / Q3_K / Q8_K",
            false,
            "no GPU GEMV kernel",
        ),
        (
            "IQ2_XXS / IQ3_* / IQ4_NL / IQ4_XS",
            false,
            "no GPU GEMV kernel; IQ also fails the CPU dequant path",
        ),
    ];

    fn render() -> String {
        let supported = gpu_supported_ops();
        let mut out = String::new();
        out.push_str("# GPU vs CPU: which models get real GPU inference\n\n");
        out.push_str("<!-- GENERATED. Do not edit by hand.\n");
        out.push_str("     Rendered from crates/aprender-serve/src/capability.rs by the test\n");
        out.push_str("     `gpu_support_doc::the_committed_doc_matches_the_capability_gate`.\n");
        out.push_str("     Regenerate: APR_WRITE_GPU_SUPPORT_DOC=1 cargo test -p aprender-serve --lib gpu_support_doc\n");
        out.push_str("     Issue: #3077 (alfredodeza) -->\n\n");
        out.push_str(
            "A model that is not GPU-eligible is **not broken** — it runs on the CPU. What this\n",
        );
        out.push_str("table exists to prevent is the surprise: assuming an RTX 4090 makes any GGUF fast.\n\n");
        out.push_str("## Architectures\n\n");
        out.push_str("| architecture | models | GPU | why not |\n|---|---|---|---|\n");
        for (arch, models) in ARCHES {
            let c = ArchConstraints::from_architecture(arch);
            let required = required_ops_for_model(&c, arch);
            let verdict = match no_cuda_forward_reason(arch) {
                Some(_) => (
                    "**refused**".to_string(),
                    "no CUDA forward at all (#3714)".to_string(),
                ),
                None => match check_capability(&required, &supported) {
                    Ok(()) => ("yes".to_string(), "—".to_string()),
                    Err(missing) => {
                        let mut names: Vec<String> =
                            missing.iter().map(ToString::to_string).collect();
                        names.sort();
                        (
                            "CPU fallback".to_string(),
                            format!("missing `{}`", names.join("`, `")),
                        )
                    },
                },
            };
            out.push_str(&format!(
                "| `{arch}` | {models} | {} | {} |\n",
                verdict.0, verdict.1
            ));
        }
        out.push_str(
            "\n**`refused` is not `CPU fallback`.** A refusal names the architecture and exits\n",
        );
        out.push_str(
            "rather than loading; a fallback runs on the CPU. `apr run` without `--gpu` uses the\n",
        );
        out.push_str("CPU path deliberately and works for every row above.\n\n");
        out.push_str("## Quantizations, on the GPU path\n\n");
        out.push_str("| quantization | GPU | note |\n|---|---|---|\n");
        for (q, ok, note) in QUANTS_GPU {
            out.push_str(&format!(
                "| {q} | {} | {note} |\n",
                if *ok { "yes" } else { "no" }
            ));
        }
        out.push_str(
            "\nAn unsupported quantization on an otherwise GPU-eligible architecture falls back\n",
        );
        out.push_str(
            "to the CPU. See #3850 for the case where it was silently reinterpreted instead.\n",
        );
        out
    }

    #[test]
    fn the_committed_doc_matches_the_capability_gate() {
        let rendered = render();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/GPU-SUPPORT.md");
        if std::env::var("APR_WRITE_GPU_SUPPORT_DOC").is_ok() {
            std::fs::write(path, &rendered).expect("write docs/GPU-SUPPORT.md");
            return;
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert_eq!(
            committed, rendered,
            "docs/GPU-SUPPORT.md has drifted from capability.rs. \
             Regenerate: APR_WRITE_GPU_SUPPORT_DOC=1 cargo test -p aprender-serve --lib gpu_support_doc"
        );
    }

    /// The table would be worthless if every row said the same thing. Prove it
    /// discriminates: at least one architecture GPU-eligible, one CPU-fallback, one
    /// refused.
    #[test]
    fn the_table_discriminates_rather_than_saying_yes_everywhere() {
        let supported = gpu_supported_ops();
        let mut eligible = 0;
        let mut fallback = 0;
        let mut refused = 0;
        for (arch, _) in ARCHES {
            let c = ArchConstraints::from_architecture(arch);
            let required = required_ops_for_model(&c, arch);
            if no_cuda_forward_reason(arch).is_some() {
                refused += 1;
            } else if check_capability(&required, &supported).is_ok() {
                eligible += 1;
            } else {
                fallback += 1;
            }
        }
        assert!(
            eligible > 0,
            "no architecture is GPU-eligible — the gate is broken"
        );
        assert!(
            fallback > 0,
            "no architecture falls back — the table cannot be discriminating"
        );
        assert!(
            refused > 0,
            "no architecture is refused — qwen3_moe should be (#3714)"
        );
    }
}

/// #3832-adjacent, operator 2026-09-22: "capability.rs need deep pv SHACL support".
///
/// These facts are DECLARED in `contracts/apr-model-capability-v1.yaml`. This module
/// makes that declaration binding: the contract is the origin and this file is its
/// consumer, so a disagreement is a test failure rather than a silent divergence.
///
/// It exists because the negative facts used to live in a COMMENT. `gpu_supported_ops()`
/// inserts the eight supported ops as code; the five unsupported ones were named only in
/// a comment trailing the `QkNorm` insert line — so the positive facts were data and the
/// negative ones were prose attached to the wrong statement (#3075's class).
#[cfg(test)]
mod capability_contract {
    use super::*;
    use std::collections::BTreeSet;

    /// The contract, resolved from this crate's manifest dir rather than the CWD —
    /// `cargo test` runs from the crate, `cargo nextest` may not, and a path that
    /// depends on the caller's directory is a test that passes for the wrong reason.
    fn contract() -> serde_yaml_ng::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/apr-model-capability-v1.yaml");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_yaml_ng::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not valid YAML: {e}", path.display()))
    }

    fn rows(doc: &serde_yaml_ng::Value, key: &str) -> Vec<serde_yaml_ng::Value> {
        doc.get(key)
            .and_then(|v| v.as_sequence())
            .unwrap_or_else(|| panic!("contract has no `{key}` sequence"))
            .clone()
    }

    fn op_name(op: RequiredOp) -> String {
        format!("{op}")
    }

    /// FALSIFY-CAP-001. The contract's `gpu_supported: true` set and
    /// `gpu_supported_ops()` are the same set, compared by equality in BOTH
    /// directions — a subset check would pass while the contract silently dropped
    /// an op, which is the failure mode a declaration exists to prevent.
    #[test]
    fn the_contract_and_gpu_supported_ops_are_the_same_set() {
        let doc = contract();
        let declared: BTreeSet<String> = rows(&doc, "ops")
            .iter()
            .filter(|r| r.get("gpu_supported").and_then(|v| v.as_bool()) == Some(true))
            .map(|r| {
                r.get("op")
                    .and_then(|v| v.as_str())
                    .expect("every ops row needs an `op`")
                    .to_string()
            })
            .collect();
        let actual: BTreeSet<String> = gpu_supported_ops().into_iter().map(op_name).collect();

        assert_eq!(
            declared, actual,
            "contract `gpu_supported: true` disagrees with gpu_supported_ops().\n\
             declared only: {:?}\nin code only: {:?}\n\
             The contract is the origin: change it first, then this function.",
            declared.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&declared).collect::<Vec<_>>(),
        );
    }

    /// Every `RequiredOp` variant is declared exactly once. An op the contract does
    /// not mention is the #3850 hole in op form: neither supported nor refused.
    #[test]
    fn every_required_op_variant_is_declared_exactly_once() {
        let doc = contract();
        let mut seen: Vec<String> = rows(&doc, "ops")
            .iter()
            .filter_map(|r| r.get("op").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        seen.sort();
        let mut deduped = seen.clone();
        deduped.dedup();
        assert_eq!(seen, deduped, "an op is declared twice: {seen:?}");

        for op in ALL_REQUIRED_OPS {
            assert!(
                seen.contains(&op_name(op)),
                "RequiredOp::{} is not declared in the contract",
                op_name(op)
            );
        }
        assert_eq!(
            seen.len(),
            ALL_REQUIRED_OPS.len(),
            "the contract declares ops that are not RequiredOp variants: {seen:?}"
        );
    }

    /// A `gpu_supported: false` row must say WHY. The whole point of moving these
    /// out of a comment is that the reason travels with the fact.
    #[test]
    fn every_unsupported_op_carries_a_reason() {
        for r in rows(&contract(), "ops") {
            if r.get("gpu_supported").and_then(|v| v.as_bool()) == Some(false) {
                let op = r.get("op").and_then(|v| v.as_str()).unwrap_or("<unnamed>");
                let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                assert!(
                    !reason.trim().is_empty(),
                    "op {op} is declared unsupported with no reason — that is the \
                     comment it replaced, in YAML"
                );
            }
        }
    }

    /// FALSIFY-CAP-003. `wired` is a claim about a call graph and may not be asserted
    /// loosely. `unestablished` is explicitly NOT a claim, so it is exempt.
    #[test]
    fn an_implementation_row_claiming_wired_names_a_symbol() {
        for r in rows(&contract(), "op_implementation") {
            let status = r.get("status").and_then(|v| v.as_str()).unwrap_or("");
            let op = r.get("op").and_then(|v| v.as_str()).unwrap_or("<unnamed>");
            assert!(
                matches!(
                    status,
                    "wired" | "unwired" | "unimplemented" | "unestablished"
                ),
                "op {op} has status {status:?}, which is not one of \
                 wired/unwired/unimplemented/unestablished"
            );
            if matches!(status, "wired" | "unwired") {
                assert!(
                    r.get("symbol").and_then(|v| v.as_str()).is_some_and(|s| !s.is_empty()),
                    "op {op} is {status} but names no symbol — {status} is a statement \
                     about a specific function"
                );
            }
            if status == "unwired" {
                assert!(
                    r.get("evidence").and_then(|v| v.as_str()).is_some_and(|s| !s.is_empty()),
                    "op {op} is unwired with no evidence — `unwired` means every caller \
                     is a test, which is a thing someone had to go and read"
                );
            }
        }
    }
}
