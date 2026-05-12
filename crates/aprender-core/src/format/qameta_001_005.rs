// `apr-qa-metamorphic-v1` algorithm-level PARTIAL discharge for the 5
// metamorphic-testing falsifiers (quant equivalence, GGUF→APR→GGUF
// roundtrip, multi-arch smoke, prompt invariance, temp=0 determinism).
//
// Contract: `contracts/apr-qa-metamorphic-v1.yaml`.
// Refs: arXiv:1807.10453 (METTLE), arXiv:2103.13630 (Quantization
// methods), arXiv:2603.23611 (LLMORPH).

/// Cosine-similarity threshold for first-token logits across quantizations.
pub const AC_QAMETA_COSINE_MIN: f64 = 0.95;

/// Top-K used by quantization-equivalence tests.
pub const AC_QAMETA_TOP_K: u32 = 5;

/// Minimum top-K overlap (top-3 of top-5).
pub const AC_QAMETA_TOP_K_OVERLAP: u32 = 3;

/// L2 norm drift tolerance for GGUF→APR→GGUF roundtrip (1%).
pub const AC_QAMETA_ROUNDTRIP_TOLERANCE: f64 = 0.01;

/// Architecture families covered by the multi-arch smoke gate.
pub const AC_QAMETA_ARCHITECTURES: [&str; 5] =
    ["qwen2", "llama", "phi", "gemma", "mistral"];

/// Minimum architectures producing coherent output (3 of 5).
pub const AC_QAMETA_MIN_COHERENT_ARCHS: u32 = 3;

// =============================================================================
// F-META-001 — quantization equivalence
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantEquivalenceVerdict {
    /// Cosine similarity > 0.95 AND top-5 overlap ≥ 3.
    Pass,
    /// Either threshold violated.
    Fail,
}

#[must_use]
pub fn verdict_from_quant_equivalence(
    cosine_similarity: f64,
    top_k_overlap_count: u32,
) -> QuantEquivalenceVerdict {
    if cosine_similarity <= AC_QAMETA_COSINE_MIN {
        return QuantEquivalenceVerdict::Fail;
    }
    if top_k_overlap_count < AC_QAMETA_TOP_K_OVERLAP {
        return QuantEquivalenceVerdict::Fail;
    }
    QuantEquivalenceVerdict::Pass
}

// =============================================================================
// F-META-002 — format roundtrip fidelity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundtripVerdict {
    /// Tensor count preserved AND no NaN AND L2 drift < 1% per tensor.
    Pass,
    /// Any of: tensor count differs, NaN detected, L2 drift exceeds tolerance.
    Fail,
}

#[must_use]
pub fn verdict_from_roundtrip(
    tensor_count_before: u32,
    tensor_count_after: u32,
    has_nan: bool,
    max_l2_drift: f64,
) -> RoundtripVerdict {
    if tensor_count_before != tensor_count_after {
        return RoundtripVerdict::Fail;
    }
    if has_nan {
        return RoundtripVerdict::Fail;
    }
    if max_l2_drift >= AC_QAMETA_ROUNDTRIP_TOLERANCE {
        return RoundtripVerdict::Fail;
    }
    RoundtripVerdict::Pass
}

// =============================================================================
// F-META-003 — multi-architecture smoke
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiArchSmokeVerdict {
    /// At least 3 architectures produce non-empty NaN-free output.
    Pass,
    /// Fewer than 3 architectures pass — architecture-specific defect.
    Fail,
}

#[must_use]
pub fn verdict_from_multi_arch_smoke(arch_results: &[(&str, bool, bool)]) -> MultiArchSmokeVerdict {
    // (architecture_name, output_nonempty, has_nan_or_panic)
    let mut coherent: u32 = 0;
    for (_arch, nonempty, broken) in arch_results {
        if *nonempty && !*broken {
            coherent += 1;
        }
    }
    if coherent >= AC_QAMETA_MIN_COHERENT_ARCHS {
        MultiArchSmokeVerdict::Pass
    } else {
        MultiArchSmokeVerdict::Fail
    }
}

// =============================================================================
// F-META-004 — prompt invariance
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptInvarianceVerdict {
    /// Both rephrased prompts yield outputs containing the expected answer.
    Pass,
    /// At least one prompt missed the expected token.
    Fail,
}

#[must_use]
pub fn verdict_from_prompt_invariance(
    output_a_contains_answer: bool,
    output_b_contains_answer: bool,
) -> PromptInvarianceVerdict {
    if output_a_contains_answer && output_b_contains_answer {
        PromptInvarianceVerdict::Pass
    } else {
        PromptInvarianceVerdict::Fail
    }
}

// =============================================================================
// F-META-005 — temperature=0 determinism
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemperatureZeroDeterminismVerdict {
    /// All N runs at temperature=0 produced exactly 1 unique output.
    Pass,
    /// At least 2 distinct outputs across N runs — non-determinism.
    Fail,
}

#[must_use]
pub fn verdict_from_temp_zero_determinism(unique_outputs: u32) -> TemperatureZeroDeterminismVerdict {
    if unique_outputs == 1 {
        TemperatureZeroDeterminismVerdict::Pass
    } else {
        TemperatureZeroDeterminismVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_cosine_threshold_095() {
        assert!((AC_QAMETA_COSINE_MIN - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_top_k_5() {
        assert_eq!(AC_QAMETA_TOP_K, 5);
    }

    #[test]
    fn provenance_top_k_overlap_3() {
        assert_eq!(AC_QAMETA_TOP_K_OVERLAP, 3);
    }

    #[test]
    fn provenance_roundtrip_tolerance_001() {
        assert!((AC_QAMETA_ROUNDTRIP_TOLERANCE - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_architectures_count_5() {
        assert_eq!(AC_QAMETA_ARCHITECTURES.len(), 5);
    }

    #[test]
    fn provenance_min_coherent_archs_3() {
        assert_eq!(AC_QAMETA_MIN_COHERENT_ARCHS, 3);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-META-001 quantization equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fm001_pass_high_similarity_full_overlap() {
        assert_eq!(verdict_from_quant_equivalence(0.99, 5), QuantEquivalenceVerdict::Pass);
    }

    #[test]
    fn fm001_pass_at_overlap_threshold() {
        assert_eq!(verdict_from_quant_equivalence(0.96, 3), QuantEquivalenceVerdict::Pass);
    }

    #[test]
    fn fm001_fail_low_similarity() {
        // Strict greater-than: 0.95 fails.
        assert_eq!(verdict_from_quant_equivalence(0.95, 5), QuantEquivalenceVerdict::Fail);
    }

    #[test]
    fn fm001_fail_low_overlap() {
        assert_eq!(verdict_from_quant_equivalence(0.99, 2), QuantEquivalenceVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: F-META-002 roundtrip fidelity.
    // -------------------------------------------------------------------------
    #[test]
    fn fm002_pass_clean_roundtrip() {
        assert_eq!(verdict_from_roundtrip(339, 339, false, 0.005), RoundtripVerdict::Pass);
    }

    #[test]
    fn fm002_fail_tensor_count_mismatch() {
        assert_eq!(verdict_from_roundtrip(339, 338, false, 0.005), RoundtripVerdict::Fail);
    }

    #[test]
    fn fm002_fail_nan_introduced() {
        assert_eq!(verdict_from_roundtrip(339, 339, true, 0.0), RoundtripVerdict::Fail);
    }

    #[test]
    fn fm002_fail_l2_drift_at_threshold() {
        // Strict less-than: 1% exactly fails.
        assert_eq!(verdict_from_roundtrip(339, 339, false, 0.01), RoundtripVerdict::Fail);
    }

    #[test]
    fn fm002_pass_l2_drift_just_under() {
        assert_eq!(verdict_from_roundtrip(339, 339, false, 0.0099), RoundtripVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 4: F-META-003 multi-arch smoke.
    // -------------------------------------------------------------------------
    #[test]
    fn fm003_pass_all_5_coherent() {
        let r = [
            ("qwen2", true, false),
            ("llama", true, false),
            ("phi", true, false),
            ("gemma", true, false),
            ("mistral", true, false),
        ];
        assert_eq!(verdict_from_multi_arch_smoke(&r), MultiArchSmokeVerdict::Pass);
    }

    #[test]
    fn fm003_pass_3_of_5() {
        let r = [
            ("qwen2", true, false),
            ("llama", true, false),
            ("phi", true, false),
            ("gemma", false, false),
            ("mistral", true, true),
        ];
        assert_eq!(verdict_from_multi_arch_smoke(&r), MultiArchSmokeVerdict::Pass);
    }

    #[test]
    fn fm003_fail_only_2_coherent() {
        let r = [
            ("qwen2", true, false),
            ("llama", true, false),
            ("phi", false, false),
            ("gemma", false, true),
            ("mistral", true, true),
        ];
        assert_eq!(verdict_from_multi_arch_smoke(&r), MultiArchSmokeVerdict::Fail);
    }

    #[test]
    fn fm003_fail_empty_results() {
        assert_eq!(verdict_from_multi_arch_smoke(&[]), MultiArchSmokeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: F-META-004 prompt invariance.
    // -------------------------------------------------------------------------
    #[test]
    fn fm004_pass_both_contain_answer() {
        assert_eq!(verdict_from_prompt_invariance(true, true), PromptInvarianceVerdict::Pass);
    }

    #[test]
    fn fm004_fail_only_a_has_answer() {
        assert_eq!(verdict_from_prompt_invariance(true, false), PromptInvarianceVerdict::Fail);
    }

    #[test]
    fn fm004_fail_only_b_has_answer() {
        assert_eq!(verdict_from_prompt_invariance(false, true), PromptInvarianceVerdict::Fail);
    }

    #[test]
    fn fm004_fail_both_miss_answer() {
        assert_eq!(verdict_from_prompt_invariance(false, false), PromptInvarianceVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: F-META-005 temperature=0 determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn fm005_pass_one_unique_output() {
        assert_eq!(
            verdict_from_temp_zero_determinism(1),
            TemperatureZeroDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fm005_fail_two_unique_outputs() {
        assert_eq!(
            verdict_from_temp_zero_determinism(2),
            TemperatureZeroDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fm005_fail_zero_unique() {
        // Empty set ⇒ no runs ⇒ determinism cannot be asserted.
        assert_eq!(
            verdict_from_temp_zero_determinism(0),
            TemperatureZeroDeterminismVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy metamorphic run + pre-fix.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_run_passes_all_5() {
        assert_eq!(verdict_from_quant_equivalence(0.98, 5), QuantEquivalenceVerdict::Pass);
        assert_eq!(verdict_from_roundtrip(339, 339, false, 0.003), RoundtripVerdict::Pass);
        let r = [
            ("qwen2", true, false),
            ("llama", true, false),
            ("phi", true, false),
            ("gemma", true, false),
            ("mistral", true, false),
        ];
        assert_eq!(verdict_from_multi_arch_smoke(&r), MultiArchSmokeVerdict::Pass);
        assert_eq!(verdict_from_prompt_invariance(true, true), PromptInvarianceVerdict::Pass);
        assert_eq!(verdict_from_temp_zero_determinism(1), TemperatureZeroDeterminismVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        assert_eq!(verdict_from_quant_equivalence(0.50, 1), QuantEquivalenceVerdict::Fail);
        assert_eq!(verdict_from_roundtrip(339, 339, true, 0.05), RoundtripVerdict::Fail);
        let r = [
            ("qwen2", false, false),
            ("llama", false, false),
            ("phi", false, false),
            ("gemma", false, false),
            ("mistral", false, false),
        ];
        assert_eq!(verdict_from_multi_arch_smoke(&r), MultiArchSmokeVerdict::Fail);
        assert_eq!(verdict_from_prompt_invariance(false, false), PromptInvarianceVerdict::Fail);
        assert_eq!(verdict_from_temp_zero_determinism(3), TemperatureZeroDeterminismVerdict::Fail);
    }
}
