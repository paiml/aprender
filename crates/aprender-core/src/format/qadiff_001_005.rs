// `apr-qa-differential-v1` algorithm-level PARTIAL discharge for the 5
// differential-testing falsifiers (ollama parity, tokenizer roundtrip,
// concurrent serve parity, perplexity budget, cross-format L2).
//
// Contract: `contracts/apr-qa-differential-v1.yaml`.
// Refs: arXiv:2207.11976 (Differential Testing for ML), arXiv:2406.07944
// (DLLens), llama.cpp perplexity tracking, ollama integration tests.

/// F-DIFF-001 minimum top-1 token agreement rate against ollama.
pub const AC_QADIFF_TOP1_AGREEMENT: f64 = 0.95;

/// F-DIFF-001 maximum perplexity gap vs reference (5%).
pub const AC_QADIFF_PPL_GAP: f64 = 0.05;

/// F-DIFF-004 perplexity budget per quantization scheme.
/// Ratio = PPL(quantized) / PPL(F16). Pass iff ratio < budget[Q].
pub const AC_QADIFF_PPL_BUDGET_Q4_K: f64 = 1.10;
pub const AC_QADIFF_PPL_BUDGET_Q5_K: f64 = 1.05;
pub const AC_QADIFF_PPL_BUDGET_Q6_K: f64 = 1.02;
pub const AC_QADIFF_PPL_BUDGET_Q8_0: f64 = 1.01;

/// F-DIFF-005 cross-format tensor L2 tolerance (0.001 = 0.1%).
pub const AC_QADIFF_L2_TOLERANCE: f64 = 0.001;

// =============================================================================
// F-DIFF-001 — ollama parity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OllamaParityVerdict {
    /// Top-1 agreement ≥ 95% AND perplexity gap < 5%.
    Pass,
    /// Either threshold violated.
    Fail,
}

#[must_use]
pub fn verdict_from_ollama_parity(top1_agreement_rate: f64, ppl_gap: f64) -> OllamaParityVerdict {
    if top1_agreement_rate + 1e-9 < AC_QADIFF_TOP1_AGREEMENT {
        return OllamaParityVerdict::Fail;
    }
    if ppl_gap >= AC_QADIFF_PPL_GAP {
        return OllamaParityVerdict::Fail;
    }
    OllamaParityVerdict::Pass
}

// =============================================================================
// F-DIFF-002 — tokenizer roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerRoundtripVerdict {
    /// `decode(encode(text)) == text` for the entire test corpus.
    Pass,
    /// Any text in the corpus failed roundtrip.
    Fail,
}

#[must_use]
pub fn verdict_from_tokenizer_roundtrip(test_corpus: &[(&str, &str)]) -> TokenizerRoundtripVerdict {
    // Each pair is (original, after_decode_encode_decode).
    for (original, decoded) in test_corpus {
        if original != decoded {
            return TokenizerRoundtripVerdict::Fail;
        }
    }
    TokenizerRoundtripVerdict::Pass
}

// =============================================================================
// F-DIFF-003 — concurrent serve parity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentParityVerdict {
    /// All N concurrent requests at temperature=0 returned identical output.
    Pass,
    /// At least 2 distinct outputs across requests.
    Fail,
}

#[must_use]
pub fn verdict_from_concurrent_parity(responses: &[&str]) -> ConcurrentParityVerdict {
    if responses.is_empty() {
        return ConcurrentParityVerdict::Fail;
    }
    let first = responses[0];
    for r in responses.iter().skip(1) {
        if *r != first {
            return ConcurrentParityVerdict::Fail;
        }
    }
    ConcurrentParityVerdict::Pass
}

// =============================================================================
// F-DIFF-004 — perplexity budget per quantization
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerplexityBudgetVerdict {
    /// PPL(quantized) / PPL(F16) < budget[Q].
    Pass,
    /// Ratio at-or-above budget — quantization quality below threshold.
    Fail,
}

#[must_use]
pub fn verdict_from_perplexity_budget(quant: &str, ppl_ratio: f64) -> PerplexityBudgetVerdict {
    let budget = match quant {
        "Q4_K" | "q4_k" | "Q4_K_M" | "q4_k_m" => AC_QADIFF_PPL_BUDGET_Q4_K,
        "Q5_K" | "q5_k" | "Q5_K_M" | "q5_k_m" => AC_QADIFF_PPL_BUDGET_Q5_K,
        "Q6_K" | "q6_k" => AC_QADIFF_PPL_BUDGET_Q6_K,
        "Q8_0" | "q8_0" => AC_QADIFF_PPL_BUDGET_Q8_0,
        _ => return PerplexityBudgetVerdict::Fail,
    };
    if ppl_ratio < budget {
        PerplexityBudgetVerdict::Pass
    } else {
        PerplexityBudgetVerdict::Fail
    }
}

// =============================================================================
// F-DIFF-005 — cross-format tensor L2 integrity
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossFormatL2Verdict {
    /// max |L2_a - L2_b| / max(L2_a, L2_b) < 0.001 across pairwise format
    /// comparisons for every tensor.
    Pass,
    /// Any tensor's relative L2 difference exceeded tolerance — format
    /// conversion corrupted data.
    Fail,
}

#[must_use]
pub fn verdict_from_cross_format_l2(per_tensor_l2: &[(f64, f64, f64)]) -> CrossFormatL2Verdict {
    // Each entry: (L2 in GGUF, L2 in APR, L2 in SafeTensors).
    for (l2_gguf, l2_apr, l2_st) in per_tensor_l2 {
        if !rel_diff(*l2_gguf, *l2_apr) {
            return CrossFormatL2Verdict::Fail;
        }
        if !rel_diff(*l2_apr, *l2_st) {
            return CrossFormatL2Verdict::Fail;
        }
        if !rel_diff(*l2_gguf, *l2_st) {
            return CrossFormatL2Verdict::Fail;
        }
    }
    CrossFormatL2Verdict::Pass
}

fn rel_diff(a: f64, b: f64) -> bool {
    let denom = a.abs().max(b.abs()).max(1e-12);
    let diff = (a - b).abs() / denom;
    diff < AC_QADIFF_L2_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_top1_agreement_095() {
        assert!((AC_QADIFF_TOP1_AGREEMENT - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_ppl_gap_005() {
        assert!((AC_QADIFF_PPL_GAP - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_quant_budgets_strict_descending() {
        // Tighter quants must have tighter budgets.
        assert!(AC_QADIFF_PPL_BUDGET_Q4_K > AC_QADIFF_PPL_BUDGET_Q5_K);
        assert!(AC_QADIFF_PPL_BUDGET_Q5_K > AC_QADIFF_PPL_BUDGET_Q6_K);
        assert!(AC_QADIFF_PPL_BUDGET_Q6_K > AC_QADIFF_PPL_BUDGET_Q8_0);
    }

    #[test]
    fn provenance_l2_tolerance_001() {
        assert!((AC_QADIFF_L2_TOLERANCE - 0.001).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-DIFF-001 ollama parity.
    // -------------------------------------------------------------------------
    #[test]
    fn fd001_pass_high_agreement_low_ppl_gap() {
        assert_eq!(verdict_from_ollama_parity(0.97, 0.03), OllamaParityVerdict::Pass);
    }

    #[test]
    fn fd001_pass_at_thresholds() {
        // Exactly 0.95 and just under 5% — pass.
        assert_eq!(verdict_from_ollama_parity(0.95, 0.04999), OllamaParityVerdict::Pass);
    }

    #[test]
    fn fd001_fail_low_agreement() {
        assert_eq!(verdict_from_ollama_parity(0.94, 0.01), OllamaParityVerdict::Fail);
    }

    #[test]
    fn fd001_fail_high_ppl_gap() {
        assert_eq!(verdict_from_ollama_parity(0.99, 0.06), OllamaParityVerdict::Fail);
    }

    #[test]
    fn fd001_fail_ppl_gap_at_threshold() {
        // 5% exactly fails (strict less-than per `<` invariant).
        assert_eq!(verdict_from_ollama_parity(0.99, 0.05), OllamaParityVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: F-DIFF-002 tokenizer roundtrip.
    // -------------------------------------------------------------------------
    #[test]
    fn fd002_pass_identity_corpus() {
        let c = [("hello", "hello"), ("world", "world")];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Pass);
    }

    #[test]
    fn fd002_pass_chatml_special_tokens() {
        let c = [
            ("<|im_start|>user\nHi<|im_end|>", "<|im_start|>user\nHi<|im_end|>"),
        ];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Pass);
    }

    #[test]
    fn fd002_pass_empty_corpus_vacuous() {
        assert_eq!(verdict_from_tokenizer_roundtrip(&[]), TokenizerRoundtripVerdict::Pass);
    }

    #[test]
    fn fd002_fail_lossy_decode() {
        let c = [("hello", "hello"), ("café", "caf?")];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Fail);
    }

    #[test]
    fn fd002_fail_special_token_dropped() {
        let c = [("<|im_end|>", "")];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: F-DIFF-003 concurrent serve parity.
    // -------------------------------------------------------------------------
    #[test]
    fn fd003_pass_3_identical_responses() {
        let r = ["Paris.", "Paris.", "Paris."];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Pass);
    }

    #[test]
    fn fd003_pass_single_response_trivial() {
        let r = ["The answer is 4."];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Pass);
    }

    #[test]
    fn fd003_fail_two_diverged_responses() {
        let r = ["Paris.", "Paris.", "Lyon."];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Fail);
    }

    #[test]
    fn fd003_fail_empty() {
        let r: [&str; 0] = [];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: F-DIFF-004 perplexity budget.
    // -------------------------------------------------------------------------
    #[test]
    fn fd004_pass_q4k_within_budget() {
        // 1.05 < 1.10 budget for Q4_K.
        assert_eq!(verdict_from_perplexity_budget("Q4_K", 1.05), PerplexityBudgetVerdict::Pass);
    }

    #[test]
    fn fd004_pass_q8_within_tight_budget() {
        // 1.005 < 1.01 budget for Q8_0.
        assert_eq!(verdict_from_perplexity_budget("Q8_0", 1.005), PerplexityBudgetVerdict::Pass);
    }

    #[test]
    fn fd004_pass_lowercase_quant_name() {
        assert_eq!(verdict_from_perplexity_budget("q4_k", 1.05), PerplexityBudgetVerdict::Pass);
    }

    #[test]
    fn fd004_pass_q4_k_m_alias() {
        assert_eq!(verdict_from_perplexity_budget("Q4_K_M", 1.05), PerplexityBudgetVerdict::Pass);
    }

    #[test]
    fn fd004_fail_q4k_above_budget() {
        // 1.15 > 1.10 budget.
        assert_eq!(verdict_from_perplexity_budget("Q4_K", 1.15), PerplexityBudgetVerdict::Fail);
    }

    #[test]
    fn fd004_fail_q8_above_tight_budget() {
        // Q8_0 at 1.05 is too loose for the 1.01 budget.
        assert_eq!(verdict_from_perplexity_budget("Q8_0", 1.05), PerplexityBudgetVerdict::Fail);
    }

    #[test]
    fn fd004_fail_unknown_quant() {
        assert_eq!(verdict_from_perplexity_budget("Q1_K", 1.05), PerplexityBudgetVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: F-DIFF-005 cross-format L2.
    // -------------------------------------------------------------------------
    #[test]
    fn fd005_pass_identical_l2() {
        let t = [(1.234, 1.234, 1.234), (5.678, 5.678, 5.678)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Pass);
    }

    #[test]
    fn fd005_pass_within_tolerance() {
        // Relative diff < 0.1%. Use tight values: max pairwise rel_diff
        // is |1.0005 - 1.0000| / 1.0005 ≈ 5e-4, well under 1e-3.
        let t = [(1.0000, 1.0005, 1.0003)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Pass);
    }

    #[test]
    fn fd005_fail_outside_tolerance() {
        // 1% relative diff = 10x tolerance.
        let t = [(1.0, 1.01, 1.0)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Fail);
    }

    #[test]
    fn fd005_fail_one_tensor_drift() {
        // First tensor matches, second drifted heavily.
        let t = [(1.0, 1.0, 1.0), (2.0, 2.5, 2.0)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Fail);
    }

    #[test]
    fn fd005_pass_empty_vacuous() {
        assert_eq!(verdict_from_cross_format_l2(&[]), CrossFormatL2Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy differential run passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_run_passes_all_5() {
        // ollama agreement at 0.97, ppl_gap 2%.
        assert_eq!(verdict_from_ollama_parity(0.97, 0.02), OllamaParityVerdict::Pass);
        // tokenizer roundtrip for ASCII + ChatML.
        let c = [
            ("Hello", "Hello"),
            ("<|im_start|>user", "<|im_start|>user"),
        ];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Pass);
        // concurrent: 5 identical responses.
        let r = ["4", "4", "4", "4", "4"];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Pass);
        // PPL budget: Q4_K at 1.07.
        assert_eq!(verdict_from_perplexity_budget("Q4_K", 1.07), PerplexityBudgetVerdict::Pass);
        // Cross-format L2: matching norms.
        let t = [(1.234, 1.234, 1.234)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: ollama divergence.
        assert_eq!(verdict_from_ollama_parity(0.85, 0.15), OllamaParityVerdict::Fail);
        // 002: tokenizer dropped UTF-8.
        let c = [("café", "caf?")];
        assert_eq!(verdict_from_tokenizer_roundtrip(&c), TokenizerRoundtripVerdict::Fail);
        // 003: concurrent state leak.
        let r = ["A", "B", "A"];
        assert_eq!(verdict_from_concurrent_parity(&r), ConcurrentParityVerdict::Fail);
        // 004: Q4_K perplexity exceeded budget.
        assert_eq!(verdict_from_perplexity_budget("Q4_K", 1.20), PerplexityBudgetVerdict::Fail);
        // 005: cross-format L2 drifted.
        let t = [(1.0, 1.5, 1.0)];
        assert_eq!(verdict_from_cross_format_l2(&t), CrossFormatL2Verdict::Fail);
    }
}
