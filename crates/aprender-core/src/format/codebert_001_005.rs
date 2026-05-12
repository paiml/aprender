// `codebert-tokenizer-validation-v1` algorithm-level PARTIAL discharge
// for the 5 RoBERTa-tokenizer-on-shell falsifiers.
//
// Contract: `contracts/codebert-tokenizer-validation-v1.yaml`.
// Refs: Feng et al. (2020) CodeBERT, Liu et al. (2019) RoBERTa.

/// CodeBERT BPE vocabulary size (microsoft/codebert-base).
pub const AC_CODEBERT_VOCAB_SIZE: u32 = 50_265;

/// Construct-preservation threshold (≥70% acceptable).
pub const AC_CODEBERT_CONSTRUCT_THRESHOLD: f32 = 0.70;

/// Maximum tokens per shell construct.
pub const AC_CODEBERT_MAX_TOKENS_PER_CONSTRUCT: usize = 20;

// =============================================================================
// FALSIFY-CTOK-001 — vocab size = 50265
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebertVocabSizeVerdict {
    /// vocab.json + merges.txt loaded exactly 50265 tokens.
    Pass,
    /// Wrong count — vocab corrupted or wrong model downloaded.
    Fail,
}

#[must_use]
pub fn verdict_from_codebert_vocab_size(loaded_vocab_size: u32) -> CodebertVocabSizeVerdict {
    if loaded_vocab_size == AC_CODEBERT_VOCAB_SIZE {
        CodebertVocabSizeVerdict::Pass
    } else {
        CodebertVocabSizeVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-CTOK-002 — every non-empty input ⇒ ≥1 token
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebertNonEmptyVerdict {
    /// Every script in corpus tokenizes to ≥ 1 token.
    Pass,
    /// At least one non-empty script produced 0 tokens — silent drop.
    Fail,
}

/// Each entry: (was_input_non_empty, token_count).
#[must_use]
pub fn verdict_from_codebert_non_empty(corpus_results: &[(bool, usize)]) -> CodebertNonEmptyVerdict {
    if corpus_results.is_empty() {
        return CodebertNonEmptyVerdict::Fail;
    }
    for (non_empty, tokens) in corpus_results {
        if *non_empty && *tokens == 0 {
            return CodebertNonEmptyVerdict::Fail;
        }
    }
    CodebertNonEmptyVerdict::Pass
}

// =============================================================================
// FALSIFY-CTOK-003 — ≥70% of constructs tokenize acceptably
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebertConstructPreservationVerdict {
    /// acceptable / total ≥ 0.70.
    Pass,
    /// Below threshold — RoBERTa tokenizer too fragmented for shell.
    Fail,
}

#[must_use]
pub fn verdict_from_codebert_construct_preservation(
    acceptable_count: u32,
    total_count: u32,
) -> CodebertConstructPreservationVerdict {
    if total_count == 0 {
        return CodebertConstructPreservationVerdict::Fail;
    }
    let rate = acceptable_count as f32 / total_count as f32;
    if rate >= AC_CODEBERT_CONSTRUCT_THRESHOLD {
        CodebertConstructPreservationVerdict::Pass
    } else {
        CodebertConstructPreservationVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-CTOK-004 — token count bounded ≤ 20 per construct
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebertTokenBoundVerdict {
    /// max(token_count_per_construct) ≤ 20.
    Pass,
    /// Some construct exploded — pathological tokenization.
    Fail,
}

#[must_use]
pub fn verdict_from_codebert_token_bound(per_construct_token_counts: &[usize]) -> CodebertTokenBoundVerdict {
    if per_construct_token_counts.is_empty() {
        return CodebertTokenBoundVerdict::Fail;
    }
    for &count in per_construct_token_counts {
        if count > AC_CODEBERT_MAX_TOKENS_PER_CONSTRUCT {
            return CodebertTokenBoundVerdict::Fail;
        }
    }
    CodebertTokenBoundVerdict::Pass
}

// =============================================================================
// FALSIFY-CTOK-005 — deterministic tokenization
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebertDeterminismVerdict {
    /// Same input → bit-identical token sequence across N calls.
    Pass,
    /// Sequences diverged — HashMap ordering leak in BPE merge.
    Fail,
}

#[must_use]
pub fn verdict_from_codebert_determinism(call_a: &[u32], call_b: &[u32]) -> CodebertDeterminismVerdict {
    if call_a != call_b {
        return CodebertDeterminismVerdict::Fail;
    }
    if call_a.is_empty() {
        // Empty matches empty — but harness defect (no input).
        return CodebertDeterminismVerdict::Fail;
    }
    CodebertDeterminismVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_vocab_size_50265() {
        assert_eq!(AC_CODEBERT_VOCAB_SIZE, 50_265);
    }

    #[test]
    fn provenance_construct_threshold_70() {
        assert!((AC_CODEBERT_CONSTRUCT_THRESHOLD - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_max_tokens_20() {
        assert_eq!(AC_CODEBERT_MAX_TOKENS_PER_CONSTRUCT, 20);
    }

    // -------------------------------------------------------------------------
    // Section 2: CTOK-001 vocab size.
    // -------------------------------------------------------------------------
    #[test]
    fn fctok001_pass_canonical() {
        assert_eq!(
            verdict_from_codebert_vocab_size(50_265),
            CodebertVocabSizeVerdict::Pass
        );
    }

    #[test]
    fn fctok001_fail_off_by_one() {
        assert_eq!(
            verdict_from_codebert_vocab_size(50_264),
            CodebertVocabSizeVerdict::Fail
        );
    }

    #[test]
    fn fctok001_fail_completely_wrong() {
        assert_eq!(
            verdict_from_codebert_vocab_size(151_936),
            CodebertVocabSizeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: CTOK-002 non-empty tokenization.
    // -------------------------------------------------------------------------
    #[test]
    fn fctok002_pass_all_non_empty_emit_tokens() {
        let r = [(true, 5), (true, 12), (true, 1)];
        assert_eq!(
            verdict_from_codebert_non_empty(&r),
            CodebertNonEmptyVerdict::Pass
        );
    }

    #[test]
    fn fctok002_fail_one_silent_drop() {
        let r = [(true, 5), (true, 0)];
        assert_eq!(
            verdict_from_codebert_non_empty(&r),
            CodebertNonEmptyVerdict::Fail
        );
    }

    #[test]
    fn fctok002_pass_empty_input_zero_tokens_ok() {
        // Genuinely-empty input may produce 0 tokens; only non-empty must produce ≥ 1.
        let r = [(false, 0), (true, 5)];
        assert_eq!(
            verdict_from_codebert_non_empty(&r),
            CodebertNonEmptyVerdict::Pass
        );
    }

    #[test]
    fn fctok002_fail_empty_corpus() {
        assert_eq!(
            verdict_from_codebert_non_empty(&[]),
            CodebertNonEmptyVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: CTOK-003 construct preservation.
    // -------------------------------------------------------------------------
    #[test]
    fn fctok003_pass_above_threshold() {
        // 80/100 = 80%.
        assert_eq!(
            verdict_from_codebert_construct_preservation(80, 100),
            CodebertConstructPreservationVerdict::Pass
        );
    }

    #[test]
    fn fctok003_pass_at_threshold() {
        // 70/100 = 70% exact.
        assert_eq!(
            verdict_from_codebert_construct_preservation(70, 100),
            CodebertConstructPreservationVerdict::Pass
        );
    }

    #[test]
    fn fctok003_fail_below_threshold() {
        assert_eq!(
            verdict_from_codebert_construct_preservation(69, 100),
            CodebertConstructPreservationVerdict::Fail
        );
    }

    #[test]
    fn fctok003_fail_zero_total() {
        assert_eq!(
            verdict_from_codebert_construct_preservation(0, 0),
            CodebertConstructPreservationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: CTOK-004 token bound.
    // -------------------------------------------------------------------------
    #[test]
    fn fctok004_pass_under_bound() {
        let counts = [3_usize, 5, 8, 12, 19];
        assert_eq!(
            verdict_from_codebert_token_bound(&counts),
            CodebertTokenBoundVerdict::Pass
        );
    }

    #[test]
    fn fctok004_pass_at_bound() {
        let counts = [10_usize, 20];
        assert_eq!(
            verdict_from_codebert_token_bound(&counts),
            CodebertTokenBoundVerdict::Pass
        );
    }

    #[test]
    fn fctok004_fail_over_bound() {
        let counts = [3_usize, 5, 25];
        assert_eq!(
            verdict_from_codebert_token_bound(&counts),
            CodebertTokenBoundVerdict::Fail
        );
    }

    #[test]
    fn fctok004_fail_empty() {
        assert_eq!(
            verdict_from_codebert_token_bound(&[]),
            CodebertTokenBoundVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: CTOK-005 determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn fctok005_pass_bit_identical() {
        let t = vec![1_u32, 2, 3, 4, 5];
        assert_eq!(
            verdict_from_codebert_determinism(&t, &t),
            CodebertDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fctok005_fail_token_drift() {
        let a = vec![1_u32, 2, 3];
        let b = vec![1_u32, 2, 4];
        assert_eq!(
            verdict_from_codebert_determinism(&a, &b),
            CodebertDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fctok005_fail_length_mismatch() {
        assert_eq!(
            verdict_from_codebert_determinism(&[1], &[1, 2]),
            CodebertDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fctok005_fail_empty() {
        assert_eq!(
            verdict_from_codebert_determinism(&[], &[]),
            CodebertDeterminismVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy CodeBERT validation passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_validation_passes_all_5() {
        // 001
        assert_eq!(
            verdict_from_codebert_vocab_size(50_265),
            CodebertVocabSizeVerdict::Pass
        );
        // 002
        let r = [(true, 8), (true, 15), (true, 3)];
        assert_eq!(
            verdict_from_codebert_non_empty(&r),
            CodebertNonEmptyVerdict::Pass
        );
        // 003
        assert_eq!(
            verdict_from_codebert_construct_preservation(85, 100),
            CodebertConstructPreservationVerdict::Pass
        );
        // 004
        let counts = [3_usize, 5, 8, 10];
        assert_eq!(
            verdict_from_codebert_token_bound(&counts),
            CodebertTokenBoundVerdict::Pass
        );
        // 005
        let t = vec![1_u32, 2, 3];
        assert_eq!(
            verdict_from_codebert_determinism(&t, &t),
            CodebertDeterminismVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: wrong vocab.
        assert_eq!(
            verdict_from_codebert_vocab_size(151_936),
            CodebertVocabSizeVerdict::Fail
        );
        // 002: silent drop.
        let r = [(true, 0)];
        assert_eq!(
            verdict_from_codebert_non_empty(&r),
            CodebertNonEmptyVerdict::Fail
        );
        // 003: too fragmented.
        assert_eq!(
            verdict_from_codebert_construct_preservation(40, 100),
            CodebertConstructPreservationVerdict::Fail
        );
        // 004: token explosion on heredoc.
        let counts = [3_usize, 50];
        assert_eq!(
            verdict_from_codebert_token_bound(&counts),
            CodebertTokenBoundVerdict::Fail
        );
        // 005: HashMap-ordering leak.
        let a = vec![1_u32, 2, 3];
        let b = vec![1_u32, 3, 2];
        assert_eq!(
            verdict_from_codebert_determinism(&a, &b),
            CodebertDeterminismVerdict::Fail
        );
    }
}
