// `classifier-pipeline-v1` algorithm-level PARTIAL discharge for the 5
// CLF-RUN classifier-pipeline falsifiers (sigmoid correctness, split
// determinism, linear probe learning, probe roundtrip, embedding
// roundtrip).
//
// Contract: `contracts/classifier-pipeline-v1.yaml`.
// Refs: SSC v11 §4.3 Classifier Infrastructure, Alain & Bengio (2016)
// "Understanding intermediate layers using linear classifier probes".
//
// ## Disambiguation
//
// `classification-finetune-v1.yaml` (FALSIFY-CLASS-001..006, already
// bound via `classification_contract_falsify`) covers fine-tuning. This
// contract — classifier-pipeline-v1 — covers the linear-probe-on-frozen-
// embeddings pipeline. Module suffix `clf_` (CLF-RUN identifier from the
// SSC spec) disambiguates from any `class_*` finetune modules.

/// Sigmoid value at zero per FALSIFY-CLF-001.
pub const AC_CLF_SIGMOID_ZERO: f32 = 0.5;

/// Tolerance for sigmoid(0) == 0.5 check.
pub const AC_CLF_SIGMOID_ZERO_TOLERANCE: f32 = 1e-6;

/// Saturation thresholds: sigmoid(±large) ≈ 0/1 within this tolerance.
pub const AC_CLF_SIGMOID_SAT_TOLERANCE: f32 = 1e-3;

/// Linear-probe synthetic-data accuracy threshold per FALSIFY-CLF-003.
pub const AC_CLF_PROBE_ACCURACY_THRESHOLD: f32 = 0.70;

// =============================================================================
// FALSIFY-CLF-001 — sigmoid correctness at boundary values
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmoidCorrectnessVerdict {
    /// sigmoid(0) ≈ 0.5, sigmoid(large) ≈ 1, sigmoid(-large) ≈ 0.
    Pass,
    /// Numerical implementation error in activation.
    Fail,
}

#[must_use]
pub fn verdict_from_sigmoid_correctness(
    sigmoid_at_zero: f32,
    sigmoid_at_large_pos: f32,
    sigmoid_at_large_neg: f32,
) -> SigmoidCorrectnessVerdict {
    // sigmoid(0) == 0.5 within 1e-6.
    if (sigmoid_at_zero - AC_CLF_SIGMOID_ZERO).abs() >= AC_CLF_SIGMOID_ZERO_TOLERANCE {
        return SigmoidCorrectnessVerdict::Fail;
    }
    // sigmoid(large_pos) ≈ 1.
    if (sigmoid_at_large_pos - 1.0).abs() >= AC_CLF_SIGMOID_SAT_TOLERANCE {
        return SigmoidCorrectnessVerdict::Fail;
    }
    // sigmoid(large_neg) ≈ 0.
    if sigmoid_at_large_neg.abs() >= AC_CLF_SIGMOID_SAT_TOLERANCE {
        return SigmoidCorrectnessVerdict::Fail;
    }
    SigmoidCorrectnessVerdict::Pass
}

// =============================================================================
// FALSIFY-CLF-002 — split determinism with same seed
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDeterminismVerdict {
    /// split_a == split_b (bit-identical) when seed is the same.
    Pass,
    /// Sets diverged — non-deterministic hash or random element.
    Fail,
}

#[must_use]
pub fn verdict_from_split_determinism(
    train_a: &[u32],
    test_a: &[u32],
    train_b: &[u32],
    test_b: &[u32],
) -> SplitDeterminismVerdict {
    if train_a.is_empty() && test_a.is_empty() {
        return SplitDeterminismVerdict::Fail;
    }
    if train_a != train_b {
        return SplitDeterminismVerdict::Fail;
    }
    if test_a != test_b {
        return SplitDeterminismVerdict::Fail;
    }
    SplitDeterminismVerdict::Pass
}

// =============================================================================
// FALSIFY-CLF-003 — linear probe learns linearly separable data
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeLearnsVerdict {
    /// Training accuracy on linearly-separable synthetic data > 70%.
    Pass,
    /// Below threshold — SGD bug or LR misconfiguration.
    Fail,
}

#[must_use]
pub fn verdict_from_probe_learns(train_accuracy: f32) -> ProbeLearnsVerdict {
    if !train_accuracy.is_finite() {
        return ProbeLearnsVerdict::Fail;
    }
    if !(0.0..=1.0).contains(&train_accuracy) {
        return ProbeLearnsVerdict::Fail;
    }
    if train_accuracy > AC_CLF_PROBE_ACCURACY_THRESHOLD {
        ProbeLearnsVerdict::Pass
    } else {
        ProbeLearnsVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-CLF-004 — probe save/load roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeRoundtripVerdict {
    /// Loaded probe weights, bias, all metadata bit-identical to saved.
    Pass,
    /// Any field differs — JSON precision loss or dropped field.
    Fail,
}

#[must_use]
pub fn verdict_from_probe_roundtrip(
    saved_weights: &[f32],
    loaded_weights: &[f32],
    saved_bias: f32,
    loaded_bias: f32,
) -> ProbeRoundtripVerdict {
    if saved_weights.len() != loaded_weights.len() {
        return ProbeRoundtripVerdict::Fail;
    }
    if saved_weights.is_empty() {
        return ProbeRoundtripVerdict::Fail;
    }
    for (a, b) in saved_weights.iter().zip(loaded_weights.iter()) {
        if a.is_nan() != b.is_nan() {
            return ProbeRoundtripVerdict::Fail;
        }
        if !a.is_nan() && a != b {
            return ProbeRoundtripVerdict::Fail;
        }
    }
    if saved_bias.is_nan() != loaded_bias.is_nan() {
        return ProbeRoundtripVerdict::Fail;
    }
    if !saved_bias.is_nan() && saved_bias != loaded_bias {
        return ProbeRoundtripVerdict::Fail;
    }
    ProbeRoundtripVerdict::Pass
}

// =============================================================================
// FALSIFY-CLF-005 — embedding save/load roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRoundtripVerdict {
    /// JSONL roundtrip preserves entry count AND every embedding value.
    Pass,
    /// Count mismatch or float precision loss.
    Fail,
}

#[must_use]
pub fn verdict_from_embedding_roundtrip(
    saved: &[Vec<f32>],
    loaded: &[Vec<f32>],
) -> EmbeddingRoundtripVerdict {
    if saved.len() != loaded.len() {
        return EmbeddingRoundtripVerdict::Fail;
    }
    if saved.is_empty() {
        return EmbeddingRoundtripVerdict::Fail;
    }
    for (sv, lv) in saved.iter().zip(loaded.iter()) {
        if sv.len() != lv.len() {
            return EmbeddingRoundtripVerdict::Fail;
        }
        for (a, b) in sv.iter().zip(lv.iter()) {
            if a != b {
                return EmbeddingRoundtripVerdict::Fail;
            }
        }
    }
    EmbeddingRoundtripVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_sigmoid_zero_05() {
        assert!((AC_CLF_SIGMOID_ZERO - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_probe_accuracy_threshold_70() {
        assert!((AC_CLF_PROBE_ACCURACY_THRESHOLD - 0.70).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: CLF-001 sigmoid correctness.
    // -------------------------------------------------------------------------
    #[test]
    fn fclf001_pass_canonical_sigmoid() {
        // sigmoid(0)=0.5, sigmoid(20)≈1, sigmoid(-20)≈0
        let s_zero = 0.5_f32;
        let s_pos = 1.0 - 1e-9;
        let s_neg = 1e-9_f32;
        assert_eq!(
            verdict_from_sigmoid_correctness(s_zero, s_pos, s_neg),
            SigmoidCorrectnessVerdict::Pass
        );
    }

    #[test]
    fn fclf001_fail_zero_off() {
        assert_eq!(
            verdict_from_sigmoid_correctness(0.6, 1.0, 0.0),
            SigmoidCorrectnessVerdict::Fail
        );
    }

    #[test]
    fn fclf001_fail_pos_saturation_off() {
        assert_eq!(
            verdict_from_sigmoid_correctness(0.5, 0.5, 0.0),
            SigmoidCorrectnessVerdict::Fail
        );
    }

    #[test]
    fn fclf001_fail_neg_saturation_off() {
        assert_eq!(
            verdict_from_sigmoid_correctness(0.5, 1.0, 0.5),
            SigmoidCorrectnessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: CLF-002 split determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn fclf002_pass_identical_splits() {
        let train = [1u32, 2, 3, 4];
        let test = [5u32, 6];
        assert_eq!(
            verdict_from_split_determinism(&train, &test, &train, &test),
            SplitDeterminismVerdict::Pass
        );
    }

    #[test]
    fn fclf002_fail_train_diverges() {
        let train_a = [1u32, 2, 3];
        let test = [4u32, 5];
        let train_b = [1u32, 3, 2];
        assert_eq!(
            verdict_from_split_determinism(&train_a, &test, &train_b, &test),
            SplitDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fclf002_fail_test_diverges() {
        let train = [1u32];
        let test_a = [2u32];
        let test_b = [3u32];
        assert_eq!(
            verdict_from_split_determinism(&train, &test_a, &train, &test_b),
            SplitDeterminismVerdict::Fail
        );
    }

    #[test]
    fn fclf002_fail_empty() {
        assert_eq!(
            verdict_from_split_determinism(&[], &[], &[], &[]),
            SplitDeterminismVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: CLF-003 probe learns.
    // -------------------------------------------------------------------------
    #[test]
    fn fclf003_pass_high_accuracy() {
        assert_eq!(verdict_from_probe_learns(0.95), ProbeLearnsVerdict::Pass);
    }

    #[test]
    fn fclf003_pass_just_above_threshold() {
        assert_eq!(verdict_from_probe_learns(0.71), ProbeLearnsVerdict::Pass);
    }

    #[test]
    fn fclf003_fail_at_threshold() {
        // Strict greater-than.
        assert_eq!(verdict_from_probe_learns(0.70), ProbeLearnsVerdict::Fail);
    }

    #[test]
    fn fclf003_fail_below_threshold() {
        assert_eq!(verdict_from_probe_learns(0.55), ProbeLearnsVerdict::Fail);
    }

    #[test]
    fn fclf003_fail_nan_accuracy() {
        assert_eq!(verdict_from_probe_learns(f32::NAN), ProbeLearnsVerdict::Fail);
    }

    #[test]
    fn fclf003_fail_out_of_range() {
        assert_eq!(verdict_from_probe_learns(1.5), ProbeLearnsVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: CLF-004 probe roundtrip.
    // -------------------------------------------------------------------------
    #[test]
    fn fclf004_pass_exact_roundtrip() {
        let w = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(
            verdict_from_probe_roundtrip(&w, &w, 0.5, 0.5),
            ProbeRoundtripVerdict::Pass
        );
    }

    #[test]
    fn fclf004_fail_weight_drift() {
        let saved = vec![0.1_f32];
        let loaded = vec![0.10000001_f32];
        assert_eq!(
            verdict_from_probe_roundtrip(&saved, &loaded, 0.5, 0.5),
            ProbeRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf004_fail_bias_drift() {
        let w = vec![0.1_f32];
        assert_eq!(
            verdict_from_probe_roundtrip(&w, &w, 0.5, 0.5000001),
            ProbeRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf004_fail_length_mismatch() {
        let saved = vec![0.1_f32, 0.2];
        let loaded = vec![0.1_f32];
        assert_eq!(
            verdict_from_probe_roundtrip(&saved, &loaded, 0.5, 0.5),
            ProbeRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf004_fail_empty() {
        assert_eq!(
            verdict_from_probe_roundtrip(&[], &[], 0.0, 0.0),
            ProbeRoundtripVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: CLF-005 embedding roundtrip.
    // -------------------------------------------------------------------------
    #[test]
    fn fclf005_pass_exact_roundtrip() {
        let saved = vec![vec![0.1_f32, 0.2], vec![0.3_f32, 0.4]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&saved, &saved),
            EmbeddingRoundtripVerdict::Pass
        );
    }

    #[test]
    fn fclf005_fail_count_mismatch() {
        let saved = vec![vec![0.1_f32]];
        let loaded = vec![vec![0.1_f32], vec![0.2_f32]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&saved, &loaded),
            EmbeddingRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf005_fail_dimension_mismatch() {
        let saved = vec![vec![0.1_f32, 0.2]];
        let loaded = vec![vec![0.1_f32]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&saved, &loaded),
            EmbeddingRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf005_fail_value_drift() {
        let saved = vec![vec![0.1_f32]];
        let loaded = vec![vec![0.10000001_f32]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&saved, &loaded),
            EmbeddingRoundtripVerdict::Fail
        );
    }

    #[test]
    fn fclf005_fail_empty() {
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(
            verdict_from_embedding_roundtrip(&empty, &empty),
            EmbeddingRoundtripVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy classifier-pipeline passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_pipeline_passes_all_5() {
        // 001
        assert_eq!(
            verdict_from_sigmoid_correctness(0.5, 1.0 - 1e-9, 1e-9),
            SigmoidCorrectnessVerdict::Pass
        );
        // 002
        let train = [1u32, 2, 3];
        let test = [4u32];
        assert_eq!(
            verdict_from_split_determinism(&train, &test, &train, &test),
            SplitDeterminismVerdict::Pass
        );
        // 003 (validated_results: linear probe at 95.2% on 500-entry BPE).
        assert_eq!(verdict_from_probe_learns(0.952), ProbeLearnsVerdict::Pass);
        // 004
        let w = vec![0.1_f32, 0.2];
        assert_eq!(
            verdict_from_probe_roundtrip(&w, &w, 0.5, 0.5),
            ProbeRoundtripVerdict::Pass
        );
        // 005
        let e = vec![vec![0.1_f32; 768]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&e, &e),
            EmbeddingRoundtripVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: sigmoid(0) returned 0.6.
        assert_eq!(
            verdict_from_sigmoid_correctness(0.6, 1.0, 0.0),
            SigmoidCorrectnessVerdict::Fail
        );
        // 002: hash leaked → split diverges.
        let a = [1u32, 2];
        let b = [2u32, 1];
        assert_eq!(
            verdict_from_split_determinism(&a, &[3u32], &b, &[3u32]),
            SplitDeterminismVerdict::Fail
        );
        // 003: SGD lr too large → 50% accuracy.
        assert_eq!(verdict_from_probe_learns(0.5), ProbeLearnsVerdict::Fail);
        // 004: f64→f32 precision loss in JSON.
        let saved = vec![0.1_f32];
        let loaded = vec![0.10000001_f32];
        assert_eq!(
            verdict_from_probe_roundtrip(&saved, &loaded, 0.5, 0.5),
            ProbeRoundtripVerdict::Fail
        );
        // 005: JSONL line skipped on parse error.
        let saved = vec![vec![0.1_f32], vec![0.2_f32]];
        let loaded = vec![vec![0.1_f32]];
        assert_eq!(
            verdict_from_embedding_roundtrip(&saved, &loaded),
            EmbeddingRoundtripVerdict::Fail
        );
    }
}
