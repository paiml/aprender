// Bundles two sister contracts in one verdict module:
//
//   `linear-probe-classifier-v1` (FALSIFY-PROBE-001..004)
//   `model-family-parity-v1` (FALSIFY-PARITY-001..005)
//
// PROBE-001: encoder weights byte-equal before/after 100 steps
// PROBE-002: softmax output sums to 1.0 AND every value > 0
// PROBE-003: trainable_params == K * d_model + K (bias)
// PROBE-004: embedding bit-determinism across calls
// PARITY-001: enum→YAML — every non-Auto variant has a YAML file
// PARITY-002: YAML→enum — every YAML family is recognized
// PARITY-003: from_model_type returns the correct variant
// PARITY-004: display_name is non-empty for all variants
// PARITY-005: is_llm classification is intentional and correct

use std::collections::HashSet;

/// PROBE-002: probability sum tolerance (softmax normalization).
pub const AC_PROBE_PROB_SUM_TOLERANCE: f32 = 1e-6;
/// PROBE-002: minimum probability — softmax must NOT underflow to 0.
pub const AC_PROBE_PROB_MIN: f32 = 0.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeParityVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// PROBE-001
// ----------------------------------------------------------------

/// PROBE-001: encoder bytes match before/after training.
#[must_use]
pub fn verdict_from_encoder_frozen(before: &[u8], after: &[u8]) -> ProbeParityVerdict {
    if before.is_empty() || after.is_empty() || before.len() != after.len() {
        return ProbeParityVerdict::Fail;
    }
    if before == after {
        ProbeParityVerdict::Pass
    } else {
        ProbeParityVerdict::Fail
    }
}

// ----------------------------------------------------------------
// PROBE-002
// ----------------------------------------------------------------

/// PROBE-002: softmax sums to 1 AND every value strictly > 0.
#[must_use]
pub fn verdict_from_softmax_valid(probs: &[f32]) -> ProbeParityVerdict {
    if probs.is_empty() {
        return ProbeParityVerdict::Fail;
    }
    let mut sum = 0.0_f32;
    for &p in probs {
        if !p.is_finite() || p <= AC_PROBE_PROB_MIN {
            return ProbeParityVerdict::Fail;
        }
        sum += p;
    }
    if (sum - 1.0).abs() <= AC_PROBE_PROB_SUM_TOLERANCE {
        ProbeParityVerdict::Pass
    } else {
        ProbeParityVerdict::Fail
    }
}

// ----------------------------------------------------------------
// PROBE-003
// ----------------------------------------------------------------

/// PROBE-003: trainable_params == K * d_model + K.
#[must_use]
pub fn verdict_from_trainable_param_count(
    k: usize,
    d_model: usize,
    observed: usize,
) -> ProbeParityVerdict {
    if k == 0 || d_model == 0 {
        return ProbeParityVerdict::Fail;
    }
    let Some(weights) = k.checked_mul(d_model) else {
        return ProbeParityVerdict::Fail;
    };
    let Some(expected) = weights.checked_add(k) else {
        return ProbeParityVerdict::Fail;
    };
    if observed == expected {
        ProbeParityVerdict::Pass
    } else {
        ProbeParityVerdict::Fail
    }
}

// ----------------------------------------------------------------
// PROBE-004
// ----------------------------------------------------------------

/// PROBE-004: embedding bit-determinism.
#[must_use]
pub fn verdict_from_embedding_determinism(
    call_a: &[f32],
    call_b: &[f32],
) -> ProbeParityVerdict {
    if call_a.is_empty() || call_b.is_empty() || call_a.len() != call_b.len() {
        return ProbeParityVerdict::Fail;
    }
    for (x, y) in call_a.iter().zip(call_b.iter()) {
        if x.to_bits() != y.to_bits() {
            return ProbeParityVerdict::Fail;
        }
    }
    ProbeParityVerdict::Pass
}

// ----------------------------------------------------------------
// PARITY-001 + PARITY-002 — enum↔YAML symmetry
// ----------------------------------------------------------------

/// PARITY-001 + 002: enum and YAML registries must be equal sets.
#[must_use]
pub fn verdict_from_enum_yaml_symmetry(
    enum_variants: &[&str],
    yaml_families: &[&str],
) -> ProbeParityVerdict {
    if enum_variants.is_empty() || yaml_families.is_empty() {
        return ProbeParityVerdict::Fail;
    }
    let e: HashSet<&str> = enum_variants.iter().copied().collect();
    let y: HashSet<&str> = yaml_families.iter().copied().collect();
    if e == y {
        ProbeParityVerdict::Pass
    } else {
        ProbeParityVerdict::Fail
    }
}

// ----------------------------------------------------------------
// PARITY-003 — from_model_type round-trip
// ----------------------------------------------------------------

/// PARITY-003: every (variant_name, model_type_str) round-trips.
///
/// `mappings` is `&[(model_type_str, expected_variant_name)]`; the
/// caller passes `(actual_variant_name)` after running
/// `from_model_type(model_type_str)`. Pass iff every actual matches
/// expected. Empty input is Fail.
#[must_use]
pub fn verdict_from_round_trip(
    mappings: &[(&str, &str, &str)],
) -> ProbeParityVerdict {
    if mappings.is_empty() {
        return ProbeParityVerdict::Fail;
    }
    for (_model_type, expected, actual) in mappings {
        if expected != actual {
            return ProbeParityVerdict::Fail;
        }
    }
    ProbeParityVerdict::Pass
}

// ----------------------------------------------------------------
// PARITY-004 — display_name non-empty
// ----------------------------------------------------------------

/// PARITY-004: every variant's `display_name()` is non-empty.
#[must_use]
pub fn verdict_from_display_name_nonempty(
    variants_with_names: &[(&str, &str)],
) -> ProbeParityVerdict {
    if variants_with_names.is_empty() {
        return ProbeParityVerdict::Fail;
    }
    for (_v, n) in variants_with_names {
        if n.is_empty() {
            return ProbeParityVerdict::Fail;
        }
    }
    ProbeParityVerdict::Pass
}

// ----------------------------------------------------------------
// PARITY-005 — is_llm classification correct
// ----------------------------------------------------------------

/// PARITY-005: `is_llm()` matches the contract for every variant.
///
/// `classifications` is `&[(variant, expected_is_llm, actual_is_llm)]`.
/// Pass iff every expected == actual.
#[must_use]
pub fn verdict_from_is_llm_classification(
    classifications: &[(&str, bool, bool)],
) -> ProbeParityVerdict {
    if classifications.is_empty() {
        return ProbeParityVerdict::Fail;
    }
    for (_v, expected, actual) in classifications {
        if expected != actual {
            return ProbeParityVerdict::Fail;
        }
    }
    ProbeParityVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_PROBE_PROB_SUM_TOLERANCE, 1e-6);
        assert_eq!(AC_PROBE_PROB_MIN, 0.0);
    }

    // -----------------------------------------------------------------
    // Section 2: PROBE-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fprobe001_pass_bytes_equal() {
        let before = vec![0xAA_u8; 64];
        let after = vec![0xAA_u8; 64];
        let v = verdict_from_encoder_frozen(&before, &after);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fprobe001_fail_one_byte_drift() {
        let before = vec![0xAA_u8; 64];
        let mut after = vec![0xAA_u8; 64];
        after[33] = 0xAB;
        let v = verdict_from_encoder_frozen(&before, &after);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe001_fail_length_mismatch() {
        let v = verdict_from_encoder_frozen(&[1, 2, 3], &[1, 2, 3, 4]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe002_pass_softmax_2_class() {
        let v = verdict_from_softmax_valid(&[0.3, 0.7]);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fprobe002_fail_underflow_zero() {
        let v = verdict_from_softmax_valid(&[1.0, 0.0]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe002_fail_sum_drift() {
        let v = verdict_from_softmax_valid(&[0.3, 0.6]); // sum = 0.9
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe002_fail_negative() {
        let v = verdict_from_softmax_valid(&[1.5, -0.5]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe003_pass_canonical_2_768() {
        // K=2, d=768: 2*768 + 2 = 1538
        let v = verdict_from_trainable_param_count(2, 768, 1538);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fprobe003_pass_canonical_10_4096() {
        let v = verdict_from_trainable_param_count(10, 4096, 10 * 4096 + 10);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fprobe003_fail_extra_params() {
        let v = verdict_from_trainable_param_count(2, 768, 1538 + 1);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe003_fail_zero_k() {
        let v = verdict_from_trainable_param_count(0, 768, 0);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fprobe004_pass_bit_identical() {
        let a = vec![1.0_f32, 2.5, -3.0];
        let b = a.clone();
        let v = verdict_from_embedding_determinism(&a, &b);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fprobe004_fail_one_ulp_drift() {
        let a = vec![1.0_f32, 2.5, -3.0];
        let bumped = f32::from_bits(2.5_f32.to_bits() + 1);
        let b = vec![1.0_f32, bumped, -3.0];
        let v = verdict_from_embedding_determinism(&a, &b);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: PARITY-001 + 002.
    // -----------------------------------------------------------------
    #[test]
    fn fparity_symmetry_pass_exact() {
        let e = ["Qwen2", "Qwen3", "Llama"];
        let y = ["Qwen3", "Qwen2", "Llama"];
        let v = verdict_from_enum_yaml_symmetry(&e, &y);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fparity_symmetry_fail_orphan() {
        let e = ["Qwen2", "Qwen3", "Llama", "Mistral"];
        let y = ["Qwen2", "Qwen3", "Llama"];
        let v = verdict_from_enum_yaml_symmetry(&e, &y);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fparity_symmetry_fail_ghost() {
        let e = ["Qwen2", "Qwen3"];
        let y = ["Qwen2", "Qwen3", "Llama"];
        let v = verdict_from_enum_yaml_symmetry(&e, &y);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fparity_symmetry_fail_both_empty() {
        let v = verdict_from_enum_yaml_symmetry(&[], &[]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: PARITY-003 round-trip.
    // -----------------------------------------------------------------
    #[test]
    fn fparity003_pass_round_trip() {
        let mappings = [
            ("qwen2", "Qwen2", "Qwen2"),
            ("qwen3", "Qwen3", "Qwen3"),
            ("llama", "Llama", "Llama"),
        ];
        let v = verdict_from_round_trip(&mappings);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fparity003_fail_one_mismatch() {
        let mappings = [
            ("qwen2", "Qwen2", "Qwen2"),
            ("llama", "Llama", "Auto"), // wrong dispatch
        ];
        let v = verdict_from_round_trip(&mappings);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: PARITY-004 + 005.
    // -----------------------------------------------------------------
    #[test]
    fn fparity004_pass_all_named() {
        let v = verdict_from_display_name_nonempty(&[
            ("Qwen2", "Qwen2.5-Coder"),
            ("Llama", "Llama 3"),
        ]);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fparity004_fail_one_empty_name() {
        let v = verdict_from_display_name_nonempty(&[("Qwen2", ""), ("Llama", "Llama 3")]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    #[test]
    fn fparity005_pass_classification_correct() {
        let v = verdict_from_is_llm_classification(&[
            ("Qwen2", true, true),
            ("Whisper", false, false),
            ("BERT", false, false),
        ]);
        assert_eq!(v, ProbeParityVerdict::Pass);
    }

    #[test]
    fn fparity005_fail_audio_classified_as_llm() {
        let v = verdict_from_is_llm_classification(&[
            ("Whisper", false, true), // bug — Whisper claimed as LLM
        ]);
        assert_eq!(v, ProbeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_003_param_formula() {
        for &(k, d) in &[(2_usize, 768), (10, 4096), (1, 64), (256, 8192)] {
            let expected = k * d + k;
            let v = verdict_from_trainable_param_count(k, d, expected);
            assert_eq!(v, ProbeParityVerdict::Pass, "k={k} d={d}");
            // Off-by-one trips the gate
            let v_off = verdict_from_trainable_param_count(k, d, expected + 1);
            assert_eq!(v_off, ProbeParityVerdict::Fail, "k={k} d={d}");
        }
    }

    #[test]
    fn mutation_survey_002_softmax_under_two_class_band() {
        // Sweep 2-class probabilities over [0.0, 1.0] in 0.1 steps.
        for p_x10 in 0_u32..=10 {
            let p = p_x10 as f32 / 10.0;
            let probs = [p, 1.0 - p];
            let v = verdict_from_softmax_valid(&probs);
            // Pass only when both > 0
            let want = if p > 0.0 && p < 1.0 {
                ProbeParityVerdict::Pass
            } else {
                ProbeParityVerdict::Fail
            };
            assert_eq!(v, want, "p={p}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_9() {
        let v1 = verdict_from_encoder_frozen(&[0xAA; 64], &[0xAA; 64]);
        let v2 = verdict_from_softmax_valid(&[0.1, 0.2, 0.7]);
        let v3 = verdict_from_trainable_param_count(2, 768, 1538);
        let v4 = verdict_from_embedding_determinism(&[1.0, 2.0], &[1.0, 2.0]);
        let v5 = verdict_from_enum_yaml_symmetry(&["Qwen2", "Llama"], &["Llama", "Qwen2"]);
        let v6 = verdict_from_round_trip(&[("qwen2", "Qwen2", "Qwen2")]);
        let v7 = verdict_from_display_name_nonempty(&[("Qwen2", "Qwen2.5-Coder")]);
        let v8 = verdict_from_is_llm_classification(&[("Qwen2", true, true)]);
        // We have 9 gates total; v9 covers PARITY-005 dual-classification.
        let v9 = verdict_from_is_llm_classification(&[("BERT", false, false)]);
        assert_eq!(v1, ProbeParityVerdict::Pass);
        assert_eq!(v2, ProbeParityVerdict::Pass);
        assert_eq!(v3, ProbeParityVerdict::Pass);
        assert_eq!(v4, ProbeParityVerdict::Pass);
        assert_eq!(v5, ProbeParityVerdict::Pass);
        assert_eq!(v6, ProbeParityVerdict::Pass);
        assert_eq!(v7, ProbeParityVerdict::Pass);
        assert_eq!(v8, ProbeParityVerdict::Pass);
        assert_eq!(v9, ProbeParityVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_9_failures() {
        // Regression class — every gate trips simultaneously.
        let mut after = vec![0xAA_u8; 64];
        after[7] = 0xCD;
        let v1 = verdict_from_encoder_frozen(&[0xAA; 64], &after);
        let v2 = verdict_from_softmax_valid(&[0.5, 0.0]); // underflow
        let v3 = verdict_from_trainable_param_count(2, 768, 1539);
        let bumped = f32::from_bits(2.0_f32.to_bits() + 1);
        let v4 = verdict_from_embedding_determinism(&[1.0, 2.0], &[1.0, bumped]);
        let v5 = verdict_from_enum_yaml_symmetry(&["Qwen2"], &["Qwen2", "Llama"]);
        let v6 = verdict_from_round_trip(&[("qwen2", "Qwen2", "Auto")]);
        let v7 = verdict_from_display_name_nonempty(&[("X", "")]);
        let v8 = verdict_from_is_llm_classification(&[("Whisper", false, true)]);
        let v9 = verdict_from_is_llm_classification(&[("Qwen2", true, false)]);
        assert_eq!(v1, ProbeParityVerdict::Fail);
        assert_eq!(v2, ProbeParityVerdict::Fail);
        assert_eq!(v3, ProbeParityVerdict::Fail);
        assert_eq!(v4, ProbeParityVerdict::Fail);
        assert_eq!(v5, ProbeParityVerdict::Fail);
        assert_eq!(v6, ProbeParityVerdict::Fail);
        assert_eq!(v7, ProbeParityVerdict::Fail);
        assert_eq!(v8, ProbeParityVerdict::Fail);
        assert_eq!(v9, ProbeParityVerdict::Fail);
    }
}
