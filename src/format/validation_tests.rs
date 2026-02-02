//! APR Format Validation Tests - Extreme TDD
//! PMAT-197: Extracted from validation.rs for file size reduction

use super::*;

#[cfg(test)]
mod tests_poka_yoke {
    use super::*;

    #[test]
    fn test_gate_pass() {
        let gate = Gate::pass("test", 10);
        assert!(gate.passed);
        assert_eq!(gate.points, 10);
        assert!(gate.error.is_none());
    }

    #[test]
    fn test_gate_fail() {
        let gate = Gate::fail("test", 10, "Fix: do something");
        assert!(!gate.passed);
        assert_eq!(gate.points, 0);
        assert!(gate.error.is_some());
        assert!(gate.error.unwrap().contains("Fix:"));
    }

    #[test]
    fn test_result_score_calculation() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("a", 50));
        result.add_gate(Gate::fail("b", 50, "error"));
        assert_eq!(result.score, 50);
        assert_eq!(result.grade(), "F");
    }

    #[test]
    fn test_result_all_pass() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("a", 50));
        result.add_gate(Gate::pass("b", 50));
        assert_eq!(result.score, 100);
        assert_eq!(result.grade(), "A+");
        assert!(result.passed());
    }

    #[test]
    fn test_result_error_summary() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::fail("gate1", 50, "Fix: action1"));
        result.add_gate(Gate::fail("gate2", 50, "Fix: action2"));
        let summary = result.error_summary();
        assert!(summary.contains("gate1"));
        assert!(summary.contains("action1"));
        assert!(summary.contains("gate2"));
    }

    #[test]
    fn test_grade_boundaries() {
        let grades = [
            (100, "A+"),
            (95, "A+"),
            (94, "A"),
            (90, "A"),
            (89, "B+"),
            (85, "B+"),
            (84, "B"),
            (80, "B"),
            (79, "C+"),
            (75, "C+"),
            (74, "C"),
            (70, "C"),
            (69, "D"),
            (60, "D"),
            (59, "F"),
            (0, "F"),
        ];
        for (score, expected_grade) in grades {
            let mut result = PokaYokeResult::new();
            // Hack to set score directly for testing
            result.score = score;
            assert_eq!(result.grade(), expected_grade, "score {score}");
        }
    }

    #[test]
    fn test_from_gates_bulk_construction() {
        let gates = vec![
            Gate::pass("check_a", 30),
            Gate::pass("check_b", 40),
            Gate::fail("check_c", 30, "Fix: implement check_c"),
        ];
        let result = PokaYokeResult::from_gates(gates);
        assert_eq!(result.score, 70); // 70/100
        assert_eq!(result.max_score, 100);
        assert_eq!(result.grade(), "C");
        assert!(result.passed());
        assert_eq!(result.gates.len(), 3);
    }

    #[test]
    fn test_from_gates_empty() {
        let result = PokaYokeResult::from_gates(vec![]);
        assert_eq!(result.score, 0);
        assert_eq!(result.max_score, 0);
        assert_eq!(result.grade(), "F");
    }

    #[test]
    fn test_fail_no_validation_rules() {
        let result = fail_no_validation_rules();
        assert_eq!(result.score, 0);
        assert_eq!(result.grade(), "F");
        assert!(!result.passed());
        assert_eq!(result.gates.len(), 1);
        assert_eq!(result.gates[0].name, "no_validation_rules");
        assert!(result.gates[0]
            .error
            .as_ref()
            .unwrap()
            .contains("Implement PokaYoke"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_no_validation_result_deprecated() {
        // Test the deprecated alias
        let result = no_validation_result();
        assert_eq!(result.score, 0);
        assert_eq!(result.grade(), "F");
        assert!(!result.passed());
    }

    #[test]
    fn test_poka_yoke_result_empty_error_summary() {
        // Test error_summary when all gates pass
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("gate1", 50));
        result.add_gate(Gate::pass("gate2", 50));
        let summary = result.error_summary();
        assert!(summary.is_empty(), "Should be empty when no failed gates");
    }

    #[test]
    fn test_poka_yoke_result_failed_gates() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("pass1", 30));
        result.add_gate(Gate::fail("fail1", 40, "Error 1"));
        result.add_gate(Gate::fail("fail2", 30, "Error 2"));
        let failed = result.failed_gates();
        assert_eq!(failed.len(), 2);
    }

    // Test PokaYoke trait default implementation
    struct MockModel {
        score: u8,
    }

    impl PokaYoke for MockModel {
        fn poka_yoke_validate(&self) -> PokaYokeResult {
            let mut result = PokaYokeResult::new();
            result.add_gate(Gate::pass("mock_check", self.score));
            result
        }
    }

    #[test]
    fn test_poka_yoke_trait_quality_score() {
        let model = MockModel { score: 80 };
        // Test the default quality_score() method
        let score = model.quality_score();
        assert_eq!(score, 100); // 80/80 = 100%
    }

    #[test]
    fn test_gate_debug() {
        let gate = Gate::pass("test", 10);
        let debug = format!("{:?}", gate);
        assert!(debug.contains("Gate"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_gate_clone() {
        let gate = Gate::fail("test", 20, "error msg");
        let cloned = gate.clone();
        assert_eq!(gate.name, cloned.name);
        assert_eq!(gate.passed, cloned.passed);
        assert_eq!(gate.points, cloned.points);
    }

    #[test]
    fn test_poka_yoke_result_clone() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("test", 50));
        let cloned = result.clone();
        assert_eq!(result.score, cloned.score);
        assert_eq!(result.gates.len(), cloned.gates.len());
    }

    #[test]
    fn test_whisper_validation_debug() {
        let wv = WhisperValidation;
        let debug = format!("{:?}", wv);
        assert!(debug.contains("WhisperValidation"));
    }

    #[test]
    fn test_whisper_validation_clone() {
        let wv = WhisperValidation;
        let _cloned = wv.clone();
    }

    #[test]
    fn test_whisper_validation_default() {
        let _wv = WhisperValidation::default();
    }
}

// ============================================================================
// Whisper Validation Tests (APR-POKA-001, D11, D12)
// ============================================================================

#[cfg(test)]
mod tests_whisper_validation {
    use super::*;

    // D11: Filterbank must be embedded
    #[test]
    fn test_filterbank_present_pass() {
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(gate.is_some());
        assert!(
            gate.unwrap().passed,
            "Filterbank should be detected as present"
        );
    }

    #[test]
    fn test_filterbank_missing_fail() {
        let result = WhisperValidation::validate_filterbank(None);
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(gate.is_some());
        assert!(!gate.unwrap().passed, "Missing filterbank should fail");
        assert!(gate
            .unwrap()
            .error
            .as_ref()
            .unwrap()
            .contains("MelFilterbankData"));
    }

    #[test]
    fn test_filterbank_empty_fail() {
        let fb: Vec<f32> = vec![];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(!gate.unwrap().passed, "Empty filterbank should fail");
    }

    // D12: Filterbank must be Slaney-normalized (max < 0.1)
    #[test]
    fn test_filterbank_normalized_pass() {
        // Slaney-normalized filterbank has max < 0.1
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(
            gate.unwrap().passed,
            "Slaney-normalized filterbank should pass"
        );
    }

    #[test]
    fn test_filterbank_not_normalized_fail() {
        // Non-normalized filterbank has max >= 0.1
        let mut fb: Vec<f32> = vec![0.05; 80 * 201];
        fb[0] = 1.0; // Bug: unnormalized value
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(!gate.unwrap().passed, "Unnormalized filterbank should fail");
        assert!(gate.unwrap().error.as_ref().unwrap().contains("Slaney"));
    }

    #[test]
    fn test_filterbank_boundary_value() {
        // Exactly 0.1 should fail (must be < 0.1)
        let fb: Vec<f32> = vec![0.1; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(
            !gate.unwrap().passed,
            "max=0.1 exactly should fail (need < 0.1)"
        );
    }

    #[test]
    fn test_filterbank_full_validation_score() {
        // Valid filterbank: 100 points
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        assert_eq!(result.score, 100);
        assert_eq!(result.grade(), "A+");
        assert!(result.passed());
    }

    #[test]
    fn test_filterbank_missing_score() {
        // Missing filterbank: 0 points
        let result = WhisperValidation::validate_filterbank(None);
        assert_eq!(result.score, 0);
        assert_eq!(result.grade(), "F");
        assert!(!result.passed());
    }

    // Tensor validation tests
    #[test]
    fn test_tensor_stats_all_valid() {
        let stats = vec![
            TensorStats::compute("encoder.layer_norm.weight", &vec![1.0f32; 384]),
            TensorStats::compute("decoder.fc1.weight", &vec![0.01f32, -0.01, 0.02, -0.02]),
        ];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        assert!(result.passed());
        assert!(result.score >= 80);
    }

    #[test]
    fn test_tensor_stats_nan_detected() {
        let stats = vec![TensorStats::compute("broken", &[1.0f32, f32::NAN, 3.0])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_nan_values");
        assert!(!gate.unwrap().passed, "NaN should be detected");
    }

    #[test]
    fn test_tensor_stats_inf_detected() {
        let stats = vec![TensorStats::compute("broken", &[1.0f32, f32::INFINITY])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_inf_values");
        assert!(!gate.unwrap().passed, "Inf should be detected");
    }

    #[test]
    fn test_tensor_stats_invalid_layernorm() {
        // LayerNorm weight with mean=11.0 (10x too high - the bug we're catching)
        let stats = vec![TensorStats::compute(
            "encoder.layer_norm.weight",
            &vec![11.0f32; 384],
        )];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "layernorm_weights_valid");
        assert!(!gate.unwrap().passed, "Invalid LayerNorm mean should fail");
    }

    #[test]
    fn test_tensor_stats_all_zeros() {
        let stats = vec![TensorStats::compute("dead_weight", &vec![0.0f32; 100])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_zero_tensors");
        assert!(!gate.unwrap().passed, "All-zero tensor should fail");
    }

    // Full validation tests
    #[test]
    fn test_full_validation_all_pass() {
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let stats = vec![
            TensorStats::compute("encoder.layer_norm.weight", &vec![1.0f32; 384]),
            TensorStats::compute("decoder.fc1.weight", &vec![0.01f32; 100]),
        ];
        let result = WhisperValidation::validate_full(Some(&fb), &stats);
        assert!(result.passed());
        assert!(result.score >= 90, "Full valid model should score >= 90");
    }

    #[test]
    fn test_full_validation_missing_filterbank() {
        let stats = vec![TensorStats::compute(
            "encoder.layer_norm.weight",
            &vec![1.0f32; 384],
        )];
        let result = WhisperValidation::validate_full(None, &stats);
        assert!(
            result.score < 60,
            "Missing filterbank should significantly reduce score"
        );
    }

    #[test]
    fn test_actionable_error_messages() {
        let result = WhisperValidation::validate_filterbank(None);
        let summary = result.error_summary();
        assert!(
            summary.contains("Fix:"),
            "Error should be actionable with Fix:"
        );
        assert!(
            summary.contains("MelFilterbankData"),
            "Error should provide solution"
        );
    }
}

// ============================================================================
// SECTION A: Format & Structural Integrity (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_a {
    use super::*;

    // Test 1: Magic bytes valid
    #[test]
    fn test_check_1_magic_valid_aprn() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_pass(), "APR\\0 magic should pass");
    }

    #[test]
    fn test_check_1_magic_valid_apr_unified() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "APR\\0 magic should pass (unified format)"
        );
    }

    #[test]
    fn test_check_1_magic_invalid() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"BAD!");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_fail(), "Invalid magic should fail");
    }

    // Test 2: Header size fixed
    #[test]
    fn test_check_2_header_complete() {
        let data = vec![0u8; 32];
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_pass(), "32-byte header should pass");
    }

    #[test]
    fn test_check_2_header_too_small() {
        let data = vec![0u8; 16]; // Only 16 bytes
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_fail(), "16-byte header should fail");
    }

    // Test 3: Version supported
    #[test]
    fn test_check_3_version_1_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 1.0 should be supported");
    }

    #[test]
    fn test_check_3_version_2_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 2; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 2.0 should be supported");
    }

    #[test]
    fn test_check_3_version_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 3; // major (unsupported)
        data[5] = 0;
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail(), "Version 3.0 should fail");
    }

    // Test 11: Flags parsed
    #[test]
    fn test_check_11_known_flags_pass() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1;
        data[8] = 0x01; // COMPRESSED flag
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 11)
            .unwrap();
        assert!(check.status.is_pass(), "Known flags should pass");
    }

    // GH-178: GGUF format support tests
    #[test]
    fn test_check_1_gguf_magic_valid() {
        // GGUF magic: [71, 71, 85, 70] = "GGUF"
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4] = 3; // GGUF version 3
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF magic [71, 71, 85, 70] should pass"
        );
    }

    #[test]
    fn test_check_3_gguf_version_3_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&3u32.to_le_bytes()); // Version 3
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 3 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_2_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&2u32.to_le_bytes()); // Version 2
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 2 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_1_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&1u32.to_le_bytes()); // Version 1
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 1 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_0_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // Version 0 (invalid)
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail(), "GH-178: GGUF version 0 should fail");
    }

    #[test]
    fn test_check_3_gguf_version_4_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&4u32.to_le_bytes()); // Version 4 (future)
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_fail(),
            "GH-178: GGUF version 4 should fail (future version)"
        );
    }
}

// ============================================================================
// SECTION B: Tensor Physics & Statistics (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_b {
    use super::*;

    // Test 26: No NaNs
    #[test]
    fn test_check_26_no_nan_pass() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_nan(), "Clean data should have no NaN");
    }

    #[test]
    fn test_check_26_nan_detected() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_nan(), "Should detect NaN");
        assert_eq!(stats.nan_count, 1);
    }

    // Test 27: No Infs
    #[test]
    fn test_check_27_no_inf_pass() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_inf(), "Clean data should have no Inf");
    }

    #[test]
    fn test_check_27_inf_detected() {
        let data = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_inf(), "Should detect Inf");
        assert_eq!(stats.inf_count, 2);
    }

    // Test 28: LayerNorm Mean in [0.5, 3.0]
    #[test]
    fn test_check_28_layernorm_mean_valid() {
        // Mean should be ~1.0 for LayerNorm weights
        let data = vec![1.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            stats.is_valid_layernorm_weight(),
            "Mean of 1.0 should be valid"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_high() {
        // Bug case: mean=11.0 (10x too high)
        let data = vec![11.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 11.0 should FAIL - this is the bug we're catching"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_low() {
        let data = vec![0.1f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 0.1 should fail"
        );
    }

    // Test 29: LayerNorm Bias in [-0.5, 0.5]
    #[test]
    fn test_check_29_layernorm_bias_valid() {
        let data = vec![0.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.bias", &data);
        assert!(
            stats.is_valid_layernorm_bias(),
            "Mean of 0.0 should be valid"
        );
    }

    #[test]
    fn test_check_29_layernorm_bias_invalid() {
        let data = vec![5.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.bias", &data);
        assert!(!stats.is_valid_layernorm_bias(), "Mean of 5.0 should fail");
    }

    // Test 31: Zero Tensors
    #[test]
    fn test_check_31_not_all_zeros_pass() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.is_not_all_zeros(), "Should pass with some non-zero");
    }

    #[test]
    fn test_check_31_all_zeros_fail() {
        let data = vec![0.0f32; 100];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.is_not_all_zeros(), "All zeros should fail");
    }

    // Test 35: Attention/Linear Mean ~0
    #[test]
    fn test_check_35_linear_weight_valid() {
        let data = vec![0.01f32, -0.02, 0.03, -0.01];
        let stats = TensorStats::compute("encoder.layers.0.self_attn.q_proj.weight", &data);
        assert!(stats.is_valid_linear_weight(), "Mean ~0 should be valid");
    }

    #[test]
    fn test_check_35_linear_weight_invalid() {
        let data = vec![1.0f32; 100];
        let stats = TensorStats::compute("encoder.layers.0.fc1.weight", &data);
        assert!(!stats.is_valid_linear_weight(), "Mean of 1.0 should fail");
    }

    // Test statistics computation
    #[test]
    fn test_stats_compute_mean() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::compute("test", &data);
        assert!((stats.mean - 3.0).abs() < 0.001, "Mean should be 3.0");
    }

    #[test]
    fn test_stats_compute_std() {
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::compute("test", &data);
        // Mean = 5.0, Variance = 32/7 ≈ 4.57, Std ≈ 2.14
        assert!(
            (stats.std - 2.14).abs() < 0.1,
            "Std should be ~2.14, got {}",
            stats.std
        );
    }

    #[test]
    fn test_stats_empty_data() {
        let data: Vec<f32> = vec![];
        let stats = TensorStats::compute("empty", &data);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }
}

// ============================================================================
// SECTION C: Tooling & Operations (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_c {
    // Test 56: Diff Identity
    #[test]
    fn test_check_56_diff_identity() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 3.0];
        let diff = compute_l2_distance(&data1, &data2);
        assert!(diff < 1e-6, "Same data should have zero L2 distance");
    }

    // Test 57: Diff Detection
    #[test]
    fn test_check_57_diff_detection() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 4.0]; // Changed last element
        let diff = compute_l2_distance(&data1, &data2);
        assert!(
            diff > 0.5,
            "Different data should have non-zero L2 distance"
        );
    }

    // Test 58: Merge Average
    #[test]
    fn test_check_58_merge_average() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![3.0f32, 4.0, 5.0];
        let merged = merge_average(&data1, &data2);
        assert_eq!(merged, vec![2.0f32, 3.0, 4.0], "Average merge failed");
    }

    /// Compute L2 distance between two tensors
    fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }

    /// Merge two tensors by averaging
    fn merge_average(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect()
    }
}

// ============================================================================
// SECTION D: Conversion & Interoperability (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_d {
    // Test 79: Roundtrip
    #[test]
    fn test_check_79_roundtrip_tolerance() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        // Simulate roundtrip with small float error
        let roundtrip: Vec<f32> = original.iter().map(|&x| x + 1e-7).collect();
        let max_diff = compute_max_diff(&original, &roundtrip);
        assert!(max_diff < 1e-5, "Roundtrip should have drift < 1e-5");
    }

    // Test 87: Tensor Name Normalization
    #[test]
    fn test_check_87_name_normalization() {
        let hf_name = "model.encoder.conv1.weight";
        let apr_name = normalize_tensor_name(hf_name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should strip 'model.' prefix"
        );
    }

    #[test]
    fn test_check_87_name_normalization_no_prefix() {
        let name = "encoder.conv1.weight";
        let apr_name = normalize_tensor_name(name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should preserve name without prefix"
        );
    }

    /// Compute max absolute difference between tensors
    fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |acc, x| if x > acc { x } else { acc })
    }

    /// Normalize tensor name to APR canonical form
    fn normalize_tensor_name(name: &str) -> &str {
        name.strip_prefix("model.").unwrap_or(name)
    }
}

// ============================================================================
// ValidationReport Tests
// ============================================================================

#[cfg(test)]
mod tests_report {
    use super::*;

    #[test]
    fn test_report_grade_a_plus() {
        let mut report = ValidationReport::new();
        for i in 1..=95 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(report.grade(), "A+");
        assert_eq!(report.total_score, 95);
    }

    #[test]
    fn test_report_grade_f() {
        let mut report = ValidationReport::new();
        for i in 1..=50 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(report.grade(), "F");
        assert_eq!(report.total_score, 50);
    }

    #[test]
    fn test_report_passed_threshold() {
        let mut report = ValidationReport::new();
        for i in 1..=90 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert!(report.passed(90));
        assert!(!report.passed(95));
    }

    #[test]
    fn test_report_failed_checks() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "pass",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "fail",
            category: Category::Structure,
            status: CheckStatus::Fail("reason".to_string()),
            points: 0,
        });

        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, 2);
    }

    #[test]
    fn test_category_scores() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "struct1",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 26,
            name: "physics1",
            category: Category::Physics,
            status: CheckStatus::Pass,
            points: 1,
        });

        assert_eq!(report.category_scores.get(&Category::Structure), Some(&1));
        assert_eq!(report.category_scores.get(&Category::Physics), Some(&1));
    }

    // ====================================================================
    // Coverage: CheckStatus methods
    // ====================================================================

    #[test]
    fn test_check_status_is_pass() {
        assert!(CheckStatus::Pass.is_pass());
        assert!(!CheckStatus::Pass.is_fail());
    }

    #[test]
    fn test_check_status_is_fail() {
        let fail = CheckStatus::Fail("bad".to_string());
        assert!(fail.is_fail());
        assert!(!fail.is_pass());
    }

    #[test]
    fn test_check_status_skip_not_pass_not_fail() {
        let skip = CheckStatus::Skip("n/a".to_string());
        assert!(!skip.is_pass());
        assert!(!skip.is_fail());
    }

    // ====================================================================
    // Coverage: Category methods
    // ====================================================================

    #[test]
    fn test_category_letter() {
        assert_eq!(Category::Structure.letter(), 'A');
        assert_eq!(Category::Physics.letter(), 'B');
        assert_eq!(Category::Tooling.letter(), 'C');
        assert_eq!(Category::Conversion.letter(), 'D');
    }

    #[test]
    fn test_category_name() {
        assert_eq!(Category::Structure.name(), "Format & Structural Integrity");
        assert_eq!(Category::Physics.name(), "Tensor Physics & Statistics");
        assert_eq!(Category::Tooling.name(), "Tooling & Operations");
        assert_eq!(Category::Conversion.name(), "Conversion & Interoperability");
    }

    // ====================================================================
    // Coverage: AprHeader flag methods
    // ====================================================================

    #[test]
    fn test_apr_header_is_compressed() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x01, // compressed bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(header.is_compressed());
        assert!(!header.is_signed());
        assert!(!header.is_encrypted());
    }

    #[test]
    fn test_apr_header_is_signed() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x20, // signed bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_compressed());
        assert!(header.is_signed());
        assert!(!header.is_encrypted());
    }

    #[test]
    fn test_apr_header_is_encrypted() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x10, // encrypted bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_compressed());
        assert!(!header.is_signed());
        assert!(header.is_encrypted());
    }

    #[test]
    fn test_apr_header_supported_versions() {
        // v1.0, v1.1, v1.2 supported
        for minor in 0..=2 {
            let h = AprHeader {
                magic: [0x41, 0x50, 0x52, 0x00],
                version_major: 1,
                version_minor: minor,
                flags: 0,
                metadata_offset: 0,
                metadata_size: 0,
                index_offset: 0,
                index_size: 0,
                data_offset: 0,
            };
            assert!(h.is_supported_version(), "v1.{minor} should be supported");
        }
        // v2.0 supported
        let h = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(h.is_supported_version());
        // v3.0 not supported
        let h = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 3,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!h.is_supported_version());
    }

    // ====================================================================
    // Coverage: ValidationReport::failed_checks
    // ====================================================================

    #[test]
    fn test_report_failed_checks_mixed() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "pass_check",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "fail_check",
            category: Category::Structure,
            status: CheckStatus::Fail("bad".to_string()),
            points: 0,
        });
        report.add_check(ValidationCheck {
            id: 3,
            name: "skip_check",
            category: Category::Physics,
            status: CheckStatus::Skip("n/a".to_string()),
            points: 0,
        });
        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, 2);
    }

    // ====================================================================
    // Coverage: CheckStatus::Warn variant
    // ====================================================================

    #[test]
    fn test_check_status_warn() {
        let warn = CheckStatus::Warn("warning message".to_string());
        assert!(!warn.is_pass());
        assert!(!warn.is_fail());
    }

    // ====================================================================
    // Coverage: AprHeader::is_valid_magic
    // ====================================================================

    #[test]
    fn test_apr_header_is_valid_magic_true() {
        let header = AprHeader {
            magic: *b"APR\0",
            version_major: 1,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(header.is_valid_magic());
    }

    #[test]
    fn test_apr_header_is_valid_magic_false() {
        let header = AprHeader {
            magic: *b"GGUF",
            version_major: 1,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_valid_magic());
    }

    // ====================================================================
    // Coverage: AprHeader::parse error path
    // ====================================================================

    #[test]
    fn test_apr_header_parse_too_small() {
        let data = vec![0u8; 16]; // Too small
        let result = AprHeader::parse(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("Header too small"));
    }

    // ====================================================================
    // Coverage: ValidationCheck with Warn status
    // ====================================================================

    #[test]
    fn test_validation_check_with_warn_status() {
        let check = ValidationCheck {
            id: 11,
            name: "unknown_flags",
            category: Category::Structure,
            status: CheckStatus::Warn("Unknown flag bits".to_string()),
            points: 0,
        };
        assert!(!check.status.is_pass());
        assert!(!check.status.is_fail());
    }

    // ====================================================================
    // Coverage: AprValidator::validate (tensor validation path)
    // ====================================================================

    #[test]
    fn test_apr_validator_validate_tensors() {
        let mut validator = AprValidator::new();
        // Add some tensor stats
        validator.add_tensor_stats(TensorStats::compute("test.weight", &vec![1.0f32; 100]));
        let report = validator.validate();
        // Should have run tensor validation
        assert!(!report.checks.is_empty());
    }

    // ====================================================================
    // Coverage: AprValidator Default trait
    // ====================================================================

    #[test]
    fn test_apr_validator_default() {
        let validator = AprValidator::default();
        assert!(validator.report().checks.is_empty());
    }

    // ====================================================================
    // Coverage: ValidationReport Default trait
    // ====================================================================

    #[test]
    fn test_validation_report_default() {
        let report = ValidationReport::default();
        assert!(report.checks.is_empty());
        assert_eq!(report.total_score, 0);
    }

    // ====================================================================
    // Coverage: File too small for magic bytes
    // ====================================================================

    #[test]
    fn test_check_magic_file_too_small() {
        let data = vec![0u8; 2]; // Only 2 bytes
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_fail());
    }

    // ====================================================================
    // Coverage: GGUF file too small for version check
    // ====================================================================

    #[test]
    fn test_gguf_version_file_too_small() {
        // GGUF magic but not enough bytes for version
        let mut data = vec![0u8; 6]; // Less than 8 bytes
        data[0..4].copy_from_slice(b"GGUF");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail());
    }

    // ====================================================================
    // Coverage: Unknown flags warning path (check 11)
    // ====================================================================

    #[test]
    fn test_check_11_unknown_flags_warn() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1; // version major
        // Set unknown flag bits (beyond bit 7)
        data[9] = 0x01; // This sets bit 8 which is unknown
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 11)
            .unwrap();
        // Should be a warning for unknown flags
        assert!(matches!(check.status, CheckStatus::Warn(_)));
    }

    // ====================================================================
    // Coverage: TensorStats with only NaN/Inf values
    // ====================================================================

    #[test]
    fn test_tensor_stats_all_nan() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let stats = TensorStats::compute("nan_tensor", &data);
        assert_eq!(stats.nan_count, 3);
        assert_eq!(stats.mean, 0.0); // No valid values, mean defaults to 0
        assert_eq!(stats.std, 0.0);  // No valid values, std defaults to 0
    }

    #[test]
    fn test_tensor_stats_all_inf() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let stats = TensorStats::compute("inf_tensor", &data);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0); // min/max default when all inf
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_tensor_stats_single_value() {
        let data = vec![42.0f32];
        let stats = TensorStats::compute("single", &data);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.std, 0.0); // std with single value is 0
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }
}
