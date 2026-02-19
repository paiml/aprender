//! APR Format Validation Tests - Extreme TDD
//! PMAT-197: Extracted from validation.rs for file size reduction

pub(crate) use super::*;

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

#[path = "validation_tests_part_02.rs"]
mod validation_tests_part_02;
#[path = "validation_tests_part_03.rs"]
mod validation_tests_part_03;
#[path = "validation_tests_part_04.rs"]
mod validation_tests_part_04;
