
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_accept_ratio_exactly_half() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // We need ratio = 0.5 exactly
        // target_prob / draft_prob = 0.5
        // exp(target_log) / exp(draft_log) = 0.5
        // exp(target_log - draft_log) = 0.5
        // target_log - draft_log = ln(0.5) ≈ -0.693
        // If draft_log = -1.0, target_log = -1.693
        let draft_prob = TokenProb::new(5, -1.0);
        let target_prob = TokenProb::new(10, -1.0 + (-0.5_f32).ln());

        // ratio = 0.5, condition is ratio > 0.5, so should NOT accept
        // But ratio >= 1.0 OR ratio > 0.5, so 0.5 exactly should NOT match > 0.5
        assert!(!decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_ratio_below_half() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // ratio needs to be < 0.5
        // target_prob / draft_prob < 0.5
        // If draft is high and target is low
        let draft_prob = TokenProb::new(5, 0.0); // prob = 1.0
        let target_prob = TokenProb::new(10, -2.0); // prob ≈ 0.135

        // ratio = 0.135 / 1.0 = 0.135 < 0.5, should NOT accept
        assert!(!decoder.should_accept(&draft_prob, &target_prob));
    }

    // === Error Handling Edge Cases ===

    #[test]
    fn test_speculative_error_verification_failed_fields() {
        let err = SpeculativeError::VerificationFailed { position: 42 };
        if let SpeculativeError::VerificationFailed { position } = err {
            assert_eq!(position, 42);
        } else {
            panic!("Expected VerificationFailed variant");
        }
    }

    #[test]
    fn test_speculative_error_draft_model_error_message() {
        let err = SpeculativeError::DraftModelError("custom error message".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Draft model error"));
        assert!(display.contains("custom error message"));
    }

    #[test]
    fn test_speculative_error_target_model_error_message() {
        let err = SpeculativeError::TargetModelError("target failed".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Target model error"));
        assert!(display.contains("target failed"));
    }

    #[test]
    fn test_speculative_error_invalid_spec_length_message() {
        let err = SpeculativeError::InvalidSpecLength(100);
        let display = format!("{}", err);
        assert!(display.contains("Invalid speculation length"));
        assert!(display.contains("100"));
    }

    // === Decoder with Rejection Scenarios ===

    /// Model where draft and target always disagree with very low target probability
    struct AlwaysRejectModel {
        token: u32,
        log_prob: f32,
    }

    impl SpeculativeModel for AlwaysRejectModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; 100])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(self.token, self.log_prob))
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_immediate_rejection() {
        // Draft has token 5 with high prob, target has token 10 with very low prob
        let draft = AlwaysRejectModel {
            token: 5,
            log_prob: 0.0, // prob = 1.0
        };
        let target = AlwaysRejectModel {
            token: 10,
            log_prob: -10.0, // prob ≈ 0.00005
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[1]).expect("decode");

        // First token should be rejected and resampled
        assert!(result.resampled_token.is_some());
        // Only 1 token accepted (the resampled one)
        assert_eq!(result.num_accepted, 1);
    }

    // === Stats Accumulation Verification ===

    #[test]
    fn test_speculative_stats_time_accumulation() {
        let mut stats = SpeculativeStats::default();

        stats.record_iteration(4, 3, 1.5, 15.0);
        assert!((stats.draft_time_ms - 1.5).abs() < 0.001);
        assert!((stats.target_time_ms - 15.0).abs() < 0.001);

        stats.record_iteration(4, 4, 2.0, 20.0);
        assert!((stats.draft_time_ms - 3.5).abs() < 0.001);
        assert!((stats.target_time_ms - 35.0).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_avg_spec_length_calculation() {
        let mut stats = SpeculativeStats::default();

        stats.record_iteration(2, 2, 1.0, 10.0);
        assert!((stats.avg_spec_length - 2.0).abs() < 0.001);

        stats.record_iteration(6, 4, 1.0, 10.0);
        // Total speculated = 8, iterations = 2, avg = 4.0
        assert!((stats.avg_spec_length - 4.0).abs() < 0.001);

        stats.record_iteration(4, 3, 1.0, 10.0);
        // Total speculated = 12, iterations = 3, avg = 4.0
        assert!((stats.avg_spec_length - 4.0).abs() < 0.001);
    }

    // === Config with Custom max_spec_length ===

    #[test]
    fn test_speculative_config_custom_max_spec_length() {
        let mut config = SpeculativeConfig {
            spec_length: 10,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 12,
        };

        config.adapt_spec_length(0.95);
        assert_eq!(config.spec_length, 11);

        config.adapt_spec_length(0.95);
        assert_eq!(config.spec_length, 12);

        config.adapt_spec_length(0.95);
        assert_eq!(config.spec_length, 12); // Capped at max
    }

    // === Deserialization Tests ===

    #[test]
    fn test_speculative_stats_deserialize_from_json() {
        let json = r#"{
            "iterations": 5,
            "tokens_speculated": 20,
            "tokens_accepted": 15,
            "acceptance_rate": 0.75,
            "avg_spec_length": 4.0,
            "time_saved_ms": 50.0,
            "draft_time_ms": 5.0,
            "target_time_ms": 50.0
        }"#;

        let stats: SpeculativeStats = serde_json::from_str(json).expect("deserialize");
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.tokens_speculated, 20);
        assert_eq!(stats.tokens_accepted, 15);
        assert!((stats.acceptance_rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_speculative_config_deserialize_from_json() {
        let json = r#"{
            "spec_length": 8,
            "min_acceptance_rate": 0.6,
            "adaptive": false,
            "max_spec_length": 16
        }"#;

        let config: SpeculativeConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(config.spec_length, 8);
        assert!((config.min_acceptance_rate - 0.6).abs() < 0.001);
        assert!(!config.adaptive);
        assert_eq!(config.max_spec_length, 16);
    }

    // === Decoder Stats Integration ===

    #[test]
    fn test_speculative_decoder_stats_after_multiple_iterations() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 3).expect("create");

        for _ in 0..10 {
            let _ = decoder.decode_iteration(&[1, 2, 3]).expect("decode");
        }

        let stats = decoder.stats();
        assert_eq!(stats.iterations, 10);
        assert!(stats.tokens_speculated >= 10);
        assert!(stats.draft_time_ms >= 0.0);
        assert!(stats.target_time_ms >= 0.0);
    }

    #[test]
    fn test_speculative_decoder_reset_clears_all_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Run some iterations
        for _ in 0..5 {
            let _ = decoder.decode_iteration(&[1]).expect("decode");
        }

        // Reset
        decoder.reset_stats();

        let stats = decoder.stats();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.tokens_speculated, 0);
        assert_eq!(stats.tokens_accepted, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
        assert_eq!(stats.avg_spec_length, 0.0);
        assert_eq!(stats.time_saved_ms, 0.0);
        assert_eq!(stats.draft_time_ms, 0.0);
        assert_eq!(stats.target_time_ms, 0.0);
    }
include!("speculative_mock_tests.rs");
include!("speculative_config.rs");
include!("speculative_tests.rs");
}
