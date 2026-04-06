
    #[test]
    fn test_speculative_config_adapt_medium_rate_no_change() {
        let mut config = SpeculativeConfig::new().with_spec_length(4);
        // 0.6 is above min_acceptance_rate (0.5) but below 0.8
        config.adapt_spec_length(0.6);
        // Should not change
        assert_eq!(config.spec_length, 4);
    }

    #[test]
    fn test_speculative_config_serialization() {
        let config = SpeculativeConfig {
            spec_length: 6,
            min_acceptance_rate: 0.4,
            adaptive: false,
            max_spec_length: 12,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: SpeculativeConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.spec_length, 6);
        assert_eq!(parsed.max_spec_length, 12);
        assert!(!parsed.adaptive);
    }

    #[test]
    fn test_speculative_config_clone() {
        let config = SpeculativeConfig::new()
            .with_spec_length(5)
            .with_adaptive(false);
        let cloned = config.clone();
        assert_eq!(config.spec_length, cloned.spec_length);
        assert_eq!(config.adaptive, cloned.adaptive);
    }

    #[test]
    fn test_speculative_config_debug() {
        let config = SpeculativeConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("spec_length"));
        assert!(debug_str.contains("adaptive"));
    }

    // === SpeculativeDecoder Extended Tests ===

    #[test]
    fn test_speculative_decoder_spec_length_too_large() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let result = SpeculativeDecoder::new(draft, target, 33);
        assert!(matches!(
            result,
            Err(SpeculativeError::InvalidSpecLength(33))
        ));
    }

    #[test]
    fn test_speculative_decoder_set_spec_length_too_large() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.set_spec_length(33);
        assert!(matches!(
            result,
            Err(SpeculativeError::InvalidSpecLength(33))
        ));
        // Original value unchanged
        assert_eq!(decoder.spec_length(), 4);
    }

    #[test]
    fn test_speculative_decoder_boundary_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        // Exactly 32 should work
        let decoder = SpeculativeDecoder::new(draft, target, 32);
        assert!(decoder.is_ok());
        assert_eq!(decoder.expect("decoder").spec_length(), 32);
    }

    #[test]
    fn test_speculative_decoder_boundary_spec_length_one() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        // Exactly 1 should work
        let decoder = SpeculativeDecoder::new(draft, target, 1);
        assert!(decoder.is_ok());
        assert_eq!(decoder.expect("decoder").spec_length(), 1);
    }

    /// Mock model that returns EOS token
    struct EosModel {
        vocab_size: usize,
        eos_token: u32,
    }

    impl EosModel {
        fn new(vocab_size: usize, eos_token: u32) -> Self {
            Self {
                vocab_size,
                eos_token,
            }
        }
    }

    impl SpeculativeModel for EosModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; self.vocab_size])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            // Always return EOS
            Ok(TokenProb::new(self.eos_token, -0.5))
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn eos_token(&self) -> u32 {
            self.eos_token
        }
    }

    #[test]
    fn test_speculative_decoder_eos_stops_draft() {
        let draft = EosModel::new(100, 0); // Returns EOS immediately
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[10]).expect("decode");

        // Should stop after first token (EOS)
        assert_eq!(result.num_speculated, 1);
    }

    /// Mock model that fails on forward
    struct FailingForwardModel;

    impl SpeculativeModel for FailingForwardModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Err(SpeculativeError::DraftModelError(
                "forward failed".to_string(),
            ))
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(1, -1.0))
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_draft_forward_error() {
        let draft = FailingForwardModel;
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[10]);

        assert!(matches!(result, Err(SpeculativeError::DraftModelError(_))));
    }

    /// Mock model that fails on sample
    struct FailingSampleModel;

    impl SpeculativeModel for FailingSampleModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; 100])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Err(SpeculativeError::DraftModelError(
                "sample failed".to_string(),
            ))
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_draft_sample_error() {
        let draft = FailingSampleModel;
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[10]);

        assert!(matches!(result, Err(SpeculativeError::DraftModelError(_))));
    }

    /// Mock model that fails target forward
    struct FailingTargetModel;

    impl SpeculativeModel for FailingTargetModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Err(SpeculativeError::TargetModelError(
                "target forward failed".to_string(),
            ))
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(1, -1.0))
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_target_forward_error() {
        let draft = MockModel::new(100, 1);
        let target = FailingTargetModel;

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[10]);

        assert!(matches!(result, Err(SpeculativeError::TargetModelError(_))));
    }

    /// Mock model that fails on target sample (after successful forward)
    struct FailingTargetSampleModel {
        sample_fail: bool,
    }

    impl SpeculativeModel for FailingTargetSampleModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; 100])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            if self.sample_fail {
                Err(SpeculativeError::TargetModelError(
                    "target sample failed".to_string(),
                ))
            } else {
                Ok(TokenProb::new(1, -1.0))
            }
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_target_sample_error() {
        let draft = MockModel::new(100, 1);
        let target = FailingTargetSampleModel { sample_fail: true };

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[10]);

        assert!(matches!(result, Err(SpeculativeError::TargetModelError(_))));
    }

    /// Mock model that returns different tokens (for rejection testing)
    struct DifferentTokenModel {
        token: u32,
        prob: f32,
    }

    impl DifferentTokenModel {
        fn new(token: u32, prob: f32) -> Self {
            Self { token, prob }
        }
    }

    impl SpeculativeModel for DifferentTokenModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            Ok(vec![0.0; 100])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(self.token, self.prob))
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn eos_token(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_speculative_decoder_rejection_resamples() {
        // Draft returns token 5 with low probability
        let draft = DifferentTokenModel::new(5, -10.0); // Very low prob
        // Target returns token 10 with high probability
        let target = DifferentTokenModel::new(10, -0.1); // High prob

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[1]).expect("decode");

        // Should have resampled token since tokens differ and ratio check may fail
        // The first token triggers the should_accept check
        assert!(!result.accepted_tokens.is_empty());
    }

    #[test]
    fn test_speculative_decoder_multiple_iterations_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 2).expect("create");

        // Run multiple iterations
        for _ in 0..5 {
            let _ = decoder.decode_iteration(&[1, 2, 3]).expect("decode");
        }

        let stats = decoder.stats();
        assert_eq!(stats.iterations, 5);
        assert!(stats.tokens_speculated >= 5);
    }

    #[test]
    fn test_speculative_decoder_empty_context() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");
        let result = decoder.decode_iteration(&[]).expect("decode");

        // Should still work with empty context
        assert!(result.num_speculated > 0);
    }

    // === should_accept Edge Cases ===

    #[test]
    fn test_should_accept_same_token() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        let draft_prob = TokenProb::new(42, -1.0);
        let target_prob = TokenProb::new(42, -2.0); // Same token, different prob

        // Same tokens should always be accepted
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_high_target_prob() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Different tokens, but target has much higher probability
        let draft_prob = TokenProb::new(5, -10.0); // Very low
        let target_prob = TokenProb::new(10, -0.1); // High

        // Ratio = exp(-0.1) / exp(-10.0) ≈ 0.9 / 0.00005 >> 1.0
        // Should accept because ratio >= 1.0
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_similar_probs() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Different tokens with similar probabilities
        let draft_prob = TokenProb::new(5, -1.0);
        let target_prob = TokenProb::new(10, -1.0);

        // ratio = 1.0, should accept (ratio >= 1.0 is true)
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_moderate_ratio() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Different tokens with ratio between 0.5 and 1.0
        let draft_prob = TokenProb::new(5, -1.0); // prob ≈ 0.368
        let target_prob = TokenProb::new(10, -1.2); // prob ≈ 0.301

        // ratio = 0.301 / 0.368 ≈ 0.82, > 0.5, should accept
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_very_low_target_prob() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Target has much lower probability
        let draft_prob = TokenProb::new(5, -0.1); // High prob ≈ 0.9
        let target_prob = TokenProb::new(10, -5.0); // Low prob ≈ 0.0067

        // ratio = 0.0067 / 0.9 ≈ 0.007, < 0.5, should reject
        assert!(!decoder.should_accept(&draft_prob, &target_prob));
    }

    #[test]
    fn test_should_accept_draft_near_zero() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);
        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("create");

        // Draft prob very close to zero (edge case for max(1e-10))
        let draft_prob = TokenProb::new(5, -100.0); // Extremely low
        let target_prob = TokenProb::new(10, -1.0);

        // ratio = 0.368 / max(tiny, 1e-10) = huge, should accept
        assert!(decoder.should_accept(&draft_prob, &target_prob));
    }
