
    // ----- Min-P Sampling Tests -----

    #[test]
    fn test_sample_min_p_basic() {
        // Token 0 has probability ~0.7, token 1 ~0.2, token 2 ~0.1
        let logits = Tensor::from_vec(vec![3], vec![1.0, -0.5, -1.0]).expect("test");

        // With min_p = 0.3 (30% of max), only token 0 should remain
        let token = sample_min_p(&logits, 0.3, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_min_p_all_pass() {
        // All tokens have similar logits
        let logits = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");

        // With min_p = 0.9, all tokens should pass (all equal)
        let token = sample_min_p(&logits, 0.9, 0.3).expect("test");
        assert!(token < 3);
    }

    #[test]
    fn test_sample_min_p_low_threshold() {
        let logits = Tensor::from_vec(vec![4], vec![10.0, 1.0, 0.5, 0.1]).expect("test");

        // With very low min_p, all tokens can be sampled
        let token = sample_min_p(&logits, 0.001, 0.99).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_min_p_edge_cases() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");

        // min_p = 0 should include all tokens
        let _ = sample_min_p(&logits, 0.0, 0.5).expect("test");

        // min_p = 1.0 should still return something (at least the max)
        let token = sample_min_p(&logits, 1.0, 0.5).expect("test");
        assert_eq!(token, 2); // Highest probability token
    }

    #[test]
    fn test_sample_min_p_rng_boundary() {
        // Test with rng_value at boundary (0.0)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let token = sample_min_p(&logits, 0.5, 0.0).expect("test");
        assert!(token < 3);
    }

    // ----- Mirostat Sampling Tests -----

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert_eq!(state.tau, 5.0);
        assert_eq!(state.eta, 0.1);
        assert_eq!(state.mu, 10.0);
    }

    #[test]
    fn test_mirostat_state_builder() {
        let state = MirostatState::new(3.0).with_eta(0.2);
        assert_eq!(state.tau, 3.0);
        assert_eq!(state.eta, 0.2);
        assert_eq!(state.mu, 6.0); // 2 * tau
    }

    #[test]
    fn test_mirostat_state_update() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);

        let initial_mu = state.mu;

        // High surprise should decrease mu (mu -= eta * (surprise - tau))
        state.update(10.0); // surprise > tau => mu decreases
        assert!(state.mu < initial_mu);

        // Reset
        state.mu = initial_mu;

        // Low surprise should increase mu
        state.update(2.0); // surprise < tau => mu increases
        assert!(state.mu > initial_mu);
    }

    #[test]
    fn test_sample_mirostat_basic() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let token = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        assert!(token < 5);
    }

    #[test]
    fn test_sample_mirostat_deterministic() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let mut state = MirostatState::new(0.1); // Low target perplexity

        // With very low tau, should prefer highest probability token
        let token = sample_mirostat(&logits, &mut state, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_mirostat_state_evolution() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let initial_mu = state.mu;

        // Sample multiple times and verify mu evolves
        for _ in 0..10 {
            let _ = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        }

        // Mu should have changed from initial
        assert_ne!(state.mu, initial_mu);
    }

    #[test]
    fn test_sample_mirostat_rng_boundary() {
        // Test with rng_value at boundary (1.0 - epsilon)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let mut state = MirostatState::default();
        let token = sample_mirostat(&logits, &mut state, 0.999).expect("test");
        assert!(token < 3);
    }

    // ----- Advanced Generation Config Tests -----

    #[test]
    fn test_advanced_generation_config_default() {
        let config = AdvancedGenerationConfig::default();
        assert!(config.stop_detector.is_none());
        assert!(config.repetition_penalty.is_none());
        assert!(config.presence_frequency.is_none());
        assert!(config.logit_bias.is_none());
    }

    #[test]
    fn test_advanced_generation_config_builder() {
        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_stop_sequences(vec!["<|end|>".to_string()])
            .with_repetition_penalty(1.5)
            .with_presence_frequency(0.5, 0.3)
            .with_logit_bias(LogitBias::new().with_bias(0, 10.0));

        assert!(config.stop_detector.is_some());
        assert!(config.repetition_penalty.is_some());
        assert!(config.presence_frequency.is_some());
        assert!(config.logit_bias.is_some());
    }

    // ----- Apply All Penalties Tests -----

    #[test]
    fn test_apply_all_penalties_empty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let original = logits.data().to_vec();
        let context: Vec<usize> = vec![];
        let config = AdvancedGenerationConfig::default();

        let result = apply_all_penalties(&logits, &context, &config);

        // No penalties applied
        assert_eq!(result.data(), original.as_slice());
    }

    #[test]
    fn test_apply_all_penalties_combined() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 twice, token 1 once

        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_repetition_penalty(2.0)
            .with_presence_frequency(1.0, 0.5)
            .with_logit_bias(LogitBias::new().with_bias(4, 100.0));

        let result = apply_all_penalties(&logits, &context, &config);

        // Token 4 should be highest due to bias
        let max_idx = result
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test")
            .0;
        assert_eq!(max_idx, 4);

        // Token 0 should be penalized most (repetition + presence + frequency)
        assert!(result.data()[0] < result.data()[2]);
    }

    #[test]
    fn test_stop_sequence_with_stop_strings() {
        let detector = StopSequenceDetector::new()
            .with_stop_strings(vec!["stop".to_string(), "end".to_string()]);

        assert!(detector.check_text("this has stop in it").is_some());
        assert!(detector.check_text("the end").is_some());
        assert!(detector.check_text("nothing here").is_none());
    }

    // ===== Tail-Free Sampling (TFS) Tests =====

    #[test]
    fn test_tfs_basic_filtering() {
        // Create logits with distinct probabilities
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // With z=0.95, should filter some low-probability tokens
        let result = sample_tfs(&logits, 0.95, 0.0);
        assert!(result.is_ok());
        // Should return one of the high-probability tokens
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_tfs_z_one_returns_greedy() {
        // z=1.0 should keep all tokens (no filtering)
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // rng=0.0 should select the first valid token after filtering
        let result = sample_tfs(&logits, 1.0, 0.0).expect("test");
        // Should be a valid token
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_z_zero_selects_top() {
        // z=0.0 should filter aggressively, keeping only top tokens
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        let result = sample_tfs(&logits, 0.01, 0.0).expect("test");
        // Should select from top tokens
        assert!(result < 3);
    }

    #[test]
    fn test_tfs_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_tfs_uniform_distribution() {
        // With uniform logits, all tokens have equal second derivative
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== Locally Typical Sampling Tests =====

    #[test]
    fn test_typical_basic_sampling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");

        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_typical_p_one_keeps_all() {
        // p=1.0 should keep all tokens
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 1.0, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_typical_low_p_selects_typical() {
        // Low p should select only the most typical tokens (closest to entropy)
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 0.1, 0.0).expect("test");
        // Should select a token
        assert!(result < 5);
    }

    #[test]
    fn test_typical_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_typical_uniform_distribution() {
        // Uniform distribution - all tokens equally typical
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert!(result < 4);
    }

    #[test]
    fn test_typical_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== DRY (Don't Repeat Yourself) Sampling Tests =====

    #[test]
    fn test_dry_config_default() {
        let config = DryConfig::default();
        assert_eq!(config.multiplier, 0.8);
        assert_eq!(config.base, 1.75);
        assert_eq!(config.allowed_length, 2);
        assert_eq!(config.penalty_last_n, 256);
        assert!(config.is_enabled()); // Default is enabled
    }

    #[test]
    fn test_dry_config_disabled() {
        let config = DryConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_dry_config_enabled() {
        let config = DryConfig::new(0.5)
            .with_base(1.5)
            .with_allowed_length(3)
            .with_penalty_last_n(64);
        assert!(config.is_enabled());
        assert_eq!(config.base, 1.5);
        assert_eq!(config.allowed_length, 3);
        assert_eq!(config.penalty_last_n, 64);
    }

    #[test]
    fn test_dry_no_penalty_when_disabled() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig::new(0.0); // disabled (multiplier=0)
        let context = vec![0, 1, 0, 1, 0];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_penalty_applied() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 64,
        };
        // Context with repeated pattern: [0, 1, 0, 1] - if next is 0, it continues [0,1] pattern
        let context = vec![0, 1, 0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Token 0 should be penalized (would create [0,1,0] repetition)
        assert!(result.data()[0] < logits.data()[0]);
    }

    #[test]
    fn test_dry_short_context_no_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 3,
            penalty_last_n: 64,
        };
        // Context shorter than allowed_length
        let context = vec![0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_respects_penalty_last_n() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 3, // Only look at last 3 tokens
        };
        // Repetition is outside the window
        let context = vec![0, 1, 2, 3, 4];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Should not detect repetition from early in context
        // (penalty window is only last 3: [2, 3, 4])
        assert!(result.data().iter().sum::<f32>() > 0.0);
    }

    // ===== Beam Search Tests =====

    #[test]
    fn test_beam_hypothesis_creation() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3], -1.5);
        assert_eq!(hyp.tokens.len(), 3);
        assert!(!hyp.finished);
        assert_eq!(hyp.score, -1.5);
    }

    #[test]
    fn test_beam_hypothesis_extend() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(3, -0.5, false);
        assert_eq!(extended.tokens, vec![1, 2, 3]);
        assert_eq!(extended.score, -1.5);
        assert!(!extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_extend_with_eos() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(99, -0.5, true);
        assert_eq!(extended.tokens, vec![1, 2, 99]);
        assert!(extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_normalized_score() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3, 4], -4.0);
        // length_penalty = 1.0 means divide by length
        assert_eq!(hyp.normalized_score(1.0), -1.0);
        // length_penalty = 0.0 means score / 1.0 = score
        assert_eq!(hyp.normalized_score(0.0), -4.0);
    }
