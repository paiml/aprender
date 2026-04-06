
    #[test]
    fn test_beam_search_config_default() {
        let config = BeamSearchConfig::default();
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 1.0);
        assert!(config.early_stopping); // Default is true
        assert_eq!(config.num_return, 1);
    }

    #[test]
    fn test_beam_search_config_new() {
        let config = BeamSearchConfig::new(8);
        assert_eq!(config.num_beams, 8);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_beam_search_config_builder() {
        let config = BeamSearchConfig::new(4)
            .with_length_penalty(0.8)
            .with_early_stopping(false)
            .with_num_return(2);
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 0.8);
        assert!(!config.early_stopping);
        assert_eq!(config.num_return, 2);
    }

    #[test]
    fn test_beam_search_state_creation() {
        let config = BeamSearchConfig::new(3)
            .with_length_penalty(0.8)
            .with_num_return(2);
        let state = BeamSearchState::new(config, vec![1, 2, 3]);
        assert_eq!(state.hypotheses.len(), 1); // Starts with one hypothesis
        assert!(state.finished.is_empty());
        assert_eq!(state.hypotheses[0].tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_beam_search_state_step() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Create log probabilities for 1 hypothesis, 5 tokens
        let log_probs = vec![vec![-0.1, -0.5, -1.0, -2.0, -3.0]];

        state.step(&log_probs, Some(4)); // EOS token is 4

        // Should have expanded to num_beams hypotheses
        assert!(!state.hypotheses.is_empty());
    }

    #[test]
    fn test_beam_search_state_with_finished() {
        let config = BeamSearchConfig::new(2);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Manually add some hypotheses
        state.hypotheses = vec![
            BeamHypothesis::new(vec![1, 2], -1.0),
            BeamHypothesis::new(vec![1, 3], -2.0),
        ];
        state.finished = vec![BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.5,
            finished: true,
        }];

        assert_eq!(state.hypotheses.len(), 2);
        assert_eq!(state.finished.len(), 1);
    }

    #[test]
    fn test_beam_search_state_should_stop_empty() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses.clear();

        // Empty hypotheses means should stop
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_should_stop_early() {
        let config = BeamSearchConfig::new(2).with_early_stopping(true);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Not done initially
        assert!(!state.should_stop());

        // Add num_beams finished hypotheses
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.0,
            finished: true,
        });
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 3, 4],
            score: -1.5,
            finished: true,
        });

        // Should be done with early_stopping=true and num_beams finished
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_all_finished() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses = vec![
            BeamHypothesis {
                tokens: vec![1],
                score: -1.0,
                finished: true,
            },
            BeamHypothesis {
                tokens: vec![2],
                score: -2.0,
                finished: true,
            },
        ];

        // All hypotheses finished
        assert!(state.should_stop());
    }

    // ===== Streaming Generation Tests =====

    #[test]
    fn test_streaming_generator_creation() {
        let generator = StreamingGenerator::new();
        assert!(generator.tokens.is_empty());
        assert!(generator.text.is_empty());
        assert!(!generator.finished);
        assert_eq!(generator.total_tokens, 0);
    }

    #[test]
    fn test_streaming_generator_default() {
        let generator = StreamingGenerator::default();
        assert!(generator.tokens.is_empty());
        assert!(!generator.finished);
    }

    #[test]
    fn test_streaming_generator_add_token() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, Some("hello"));
        assert_eq!(generator.tokens, vec![1, 2]);
        assert_eq!(generator.text, "hello");
        assert_eq!(generator.total_tokens, 2);
    }

    #[test]
    fn test_streaming_generator_add_token_with_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(0, Some("Hello "));
        generator.add_token(1, Some("world"));
        generator.add_token(2, Some("!"));
        assert_eq!(generator.text, "Hello world!");
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_token_count() {
        let mut generator = StreamingGenerator::new();
        assert_eq!(generator.token_count(), 0);
        generator.add_token(1, None);
        assert_eq!(generator.token_count(), 1);
        generator.add_token(2, None);
        generator.add_token(3, None);
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_finish() {
        let mut generator = StreamingGenerator::new();
        assert!(!generator.finished);
        generator.add_token(1, Some("test"));
        generator.finish();
        assert!(generator.finished);
    }

    #[test]
    fn test_streaming_generator_accumulates_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, Some("The "));
        generator.add_token(2, Some("quick "));
        generator.add_token(3, Some("brown "));
        generator.add_token(4, Some("fox"));
        assert_eq!(generator.text, "The quick brown fox");
    }

    #[test]
    fn test_streaming_generator_none_text_no_accumulation() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, None);
        assert!(generator.text.is_empty());
        assert_eq!(generator.tokens, vec![1, 2]);
    }

    // ===== XTC (Exclude Top Choices) Sampling Tests =====

    #[test]
    fn test_xtc_config_default() {
        let config = XtcConfig::default();
        assert_eq!(config.probability, 0.0);
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.min_keep, 1);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_xtc_config_enabled() {
        let config = XtcConfig::new(0.5).with_threshold(0.3).with_min_keep(2);
        assert!(config.is_enabled());
        assert_eq!(config.probability, 0.5);
        assert_eq!(config.threshold, 0.3);
        assert_eq!(config.min_keep, 2);
    }

    #[test]
    fn test_xtc_disabled_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::default(); // disabled
        let result = apply_xtc(&logits, &config, 0.5);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_rng_above_probability_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(0.5); // 50% probability
        let result = apply_xtc(&logits, &config, 0.8); // rng > probability
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_excludes_top_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.5); // Always exclude, high threshold
        let result = apply_xtc(&logits, &config, 0.0); // rng < probability
        // Top token (index 0) should be excluded (set to NEG_INFINITY)
        assert_eq!(result.data()[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_xtc_respects_min_keep() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 9.0, 8.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(2);
        let result = apply_xtc(&logits, &config, 0.0);
        // Should keep at least 2 tokens (not set all to NEG_INFINITY)
        let finite_count = result.data().iter().filter(|&&x| x.is_finite()).count();
        assert!(finite_count >= 2);
    }

    // ===== Eta Sampling Tests =====

    #[test]
    fn test_eta_config_default() {
        let config = EtaConfig::default();
        assert_eq!(config.eta, 0.3);
        assert_eq!(config.min_p, 0.0001);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_eta_config_disabled() {
        let config = EtaConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_eta_config_builder() {
        let config = EtaConfig::new(0.5).with_min_p(0.001);
        assert_eq!(config.eta, 0.5);
        assert_eq!(config.min_p, 0.001);
    }

    #[test]
    fn test_eta_sampling_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_eta_sampling_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_eta_sampling_uniform() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert!(result < 4);
    }

    // ===== Token Healing Tests =====

    #[test]
    fn test_token_healing_config_default() {
        let config = TokenHealingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_backup_chars, 0);
    }

    #[test]
    fn test_token_healing_config_enabled() {
        let config = TokenHealingConfig::new(true).with_max_backup(15);
        assert!(config.enabled);
        assert_eq!(config.max_backup_chars, 15);
    }

    #[test]
    fn test_token_healing_no_heal_needed() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("hello"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
        assert_eq!(result.tokens_removed, 0);
    }

    #[test]
    fn test_token_healing_partial_word() {
        let tokens = vec![1, 2, 3, 4, 5];
        // "wo" is a short alphanumeric token without leading space - should heal
        let result = analyze_token_healing(&tokens, Some("wo"));
        assert_eq!(result.adjusted_tokens, vec![1, 2, 3, 4]);
        assert_eq!(result.prefix_constraint, Some("wo".to_string()));
        assert_eq!(result.tokens_removed, 1);
    }

    #[test]
    fn test_token_healing_empty_tokens() {
        let tokens: Vec<usize> = vec![];
        let result = analyze_token_healing(&tokens, Some("a"));
        assert!(result.adjusted_tokens.is_empty());
        assert!(result.prefix_constraint.is_none());
    }

    #[test]
    fn test_token_healing_space_prefix_no_heal() {
        let tokens = vec![1, 2, 3];
        // Token starting with space - no healing needed
        let result = analyze_token_healing(&tokens, Some(" word"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
    }

    // ===== Classifier-Free Guidance (CFG) Tests =====

    #[test]
    fn test_cfg_config_default() {
        let config = CfgConfig::default();
        assert_eq!(config.scale, 1.0);
        assert!(config.negative_prompt_tokens.is_empty());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_cfg_config_enabled() {
        let config = CfgConfig::new(1.5).with_negative_prompt(vec![1, 2, 3]);
        assert!(config.is_enabled());
        assert_eq!(config.scale, 1.5);
        assert_eq!(config.negative_prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_cfg_scale_one_no_change() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.0).expect("test");
        // scale=1.0: uncond + 1.0 * (cond - uncond) = cond
        assert_eq!(result.data(), cond.data());
    }

    #[test]
    fn test_cfg_scale_zero_returns_uncond() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 0.0).expect("test");
        // scale=0.0: uncond + 0.0 * (cond - uncond) = uncond
        assert_eq!(result.data(), uncond.data());
    }

    #[test]
    fn test_cfg_amplifies_difference() {
        let cond = Tensor::from_vec(vec![3], vec![2.0, 1.0, 0.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).expect("test");
        let result = apply_cfg(&cond, &uncond, 2.0).expect("test");
        // scale=2.0: uncond + 2.0 * (cond - uncond)
        // = [1,1,1] + 2*([2,1,0] - [1,1,1])
        // = [1,1,1] + 2*[1,0,-1]
        // = [1,1,1] + [2,0,-2]
        // = [3,1,-1]
        assert_eq!(result.data(), &[3.0, 1.0, -1.0]);
    }

    #[test]
    fn test_cfg_shape_mismatch_error() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![0.5, 1.5, 2.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.5);
        assert!(result.is_err());
    }

    // ===== Prompt Cache Tests =====

    #[test]
    fn test_prompt_cache_creation() {
        let cache = PromptCache::new(50);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_prompt_cache_default() {
        let cache = PromptCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_add_and_find() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        assert_eq!(cache.len(), 1);

        // Find exact match
        let result = cache.find_prefix(&[1, 2, 3]);
        assert!(result.is_some());
        let (len, kv_hash) = result.expect("test");
        assert_eq!(len, 3);
        assert_eq!(kv_hash, 12345);
    }
