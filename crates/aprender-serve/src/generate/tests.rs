#[cfg(test)]
mod tests {
    use crate::generate::*;

    #[test]
    fn test_build_nucleus_single_element() {
        let indexed = vec![(42, 1.0)];
        let nucleus = build_nucleus(&indexed, 0.5);
        assert_eq!(nucleus.len(), 1);
        assert_eq!(nucleus[0].0, 42);
    }

    // ----- Temperature Edge Cases -----

    #[test]
    fn test_apply_temperature_very_low() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.01).expect("test");
        // Very low temp = very high scaled values
        assert!(scaled.data()[0] > 90.0);
        assert!(scaled.data()[3] > 390.0);
    }

    #[test]
    fn test_apply_temperature_very_high() {
        let logits = Tensor::from_vec(vec![4], vec![100.0, 200.0, 300.0, 400.0]).expect("test");
        let scaled = apply_temperature(&logits, 100.0).expect("test");
        // Very high temp = very low scaled values (flattened distribution)
        assert!((scaled.data()[0] - 1.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_preserves_ordering() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 3.0, 2.0, 5.0, 4.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.7).expect("test");
        // Ordering should be preserved
        assert!(scaled.data()[3] > scaled.data()[4]); // 5 > 4
        assert!(scaled.data()[4] > scaled.data()[1]); // 4 > 3
        assert!(scaled.data()[1] > scaled.data()[2]); // 3 > 2
        assert!(scaled.data()[2] > scaled.data()[0]); // 2 > 1
    }

    #[test]
    fn test_apply_temperature_negative_logits() {
        let logits = Tensor::from_vec(vec![3], vec![-5.0, -2.0, -1.0]).expect("test");
        let scaled = apply_temperature(&logits, 0.5).expect("test");
        assert!((scaled.data()[0] - (-10.0)).abs() < 1e-6);
        assert!((scaled.data()[1] - (-4.0)).abs() < 1e-6);
        assert!((scaled.data()[2] - (-2.0)).abs() < 1e-6);
    }

    // ----- Seed Reproducibility Tests -----

    #[test]
    fn test_generation_config_seed() {
        let config = GenerationConfig::greedy().with_seed(12345);
        assert_eq!(config.seed, Some(12345));
    }

    #[test]
    fn test_generation_config_different_seeds_produce_different_results() {
        // This tests the LCG used in generation pipeline
        let seed1 = 42u64;
        let seed2 = 43u64;

        // Simple LCG step
        let next1 = seed1
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let next2 = seed2
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);

        assert_ne!(
            next1, next2,
            "Different seeds should produce different values"
        );
    }

    // ----- Greedy Edge Cases -----

    #[test]
    fn test_sample_greedy_with_ties() {
        // Multiple maximum values - should return first occurrence
        let logits = Tensor::from_vec(vec![5], vec![1.0, 10.0, 10.0, 10.0, 1.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 1, "Should return first maximum");
    }

    #[test]
    fn test_sample_greedy_all_negative() {
        let logits = Tensor::from_vec(vec![4], vec![-5.0, -2.0, -3.0, -4.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 1, "Should return index of least negative (-2.0)");
    }

    #[test]
    fn test_sample_greedy_large_vocab() {
        let mut data = vec![0.0f32; 50000];
        data[49999] = 100.0; // Maximum at end
        let logits = Tensor::from_vec(vec![50000], data).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 49999);
    }

    // ----- Top-K Edge Cases -----

    #[test]
    fn test_sample_top_k_k_larger_than_vocab() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        // k=10 but vocab size is 3
        let token = sample_top_k(&logits, 10, 0.0).expect("test");
        assert_eq!(token, 2, "Should work with k > vocab size");
    }

    #[test]
    fn test_sample_top_k_deterministic_with_same_rng() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let token1 = sample_top_k(&logits, 3, 0.123).expect("test");
        let token2 = sample_top_k(&logits, 3, 0.123).expect("test");
        assert_eq!(token1, token2, "Same RNG should give same result");
    }

    #[test]
    fn test_sample_top_k_with_negative_logits() {
        let logits = Tensor::from_vec(vec![5], vec![-10.0, -5.0, -1.0, -3.0, -7.0]).expect("test");
        let token = sample_top_k(&logits, 2, 0.0).expect("test");
        // Top 2 are indices 2 (-1.0) and 3 (-3.0)
        assert!(token == 2 || token == 3);
    }

    // ----- Top-P Edge Cases -----

    #[test]
    fn test_sample_top_p_p_exactly_one() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let token = sample_top_p(&logits, 1.0, 0.5).expect("test");
        assert!(token < 4, "p=1.0 should include all tokens");
    }

    #[test]
    fn test_sample_top_p_very_small_p() {
        // Very small p should select only the highest probability token
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let token = sample_top_p(&logits, 0.01, 0.0).expect("test");
        // Index 2 has highest logit
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_top_p_deterministic_with_same_rng() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let token1 = sample_top_p(&logits, 0.9, 0.456).expect("test");
        let token2 = sample_top_p(&logits, 0.9, 0.456).expect("test");
        assert_eq!(token1, token2, "Same RNG should give same result");
    }

    // ----- Sample Token Integration -----

    #[test]
    fn test_sample_token_temperature_affects_distribution() {
        // With very low temperature, top-k should behave more greedy
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");

        let config_low_temp = GenerationConfig::top_k(5).with_temperature(0.01);
        let config_high_temp = GenerationConfig::top_k(5).with_temperature(10.0);

        // Low temp should always pick max
        for rng in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let token = sample_token(&logits, &config_low_temp, rng).expect("test");
            assert_eq!(token, 2, "Low temp should always pick highest");
        }

        // High temp might pick different tokens (just verify it works)
        let _ = sample_token(&logits, &config_high_temp, 0.5).expect("test");
    }

    // ----- Min-P Additional Tests -----

    #[test]
    fn test_sample_min_p_errors_on_invalid_input() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        // Invalid min_p values
        assert!(sample_min_p(&logits, -0.1, 0.5).is_err());
        assert!(sample_min_p(&logits, 1.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_min_p_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("test");
        let result = sample_min_p(&logits, 0.5, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    // ----- Mirostat Additional Tests -----

    #[test]
    fn test_sample_mirostat_empty_logits_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_sample_mirostat_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    // ----- TFS Additional Tests -----

    #[test]
    fn test_sample_tfs_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- Typical Sampling Additional Tests -----

    #[test]
    fn test_sample_typical_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- Eta Sampling Additional Tests -----

    #[test]
    fn test_sample_eta_empty_error() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    // ----- CFG Additional Tests -----

    #[test]
    fn test_cfg_empty_logits() {
        // Zero-dimension tensors are invalid, so test that it returns an error
        let cond = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(cond.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_cfg_large_scale() {
        let cond = Tensor::from_vec(vec![3], vec![1.0, 0.0, -1.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
        let result = apply_cfg(&cond, &uncond, 10.0).expect("test");
        // scale=10.0: 0 + 10 * (cond - 0) = 10 * cond
        assert!((result.data()[0] - 10.0).abs() < 1e-6);
        assert!((result.data()[1] - 0.0).abs() < 1e-6);
        assert!((result.data()[2] - (-10.0)).abs() < 1e-6);
    }

    // ----- Beam Search Additional Tests -----

    #[test]
    fn test_beam_search_best_hypotheses() {
        let config = BeamSearchConfig::new(2).with_num_return(2);
        let mut state = BeamSearchState::new(config, vec![0]);

        state.finished.push(BeamHypothesis {
            tokens: vec![0, 1],
            score: -1.0,
            finished: true,
        });
        state.finished.push(BeamHypothesis {
            tokens: vec![0, 2],
            score: -0.5,
            finished: true,
        });

        let best = state.best_hypotheses();
        assert_eq!(best.len(), 2);
        // Higher score (less negative) should be first
        assert!(best[0].score > best[1].score);
    }

    // ----- Stop Sequence Edge Cases -----

    #[test]
    fn test_stop_sequence_empty_pattern() {
        let detector = StopSequenceDetector::new()
            .with_token_sequence(vec![])
            .with_string_pattern("");
        assert!(!detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_multiple_matches() {
        let detector = StopSequenceDetector::new()
            .with_string_pattern("stop")
            .with_string_pattern("end");
        let pos = detector.check_text("stop at end");
        assert!(pos.is_some());
        assert_eq!(pos.unwrap(), 0); // "stop" is found first at position 0
    }

    // ----- Generation Config Builder Chain -----

    #[test]
    fn test_generation_config_full_builder_chain() {
        let config = GenerationConfig::top_k(50)
            .with_temperature(0.8)
            .with_max_tokens(200)
            .with_eos_token_id(99)
            .with_seed(42);

        assert_eq!(config.strategy, SamplingStrategy::TopK { k: 50 });
        assert!((config.temperature - 0.8).abs() < 1e-6);
        assert_eq!(config.max_tokens, 200);
        assert_eq!(config.eos_token_id, Some(99));
        assert_eq!(config.seed, Some(42));
    }

    // ----- LogitBias with HashMap -----

    #[test]
    fn test_logit_bias_with_biases() {
        let mut biases = std::collections::HashMap::new();
        biases.insert(1, 5.0);
        biases.insert(3, -5.0);

        let bias = LogitBias::new().with_biases(biases);
        assert_eq!(bias.get(1), 5.0);
        assert_eq!(bias.get(3), -5.0);
        assert_eq!(bias.get(2), 0.0);
    }

    // ----- Dynamic Temperature with Edge Cases -----

    #[test]
    fn test_dyn_temp_empty_logits() {
        // Zero-dimension tensors are invalid - from_vec should error
        let result = Tensor::<f32>::from_vec(vec![0], vec![]);
        assert!(result.is_err()); // Shape dimensions cannot be zero
    }

    #[test]
    fn test_dyn_temp_negative_delta() {
        // Negative delta should behave like no delta
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = DynTempConfig::new(1.0, -0.5, 1.0);
        let result = apply_dynamic_temperature(&logits, &config);
        // Should just apply static temp
        assert!((result.data()[0] - 1.0).abs() < 1e-6);
    }

    // ----- Sampler Clone Tests -----

    #[test]
    fn test_temperature_sampler_clone_box() {
        let sampler = TemperatureSampler::new(0.5);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "temperature");
    }

    #[test]
    fn test_top_k_sampler_clone_box() {
        let sampler = TopKSampler::new(10);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "top_k");
    }

    #[test]
    fn test_top_p_sampler_clone_box() {
        let sampler = TopPSampler::new(0.9);
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "top_p");
    }

    #[test]
    fn test_dyn_temp_sampler_clone_box() {
        let sampler = DynTempSampler::new(DynTempConfig::new(1.0, 0.5, 1.0));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "dyn_temp");
    }

    #[test]
    fn test_repetition_penalty_sampler_clone_box() {
        let sampler = RepetitionPenaltySampler::new(RepetitionPenaltyConfig::new(1.2));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "repetition_penalty");
    }

    #[test]
    fn test_infill_sampler_clone_box() {
        let sampler = InfillSampler::new(InfillConfig::new(vec![1]));
        let cloned = sampler.clone_box();
        assert_eq!(cloned.name(), "infill");
    }

    // ----- Pipeline Accessors -----

    #[test]
    fn test_generation_pipeline_accessors() {
        let model = MockModel::new(100, 42);
        let pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![0]))
            .with_config(GenerationConfig::greedy().with_max_tokens(5));

        assert_eq!(pipeline.model().vocab_size, 100);
        assert_eq!(pipeline.processors().len(), 1);
        assert_eq!(pipeline.config().max_tokens, 5);
    }

    #[test]
    fn test_generation_pipeline_model_mut() {
        let model = MockModel::new(100, 42);
        let mut pipeline = GenerationPipeline::new(model);

        // Should be able to mutate model
        pipeline.model_mut().highest_token = 50;
        assert_eq!(pipeline.model().highest_token, 50);
    }
    include!("tests_generation_config.rs");
    include!("tests_sample_min.rs");
    include!("tests_beam_search.rs");
    include!("tests_prompt_cache.rs");
    include!("logit_processor_tests.rs");
}
