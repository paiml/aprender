
    #[test]
    fn test_prompt_cache_find_prefix() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2], 111);
        cache.add(vec![1, 2, 3], 222);

        // Should find longer prefix first
        let result = cache.find_prefix(&[1, 2, 3, 4]);
        assert!(result.is_some());
        let (len, _) = result.expect("test");
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prompt_cache_miss() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);

        // No matching prefix
        let result = cache.find_prefix(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_prompt_cache_clear() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        cache.add(vec![4, 5, 6], 67890);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_stats() {
        let mut cache = PromptCache::new(100);
        cache.add(vec![1, 2, 3], 12345);

        // Hit the cache
        cache.find_prefix(&[1, 2, 3]);
        cache.find_prefix(&[1, 2, 3]);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.max_entries, 100);
    }

    #[test]
    fn test_prompt_cache_eviction() {
        let mut cache = PromptCache::new(2);
        cache.add(vec![1], 111);
        cache.add(vec![2], 222);
        assert_eq!(cache.len(), 2);

        // Adding third entry should evict LRU
        cache.add(vec![3], 333);
        assert_eq!(cache.len(), 2);
    }

    // ========================================================================
    // Dynamic Temperature Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_dyn_temp_config_default() {
        let config = DynTempConfig::default();
        assert!((config.temp - 1.0).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
        assert!((config.exponent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_new() {
        let config = DynTempConfig::new(0.8, 0.2, 1.5);
        assert!((config.temp - 0.8).abs() < 1e-6);
        assert!((config.delta - 0.2).abs() < 1e-6);
        assert!((config.exponent - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_static() {
        let config = DynTempConfig::static_temp(0.5);
        assert!((config.temp - 0.5).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_no_delta_uses_static() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DynTempConfig::static_temp(0.5);

        let result = apply_dynamic_temperature(&logits, &config);
        let static_result = apply_temperature(&logits, 0.5).expect("test");

        // Should be identical to static temperature
        for (a, b) in result.data().iter().zip(static_result.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dyn_temp_single_element() {
        let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Single element should return unchanged
        assert!((result.data()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_low_entropy_higher_temp() {
        // Low entropy (one dominant logit) should use higher temperature
        let logits = Tensor::from_vec(vec![5], vec![10.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Result should be scaled, but logits should still be ordered
        assert!(result.data()[0] > result.data()[1]);
    }

    #[test]
    fn test_dyn_temp_high_entropy_lower_temp() {
        // High entropy (uniform logits) should use lower temperature
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // With uniform logits and high entropy, should use max temp
        // All values should be close to 1.0 (uniform scaled)
        let sum: f32 = result.data().iter().sum();
        assert!(sum.abs() > 0.0); // Non-degenerate
    }

    #[test]
    fn test_dyn_temp_exponent_affects_scaling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");
        let config_exp1 = DynTempConfig::new(1.0, 0.5, 1.0);
        let config_exp2 = DynTempConfig::new(1.0, 0.5, 2.0);

        let result1 = apply_dynamic_temperature(&logits, &config_exp1);
        let result2 = apply_dynamic_temperature(&logits, &config_exp2);

        // Different exponents should produce different results
        let diff: f32 = result1
            .data()
            .iter()
            .zip(result2.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6);
    }

    // ========================================================================
    // Infill/FIM Sampler Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_infill_config_default() {
        let config = InfillConfig::default();
        assert!(config.eog_tokens.is_empty());
        assert!((config.eog_ratio_threshold - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_config_new() {
        let config = InfillConfig::new(vec![1, 2, 3]);
        assert_eq!(config.eog_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_infill_config_with_threshold() {
        let config = InfillConfig::new(vec![1]).with_threshold(5.0);
        assert!((config.eog_ratio_threshold - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_empty_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = InfillConfig::default();

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!((result.p_txt - 1.0).abs() < 1e-6);
        assert!((result.p_eog - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_no_force_eog_when_text_dominant() {
        // Text tokens have much higher probability than EOG
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 0.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!(result.p_txt > result.p_eog);
    }

    #[test]
    fn test_infill_force_eog_when_eog_dominant() {
        // EOG token has high probability relative to text
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(result.force_eog);
        assert!(result.p_eog > 0.5);
    }

    #[test]
    fn test_infill_modified_logits_when_force_eog() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]);

        let result = apply_infill_sampling(&logits, &config);
        if result.force_eog {
            // Non-EOG tokens should be -inf
            assert!(result.logits.data()[0] == f32::NEG_INFINITY);
            assert!(result.logits.data()[1] == f32::NEG_INFINITY);
            // EOG token should remain
            assert!(result.logits.data()[4] > f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_infill_multiple_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 5.0, 5.0]).expect("test");
        let config = InfillConfig::new(vec![3, 4]); // Tokens 3 and 4 are EOG

        let result = apply_infill_sampling(&logits, &config);
        // Check that both EOG tokens contribute to p_eog
        assert!(result.p_eog > 0.0);
    }

    // ========================================================================
    // Sampler Chain Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_sampler_chain_new() {
        let chain = SamplerChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_sampler_chain_default() {
        let chain = SamplerChain::default();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_sampler_chain_with_sampler() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.8));
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec!["temperature"]);
    }

    #[test]
    fn test_sampler_chain_multiple_samplers() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.9));

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.names(), vec!["temperature", "top_k", "top_p"]);
    }

    #[test]
    fn test_sampler_chain_push() {
        let mut chain = SamplerChain::new();
        chain.push(Box::new(TemperatureSampler::new(0.5)));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_sampler_chain_apply() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.5));

        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        chain.apply(&mut logits, &context);

        // Temperature 0.5 should double the logits
        assert!((logits.data()[0] - 2.0).abs() < 1e-6);
        assert!((logits.data()[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampler_chain_sample() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(1.0));

        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 4); // Greedy should pick max
    }

    #[test]
    fn test_sampler_chain_clone() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(10));

        let cloned = chain.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.names(), vec!["temperature", "top_k"]);
    }

    #[test]
    fn test_sampler_context_default() {
        let ctx = SamplerContext::default();
        assert!(ctx.tokens.is_empty());
        assert!((ctx.rng_value - 0.0).abs() < 1e-6);
        assert_eq!(ctx.step, 0);
    }

    #[test]
    fn test_sampler_context_builders() {
        let ctx = SamplerContext::new()
            .with_tokens(vec![1, 2, 3])
            .with_rng(0.5)
            .with_step(10);

        assert_eq!(ctx.tokens, vec![1, 2, 3]);
        assert!((ctx.rng_value - 0.5).abs() < 1e-6);
        assert_eq!(ctx.step, 10);
    }

    #[test]
    fn test_temperature_sampler() {
        let sampler = TemperatureSampler::new(0.5);
        assert_eq!(sampler.name(), "temperature");
    }

    #[test]
    fn test_dyn_temp_sampler() {
        let sampler = DynTempSampler::new(DynTempConfig::new(1.0, 0.5, 1.0));
        assert_eq!(sampler.name(), "dyn_temp");
    }

    #[test]
    fn test_top_k_sampler() {
        let sampler = TopKSampler::new(10);
        assert_eq!(sampler.name(), "top_k");
        assert_eq!(sampler.k, 10);
    }

    #[test]
    fn test_top_p_sampler() {
        let sampler = TopPSampler::new(0.9);
        assert_eq!(sampler.name(), "top_p");
        assert!((sampler.p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_sampler() {
        let sampler = RepetitionPenaltySampler::new(RepetitionPenaltyConfig::new(1.2));
        assert_eq!(sampler.name(), "repetition_penalty");
    }

    #[test]
    fn test_infill_sampler() {
        let sampler = InfillSampler::new(InfillConfig::new(vec![1, 2]));
        assert_eq!(sampler.name(), "infill");
    }

    #[test]
    fn test_top_k_sampler_apply() {
        let sampler = TopKSampler::new(2);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 3.0, 2.0, 4.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Only top 2 (indices 1 and 4) should remain
        let data = logits.data();
        assert!(data[0] == f32::NEG_INFINITY);
        assert!(data[1] > f32::NEG_INFINITY); // 5.0 is top
        assert!(data[2] == f32::NEG_INFINITY);
        assert!(data[3] == f32::NEG_INFINITY);
        assert!(data[4] > f32::NEG_INFINITY); // 4.0 is second
    }

    #[test]
    fn test_top_p_sampler_apply() {
        let sampler = TopPSampler::new(0.5);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 2.0, 0.0, 0.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Top token (index 1 with 5.0) should definitely remain
        let data = logits.data();
        assert!(data[1] > f32::NEG_INFINITY);
    }

    #[test]
    fn test_full_sampler_pipeline() {
        // Build a realistic pipeline: temp -> top_k -> top_p
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.95));

        let logits = Tensor::from_vec(
            vec![10],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 9); // Should still pick max after pipeline
    }

    // =========================================================================
    // LogitProcessor Tests (RLZR-GEN-001)
    // =========================================================================

    #[test]
    fn test_logit_processor_context() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let ctx = LogitProcessorContext::new(&tokens, 3, 1000);

        assert_eq!(ctx.tokens, &[1, 2, 3, 4, 5]);
        assert_eq!(ctx.step, 3);
        assert_eq!(ctx.n_vocab, 1000);
    }

    #[test]
    fn test_token_suppressor_basic() {
        let suppressor = TokenSuppressor::new(vec![0, 5, 9]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = LogitProcessorContext::new(&[], 0, 10);

        suppressor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!(logits[5].is_infinite() && logits[5] < 0.0);
        assert!(logits[9].is_infinite() && logits[9] < 0.0);
    }
