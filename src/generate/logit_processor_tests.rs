
    #[test]
    fn test_token_suppressor_out_of_bounds() {
        let suppressor = TokenSuppressor::new(vec![100, 200]); // Out of bounds
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        // Should not panic
        suppressor.process(&mut logits, &ctx);

        // Logits unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_token_suppressor_name() {
        let suppressor = TokenSuppressor::new(vec![]);
        assert_eq!(suppressor.name(), "token_suppressor");
    }

    #[test]
    fn test_repetition_penalty_basic() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![1u32, 3, 5];
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 6);

        penalty.process(&mut logits, &ctx);

        // Token 1 (logit 2.0) should be halved: 2.0 / 2.0 = 1.0
        assert!((logits[1] - 1.0).abs() < 1e-6);
        // Token 3 (logit 4.0) should be halved: 4.0 / 2.0 = 2.0
        assert!((logits[3] - 2.0).abs() < 1e-6);
        // Token 5 (logit 6.0) should be halved: 6.0 / 2.0 = 3.0
        assert!((logits[5] - 3.0).abs() < 1e-6);
        // Token 0 unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![0u32];
        let mut logits = vec![-2.0, 1.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 2);

        penalty.process(&mut logits, &ctx);

        // Negative logit should be multiplied: -2.0 * 2.0 = -4.0
        assert!((logits[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_with_window() {
        let penalty = RepetitionPenalty::new(2.0, 2); // Window of 2
        let tokens = vec![1u32, 2, 3, 4]; // Only last 2 (3, 4) should be penalized
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 5);

        penalty.process(&mut logits, &ctx);

        // Token 1, 2 NOT penalized (outside window)
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        // Token 3, 4 penalized (inside window)
        assert!((logits[3] - 2.0).abs() < 1e-6); // 4.0 / 2.0
        assert!((logits[4] - 2.5).abs() < 1e-6); // 5.0 / 2.0
    }

    #[test]
    fn test_temperature_scaler_basic() {
        let scaler = TemperatureScaler::new(2.0);
        let mut logits = vec![2.0, 4.0, 6.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scaler_no_effect_at_1() {
        let scaler = TemperatureScaler::new(1.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_temperature_scaler_panics_on_zero() {
        let _ = TemperatureScaler::new(0.0);
    }

    #[test]
    fn test_processor_chain_empty() {
        let chain = LogitProcessorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_processor_chain_add() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_processor_chain_names() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5))
            .with_processor(TemperatureScaler::new(0.8));

        let names = chain.processor_names();
        assert_eq!(
            names,
            vec![
                "token_suppressor",
                "repetition_penalty",
                "temperature_scaler"
            ]
        );
    }

    #[test]
    fn test_processor_chain_applies_in_order() {
        // Suppress token 0, then apply temp scaling
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(TemperatureScaler::new(2.0));

        let mut logits = vec![10.0, 4.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        chain.process(&mut logits, &ctx);

        // Token 0 suppressed (still -inf after scaling)
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        // Other logits scaled
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_processor_chain_as_logit_processor() {
        let chain = LogitProcessorChain::new().with_processor(TokenSuppressor::new(vec![0]));

        // Use as dyn LogitProcessor
        let processor: &dyn LogitProcessor = &chain;
        assert_eq!(processor.name(), "processor_chain");

        let mut logits = vec![1.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 2);
        processor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite());
    }

    // =========================================================================
    // GenerationPipeline Tests
    // =========================================================================

    /// Mock model for testing GenerationPipeline
    struct MockModel {
        vocab_size: usize,
        /// Returns logits with this token as highest
        highest_token: usize,
        call_count: usize,
    }

    impl MockModel {
        fn new(vocab_size: usize, highest_token: usize) -> Self {
            Self {
                vocab_size,
                highest_token,
                call_count: 0,
            }
        }
    }

    impl GenerativeModel for MockModel {
        fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            self.call_count += 1;
            let mut logits = vec![0.0f32; self.vocab_size];
            logits[self.highest_token] = 10.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn test_generation_pipeline_basic() {
        let model = MockModel::new(100, 42);
        let mut pipeline = GenerationPipeline::new(model)
            .with_config(GenerationConfig::greedy().with_max_tokens(3));

        let result = pipeline.generate(&[1, 2]).expect("test");

        // Initial tokens + 3 generated
        assert_eq!(result.len(), 5);
        // All generated tokens should be 42 (highest)
        assert_eq!(result[2], 42);
        assert_eq!(result[3], 42);
        assert_eq!(result[4], 42);
    }

    #[test]
    fn test_generation_pipeline_with_eos() {
        // Model that returns EOS token (99) on third call
        struct EosModel {
            call_count: usize,
        }
        impl GenerativeModel for EosModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 100];
                if self.call_count >= 3 {
                    logits[99] = 10.0; // EOS
                } else {
                    logits[50] = 10.0; // Regular token
                }
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                100
            }
        }

        let model = EosModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model).with_config(
            GenerationConfig::greedy()
                .with_max_tokens(10)
                .with_eos_token_id(99),
        );

        let result = pipeline.generate(&[1]).expect("test");

        // Should stop at EOS: [1, 50, 50, 99]
        assert_eq!(result.len(), 4);
        assert_eq!(result[result.len() - 1], 99);
    }

    #[test]
    fn test_generation_pipeline_with_token_suppression() {
        // Model that would return token 0 if not suppressed
        struct ZeroModel;
        impl GenerativeModel for ZeroModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                let mut logits = vec![0.0f32; 10];
                logits[0] = 10.0; // Token 0 is highest
                logits[5] = 5.0; // Token 5 is second highest
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                10
            }
        }

        let model = ZeroModel;
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![0])) // Suppress token 0
            .with_config(GenerationConfig::greedy().with_max_tokens(1));

        let result = pipeline.generate(&[1]).expect("test");

        // Should pick token 5 (second highest) since 0 is suppressed
        assert_eq!(result, vec![1, 5]);
    }

    #[test]
    fn test_generation_pipeline_whisper_use_case() {
        // Simulate Whisper: suppress SOT (50257) to prevent hallucination
        const SOT: u32 = 50257;
        const EOT: u32 = 50256;

        struct WhisperMockModel {
            call_count: usize,
        }
        impl GenerativeModel for WhisperMockModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 51865];

                // Test scenario: SOT has highest logit (intentional for testing SOT suppression)
                logits[SOT as usize] = 11.0;

                // Text token has second highest
                logits[440] = 10.0; // "The" token

                // EOT after 3 calls
                if self.call_count >= 4 {
                    logits[EOT as usize] = 20.0;
                }

                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                51865
            }
        }

        let model = WhisperMockModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![SOT])) // Suppress SOT
            .with_config(
                GenerationConfig::greedy()
                    .with_max_tokens(10)
                    .with_eos_token_id(EOT as usize),
            );

        let result = pipeline.generate(&[50257, 50258]).expect("test");

        // Should NOT contain SOT (50257) in generated tokens
        for &token in &result[2..] {
            // Skip initial tokens
            assert_ne!(token, SOT, "SOT should be suppressed");
        }

        // Should contain the text token and EOT
        assert!(result.contains(&440), "Should contain text token");
        assert!(result.contains(&EOT), "Should end with EOT");
    }

    // =========================================================================
    // Additional Coverage Tests (PMAT-802 Phase 2)
    // =========================================================================

    // ----- Helper Function Tests -----

    #[test]
    fn test_sample_from_distribution_first_element() {
        let probs = vec![0.5, 0.3, 0.2];
        let indices = vec![10, 20, 30];
        // rng=0.0 should hit first bucket
        let result = sample_from_distribution(&probs, &indices, 0.0);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_sample_from_distribution_middle_element() {
        let probs = vec![0.3, 0.4, 0.3];
        let indices = vec![10, 20, 30];
        // rng=0.5 (cumsum: 0.3, 0.7, 1.0) -> should hit second bucket
        let result = sample_from_distribution(&probs, &indices, 0.5);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_sample_from_distribution_last_element() {
        let probs = vec![0.2, 0.3, 0.5];
        let indices = vec![10, 20, 30];
        // rng=0.99 should hit last bucket
        let result = sample_from_distribution(&probs, &indices, 0.99);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_sample_from_distribution_fallback() {
        let probs = vec![0.5, 0.5];
        let indices = vec![100, 200];
        // rng=1.0 should fall through to last element
        let result = sample_from_distribution(&probs, &indices, 1.0);
        assert_eq!(result, 200);
    }

    #[test]
    fn test_logits_to_probs_sum_to_one() {
        let indexed = vec![(0, 2.0), (1, 1.0), (2, 0.5)];
        let probs = logits_to_probs(&indexed);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1.0");
    }

    #[test]
    fn test_logits_to_probs_ordering_preserved() {
        let indexed = vec![(0, 3.0), (1, 2.0), (2, 1.0)];
        let probs = logits_to_probs(&indexed);
        // Higher logits should have higher probabilities
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_logits_to_probs_single_element() {
        let indexed = vec![(42, 5.0)];
        let probs = logits_to_probs(&indexed);
        assert_eq!(probs.len(), 1);
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "Single element should have prob 1.0"
        );
    }

    #[test]
    fn test_logits_to_probs_equal_logits() {
        let indexed = vec![(0, 1.0), (1, 1.0), (2, 1.0)];
        let probs = logits_to_probs(&indexed);
        // Equal logits should give equal probabilities
        let expected = 1.0 / 3.0;
        for p in &probs {
            assert!((p - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_build_nucleus_p_one() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 1.0);
        // Should include all tokens
        assert_eq!(nucleus.len(), 3);
    }

    #[test]
    fn test_build_nucleus_p_zero_five() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 0.5);
        // Should include just first token (0.5 >= 0.5)
        assert_eq!(nucleus.len(), 1);
        assert_eq!(nucleus[0].0, 0);
    }

    #[test]
    fn test_build_nucleus_p_zero_eight() {
        let indexed = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let nucleus = build_nucleus(&indexed, 0.8);
        // Should include first two tokens (0.5 + 0.3 = 0.8 >= 0.8)
        assert_eq!(nucleus.len(), 2);
    }
