
    #[test]
    fn test_trace_step_parse() {
        assert_eq!(TraceStep::parse("encode"), Some(TraceStep::Tokenize));
        assert_eq!(TraceStep::parse("TOKENIZE"), Some(TraceStep::Tokenize));
        assert_eq!(TraceStep::parse("sample"), Some(TraceStep::Sample));
        assert_eq!(TraceStep::parse("decode"), Some(TraceStep::Decode));
        assert_eq!(TraceStep::parse("invalid"), None);
    }

    #[test]
    fn test_tensor_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(!stats.has_nan);
        assert!(stats.has_inf);
    }

    #[test]
    fn test_garbage_detection() {
        assert!(!is_garbage_output("Hello world"));
        assert!(!is_garbage_output(""));
        assert!(is_garbage_output("\u{FFFD}\u{FFFD}\u{FFFD}"));
    }

    #[test]
    fn test_trace_config_parse_steps() {
        let steps = TraceConfig::parse_steps("encode,decode,sample");
        assert!(steps.contains(&TraceStep::Tokenize));
        assert!(steps.contains(&TraceStep::Decode));
        assert!(steps.contains(&TraceStep::Sample));
        assert!(!steps.contains(&TraceStep::Embed));
    }

    #[test]
    fn test_tracer_basic() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test-model".to_string(),
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 32000,
            num_heads: 12,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2, 3], 32000);

        assert_eq!(tracer.events().len(), 2); // Entry + Exit (AWS F-AWS-01)
        assert_eq!(tracer.error_count(), 0);
    }

    #[test]
    fn test_tracer_vocab_overflow() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "garbage", 32000); // token > vocab

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_disabled_tracer() {
        let tracer = InferenceTracer::disabled();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_trace_step_names() {
        assert_eq!(TraceStep::Tokenize.name(), "TOKENIZE");
        assert_eq!(TraceStep::Embed.name(), "EMBED");
        assert_eq!(TraceStep::LayerNorm.name(), "LAYER_NORM");
        assert_eq!(TraceStep::Attention.name(), "ATTENTION");
        assert_eq!(TraceStep::FFN.name(), "FFN");
        assert_eq!(TraceStep::TransformerBlock.name(), "TRANSFORMER_BLOCK");
        assert_eq!(TraceStep::LmHead.name(), "LM_HEAD");
        assert_eq!(TraceStep::Sample.name(), "SAMPLE");
        assert_eq!(TraceStep::Decode.name(), "DECODE");
    }

    #[test]
    fn test_trace_step_numbers() {
        assert_eq!(TraceStep::Tokenize.step_number(), 1);
        assert_eq!(TraceStep::Embed.step_number(), 2);
        assert_eq!(TraceStep::Decode.step_number(), 6);
    }

    #[test]
    fn test_trace_step_parse_all() {
        assert_eq!(TraceStep::parse("embed"), Some(TraceStep::Embed));
        assert_eq!(TraceStep::parse("embedding"), Some(TraceStep::Embed));
        assert_eq!(TraceStep::parse("layernorm"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("ln"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("norm"), Some(TraceStep::LayerNorm));
        assert_eq!(TraceStep::parse("attention"), Some(TraceStep::Attention));
        assert_eq!(TraceStep::parse("attn"), Some(TraceStep::Attention));
        assert_eq!(TraceStep::parse("ffn"), Some(TraceStep::FFN));
        assert_eq!(TraceStep::parse("mlp"), Some(TraceStep::FFN));
        assert_eq!(
            TraceStep::parse("transformer"),
            Some(TraceStep::TransformerBlock)
        );
        assert_eq!(TraceStep::parse("layer"), Some(TraceStep::TransformerBlock));
        assert_eq!(TraceStep::parse("lmhead"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("lm_head"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("head"), Some(TraceStep::LmHead));
        assert_eq!(TraceStep::parse("sampling"), Some(TraceStep::Sample));
        assert_eq!(TraceStep::parse("detokenize"), Some(TraceStep::Decode));
    }

    #[test]
    fn test_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = TensorStats::from_slice(&data);
        // Empty slice returns default, which has min=0.0
        assert!((stats.min - 0.0).abs() < f32::EPSILON);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
    }

    #[test]
    fn test_tensor_stats_has_error() {
        let normal = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        assert!(!normal.has_error());

        let nan = TensorStats::from_slice(&[1.0, f32::NAN, 3.0]);
        assert!(nan.has_error());

        let inf = TensorStats::from_slice(&[1.0, f32::INFINITY, 3.0]);
        assert!(inf.has_error());
    }

    #[test]
    fn test_trace_error_display() {
        let err1 = TraceError::VocabOverflow {
            token_id: 50000,
            vocab_size: 32000,
        };
        assert!(err1.to_string().contains("50000"));
        assert!(err1.to_string().contains("32000"));

        let err2 = TraceError::NaNDetected { layer: Some(5) };
        assert!(err2.to_string().contains("layer 5"));

        let err3 = TraceError::NaNDetected { layer: None };
        assert!(err3.to_string().contains("NaN"));

        let err4 = TraceError::InfDetected { layer: Some(3) };
        assert!(err4.to_string().contains("Inf"));

        let err5 = TraceError::GarbageOutput {
            sample: "garbage".to_string(),
        };
        assert!(err5.to_string().contains("garbage"));

        let err6 = TraceError::UnknownToken { token_id: 99999 };
        assert!(err6.to_string().contains("99999"));

        let err7 = TraceError::ShapeMismatch {
            expected: vec![1, 2],
            actual: vec![3, 4],
        };
        assert!(err7.to_string().contains("mismatch"));
    }

    #[test]
    fn test_trace_embed() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, 0.2, 0.3, 0.4];
        tracer.trace_embed(1, 4, Some(&embeddings));

        // 2 events: Entry + Exit (AWS F-AWS-01)
        assert_eq!(tracer.events().len(), 2);
        assert_eq!(tracer.events()[1].step, TraceStep::Embed); // [1] = Exit
    }

    #[test]
    fn test_trace_layer() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, 0.2, 0.3, 0.4];
        tracer.trace_layer(0, 0, Some(&hidden), 1, 4);

        // 2 events: Entry + Exit (AWS F-AWS-01)
        assert_eq!(tracer.events().len(), 2);
        assert_eq!(tracer.events()[1].step, TraceStep::TransformerBlock); // [1] = Exit
    }

    #[test]
    fn test_trace_lm_head() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, 2.0, 10.0, 3.0, 4.0];
        tracer.trace_lm_head(0, &logits, 5);

        // 2 events: Entry + Exit (AWS F-AWS-01)
        assert_eq!(tracer.events().len(), 2);
        assert_eq!(tracer.events()[1].step, TraceStep::LmHead); // [1] = Exit
    }

    #[test]
    fn test_trace_sample() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        let logits = vec![1.0, 2.0, 10.0, 3.0, 4.0];
        tracer.trace_sample(0, &logits, 2, 1.0, 5);

        // 2 events: Entry + Exit (AWS F-AWS-01)
        assert_eq!(tracer.events().len(), 2);
        assert_eq!(tracer.events()[1].step, TraceStep::Sample); // [1] = Exit
    }

    #[test]
    fn test_format_text_output() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test-model".to_string(),
            num_layers: 2,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2], 100);

        let text = tracer.format_text();
        assert!(text.contains("APR Inference Trace"));
        assert!(text.contains("test-model"));
    }

    #[test]
    fn test_to_json_output() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "json-test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Test", &[1], 100);

        let json = tracer.to_json();
        assert!(json.contains("json-test"));
        assert!(json.contains("events"));
    }

    #[test]
    fn test_trace_config_should_trace() {
        let mut config = TraceConfig::enabled();
        assert!(config.should_trace(TraceStep::Tokenize));
        assert!(config.should_trace(TraceStep::Decode));

        config.steps.insert(TraceStep::Tokenize);
        assert!(config.should_trace(TraceStep::Tokenize));
        assert!(!config.should_trace(TraceStep::Decode));

        let disabled = TraceConfig::default();
        assert!(!disabled.should_trace(TraceStep::Tokenize));
    }

    #[test]
    fn test_garbage_detection_various() {
        assert!(!is_garbage_output("Normal text"));
        assert!(!is_garbage_output("123 numbers"));
        assert!(!is_garbage_output("code();"));
        assert!(is_garbage_output("⚠\u{FFFD}⚠\u{FFFD}⚠"));
    }

    #[test]
    fn test_tracer_with_nan_in_embed() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, f32::NAN, 0.3];
        tracer.trace_embed(1, 3, Some(&embeddings));

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_tracer_with_inf_in_layer() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, f32::INFINITY, 0.3];
        tracer.trace_layer(0, 0, Some(&hidden), 1, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_tracer_garbage_in_decode() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "\u{FFFD}\u{FFFD}\u{FFFD}", 100);

        assert_eq!(tracer.error_count(), 1);
    }

    // Coverage tests for helper functions
    #[test]
    fn test_cov_get_top_k_indices() {
        let logits = vec![1.0, 5.0, 2.0, 10.0, 3.0];
        let top_k = get_top_k_indices(&logits, 3);
        // Should return indices sorted by value descending: 10.0 (idx 3), 5.0 (idx 1), 3.0 (idx 4)
        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 3); // index 3 has value 10.0
        assert_eq!(top_k[1].0, 1); // index 1 has value 5.0
        assert_eq!(top_k[2].0, 4); // index 4 has value 3.0
    }

    #[test]
    fn test_cov_get_top_k_indices_with_nan() {
        let logits = vec![1.0, f32::NAN, 2.0];
        let top_k = get_top_k_indices(&logits, 2);
        assert_eq!(top_k.len(), 2);
    }

    #[test]
    fn test_cov_compute_top_k_probs() {
        let logits = vec![1.0, 2.0, 3.0];
        let top_k = vec![(2u32, 3.0f32), (1, 2.0)];
        let probs = compute_top_k_probs(&logits, &top_k);
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to less than 1 since we only have top-2
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        assert!(sum > 0.5 && sum <= 1.0);
    }

    #[test]
    fn test_cov_get_error_hint() {
        let hint1 = get_error_hint(&TraceError::VocabOverflow {
            token_id: 0,
            vocab_size: 0,
        });
        assert!(hint1.contains("vocab"));

        let hint2 = get_error_hint(&TraceError::NaNDetected { layer: None });
        assert!(hint2.contains("overflow") || hint2.contains("softmax"));

        let hint3 = get_error_hint(&TraceError::InfDetected { layer: None });
        assert!(hint3.contains("division") || hint3.contains("zero"));

        let hint4 = get_error_hint(&TraceError::GarbageOutput {
            sample: String::new(),
        });
        assert!(hint4.contains("vocab"));

        let hint5 = get_error_hint(&TraceError::UnknownToken { token_id: 0 });
        assert!(hint5.contains("vocabulary"));

        let hint6 = get_error_hint(&TraceError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
        });
        assert!(hint6.contains("dimensions") || hint6.contains("architecture"));
    }

    #[test]
    fn test_cov_format_json_float() {
        assert_eq!(format_json_float(1.5), "1.500000");
        assert_eq!(format_json_float(f32::NAN), "null");
        assert_eq!(format_json_float(f32::INFINITY), "\"Infinity\"");
        assert_eq!(format_json_float(f32::NEG_INFINITY), "\"-Infinity\"");
    }

    #[test]
    fn test_cov_format_text_long_input() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 2,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        // Long input text (>50 chars) should be truncated
        let long_text = "A".repeat(100);
        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode(&long_text, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100);

        let text = tracer.format_text();
        assert!(text.contains("...")); // truncation indicator
    }
