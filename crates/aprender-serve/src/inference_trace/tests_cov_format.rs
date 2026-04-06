
    #[test]
    fn test_cov_format_text_many_layers() {
        let mut config = TraceConfig::enabled();
        config.verbose = false;
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 10,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 2,
            quant_type: None,
        });

        // Add multiple transformer layer events followed by a different step
        // The "layers total" message only shows when transitioning away from transformer layers
        for i in 0..5 {
            tracer.start_step(TraceStep::TransformerBlock);
            tracer.trace_layer(i, 0, Some(&[0.1, 0.2, 0.3, 0.4]), 1, 4);
        }

        // Add a different step to trigger the layer count summary
        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 3.0, 4.0], 4);

        let text = tracer.format_text();
        assert!(text.contains("TRANSFORMER")); // layers were traced
        assert!(text.contains("LM_HEAD")); // next step was traced
    }

    #[test]
    fn test_cov_format_text_sample_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        tracer.trace_sample(0, &[1.0, 5.0, 2.0], 1, 0.8, 3);

        let text = tracer.format_text();
        assert!(text.contains("SAMPLE"));
        assert!(text.contains("Sampled"));
    }

    #[test]
    fn test_cov_format_text_decode_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "hello", 100);

        let text = tracer.format_text();
        assert!(text.contains("DECODE"));
        assert!(text.contains("Token ID"));
        assert!(text.contains("Decoded"));
    }

    #[test]
    fn test_cov_format_text_lm_head_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 5,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 10.0, 3.0, 4.0], 5);

        let text = tracer.format_text();
        assert!(text.contains("LM_HEAD"));
        assert!(text.contains("logits"));
    }

    #[test]
    fn test_cov_format_text_embed_step() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(2, 4, Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));

        let text = tracer.format_text();
        assert!(text.contains("EMBED"));
        assert!(text.contains("Range"));
    }

    #[test]
    fn test_cov_format_text_with_error() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "garbage", 32000); // OOV

        let text = tracer.format_text();
        assert!(text.contains("ERROR"));
        assert!(text.contains("Hint"));
        assert!(text.contains("errors"));
    }

    #[test]
    fn test_cov_trace_inf_detected_no_layer() {
        let err = TraceError::InfDetected { layer: None };
        let display = err.to_string();
        assert!(display.contains("Inf"));
        assert!(!display.contains("layer"));
    }

    #[test]
    fn test_cov_tracer_lm_head_with_nan() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, f32::NAN, 3.0];
        tracer.trace_lm_head(0, &logits, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_tracer_lm_head_with_inf() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::LmHead);
        let logits = vec![1.0, f32::INFINITY, 3.0];
        tracer.trace_lm_head(0, &logits, 3);

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_tracer_layer_with_nan() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::TransformerBlock);
        let hidden = vec![0.1, f32::NAN, 0.3];
        tracer.trace_layer(5, 0, Some(&hidden), 1, 3);

        assert_eq!(tracer.error_count(), 1);
        let event = &tracer.events()[0];
        if let Some(TraceError::NaNDetected { layer }) = &event.error {
            assert_eq!(*layer, Some(5));
        }
    }

    #[test]
    fn test_cov_tracer_embed_with_inf() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        let embeddings = vec![0.1, f32::INFINITY, 0.3];
        tracer.trace_embed(1, 3, Some(&embeddings));

        assert_eq!(tracer.error_count(), 1);
    }

    #[test]
    fn test_cov_disabled_tracer_no_events() {
        let config = TraceConfig::default(); // disabled
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("Hello", &[1, 2], 100);

        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_cov_to_json_with_error() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "json-err".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 50000, "bad", 100);

        let json = tracer.to_json();
        assert!(json.contains("error_count"));
        assert!(json.contains("50000"));
    }

    #[test]
    fn test_cov_to_json_with_special_floats() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.set_model_info(ModelInfo {
            name: "float-test".to_string(),
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 100,
            num_heads: 1,
            quant_type: None,
        });

        // Add event with NaN stats
        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(1, 4, Some(&[f32::NAN, f32::INFINITY, 0.5, -0.5]));

        let json = tracer.to_json();
        // Stats contain NaN/Inf, but the min/max/mean/std might be computed excluding those
        // The JSON contains has_nan and has_inf fields
        assert!(json.contains("\"has_nan\": true") || json.contains("\"has_inf\": true"));
    }

    #[test]
    fn test_cov_garbage_detection_private_use_area() {
        // Test Private Use Area characters
        let pua = "\u{E000}\u{E001}\u{E002}";
        assert!(is_garbage_output(pua));
    }

    #[test]
    fn test_cov_garbage_detection_cjk_extension() {
        // Test CJK Extension B characters
        let cjk = "\u{20000}\u{20001}\u{20002}";
        assert!(is_garbage_output(cjk));
    }

    #[test]
    fn test_cov_garbage_detection_normal_cjk() {
        // Normal CJK should NOT be flagged as garbage
        let normal_cjk = "你好世界"; // Hello World in Chinese
        assert!(!is_garbage_output(normal_cjk));
    }

    #[test]
    fn test_cov_tensor_stats_std_calculation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::from_slice(&data);
        // Mean should be 5.0, std should be 2.0
        assert!((stats.mean - 5.0).abs() < 0.1);
        assert!((stats.std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_cov_tensor_stats_neg_inf() {
        let data = vec![1.0, f32::NEG_INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_inf);
        assert!(!stats.has_nan);
    }

    // ============================================================
    // AWS Step Functions Parity Tests (per spec v3.1.0)
    // ============================================================

    #[test]
    fn test_aws_event_type_names() {
        // F-AWS-01: Verify event type names match AWS Step Functions format
        assert_eq!(AwsEventType::TaskStateEntered.name(), "TaskStateEntered");
        assert_eq!(AwsEventType::TaskStateExited.name(), "TaskStateExited");
        assert_eq!(AwsEventType::ExecutionFailed.name(), "ExecutionFailed");
    }

    #[test]
    fn test_aws_event_ids_monotonic() {
        // F-AWS-01: Event IDs should be monotonically increasing
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("hello", &[1, 2, 3], 1000);

        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(3, 4, Some(&[0.1, 0.2, 0.3, 0.4]));

        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(0, &[1.0, 2.0, 3.0], 3);

        let events = tracer.events();
        // 6 events: 3 Entry + 3 Exit (AWS F-AWS-01)
        assert_eq!(events.len(), 6);

        // IDs should be 1..6 (monotonically increasing)
        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.id, (i + 1) as u64);
        }
    }

    #[test]
    fn test_aws_event_type_exited() {
        // F-AWS-01: Trace methods emit paired TaskStateEntered/TaskStateExited events
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("test", &[1], 100);

        // [0] = Entry, [1] = Exit
        let entry = &tracer.events()[0];
        let exit = &tracer.events()[1];
        assert_eq!(entry.event_type, AwsEventType::TaskStateEntered);
        assert_eq!(exit.event_type, AwsEventType::TaskStateExited);
    }

    #[test]
    fn test_aws_timestamp_iso8601() {
        // F-JSON-03: Timestamp should be ISO 8601 format
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Decode);
        tracer.trace_decode(0, 5, "hello", 100);

        // Check both Entry and Exit have valid timestamps
        for event in tracer.events() {
            // ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ
            assert!(event.timestamp.contains("T"));
            assert!(event.timestamp.ends_with("Z"));

            // Should be parseable as RFC3339
            chrono::DateTime::parse_from_rfc3339(&event.timestamp)
                .expect("Timestamp should be valid RFC3339/ISO8601");
        }
    }

    #[test]
    fn test_aws_previous_event_id() {
        // F-AWS-02: TaskStateExited MUST have previous_event_id linking to Entry
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Sample);
        tracer.trace_sample(0, &[1.0, 2.0, 3.0], 1, 0.8, 3);

        let events = tracer.events();
        let entry = &events[0];
        let exit = &events[1];

        // Entry has no predecessor
        assert!(entry.previous_event_id.is_none());

        // Exit MUST link back to Entry (F-AWS-02 CRITICAL)
        assert_eq!(exit.previous_event_id, Some(entry.id));
    }

    #[test]
    fn test_aws_json_contains_type() {
        // F-JSON-01: JSON output should contain event type
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("hi", &[1], 100);

        let json = tracer.to_json();
        // Should contain event type in output
        assert!(json.contains("TOKENIZE") || json.contains("events"));
    }

    // =========================================================================
    // Additional Coverage Tests (95% push)
    // =========================================================================

    #[test]
    fn test_write_output_to_file() {
        use std::fs;
        use std::path::PathBuf;
        let path = std::env::temp_dir().join("trace_test_output.json");

        let config = TraceConfig {
            enabled: true,
            steps: HashSet::new(),
            verbose: false,
            output: Some(PathBuf::from(&path)),
        };

        let mut tracer = InferenceTracer::new(config);
        tracer.start_step(TraceStep::Tokenize);
        tracer.trace_encode("test", &[1, 2], 100);

        tracer.write_output().expect("write output");

        // Verify file was created and contains JSON
        assert!(path.exists());
        let content = fs::read_to_string(&path).expect("read");
        assert!(content.contains("events"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_write_output_text_mode() {
        // Text mode (no output path) writes to stderr
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.start_step(TraceStep::Embed);
        tracer.trace_embed(3, 4, Some(&[0.1, 0.2]));

        // This should not panic
        tracer.write_output().expect("write output");
    }

    #[test]
    fn test_set_model_info() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        let info = ModelInfo {
            name: "TestModel".to_string(),
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50000,
            num_heads: 12,
            quant_type: Some("Q4_K_M".to_string()),
        };
        tracer.set_model_info(info);

        // Model info should affect JSON output
        let json = tracer.to_json();
        assert!(json.contains("TestModel"));
    }

    #[test]
    fn test_is_verbose() {
        let mut config = TraceConfig::enabled();
        config.verbose = true;
        let tracer = InferenceTracer::new(config);
        assert!(tracer.is_verbose());

        let config2 = TraceConfig::enabled();
        let tracer2 = InferenceTracer::new(config2);
        assert!(!tracer2.is_verbose());
    }
