
    // ========================================================================
    // TensorStats::from_slice: additional edge cases
    // ========================================================================

    #[test]
    fn test_tensor_stats_all_zeros() {
        let data = vec![0.0; 100];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.count, 100);
        assert!((stats.mean - 0.0).abs() < 1e-8);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.max_abs, 0.0);
    }

    #[test]
    fn test_tensor_stats_all_same_value() {
        let data = vec![7.0; 50];
        let stats = TensorStats::from_slice(&data);
        assert!((stats.mean - 7.0).abs() < 1e-5);
        assert_eq!(stats.min, 7.0);
        assert_eq!(stats.max, 7.0);
    }

    #[test]
    fn test_tensor_stats_mixed_nan_and_inf() {
        let data = vec![1.0, f32::NAN, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.count, 5);
        // Mean of [1.0, 3.0] = 2.0
        assert!((stats.mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_large_values() {
        let data = vec![1e10, -1e10];
        let stats = TensorStats::from_slice(&data);
        assert!((stats.mean - 0.0).abs() < 1.0); // Close to zero
        assert!(stats.max_abs > 1e9);
    }

    #[test]
    fn test_tensor_stats_very_small_values() {
        let data = vec![1e-10, 2e-10, 3e-10];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.mean > 0.0);
        assert!(stats.mean < 1.0);
    }

    // ========================================================================
    // TensorStats::detect_anomalies: additional branches
    // ========================================================================

    #[test]
    fn test_anomaly_detection_large_max_abs() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 150.0,
            min: -150.0,
            max: 150.0,
            max_abs: 150.0,
            nan_count: 0,
            inf_count: 0,
        };
        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("large values")));
    }

    #[test]
    fn test_anomaly_detection_multiple_anomalies() {
        let stats = TensorStats {
            count: 100,
            mean: 15.0, // Large mean
            std: 0.0,   // Zero std
            l2_norm: 200.0,
            min: -200.0,
            max: 200.0,
            max_abs: 200.0, // Large values
            nan_count: 5,   // NaN
            inf_count: 3,   // Inf
        };
        let anomalies = stats.detect_anomalies("layer_7");
        // Should detect NaN, Inf, zero variance, large values, large mean
        assert!(anomalies.len() >= 4);
    }

    #[test]
    fn test_anomaly_detection_single_element_no_zero_std() {
        // count = 1, std < 1e-8 but count is not > 1, so no zero-variance anomaly
        let stats = TensorStats {
            count: 1,
            mean: 5.0,
            std: 0.0,
            l2_norm: 5.0,
            min: 5.0,
            max: 5.0,
            max_abs: 5.0,
            nan_count: 0,
            inf_count: 0,
        };
        let anomalies = stats.detect_anomalies("test");
        // Should NOT flag zero-variance for single element
        assert!(!anomalies.iter().any(|a| a.contains("variance")));
    }

    #[test]
    fn test_anomaly_detection_negative_large_mean() {
        let stats = TensorStats {
            count: 100,
            mean: -15.0, // Large negative mean
            std: 1.0,
            l2_norm: 100.0,
            min: -20.0,
            max: 0.0,
            max_abs: 20.0,
            nan_count: 0,
            inf_count: 0,
        };
        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("large mean")));
    }

    // ========================================================================
    // validate_path tests
    // ========================================================================

    #[test]
    fn test_validate_path_nonexistent() {
        let result = validate_path(Path::new("/nonexistent/file.apr"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_valid_file() {
        let file = NamedTempFile::new().expect("create temp file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // trace_layers: with valid and invalid metadata
    // ========================================================================

    #[test]
    fn test_trace_layers_empty_metadata() {
        let layers = trace_layers(&[], None, false);
        // Invalid metadata, should return default layer
        assert_eq!(layers.len(), 1);
        assert!(layers[0].name.contains("not available"));
    }

    #[test]
    fn test_trace_layers_invalid_metadata() {
        let layers = trace_layers(b"not valid msgpack", None, false);
        // Should fall back to default layer
        assert_eq!(layers.len(), 1);
        assert!(layers[0].name.contains("not available"));
    }

    #[test]
    fn test_trace_layers_valid_metadata_no_hyperparameters() {
        // Valid msgpack but no hyperparameters key
        let map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let bytes = rmp_serde::to_vec(&map).expect("serialize msgpack");
        let layers = trace_layers(&bytes, None, false);
        // No hyperparameters → default layer
        assert_eq!(layers.len(), 1);
        assert!(layers[0].name.contains("not available"));
    }

    #[test]
    fn test_trace_layers_valid_metadata_with_hyperparameters() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(2));
        hp.insert("n_embd".to_string(), serde_json::json!(128));

        let mut map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        map.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));
        let bytes = rmp_serde::to_vec(&map).expect("serialize msgpack");
        let layers = trace_layers(&bytes, None, false);
        // embedding + 2 transformer blocks + final_layer_norm = 4
        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0].name, "embedding");
        assert_eq!(layers[1].name, "transformer_block_0");
        assert_eq!(layers[2].name, "transformer_block_1");
        assert_eq!(layers[3].name, "final_layer_norm");
    }

    #[test]
    fn test_trace_layers_with_filter() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(5));
        hp.insert("n_embd".to_string(), serde_json::json!(256));

        let mut map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        map.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));
        let bytes = rmp_serde::to_vec(&map).expect("serialize msgpack");
        let layers = trace_layers(&bytes, Some("block_2"), false);
        // Only block_2 should match the filter
        assert!(layers.iter().any(|l| l.name == "transformer_block_2"));
    }

    // ========================================================================
    // print_stats: smoke test (no crash)
    // ========================================================================

    #[test]
    fn test_print_stats_no_panic() {
        let stats = compute_vector_stats(&[1.0, 2.0, 3.0]);
        print_stats("  ", &stats);
    }

    #[test]
    fn test_print_stats_with_nan_no_panic() {
        let stats = VectorStats {
            l2_norm: 5.0,
            min: 0.0,
            max: 10.0,
            mean: 5.0,
            nan_count: 3,
            inf_count: 2,
        };
        print_stats("  ", &stats);
    }

    #[test]
    fn test_print_stats_no_anomalies_no_extra_output() {
        let stats = VectorStats {
            l2_norm: 5.0,
            min: 0.0,
            max: 10.0,
            mean: 5.0,
            nan_count: 0,
            inf_count: 0,
        };
        // Should not panic, and should skip NaN/Inf line
        print_stats("", &stats);
    }

    // ========================================================================
    // LayerTrace: construction with stats
    // ========================================================================

    #[test]
    fn test_layer_trace_with_all_stats() {
        let stats = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        let trace = LayerTrace {
            name: "full_layer".to_string(),
            index: Some(5),
            input_stats: Some(stats.clone()),
            output_stats: Some(stats.clone()),
            weight_stats: Some(stats),
            anomalies: vec!["test anomaly".to_string()],
        };
        assert!(trace.input_stats.is_some());
        assert!(trace.output_stats.is_some());
        assert!(trace.weight_stats.is_some());
        assert_eq!(trace.anomalies.len(), 1);
    }

    #[test]
    fn test_layer_trace_serialize_with_stats() {
        let stats = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        let trace = LayerTrace {
            name: "layer_with_stats".to_string(),
            index: Some(0),
            input_stats: Some(stats.clone()),
            output_stats: None,
            weight_stats: None,
            anomalies: vec!["anomaly1".to_string()],
        };
        let json = serde_json::to_string(&trace).expect("serialize");
        assert!(json.contains("layer_with_stats"));
        assert!(json.contains("anomaly1"));
        assert!(json.contains("input_stats"));
    }

    // ========================================================================
    // TraceSummary and TraceResult serialization
    // ========================================================================

    #[test]
    fn test_trace_summary_serialize() {
        let summary = TraceSummary {
            total_layers: 12,
            total_parameters: 1_000_000,
            anomaly_count: 2,
            anomalies: vec![
                "NaN in layer 3".to_string(),
                "Large mean in layer 7".to_string(),
            ],
        };
        let json = serde_json::to_string(&summary).expect("serialize");
        assert!(json.contains("\"total_layers\":12"));
        assert!(json.contains("\"total_parameters\":1000000"));
        assert!(json.contains("\"anomaly_count\":2"));
    }

    // ========================================================================
    // handle_special_modes: interactive precedence over payload
    // ========================================================================

    #[test]
    fn test_handle_special_modes_interactive_takes_precedence() {
        let path = Path::new("/tmp/model.apr");
        // Both interactive and payload set - interactive should win (checked first)
        let result = handle_special_modes(path, None, true, false, true);
        assert!(result.is_some());
        assert!(result.expect("should be Some").is_ok());
    }

    // ========================================================================
    // GGUF layer filter tests
    // ========================================================================

    #[test]
    fn test_run_valid_gguf_with_layer_filter() {
        let file = build_test_gguf();
        let result = run(
            file.path(),
            Some("block_0"),
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "trace on valid GGUF with filter failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_gguf_verbose() {
        let file = build_test_gguf();
        let result = run(
            file.path(),
            None,
            None,
            false,
            true, // verbose
            false,
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "trace on valid GGUF verbose failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_with_filter() {
        let file = build_test_safetensors();
        let result = run(
            file.path(),
            Some("block_0"),
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "trace on valid SafeTensors with filter failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_verbose() {
        let file = build_test_safetensors();
        let result = run(
            file.path(),
            None,
            None,
            false,
            true, // verbose
            false,
            false,
            false,
        );
        assert!(
            result.is_ok(),
            "trace on valid SafeTensors verbose failed: {result:?}"
        );
    }

    // ========================================================================
    // GGUF with layer filter returning no matches
    // ========================================================================

    #[test]
    fn test_trace_gguf_filter_no_match() {
        let file = build_test_gguf();
        let (_, layers, _) =
            detect_and_trace(file.path(), Some("nonexistent"), false).expect("detect_and_trace");
        // Should still have at least one layer (default or embedding)
        assert!(!layers.is_empty());
    }

    // ========================================================================
    // is_likely_garbage: mixed cases
    // ========================================================================

    #[test]
    fn test_is_likely_garbage_short_two_repeated() {
        // Only 2 words, repeated check needs len > 2
        assert!(!is_likely_garbage("foo foo"));
    }

    #[test]
    fn test_is_likely_garbage_three_different_unknown() {
        // 3 different words, none common, no numbers → garbage
        assert!(is_likely_garbage("xyzzy plugh plover"));
    }

    #[test]
    fn test_is_likely_garbage_has_digit_in_text() {
        // Has number, so the no-normal-words check is skipped
        assert!(!is_likely_garbage("xyzzy plugh 42"));
    }
