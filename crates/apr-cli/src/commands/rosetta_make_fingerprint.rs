
    // ========================================================================
    // NEW: validate_fingerprints comprehensive tests
    // ========================================================================

    fn make_fingerprint(
        name: &str,
        mean: f32,
        std: f32,
        nan_count: u32,
        inf_count: u32,
    ) -> TensorFingerprint {
        TensorFingerprint {
            name: name.to_string(),
            shape: vec![10, 20],
            dtype: "F32".to_string(),
            mean,
            std,
            min: -1.0,
            max: 1.0,
            p5: -0.9,
            p25: -0.25,
            p50: 0.0,
            p75: 0.25,
            p95: 0.9,
            nan_count,
            inf_count,
            zero_fraction: 0.0,
            checksum: 0,
        }
    }

    #[test]
    fn test_validate_fingerprints_identical() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_mean_deviation_above_threshold() {
        let actual = vec![make_fingerprint("tensor_a", 5.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].field, "mean");
    }

    #[test]
    fn test_validate_fingerprints_mean_deviation_below_threshold() {
        let actual = vec![make_fingerprint("tensor_a", 0.6, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // 0.1 sigma deviation < 3.0 threshold
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_nan_anomaly() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // NaN in actual but not in reference = anomaly
        let nan_anomaly = anomalies.iter().find(|a| a.field == "nan_count");
        assert!(nan_anomaly.is_some());
        assert_eq!(
            nan_anomaly.expect("nan anomaly").deviation_sigma,
            f32::INFINITY
        );
    }

    #[test]
    fn test_validate_fingerprints_inf_anomaly() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let inf_anomaly = anomalies.iter().find(|a| a.field == "inf_count");
        assert!(inf_anomaly.is_some());
    }

    #[test]
    fn test_validate_fingerprints_nan_not_anomaly_when_reference_has_nan() {
        // Both have NaN => not anomalous
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let nan_anomaly = anomalies.iter().find(|a| a.field == "nan_count");
        assert!(nan_anomaly.is_none());
    }

    #[test]
    fn test_validate_fingerprints_missing_reference_tensor() {
        let actual = vec![make_fingerprint("tensor_only_in_actual", 0.5, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_only_in_ref", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // No matching tensor name => no anomalies (tensor is just skipped)
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_layernorm() {
        // LayerNorm has tighter threshold (2.0) in strict mode
        let actual = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            3.5,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            1.0,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 5.0, true);
        // Deviation = 2.5 sigma, strict threshold for layernorm = 2.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_embedding() {
        // Embeddings have looser threshold (5.0) in strict mode
        let actual = vec![make_fingerprint(
            "model.embed_tokens.weight",
            3.5,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.embed_tokens.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 2.0, true);
        // Deviation = 3.0 sigma, strict threshold for embed = 5.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_zero_std_reference() {
        // When reference std is near zero, deviation is scaled up
        let actual = vec![make_fingerprint("tensor_a", 0.001, 0.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 0.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // deviation = 0.001 * 1000 = 1.0 < threshold 3.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_cross_format_names() {
        // GGUF name should match APR name via normalize_tensor_name
        let actual = vec![make_fingerprint("blk.0.attn_q.weight", 5.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint(
            "model.layers.0.self_attn.q_proj.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // Should match and detect the mean deviation
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_multiple_tensors() {
        let actual = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 10.0, 1.0, 0, 0),
            make_fingerprint("tensor_c", 0.5, 1.0, 3, 0),
        ];
        let reference = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_c", 0.5, 1.0, 0, 0),
        ];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // tensor_b has mean deviation, tensor_c has NaN anomaly
        assert!(anomalies.len() >= 2);
    }

    // ========================================================================
    // NEW: get_role_threshold specific return value tests
    // ========================================================================

    #[test]
    fn test_get_role_threshold_layernorm_value() {
        assert_eq!(
            get_role_threshold("model.layers.0.input_layernorm.weight"),
            2.0
        );
    }

    #[test]
    fn test_get_role_threshold_layer_norm_underscore_value() {
        assert_eq!(get_role_threshold("some.layer_norm.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_ln_prefix_value() {
        assert_eq!(get_role_threshold("ln_1.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_embed_value() {
        assert_eq!(get_role_threshold("model.embed_tokens.weight"), 5.0);
    }

    #[test]
    fn test_get_role_threshold_lm_head_value() {
        assert_eq!(get_role_threshold("lm_head.weight"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_output_value() {
        assert_eq!(get_role_threshold("output.weight"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_default_value() {
        assert_eq!(get_role_threshold("some.random.tensor"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_case_insensitive() {
        // Should detect "LAYERNORM" even though check uses lowercase
        assert_eq!(get_role_threshold("model.LAYERNORM.weight"), 2.0);
        assert_eq!(get_role_threshold("model.EMBED.weight"), 5.0);
    }

    // ========================================================================
    // NEW: fingerprints_to_json comprehensive tests
    // ========================================================================

    #[test]
    fn test_fingerprints_to_json_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 1.5, 2.0, 1, 2),
        ];
        let json = fingerprints_to_json(&fingerprints);
        assert!(json.contains("tensor_a"));
        assert!(json.contains("tensor_b"));
        // First entry should have trailing comma, last should not (between entries)
        assert!(json.contains("},\n"));
    }

    #[test]
    fn test_fingerprints_to_json_special_values() {
        let fp = TensorFingerprint {
            name: "test".to_string(),
            shape: vec![],
            dtype: "Q4_K".to_string(),
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p95: 0.0,
            nan_count: 100,
            inf_count: 50,
            zero_fraction: 0.99,
            checksum: 0xDEADBEEF,
        };
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"nan_count\": 100"));
        assert!(json.contains("\"inf_count\": 50"));
        assert!(json.contains("Q4_K"));
        assert!(json.contains(&format!("{}", 0xDEADBEEF_u32)));
    }

    #[test]
    fn test_fingerprints_to_json_roundtrip_structure() {
        let fps = vec![make_fingerprint("t1", 0.1, 0.2, 0, 0)];
        let json = fingerprints_to_json(&fps);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"fingerprints\""));
    }

    // ========================================================================
    // NEW: load_fingerprints_from_json with valid content
    // ========================================================================

    #[test]
    fn test_load_fingerprints_from_json_valid_name_fields() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"{\n  \"fingerprints\": [\n    {\"name\": \"tensor_a\", \"mean\": 0.5},\n    {\"name\": \"tensor_b\", \"mean\": 1.0}\n  ]\n}").expect("write");

        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 2);
        assert_eq!(fps[0].name, "tensor_a");
        assert_eq!(fps[1].name, "tensor_b");
        // All other fields are placeholder defaults
        assert_eq!(fps[0].std, 1.0);
        assert_eq!(fps[0].dtype, "unknown");
    }

    #[test]
    fn test_load_fingerprints_from_json_no_name_fields() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"{\"data\": [1, 2, 3]}").expect("write");

        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        assert!(result.expect("parsed").is_empty());
    }

    // ========================================================================
    // NEW: parse_tensor_stats_json always returns None
    // ========================================================================

    #[test]
    fn test_parse_tensor_stats_json_placeholder() {
        assert!(parse_tensor_stats_json("{}").is_none());
        assert!(parse_tensor_stats_json("{\"tensors\": {}}").is_none());
        assert!(parse_tensor_stats_json("").is_none());
    }

    // ========================================================================
    // NEW: normalize_tensor_name edge cases
    // ========================================================================

    #[test]
    fn test_normalize_tensor_name_output_weight() {
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_not_weight() {
        // "output.bias" should NOT map to lm_head
        let result = normalize_tensor_name("output.bias");
        assert_ne!(result, "lm_head.weight");
        assert_eq!(result, "output.bias"); // stays as-is (falls through)
    }

    #[test]
    fn test_normalize_tensor_name_deeply_nested() {
        // Only first occurrence of prefixes is stripped
        let result = normalize_tensor_name("model.layers.10.self_attn.q_proj.weight");
        assert_eq!(result, "10.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_no_match() {
        // Name with no recognized patterns should pass through mostly unchanged
        let result = normalize_tensor_name("custom_tensor_name");
        assert_eq!(result, "custom_tensor_name");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_all_mappings() {
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
    }

    // ========================================================================
    // NEW: is_transposed_dims edge cases
    // ========================================================================

    #[test]
    fn test_is_transposed_dims_square_matrix() {
        // [512, 512] vs [512, 512] - same shape, NOT transposed
        assert!(!is_transposed_dims(&[512, 512], &[512, 512]));
    }

    #[test]
    fn test_is_transposed_dims_empty_shapes() {
        assert!(!is_transposed_dims(&[], &[]));
    }

    #[test]
    fn test_is_transposed_dims_3d_shapes() {
        assert!(!is_transposed_dims(&[2, 3, 4], &[4, 3, 2]));
    }

    #[test]
    fn test_is_transposed_dims_one_empty_one_not() {
        assert!(!is_transposed_dims(&[768, 3072], &[]));
    }

    #[test]
    fn test_is_transposed_dims_different_sizes() {
        // Shapes that are NOT transposed versions of each other
        assert!(!is_transposed_dims(&[768, 3072], &[768, 1024]));
    }

    // ========================================================================
    // NEW: strip_ansi edge cases
    // ========================================================================

    #[test]
    fn test_strip_ansi_escape_without_bracket() {
        // ESC not followed by [ should just skip the ESC char
        let text = "\x1b Hello";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, " Hello");
    }

    #[test]
    fn test_strip_ansi_nested_escape_sequences() {
        let text = "\x1b[1;31;42mColored\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Colored");
    }

    #[test]
    fn test_strip_ansi_only_escape_sequences() {
        let text = "\x1b[31m\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "");
    }

    #[test]
    fn test_strip_ansi_preserves_non_ansi_content() {
        let text = "Hello [World] (test)";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Hello [World] (test)");
    }

    // ========================================================================
    // NEW: truncate_path edge cases
    // ========================================================================

    #[test]
    fn test_truncate_path_empty_string() {
        let result = truncate_path(String::new(), 10);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_path_single_char() {
        let result = truncate_path("a".to_string(), 1);
        assert_eq!(result, "a");
    }
