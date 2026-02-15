
    #[test]
    fn test_print_fingerprint_diff_empty_both_json() {
        let fps_a: Vec<TensorFingerprint> = vec![];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_inf_mismatch() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 5)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_multiple_tensors_mixed() {
        let fps_a = vec![
            make_fingerprint("t1", 0.5, 1.0, 0, 0),
            make_fingerprint("t2", 0.5, 1.0, 0, 0),
            make_fingerprint("t3_only_in_a", 0.5, 1.0, 0, 0),
        ];
        let fps_b = vec![
            make_fingerprint("t1", 0.5, 1.0, 0, 0),  // matches
            make_fingerprint("t2", 50.0, 1.0, 3, 0), // anomaly
        ];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    // ====================================================================
    // Coverage-boost: print_fingerprints more coverage
    // ====================================================================

    #[test]
    fn test_print_fingerprints_verbose_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
            make_fingerprint("tensor_c", 100.0, 50.0, 0, 0),
        ];
        let result = print_fingerprints(&fingerprints, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_non_verbose_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    // ====================================================================
    // Coverage-boost: dequantize with nonzero scale/data patterns
    // ====================================================================

    #[test]
    fn test_dequantize_q4k_for_stats_with_nonzero_scales() {
        let mut data = vec![0u8; 144];
        // Set d (f16 1.0) at bytes 0-1
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set dmin (f16 0.5) at bytes 2-3
        data[2] = 0x00;
        data[3] = 0x38;
        // Set some scale values
        for i in 4..16 {
            data[i] = 0x21; // scale = 0x21 & 0x3F = 33
        }
        // Set some quantized values (alternating patterns)
        for i in 16..144 {
            data[i] = 0xA5; // nibbles: 5 and 0xA
        }
        let result = dequantize_q4k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
        // Values should not all be zero
        let nonzero = result.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "Expected nonzero dequantized values");
    }

    #[test]
    fn test_dequantize_q4k_for_stats_request_fewer_than_block() {
        let mut data = vec![0u8; 144];
        data[0] = 0x00;
        data[1] = 0x3C; // d = 1.0
        let result = dequantize_q4k_for_stats(&data, 128);
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_dequantize_q4k_for_stats_request_zero_elements() {
        let data = vec![0u8; 144];
        let result = dequantize_q4k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_for_stats_with_nonzero_d() {
        let mut data = vec![0u8; 210];
        // Set d (f16 2.0) at bytes 208-209
        data[208] = 0x00;
        data[209] = 0x40;
        // Set some quantized data
        for i in 0..208 {
            data[i] = 0x55;
        }
        let result = dequantize_q6k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
        let nonzero = result.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "Expected nonzero dequantized values");
    }

    #[test]
    fn test_dequantize_q6k_for_stats_request_fewer_than_block() {
        let mut data = vec![0u8; 210];
        data[208] = 0x00;
        data[209] = 0x3C; // d = 1.0
        let result = dequantize_q6k_for_stats(&data, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_request_zero_elements() {
        let data = vec![0u8; 210];
        let result = dequantize_q6k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4k_for_stats_three_blocks() {
        let data = vec![0u8; 432]; // 3 * 144
        let result = dequantize_q4k_for_stats(&data, 768);
        assert_eq!(result.len(), 768);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_three_blocks() {
        let data = vec![0u8; 630]; // 3 * 210
        let result = dequantize_q6k_for_stats(&data, 768);
        assert_eq!(result.len(), 768);
    }

    // ====================================================================
    // Coverage-boost: compute_tensor_stats edge cases
    // ====================================================================

    #[test]
    fn test_compute_tensor_stats_alternating_nan_and_valid() {
        let data = vec![f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN, 3.0];
        let (mean, _std, min, max, _, _, _, _, _, nan_count, _, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 3);
        assert!((mean - 2.0).abs() < 0.001);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_very_small_values() {
        let data = vec![1e-30, 2e-30, 3e-30];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean > 0.0);
        assert!(min > 0.0);
        assert!(max > 0.0);
        assert!(min <= mean && mean <= max);
    }

    #[test]
    fn test_compute_tensor_stats_very_large_values() {
        let data = vec![1e30, 2e30, 3e30];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean > 1e29);
        assert!(min <= mean && mean <= max);
    }

    #[test]
    fn test_compute_tensor_stats_single_nan() {
        let data = vec![f32::NAN];
        let (mean, std, _, _, _, _, _, _, _, nan_count, _, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 1);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_single_inf() {
        let data = vec![f32::INFINITY];
        let (mean, std, _, _, _, _, _, _, _, _, inf_count, _, _) = compute_tensor_stats(&data);
        assert_eq!(inf_count, 1);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ====================================================================
    // Coverage-boost: validate_fingerprints strict mode for all roles
    // ====================================================================

    #[test]
    fn test_validate_fingerprints_strict_mode_lm_head() {
        // lm_head has threshold 3.0 in strict mode
        let actual = vec![make_fingerprint("lm_head.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("lm_head.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for lm_head = 3.0, deviation = 4.0 > 3.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_ln_prefix() {
        // ln_ prefix has threshold 2.0 in strict mode
        let actual = vec![make_fingerprint("ln_1.weight", 3.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("ln_1.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for ln_ = 2.0, deviation = 3.0 > 2.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_embed_below_threshold() {
        // Embeddings have loose threshold (5.0) - deviation 4.0 should pass
        let actual = vec![make_fingerprint("embed_tokens.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("embed_tokens.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for embed = 5.0, deviation = 4.0 < 5.0
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_non_strict_ignores_role() {
        // Non-strict mode: all tensors use the same threshold
        let actual = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            4.0,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            0.0,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 5.0, false);
        // Non-strict: threshold = 5.0, deviation = 4.0 < 5.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    // ====================================================================
    // Coverage-boost: load_fingerprints_from_json with varied JSON content
    // ====================================================================

    #[test]
    fn test_load_fingerprints_from_json_multiple_tensors() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        let content = r#"{
  "fingerprints": [
    {"name": "t1", "mean": 0.1},
    {"name": "t2", "mean": 0.2},
    {"name": "t3", "mean": 0.3}
  ]
}"#;
        file.write_all(content.as_bytes()).expect("write");
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 3);
        assert_eq!(fps[0].name, "t1");
        assert_eq!(fps[1].name, "t2");
        assert_eq!(fps[2].name, "t3");
    }

    #[test]
    fn test_load_fingerprints_from_json_with_quoted_name() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        let content = r#"{"name": "layer.0.attn.q_proj.weight"}"#;
        file.write_all(content.as_bytes()).expect("write");
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 1);
        assert_eq!(fps[0].name, "layer.0.attn.q_proj.weight");
    }

    #[test]
    fn test_load_fingerprints_from_json_empty_file() {
        let file = NamedTempFile::with_suffix(".json").expect("create temp file");
        // Empty file - no "name" fields
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        assert!(result.expect("parsed").is_empty());
    }

    // ====================================================================
    // Coverage-boost: parse_tensor_stats_json placeholder
    // ====================================================================

    #[test]
    fn test_parse_tensor_stats_json_with_valid_looking_json() {
        // Even valid-looking JSON returns None (placeholder implementation)
        let json_str = r#"{"tensors": {"layer.0.weight": [1.0, 2.0, 3.0]}}"#;
        assert!(parse_tensor_stats_json(json_str).is_none());
    }

    // ====================================================================
    // Coverage-boost: InspectionReport creation and field access
    // ====================================================================

    #[test]
    fn test_inspection_report_with_long_metadata_values() {
        let report = make_inspection_report(1, None, None);
        // Metadata should contain the long value
        let long_val = report.metadata.get("long_key").expect("long_key exists");
        assert_eq!(long_val.len(), 80);
    }

    #[test]
    fn test_inspection_report_exactly_12_tensors() {
        // 12 tensors: first 10 + last 2 = all printed, no "..." line
        let report = make_inspection_report(12, None, None);
        assert_eq!(report.tensors.len(), 12);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_inspection_report_exactly_13_tensors() {
        // 13 tensors: first 10 + "..." + last 2 = exercises the "..." branch
        let report = make_inspection_report(13, None, None);
        assert_eq!(report.tensors.len(), 13);
        print_inspection_report(&report, false);
    }

    // ====================================================================
    // Coverage-boost: fingerprints_to_json field validation
    // ====================================================================

    #[test]
    fn test_fingerprints_to_json_with_zero_values() {
        let fp = TensorFingerprint {
            name: "zeros".to_string(),
            shape: vec![0],
            dtype: "F32".to_string(),
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p95: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 1.0,
            checksum: 0,
        };
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"name\": \"zeros\""));
        assert!(json.contains("\"zero_fraction\": 1"));
    }

    #[test]
    fn test_fingerprints_to_json_with_large_checksum() {
        let mut fp = make_fingerprint("test", 0.0, 0.0, 0, 0);
        fp.checksum = u32::MAX;
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains(&format!("{}", u32::MAX)));
    }

    // ====================================================================
    // Coverage-boost: ConversionPath edge cases
    // ====================================================================

    #[test]
    fn test_conversion_path_direct_display_format() {
        let path = ConversionPath::direct(FormatType::SafeTensors, FormatType::Gguf);
        let display = format!("{path}");
        assert!(display.contains("SafeTensors"));
        assert!(display.contains("GGUF"));
    }

    #[test]
    fn test_conversion_path_long_chain_display() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors, FormatType::Apr],
            FormatType::Gguf,
        );
        let display = format!("{path}");
        assert!(display.contains("GGUF"));
        assert!(display.contains("SafeTensors"));
        assert!(display.contains("APR"));
    }

    #[test]
    fn test_conversion_path_steps_direct() {
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
        let steps = path.steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0], FormatType::Apr);
        assert_eq!(steps[1], FormatType::Gguf);
    }

    #[test]
    fn test_conversion_path_has_cycle_no_intermediates() {
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
        assert!(!path.has_cycle());
    }

    #[test]
    fn test_conversion_path_is_roundtrip_direct_same() {
        // Same source and target without intermediates is NOT a roundtrip
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Apr);
        assert!(!path.is_roundtrip());
    }
