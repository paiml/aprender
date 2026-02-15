
    #[test]
    fn test_compute_tensor_stats_negative_only() {
        let data = vec![-10.0, -5.0, -1.0];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean < 0.0);
        assert!((min - (-10.0)).abs() < 0.001);
        assert!((max - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_all_inf() {
        let data = vec![f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY];
        let (mean, std, _, _, _, _, _, _, _, nan, inf, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan, 0);
        assert_eq!(inf, 3);
        // No valid values, so mean/std should be 0
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ====================================================================
    // Coverage-boost tests: get_role_threshold more patterns
    // ====================================================================

    #[test]
    fn test_get_role_threshold_post_attention_layernorm() {
        assert_eq!(
            get_role_threshold("model.layers.0.post_attention_layernorm.weight"),
            2.0
        );
    }

    #[test]
    fn test_get_role_threshold_ln_f() {
        // ln_f is a common name for final layer norm (GPT-style)
        assert_eq!(get_role_threshold("ln_f.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_embed_with_suffix() {
        assert_eq!(get_role_threshold("wte.embed.weight"), 5.0);
    }

    #[test]
    fn test_get_role_threshold_output_proj() {
        // "output" in the name should match
        assert_eq!(get_role_threshold("output_proj.weight"), 3.0);
    }

    // ====================================================================
    // Coverage-boost tests: f16_to_f32 more values
    // ====================================================================

    #[test]
    fn test_f16_to_f32_two() {
        // f16 2.0: 0x4000
        let bytes = [0x00, 0x40];
        let result = f16_to_f32(&bytes);
        assert!((result - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_smallest_subnormal() {
        // f16 smallest subnormal: 0x0001
        let bytes = [0x01, 0x00];
        let result = f16_to_f32(&bytes);
        assert!(result > 0.0);
        assert!(result < 0.001);
    }

    #[test]
    fn test_f16_to_f32_max_normal() {
        // f16 max normal: 0x7BFF (65504.0)
        let bytes = [0xFF, 0x7B];
        let result = f16_to_f32(&bytes);
        assert!((result - 65504.0).abs() < 1.0);
    }

    // ====================================================================
    // Coverage-boost tests: validate_fingerprints more cases
    // ====================================================================

    #[test]
    fn test_validate_fingerprints_empty_actual() {
        let actual: Vec<TensorFingerprint> = vec![];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_empty_reference() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let reference: Vec<TensorFingerprint> = vec![];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_both_empty() {
        let actual: Vec<TensorFingerprint> = vec![];
        let reference: Vec<TensorFingerprint> = vec![];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_exact_threshold_boundary() {
        // Deviation exactly at threshold should not trigger anomaly
        // ref mean=0, ref std=1, actual mean=3 => deviation=3.0, threshold=3.0
        let actual = vec![make_fingerprint("tensor_a", 3.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // deviation == threshold, not > threshold, so no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_just_above_threshold() {
        // Deviation just above threshold should trigger anomaly
        let actual = vec![make_fingerprint("tensor_a", 3.01, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_inf_count_not_anomaly_when_both_have() {
        // Both have inf => not anomalous for inf_count
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let inf_anomaly = anomalies.iter().find(|a| a.field == "inf_count");
        assert!(inf_anomaly.is_none());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_default_tensor() {
        // Non-special tensor in strict mode should use default threshold (3.0)
        let actual = vec![make_fingerprint("random_tensor.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("random_tensor.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict mode: default threshold is 3.0, deviation is 4.0 > 3.0
        assert!(!anomalies.is_empty());
    }

    // ====================================================================
    // Coverage-boost tests: fingerprints_to_json structure
    // ====================================================================

    #[test]
    fn test_fingerprints_to_json_contains_all_fields() {
        let fp = make_fingerprint("test_tensor", 0.5, 1.0, 2, 3);
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"name\": \"test_tensor\""));
        assert!(json.contains("\"mean\":"));
        assert!(json.contains("\"std\":"));
        assert!(json.contains("\"min\":"));
        assert!(json.contains("\"max\":"));
        assert!(json.contains("\"p5\":"));
        assert!(json.contains("\"p25\":"));
        assert!(json.contains("\"p50\":"));
        assert!(json.contains("\"p75\":"));
        assert!(json.contains("\"p95\":"));
        assert!(json.contains("\"nan_count\": 2"));
        assert!(json.contains("\"inf_count\": 3"));
        assert!(json.contains("\"zero_fraction\":"));
        assert!(json.contains("\"checksum\":"));
        assert!(json.contains("\"shape\":"));
        assert!(json.contains("\"dtype\": \"F32\""));
    }

    #[test]
    fn test_fingerprints_to_json_three_items_comma_placement() {
        let fps = vec![
            make_fingerprint("a", 0.0, 0.0, 0, 0),
            make_fingerprint("b", 0.0, 0.0, 0, 0),
            make_fingerprint("c", 0.0, 0.0, 0, 0),
        ];
        let json = fingerprints_to_json(&fps);
        // Count commas between entries (should be exactly 2 for 3 items)
        let entry_separators = json.matches("},\n").count();
        assert_eq!(entry_separators, 2);
    }

    // ====================================================================
    // Coverage-boost tests: is_transposed_dims symmetry
    // ====================================================================

    #[test]
    fn test_is_transposed_dims_symmetric() {
        // If a and b are transposed, b and a should also be transposed
        let a = &[768, 3072];
        let b = &[3072, 768];
        assert_eq!(is_transposed_dims(a, b), is_transposed_dims(b, a));
    }

    #[test]
    fn test_is_transposed_dims_square_large() {
        // Large square matrix should not be considered transposed
        assert!(!is_transposed_dims(&[4096, 4096], &[4096, 4096]));
    }

    // ====================================================================
    // Coverage-boost tests: normalize_tensor_name idempotency
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_idempotent() {
        // Normalizing an already-normalized name should be stable
        let names = [
            "0.q_proj.weight",
            "0.gate_proj.weight",
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
        ];
        for name in &names {
            let once = normalize_tensor_name(name);
            let twice = normalize_tensor_name(&once);
            assert_eq!(
                once, twice,
                "normalize_tensor_name not idempotent for {name}"
            );
        }
    }

    #[test]
    fn test_normalize_tensor_name_with_dots_only() {
        let result = normalize_tensor_name("...");
        assert_eq!(result, "...");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_all_attn_variants_same_layer() {
        // All attention-related tensors for layer 0 normalize correctly
        let mappings = [
            ("blk.0.attn_q.weight", "0.q_proj.weight"),
            ("blk.0.attn_k.weight", "0.k_proj.weight"),
            ("blk.0.attn_v.weight", "0.v_proj.weight"),
            ("blk.0.attn_output.weight", "0.o_proj.weight"),
            ("blk.0.attn_norm.weight", "0.input_layernorm.weight"),
        ];
        for (gguf, expected) in &mappings {
            assert_eq!(
                normalize_tensor_name(gguf),
                *expected,
                "Failed for GGUF name: {gguf}"
            );
        }
    }

    // ====================================================================
    // Coverage-boost: print_inspection_report / summary / json
    // ====================================================================

    fn make_inspection_report(
        tensor_count: usize,
        arch: Option<&str>,
        quant: Option<&str>,
    ) -> InspectionReport {
        use aprender::format::rosetta::TensorInfo;
        use std::collections::BTreeMap;

        let mut metadata = BTreeMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert(
            "long_key".to_string(),
            "a".repeat(80), // long value to trigger truncation
        );

        let tensors: Vec<TensorInfo> = (0..tensor_count)
            .map(|i| TensorInfo {
                name: format!("layer.{i}.weight"),
                dtype: "F32".to_string(),
                shape: vec![768, 3072],
                size_bytes: 768 * 3072 * 4,
                stats: None,
            })
            .collect();

        InspectionReport {
            format: FormatType::Apr,
            file_size: 1_000_000,
            metadata,
            tensors,
            total_params: 1_000_000,
            quantization: quant.map(String::from),
            architecture: arch.map(String::from),
        }
    }

    #[test]
    fn test_print_inspection_report_basic() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        // Should not panic - exercises format, arch, quant branches
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_with_hexdump_flag() {
        let report = make_inspection_report(3, None, None);
        // Exercises the hexdump branch (just prints a note)
        print_inspection_report(&report, true);
    }

    #[test]
    fn test_print_inspection_report_many_tensors() {
        // >12 tensors triggers the "... (N more tensors) ..." branch
        let report = make_inspection_report(20, Some("qwen2"), None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_no_arch_no_quant() {
        let report = make_inspection_report(2, None, None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_zero_tensors() {
        let report = make_inspection_report(0, None, None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_summary_basic() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        print_inspection_summary(&report);
    }

    #[test]
    fn test_print_inspection_summary_no_optionals() {
        let report = make_inspection_report(3, None, None);
        print_inspection_summary(&report);
    }

    #[test]
    fn test_print_inspection_json_with_arch_and_quant() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        print_inspection_json(&report);
    }

    #[test]
    fn test_print_inspection_json_no_arch_no_quant() {
        let report = make_inspection_report(3, None, None);
        print_inspection_json(&report);
    }

    #[test]
    fn test_print_inspection_json_empty_tensors() {
        let report = make_inspection_report(0, None, None);
        print_inspection_json(&report);
    }

    // ====================================================================
    // Coverage-boost: print_conversion_json
    // ====================================================================

    #[test]
    fn test_print_conversion_json_direct_path() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let source = make_inspection_report(10, Some("llama"), None);
        let target = make_inspection_report(10, Some("llama"), None);
        print_conversion_json(&path, &source, &target);
    }

    #[test]
    fn test_print_conversion_json_chain_path() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let source = make_inspection_report(5, None, Some("Q4_K"));
        let target = make_inspection_report(5, None, None);
        print_conversion_json(&path, &source, &target);
    }

    // ====================================================================
    // Coverage-boost: print_verification_json
    // ====================================================================

    #[test]
    fn test_print_verification_json_passing() {
        let report = VerificationReport::passing();
        print_verification_json(&report);
    }

    #[test]
    fn test_print_verification_json_with_failures() {
        let mut report = VerificationReport::passing();
        report.is_equivalent = false;
        report.max_diff = 0.5;
        report.mean_diff = 0.01;
        report.failed_tensors = vec!["tensor_a".to_string(), "tensor_b".to_string()];
        print_verification_json(&report);
    }

    // ====================================================================
    // Coverage-boost: print_fingerprint_diff JSON with anomalies
    // ====================================================================

    #[test]
    fn test_print_fingerprint_diff_json_with_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 10.0, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json_no_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_verbose_with_anomaly() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 20.0, 1.0, 5, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json_missing_in_b() {
        let fps_a = vec![make_fingerprint("only_in_a", 0.5, 1.0, 0, 0)];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_empty_both() {
        let fps_a: Vec<TensorFingerprint> = vec![];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }
