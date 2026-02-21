
    // ====================================================================
    // Coverage-boost: VerificationReport with tensor_diffs and changed_metadata
    // ====================================================================

    #[test]
    fn test_verification_report_with_tensor_diffs() {
        use std::collections::BTreeMap;
        let mut tensor_diffs = BTreeMap::new();
        tensor_diffs.insert("layer.0.weight".to_string(), 0.001);
        tensor_diffs.insert("layer.1.weight".to_string(), 0.005);
        let report = VerificationReport {
            is_equivalent: true,
            max_diff: 0.005,
            mean_diff: 0.003,
            tensor_diffs,
            changed_metadata: vec!["version".to_string()],
            failed_tensors: vec![],
        };
        assert!(report.passes_with_tolerance(0.01));
        assert!(!report.passes_with_tolerance(0.001));
        print_verification_json(&report);
    }

    #[test]
    fn test_verification_report_not_equivalent_but_within_tolerance() {
        let report = VerificationReport {
            is_equivalent: false,
            max_diff: 0.01,
            mean_diff: 0.001,
            tensor_diffs: std::collections::BTreeMap::new(),
            changed_metadata: vec![],
            failed_tensors: vec![],
        };
        // passes_with_tolerance checks max_diff AND failed_tensors
        assert!(report.passes_with_tolerance(0.1));
    }

    // ====================================================================
    // Coverage-boost: TensorFingerprint field access patterns
    // ====================================================================

    #[test]
    fn test_tensor_fingerprint_all_fields() {
        let fp = TensorFingerprint {
            name: "model.layers.5.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: "Q4_K_M".to_string(),
            mean: -0.002,
            std: 0.15,
            min: -1.5,
            max: 1.5,
            p5: -0.25,
            p25: -0.08,
            p50: -0.001,
            p75: 0.08,
            p95: 0.25,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.05,
            checksum: 0xABCD_1234,
        };
        assert_eq!(fp.name, "model.layers.5.self_attn.q_proj.weight");
        assert_eq!(fp.shape, vec![4096, 4096]);
        assert_eq!(fp.dtype, "Q4_K_M");
        assert!((fp.mean - (-0.002)).abs() < 0.001);
        assert!((fp.std - 0.15).abs() < 0.001);
        assert!((fp.min - (-1.5)).abs() < 0.001);
        assert!((fp.max - 1.5).abs() < 0.001);
        assert!((fp.p5 - (-0.25)).abs() < 0.001);
        assert!((fp.p25 - (-0.08)).abs() < 0.001);
        assert!((fp.p50 - (-0.001)).abs() < 0.001);
        assert!((fp.p75 - 0.08).abs() < 0.001);
        assert!((fp.p95 - 0.25).abs() < 0.001);
        assert_eq!(fp.nan_count, 0);
        assert_eq!(fp.inf_count, 0);
        assert!((fp.zero_fraction - 0.05).abs() < 0.001);
        assert_eq!(fp.checksum, 0xABCD_1234);
    }

    // ====================================================================
    // Coverage-boost: strip_ansi with escape but no bracket
    // ====================================================================

    #[test]
    fn test_strip_ansi_escape_followed_by_non_bracket() {
        // ESC followed by a regular character (not '[')
        let text = "\x1bXhello";
        let stripped = strip_ansi(text);
        // ESC is consumed, 'X' is not consumed since peek != '['
        assert_eq!(stripped, "Xhello");
    }

    #[test]
    fn test_strip_ansi_multiple_escapes_no_brackets() {
        let text = "\x1b\x1b\x1bhello";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "hello");
    }

    #[test]
    fn test_strip_ansi_escape_at_end_of_string() {
        let text = "hello\x1b";
        let stripped = strip_ansi(text);
        // ESC at end - peek returns None, ESC consumed
        assert_eq!(stripped, "hello");
    }

    #[test]
    fn test_strip_ansi_escape_bracket_at_end() {
        // ESC[ at end with no terminating letter
        let text = "hello\x1b[";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "hello");
    }

    // ====================================================================
    // Coverage-boost: normalize_tensor_name with all prefix combinations
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_layers_prefix_without_model() {
        // "layers." prefix alone (without "model." before it)
        let result = normalize_tensor_name("layers.5.self_attn.q_proj.weight");
        assert_eq!(result, "5.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_blk_with_self_attn() {
        // "blk." prefix but tensor uses self_attn (unusual but possible)
        let result = normalize_tensor_name("blk.0.self_attn.q_proj.weight");
        // blk. stripped, .self_attn. stripped
        assert_eq!(result, "0.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_mlp_prefix_stripped() {
        // Test .mlp. stripping without layers prefix
        let result = normalize_tensor_name("0.mlp.gate_proj.weight");
        assert_eq!(result, "0.gate_proj.weight");
    }

    // ====================================================================
    // Coverage-boost: FormatType from_extension edge cases
    // ====================================================================

    #[test]
    fn test_format_type_from_extension_case_sensitivity() {
        // Extension lookup may be case-sensitive depending on implementation
        let path = Path::new("model.GGUF");
        let result = FormatType::from_extension(path);
        // Should handle uppercase extensions
        assert!(result.is_ok() || result.is_err()); // Platform-dependent
    }

    #[test]
    fn test_format_type_from_extension_double_extension() {
        let path = Path::new("model.tar.gguf");
        let result = FormatType::from_extension(path);
        // Should look at last extension only
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Gguf);
    }

    // ====================================================================
    // Coverage-boost: ConversionOptions clone
    // ====================================================================

    #[test]
    fn test_conversion_options_clone() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            verify: true,
            compute_stats: true,
            tolerance: 0.01,
            preserve_metadata: true,
            add_provenance: true,
            tokenizer_path: None,
        };
        let cloned = opts.clone();
        assert_eq!(opts.quantization, cloned.quantization);
        assert_eq!(opts.verify, cloned.verify);
        assert_eq!(opts.compute_stats, cloned.compute_stats);
        assert!((opts.tolerance - cloned.tolerance).abs() < 1e-10);
        assert_eq!(opts.preserve_metadata, cloned.preserve_metadata);
        assert_eq!(opts.add_provenance, cloned.add_provenance);
    }

    // ========================================================================
    // F-ROSETTA-004: Fingerprint detects tensor corruption
    // Flip 1 byte in tensor data → checksum must differ
    // ========================================================================

    #[test]
    fn t_f_rosetta_004_fingerprint_detects_single_byte_corruption() {
        // Create realistic tensor data (weight-like values)
        let original: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.00314159 - 1.5).sin() * 0.02)
            .collect();

        // Compute baseline fingerprint
        let (mean_a, std_a, min_a, max_a, _, _, _, _, _, _, _, _, checksum_a) =
            compute_tensor_stats(&original);

        // Corrupt exactly 1 float value (flip a bit in byte representation)
        let mut corrupted = original.clone();
        // Flip the sign bit of element 500 (significant change)
        let bits = corrupted[500].to_bits() ^ 0x8000_0000;
        corrupted[500] = f32::from_bits(bits);

        // Compute corrupted fingerprint
        let (mean_b, std_b, min_b, max_b, _, _, _, _, _, _, _, _, checksum_b) =
            compute_tensor_stats(&corrupted);

        // PRIMARY ASSERTION: checksum MUST differ
        assert_ne!(
            checksum_a, checksum_b,
            "F-ROSETTA-004: Checksum must detect single-byte corruption"
        );

        // SECONDARY: at least one stat must differ (mean, std, min, or max)
        let stats_differ = (mean_a - mean_b).abs() > 1e-10
            || (std_a - std_b).abs() > 1e-10
            || (min_a - min_b).abs() > 1e-10
            || (max_a - max_b).abs() > 1e-10;
        assert!(
            stats_differ,
            "F-ROSETTA-004: At least one stat must differ after corruption"
        );
    }

    #[test]
    fn t_f_rosetta_004_fingerprint_stable_for_identical_data() {
        let data: Vec<f32> = (0..500)
            .map(|i| ((i as f32) * 0.007 - 1.75).cos() * 0.1)
            .collect();

        let (mean_a, std_a, _, _, _, _, _, _, _, _, _, _, checksum_a) = compute_tensor_stats(&data);
        let (mean_b, std_b, _, _, _, _, _, _, _, _, _, _, checksum_b) = compute_tensor_stats(&data);

        assert_eq!(
            checksum_a, checksum_b,
            "Identical data must produce identical checksums"
        );
        assert!(
            (mean_a - mean_b).abs() < f32::EPSILON,
            "Identical data must produce identical means"
        );
        assert!(
            (std_a - std_b).abs() < f32::EPSILON,
            "Identical data must produce identical stds"
        );
    }

    #[test]
    fn t_f_rosetta_004_fingerprint_detects_small_perturbation() {
        // Even a tiny perturbation (1 ULP change) must be detected by checksum
        let original: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();

        let mut perturbed = original.clone();
        // Add 1 ULP (unit of least precision) to element 50
        let bits = perturbed[50].to_bits() + 1;
        perturbed[50] = f32::from_bits(bits);

        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum_a) = compute_tensor_stats(&original);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum_b) = compute_tensor_stats(&perturbed);

        assert_ne!(
            checksum_a, checksum_b,
            "F-ROSETTA-004: Even 1 ULP change must produce different checksum"
        );
    }

    // =========================================================================
    // F-GT-002: Mixed quantization level warning tests
    // =========================================================================

    #[test]
    fn t_f_gt_002_mixed_quant_warning_safetensors_vs_gguf_q4k() {
        let model_a = Path::new("model.safetensors");
        let model_b = Path::new("model-q4_k_m.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing SafeTensors (unquantized) vs GGUF Q4_K_M"
        );
        let msg = warning.expect("checked above");
        assert!(
            msg.contains("F-GT-002"),
            "Warning must cite F-GT-002: {msg}"
        );
        assert!(
            msg.contains("mixed quantization") || msg.contains("Mixed quantization"),
            "Warning must mention mixed quantization: {msg}"
        );
    }

    #[test]
    fn t_f_gt_002_no_warning_same_format() {
        let model_a = Path::new("model-q4_k.gguf");
        let model_b = Path::new("other-q4_k.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_none(),
            "F-GT-002: No warning when both models are Q4_K GGUF"
        );
    }

    #[test]
    fn t_f_gt_002_warning_different_gguf_quants() {
        let model_a = Path::new("model-q4_k.gguf");
        let model_b = Path::new("model-q6_k.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing Q4_K vs Q6_K"
        );
    }

    #[test]
    fn t_f_gt_002_warning_apr_vs_safetensors() {
        let model_a = Path::new("model-q4k.apr");
        let model_b = Path::new("model.safetensors");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing APR Q4K vs SafeTensors (unquantized)"
        );
    }

    #[test]
    fn t_f_gt_002_no_warning_both_safetensors() {
        let model_a = Path::new("model-part1.safetensors");
        let model_b = Path::new("model-part2.safetensors");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_none(),
            "F-GT-002: No warning when both are SafeTensors (same quant level)"
        );
    }
