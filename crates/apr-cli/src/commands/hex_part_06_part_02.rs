
    // ========================================================================
    // HexOptions defaults
    // ========================================================================

    #[test]
    fn test_hex_options_defaults() {
        let opts = HexOptions::default();
        assert_eq!(opts.limit, 64);
        assert_eq!(opts.width, 16);
        assert_eq!(opts.offset, 0);
        assert!(!opts.header);
        assert!(!opts.blocks);
        assert!(!opts.distribution);
        assert!(!opts.contract);
        assert!(!opts.entropy);
        assert!(!opts.raw);
        assert!(!opts.stats);
        assert!(!opts.list);
        assert!(!opts.json);
        assert!(opts.tensor.is_none());
    }

    // ========================================================================
    // Format detection
    // ========================================================================

    #[test]
    fn test_detect_format_gguf() {
        let bytes = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), Some(FileFormat::Gguf));
    }

    #[test]
    fn test_detect_format_apr_aprn() {
        let bytes = b"APRN\x02\x00\x00\x00";
        assert_eq!(detect_format(bytes), Some(FileFormat::Apr));
    }

    #[test]
    fn test_detect_format_apr_apr0() {
        let bytes = [0x41, 0x50, 0x52, 0x00, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&bytes), Some(FileFormat::Apr));
    }

    #[test]
    fn test_detect_format_safetensors() {
        // header_length = 50 (LE u64), then '{' starts JSON
        let mut bytes = vec![50, 0, 0, 0, 0, 0, 0, 0, b'{'];
        bytes.extend_from_slice(b"\"test\":{}");
        assert_eq!(detect_format(&bytes), Some(FileFormat::SafeTensors));
    }

    #[test]
    fn test_detect_format_unknown() {
        let bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), None);
    }

    #[test]
    fn test_detect_format_too_short() {
        assert_eq!(detect_format(&[0x41, 0x50]), None);
        assert_eq!(detect_format(&[]), None);
    }

    // ========================================================================
    // f16 to f32 conversion
    // ========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0_f32);
    }

    #[test]
    fn test_f16_to_f32_neg_zero() {
        let val = f16_to_f32(0x8000);
        assert_eq!(val.to_bits(), 0x8000_0000); // -0.0
    }

    #[test]
    fn test_f16_to_f32_one() {
        let val = f16_to_f32(0x3C00); // 1.0 in f16
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_neg_one() {
        let val = f16_to_f32(0xBC00); // -1.0 in f16
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        let val = f16_to_f32(0x7C00); // +Inf in f16
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        let val = f16_to_f32(0xFC00); // -Inf in f16
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let val = f16_to_f32(0x7C01); // NaN in f16
        assert!(val.is_nan());
    }

    #[test]
    fn test_f16_to_f32_half() {
        let val = f16_to_f32(0x3800); // 0.5 in f16
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let val = f16_to_f32(0x0001); // Smallest subnormal f16
        assert!(val > 0.0 && val < 1e-6);
    }

    // ========================================================================
    // ggml_dtype_name
    // ========================================================================

    #[test]
    fn test_ggml_dtype_name_known() {
        assert_eq!(ggml_dtype_name(0), "F32");
        assert_eq!(ggml_dtype_name(1), "F16");
        assert_eq!(ggml_dtype_name(2), "Q4_0");
        assert_eq!(ggml_dtype_name(8), "Q8_0");
        assert_eq!(ggml_dtype_name(12), "Q4_K");
        assert_eq!(ggml_dtype_name(14), "Q6_K");
    }

    #[test]
    fn test_ggml_dtype_name_unknown() {
        assert_eq!(ggml_dtype_name(99), "Unknown");
        assert_eq!(ggml_dtype_name(255), "Unknown");
    }

    // ========================================================================
    // parse_hex_offset
    // ========================================================================

    #[test]
    fn test_parse_hex_offset_decimal() {
        assert_eq!(parse_hex_offset("256"), Ok(256));
        assert_eq!(parse_hex_offset("0"), Ok(0));
    }

    #[test]
    fn test_parse_hex_offset_hex() {
        assert_eq!(parse_hex_offset("0x100"), Ok(256));
        assert_eq!(parse_hex_offset("0xFF"), Ok(255));
        assert_eq!(parse_hex_offset("0X1A"), Ok(26));
    }

    #[test]
    fn test_parse_hex_offset_invalid() {
        assert!(parse_hex_offset("0xGG").is_err());
        assert!(parse_hex_offset("abc").is_err());
    }

    // ========================================================================
    // compute_byte_entropy
    // ========================================================================

    #[test]
    fn test_byte_entropy_empty() {
        assert_eq!(compute_byte_entropy(&[]), 0.0);
    }

    #[test]
    fn test_byte_entropy_all_same() {
        let bytes = vec![0x42; 1000];
        assert_eq!(compute_byte_entropy(&bytes), 0.0);
    }

    #[test]
    fn test_byte_entropy_two_values() {
        // Equal distribution of two values → entropy = 1.0
        let mut bytes = vec![0u8; 500];
        bytes.extend(vec![1u8; 500]);
        let entropy = compute_byte_entropy(&bytes);
        assert!((entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_entropy_uniform() {
        // All 256 byte values equally represented → entropy = 8.0
        let bytes: Vec<u8> = (0..=255).cycle().take(256 * 100).collect();
        let entropy = compute_byte_entropy(&bytes);
        assert!((entropy - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_entropy_single_byte() {
        assert_eq!(compute_byte_entropy(&[42]), 0.0);
    }

    // ========================================================================
    // compute_distribution
    // ========================================================================

    #[test]
    fn test_distribution_empty() {
        let analysis = compute_distribution(&[]);
        assert_eq!(analysis.total, 0);
        assert_eq!(analysis.entropy, 0.0);
        assert!(analysis.histogram.is_empty());
    }

    #[test]
    fn test_distribution_single_value() {
        let analysis = compute_distribution(&[1.0]);
        assert_eq!(analysis.total, 1);
        assert_eq!(analysis.min, 1.0);
        assert_eq!(analysis.max, 1.0);
        assert_eq!(analysis.nan_count, 0);
    }

    #[test]
    fn test_distribution_uniform() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.total, 1000);
        assert!(analysis.min >= 0.0);
        assert!(analysis.max < 1.0);
        assert!(analysis.entropy > 2.0); // Should be high entropy
        assert_eq!(analysis.histogram.len(), 10);
    }

    #[test]
    fn test_distribution_with_nan_inf() {
        let data = [1.0, 2.0, f32::NAN, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.nan_count, 1);
        assert_eq!(analysis.inf_count, 2);
        assert_eq!(analysis.total, 6);
    }

    #[test]
    fn test_distribution_all_zeros() {
        let data = vec![0.0_f32; 100];
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.zero_count, 100);
        assert_eq!(analysis.min, 0.0);
        assert_eq!(analysis.max, 0.0);
    }

    #[test]
    fn test_distribution_skewed() {
        // Heavily right-skewed: many small values, few large
        let mut data: Vec<f32> = vec![0.1; 900];
        data.extend(vec![10.0; 100]);
        let analysis = compute_distribution(&data);
        assert!(analysis.skewness > 0.0, "Should be right-skewed");
    }

    // ========================================================================
    // print_annotated_field
    // ========================================================================

    #[test]
    fn test_print_annotated_field_short() {
        // Should not panic
        print_annotated_field(0, &[0x41, 0x50], "magic", "AP");
    }

    #[test]
    fn test_print_annotated_field_long() {
        // More than 8 bytes → should show ".."
        print_annotated_field(0x100, &[0u8; 16], "data", "16 bytes");
    }

    #[test]
    fn test_print_annotated_field_exact_8() {
        print_annotated_field(0, &[1, 2, 3, 4, 5, 6, 7, 8], "field", "value");
    }

    // ========================================================================
    // Statistics Tests (preserved)
    // ========================================================================

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let (min, max, mean, std) = compute_stats(&[5.0]);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_simple_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        assert!((std - 1.4142).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_all_same() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_values() {
        let data = [1e6, 2e6, 3e6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, 1e6);
        assert_eq!(max, 3e6);
        assert_eq!(mean, 2e6);
    }

    #[test]
    fn test_compute_stats_tiny_values() {
        let data = [1e-6, 2e-6, 3e-6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!((min - 1e-6).abs() < 1e-9);
        assert!((max - 3e-6).abs() < 1e-9);
        assert!((mean - 2e-6).abs() < 1e-9);
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert!(min.is_nan() || min == 1.0);
        assert!(mean.is_nan());
        let _ = (max, std);
    }

    #[test]
    fn test_compute_stats_all_nan() {
        let data = [f32::NAN, f32::NAN, f32::NAN];
        let (_min, _max, mean, std) = compute_stats(&data);
        assert!(mean.is_nan());
        assert!(std.is_nan());
    }

    #[test]
    fn test_compute_stats_with_infinity() {
        let data = [1.0, f32::INFINITY, -1.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -1.0);
        assert_eq!(max, f32::INFINITY);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_with_neg_infinity() {
        let data = [1.0, f32::NEG_INFINITY, 3.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::NEG_INFINITY);
        assert_eq!(max, 3.0);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_two_values() {
        let data = [0.0, 10.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 10.0);
        assert_eq!(mean, 5.0);
        assert!((std - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_array() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 999.0);
        assert!((mean - 499.5).abs() < 0.1);
        assert!((std - 288.67).abs() < 1.0);
    }

    #[test]
    fn test_compute_stats_mixed_positive_negative() {
        let data = [-100.0, -50.0, 0.0, 50.0, 100.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -100.0);
        assert_eq!(max, 100.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_subnormal_values() {
        let data = [
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 2.0,
            f32::MIN_POSITIVE * 3.0,
        ];
        let (min, max, _mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::MIN_POSITIVE);
        assert_eq!(max, f32::MIN_POSITIVE * 3.0);
    }

    // ========================================================================
    // print_tensor_anomalies tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_tensor_anomalies_no_issues() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_min() {
        print_tensor_anomalies(f32::NAN, 1.0, 0.5, 0.3);
    }
