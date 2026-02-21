
    #[test]
    fn test_truncate_path_boundary() {
        let path = "12345".to_string();
        // Exactly at boundary
        assert_eq!(truncate_path(path.clone(), 5), "12345");
        // One less than boundary
        let truncated = truncate_path(path, 4);
        assert!(truncated.starts_with("..."));
    }

    // ========================================================================
    // NEW: f16_to_f32 edge cases
    // ========================================================================

    #[test]
    fn test_f16_to_f32_empty_bytes() {
        let bytes: [u8; 0] = [];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_single_byte() {
        let bytes = [0x00];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // f16 +infinity: 0x7C00
        let bytes = [0x00, 0x7C];
        let result = f16_to_f32(&bytes);
        assert!(result.is_infinite() && result > 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // f16 -infinity: 0xFC00
        let bytes = [0x00, 0xFC];
        let result = f16_to_f32(&bytes);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // f16 NaN: 0x7E00
        let bytes = [0x00, 0x7E];
        let result = f16_to_f32(&bytes);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        // f16 -0.0: 0x8000
        let bytes = [0x00, 0x80];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    // ========================================================================
    // NEW: dequantize_q4k_for_stats tests
    // ========================================================================

    #[test]
    fn test_dequantize_q4k_for_stats_short_data() {
        // Data shorter than one block => no output
        let data = vec![0u8; 100]; // Less than 144 bytes (one Q4_K block)
        let result = dequantize_q4k_for_stats(&data, 256);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4k_for_stats_one_block() {
        // One complete Q4_K block: 144 bytes
        let mut data = vec![0u8; 144];
        // Set d (f16 1.0) at bytes 0-1
        data[0] = 0x00;
        data[1] = 0x3C;
        // dmin = 0
        // scales and qs all zero => all values should be d * scale * (0 - 8) = negative
        let result = dequantize_q4k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4k_for_stats_limits_to_num_elements() {
        // Request fewer elements than one block produces
        let data = vec![0u8; 144];
        let result = dequantize_q4k_for_stats(&data, 10);
        assert_eq!(result.len(), 10);
    }

    // ========================================================================
    // NEW: dequantize_q6k_for_stats tests
    // ========================================================================

    #[test]
    fn test_dequantize_q6k_for_stats_short_data() {
        let data = vec![0u8; 100]; // Less than 210 bytes (one Q6_K block)
        let result = dequantize_q6k_for_stats(&data, 256);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_for_stats_one_block() {
        let mut data = vec![0u8; 210];
        // Set d (f16 1.0) at bytes 208-209
        data[208] = 0x00;
        data[209] = 0x3C;
        let result = dequantize_q6k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_limits_to_num_elements() {
        let data = vec![0u8; 210];
        let result = dequantize_q6k_for_stats(&data, 10);
        assert_eq!(result.len(), 10);
    }

    // ========================================================================
    // NEW: ConversionOptions default tests
    // ========================================================================

    #[test]
    fn test_conversion_options_default() {
        let opts = ConversionOptions::default();
        assert!(opts.quantization.is_none());
        assert!(opts.verify);
        assert!(!opts.compute_stats);
        assert!((opts.tolerance - 1e-6).abs() < 1e-10);
        assert!(opts.preserve_metadata);
        assert!(opts.add_provenance);
    }

    #[test]
    fn test_conversion_options_custom() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            verify: false,
            compute_stats: true,
            tolerance: 0.01,
            preserve_metadata: false,
            add_provenance: false,
            tokenizer_path: None,
        };
        assert_eq!(opts.quantization.as_deref(), Some("int8"));
        assert!(!opts.verify);
        assert!(opts.compute_stats);
    }

    // ========================================================================
    // NEW: ConversionPath tests
    // ========================================================================

    #[test]
    fn test_conversion_path_direct() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert_eq!(path.source, FormatType::Gguf);
        assert_eq!(path.target, FormatType::Apr);
        assert!(path.intermediates.is_empty());
    }

    #[test]
    fn test_conversion_path_chain() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        assert_eq!(path.intermediates.len(), 1);
        assert_eq!(path.intermediates[0], FormatType::SafeTensors);
    }

    #[test]
    fn test_conversion_path_steps() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let steps = path.steps();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], FormatType::Gguf);
        assert_eq!(steps[1], FormatType::SafeTensors);
        assert_eq!(steps[2], FormatType::Apr);
    }

    #[test]
    fn test_conversion_path_is_roundtrip() {
        let roundtrip =
            ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
        assert!(roundtrip.is_roundtrip());

        let direct = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert!(!direct.is_roundtrip());

        // Same source/target but no intermediates
        let same = ConversionPath::direct(FormatType::Gguf, FormatType::Gguf);
        assert!(!same.is_roundtrip());
    }

    #[test]
    fn test_conversion_path_has_cycle() {
        // A→B→A is roundtrip but no cycle (middle doesn't repeat)
        let roundtrip =
            ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
        assert!(!roundtrip.has_cycle());

        // A→B→B→C has cycle (B repeated in middle)
        let cyclic = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors, FormatType::SafeTensors],
            FormatType::Apr,
        );
        assert!(cyclic.has_cycle());
    }

    #[test]
    fn test_conversion_path_display() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let display = format!("{path}");
        assert!(display.contains("GGUF"));
        assert!(display.contains("APR"));
    }

    #[test]
    fn test_conversion_path_display_chain() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let display = format!("{path}");
        assert!(display.contains("SafeTensors"));
    }

    // ========================================================================
    // NEW: FormatType additional tests
    // ========================================================================

    #[test]
    fn test_format_type_extension() {
        assert_eq!(FormatType::Gguf.extension(), "gguf");
        assert_eq!(FormatType::SafeTensors.extension(), "safetensors");
        assert_eq!(FormatType::Apr.extension(), "apr");
    }

    #[test]
    fn test_format_type_debug() {
        let fmt = format!("{:?}", FormatType::Gguf);
        assert_eq!(fmt, "Gguf");
    }

    #[test]
    fn test_format_type_clone_eq() {
        let a = FormatType::SafeTensors;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_format_type_from_extension_no_extension() {
        let path = Path::new("model");
        let result = FormatType::from_extension(path);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: Chain format parsing tests
    // ========================================================================

    #[test]
    fn test_chain_format_parsing_st_alias() {
        // "st" is alias for "safetensors"
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["st".to_string(), "apr".to_string()];

        // Will fail due to invalid file, but exercises format parsing
        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_format_parsing_invalid_format() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["pytorch".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{err}");
        assert!(err_str.contains("Unknown format"));
    }

    #[test]
    fn test_chain_format_parsing_case_insensitive() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["GGUF".to_string(), "APR".to_string()];

        // Will fail due to invalid file, but exercises case-insensitive parsing
        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_single_format_too_short() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("at least 2 formats"));
    }

    #[test]
    fn test_chain_with_cycle_detection() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        // GGUF→SafeTensors→SafeTensors→APR has a cycle (SafeTensors repeated)
        let formats = vec![
            "gguf".to_string(),
            "safetensors".to_string(),
            "safetensors".to_string(),
            "apr".to_string(),
        ];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("cycle"));
    }

    #[test]
    fn test_chain_json_output_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), true);
        // Fails due to invalid file, but exercises json=true path
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: Verify intermediate format parsing tests
    // ========================================================================

    #[test]
    fn test_verify_intermediate_gguf() {
        let mut source = NamedTempFile::with_suffix(".safetensors").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "gguf", 1e-5, false);
        assert!(result.is_err()); // Invalid file, but exercises gguf intermediate
    }

    #[test]
    fn test_verify_intermediate_st_alias() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "st", 1e-5, false);
        assert!(result.is_err()); // Invalid file, but exercises st alias
    }

    #[test]
    fn test_verify_intermediate_invalid() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "pytorch", 1e-5, false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Unknown format"));
    }

    #[test]
    fn test_verify_json_output_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "safetensors", 1e-5, true);
        assert!(result.is_err()); // Invalid file, exercises json=true
    }

    // ========================================================================
    // NEW: run_validate_stats missing reference and fingerprints
    // ========================================================================

    #[test]
    fn test_run_validate_stats_no_reference_no_fingerprints() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 3.0, false, false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("--reference") || err_str.contains("--fingerprints"));
    }

    #[test]
    fn test_run_validate_stats_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(file.path(), None, None, 3.0, false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_reference_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(
            file.path(),
            Some(Path::new("/nonexistent/ref.gguf")),
            None,
            3.0,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_fingerprints_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(
            file.path(),
            None,
            Some(Path::new("/nonexistent/fp.json")),
            3.0,
            false,
            false,
        );
        assert!(result.is_err());
    }
