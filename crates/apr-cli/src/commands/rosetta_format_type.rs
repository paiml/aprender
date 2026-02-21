
    // ========================================================================
    // FormatType Library Tests
    // ========================================================================

    #[test]
    fn test_format_type_from_extension_gguf() {
        let path = Path::new("model.gguf");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::Gguf);
    }

    #[test]
    fn test_format_type_from_extension_safetensors() {
        let path = Path::new("model.safetensors");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::SafeTensors);
    }

    #[test]
    fn test_format_type_from_extension_apr() {
        let path = Path::new("model.apr");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::Apr);
    }

    #[test]
    fn test_format_type_from_extension_invalid() {
        let path = Path::new("model.pytorch");
        let result = FormatType::from_extension(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_display() {
        assert_eq!(FormatType::Gguf.to_string(), "GGUF");
        assert_eq!(FormatType::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(FormatType::Apr.to_string(), "APR");
    }

    // ========================================================================
    // Run Inspect Tests
    // ========================================================================

    #[test]
    fn test_run_inspect_file_not_found() {
        let result = run_inspect(Path::new("/nonexistent/model.gguf"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run_inspect(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run_inspect(dir.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_with_hexdump_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf").expect("write");

        let result = run_inspect(file.path(), true, false);
        // Should fail (invalid file) but tests hexdump path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_with_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf").expect("write");

        let result = run_inspect(file.path(), false, true);
        // Should fail (invalid file) but tests json path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Convert Tests
    // ========================================================================

    #[test]
    fn test_run_convert_source_not_found() {
        let target = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run_convert(
            Path::new("/nonexistent/model.gguf"),
            target.path(),
            None,
            false,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, false, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_with_quantize_option() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("int8"),
            false,
            false,
            None,
        );
        // Should fail (invalid file) but tests quantize path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_with_verify_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, true, false, None);
        // Should fail (invalid file) but tests verify path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Chain Tests
    // ========================================================================

    #[test]
    fn test_run_chain_source_not_found() {
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];
        let result = run_chain(
            Path::new("/nonexistent/model.gguf"),
            &formats,
            work_dir.path(),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_chain_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_chain_empty_formats() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats: Vec<String> = vec![];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        // Should fail - empty chain
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Verify Tests
    // ========================================================================

    #[test]
    fn test_run_verify_source_not_found() {
        let result = run_verify(
            Path::new("/nonexistent/model.gguf"),
            "safetensors",
            1e-5,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");

        let result = run_verify(source.path(), "safetensors", 1e-5, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_with_different_tolerance() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");

        let result = run_verify(source.path(), "safetensors", 1e-3, false);
        // Should fail (invalid file) but tests tolerance path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_with_apr_intermediate() {
        let mut source = NamedTempFile::with_suffix(".safetensors").expect("create source");
        source.write_all(b"not valid safetensors").expect("write");

        let result = run_verify(source.path(), "apr", 1e-5, false);
        // Should fail (invalid file) but tests apr intermediate path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Fingerprint Tests
    // ========================================================================

    #[test]
    fn test_run_fingerprint_file_not_found() {
        // run_fingerprint(model, model_b, output, filter, verbose, json)
        let result = run_fingerprint(
            Path::new("/nonexistent/model.gguf"),
            None,  // model_b
            None,  // output
            None,  // filter
            false, // verbose
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_with_output() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let output = NamedTempFile::with_suffix(".json").expect("create output");

        let result = run_fingerprint(file.path(), None, Some(output.path()), None, false, false);
        // Should fail (invalid file) but tests output path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_with_filter() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, Some("encoder"), false, false);
        // Should fail (invalid file) but tests filter path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Validate Stats Tests
    // ========================================================================

    #[test]
    fn test_run_validate_stats_file_not_found() {
        // run_validate_stats(model, reference, fingerprints_file, threshold, strict, json)
        let result = run_validate_stats(
            Path::new("/nonexistent/model.gguf"),
            None,  // reference
            None,  // fingerprints_file
            1e-5,  // threshold
            false, // strict
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 1e-5, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_with_reference() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".gguf").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result =
            run_validate_stats(file.path(), Some(ref_file.path()), None, 1e-5, false, false);
        // Should fail (invalid files) but tests reference path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_with_strict() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 1e-5, true, false);
        // Should fail (invalid file) but tests strict path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Diff Tensors Tests
    // ========================================================================

    #[test]
    fn test_run_diff_tensors_model1_not_found() {
        let model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // run_diff_tensors(model_a, model_b, mismatches_only, show_values, filter, json)
        let result = run_diff_tensors(
            Path::new("/nonexistent/model1.gguf"),
            model2.path(),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_model2_not_found() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid gguf").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            Path::new("/nonexistent/model2.gguf"),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_both_invalid() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(model1.path(), model2.path(), false, 0, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_filter() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false,           // mismatches_only
            0,               // show_values
            Some("encoder"), // filter
            false,           // json
        );
        // Should fail (invalid files) but tests filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_mismatches_only() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            true,  // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        // Should fail (invalid files) but tests mismatches_only path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_show_values() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false, // mismatches_only
            10,    // show_values (show 10 sample values)
            None,  // filter
            false, // json
        );
        // Should fail (invalid files) but tests show_values path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_json() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            true,  // json
        );
        // Should fail (invalid files) but tests json path
        assert!(result.is_err());
    }
