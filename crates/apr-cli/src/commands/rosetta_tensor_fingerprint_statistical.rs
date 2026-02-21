
    // ========================================================================
    // NEW: TensorFingerprint struct tests
    // ========================================================================

    #[test]
    fn test_tensor_fingerprint_clone() {
        let fp = make_fingerprint("test", 0.5, 1.0, 0, 0);
        let fp_clone = fp.clone();
        assert_eq!(fp.name, fp_clone.name);
        assert_eq!(fp.mean, fp_clone.mean);
        assert_eq!(fp.shape, fp_clone.shape);
    }

    #[test]
    fn test_tensor_fingerprint_debug() {
        let fp = make_fingerprint("test", 0.5, 1.0, 0, 0);
        let debug_str = format!("{fp:?}");
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("TensorFingerprint"));
    }

    // ========================================================================
    // NEW: StatisticalAnomaly tests
    // ========================================================================

    #[test]
    fn test_statistical_anomaly_construction() {
        let anomaly = StatisticalAnomaly {
            tensor: "test_tensor".to_string(),
            field: "mean".to_string(),
            expected: 0.5,
            actual: 5.0,
            deviation_sigma: 4.5,
        };
        assert_eq!(anomaly.tensor, "test_tensor");
        assert_eq!(anomaly.field, "mean");
        assert!((anomaly.deviation_sigma - 4.5).abs() < 0.001);
    }

    #[test]
    fn test_statistical_anomaly_debug() {
        let anomaly = StatisticalAnomaly {
            tensor: "t".to_string(),
            field: "std".to_string(),
            expected: 1.0,
            actual: 10.0,
            deviation_sigma: 9.0,
        };
        let debug = format!("{anomaly:?}");
        assert!(debug.contains("StatisticalAnomaly"));
    }

    // ========================================================================
    // NEW: InferenceResult struct tests
    // ========================================================================

    #[test]
    fn test_inference_result_construction() {
        let result = InferenceResult {
            tokens: vec![1, 2, 3],
            logits: vec![0.5, 0.6, 0.7],
            top5: vec![vec![1, 2, 3, 4, 5]],
            output_text: "hello world".to_string(),
        };
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.logits.len(), 3);
        assert_eq!(result.top5.len(), 1);
        assert_eq!(result.output_text, "hello world");
    }

    // ========================================================================
    // NEW: print_fingerprint_diff tests
    // ========================================================================

    #[test]
    fn test_print_fingerprint_diff_no_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_with_anomaly() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 10.0, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_missing_in_b() {
        let fps_a = vec![make_fingerprint("only_in_a", 0.5, 1.0, 0, 0)];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_verbose() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_nan_mismatch() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_cross_format_matching() {
        // GGUF name in A, HF name in B - should still match
        let fps_a = vec![make_fingerprint("blk.0.attn_q.weight", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint(
            "model.layers.0.self_attn.q_proj.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_zero_std() {
        // When std is near zero, mean diff uses absolute value
        let fps_a = vec![make_fingerprint("tensor_a", 0.001, 0.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.0, 0.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: print_fingerprints non-verbose with data
    // ========================================================================

    #[test]
    fn test_print_fingerprints_non_verbose_with_data() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_with_data() {
        let fingerprints = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: VerificationReport tests
    // ========================================================================

    #[test]
    fn test_verification_report_passing() {
        let report = VerificationReport::passing();
        assert!(report.is_equivalent);
        assert_eq!(report.max_diff, 0.0);
        assert_eq!(report.mean_diff, 0.0);
        assert!(report.tensor_diffs.is_empty());
        assert!(report.changed_metadata.is_empty());
        assert!(report.failed_tensors.is_empty());
    }

    #[test]
    fn test_verification_report_passes_with_tolerance() {
        let report = VerificationReport::passing();
        assert!(report.passes_with_tolerance(1e-5));
        assert!(report.passes_with_tolerance(0.0));
    }

    #[test]
    fn test_verification_report_fails_with_tolerance() {
        let mut report = VerificationReport::passing();
        report.max_diff = 0.01;
        assert!(!report.passes_with_tolerance(0.001));
        assert!(report.passes_with_tolerance(0.1));
    }

    #[test]
    fn test_verification_report_fails_with_failed_tensors() {
        let mut report = VerificationReport::passing();
        report.failed_tensors.push("bad_tensor".to_string());
        assert!(!report.passes_with_tolerance(1.0));
    }

    // ========================================================================
    // NEW: RosettaCommands additional variant tests
    // ========================================================================

    #[test]
    fn test_rosetta_commands_inspect_with_hexdump() {
        let cmd = RosettaCommands::Inspect {
            file: PathBuf::from("model.gguf"),
            hexdump: true,
            json: true,
        };
        match cmd {
            RosettaCommands::Inspect { hexdump, json, .. } => {
                assert!(hexdump);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_convert_with_all_options() {
        let cmd = RosettaCommands::Convert {
            source: PathBuf::from("in.safetensors"),
            target: PathBuf::from("out.apr"),
            quantize: Some("int4".to_string()),
            verify: true,
            json: true,
            tokenizer: None,
        };
        match cmd {
            RosettaCommands::Convert {
                quantize,
                verify,
                json,
                ..
            } => {
                assert_eq!(quantize, Some("int4".to_string()));
                assert!(verify);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_fingerprint_with_model_b() {
        let cmd = RosettaCommands::Fingerprint {
            model: PathBuf::from("model_a.gguf"),
            model_b: Some(PathBuf::from("model_b.apr")),
            output: None,
            filter: Some("attn".to_string()),
            verbose: false,
            json: true,
        };
        match cmd {
            RosettaCommands::Fingerprint {
                model_b,
                filter,
                json,
                ..
            } => {
                assert!(model_b.is_some());
                assert_eq!(filter, Some("attn".to_string()));
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    // ========================================================================
    // NEW: run_convert with JSON flag
    // ========================================================================

    #[test]
    fn test_run_convert_json_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, false, true, None);
        // Fails due to invalid file, but exercises json=true path
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: run_fingerprint edge cases
    // ========================================================================

    #[test]
    fn test_run_fingerprint_model_b_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(
            file.path(),
            Some(Path::new("/nonexistent/model_b.gguf")),
            None,
            None,
            false,
            false,
        );
        // Fails because model A is invalid (inspection fails before model_b check)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_verbose_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, true, false);
        assert!(result.is_err()); // Invalid file, but exercises verbose path
    }

    #[test]
    fn test_run_fingerprint_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, false, true);
        assert!(result.is_err()); // Invalid file, but exercises json path
    }

    // ========================================================================
    // NEW: run_compare_inference error paths
    // ========================================================================

    #[test]
    fn test_run_compare_inference_both_not_found() {
        let result = run_compare_inference(
            Path::new("/nonexistent/a.gguf"),
            Path::new("/nonexistent/b.apr"),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compare_inference_json_flag() {
        let result = run_compare_inference(
            Path::new("/nonexistent/a.gguf"),
            Path::new("/nonexistent/b.apr"),
            "test",
            5,
            0.0,
            0.1,
            true,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: load_tensor_data_direct edge cases
    // ========================================================================

    #[test]
    fn test_load_tensor_data_direct_unknown_extension() {
        let mut file = NamedTempFile::with_suffix(".unknown").expect("create temp file");
        file.write_all(b"data").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid apr").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_apr_too_short() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APR\0short").expect("write"); // Less than 40 bytes
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_apr_wrong_magic() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let data = vec![0u8; 50]; // 50 bytes but wrong magic
        file.write_all(&data).expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_safetensors() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_no_extension() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("model");
        std::fs::write(&file_path, b"data").expect("write");
        let result = load_tensor_data_direct(&file_path);
        assert!(result.is_none());
    }

    // ========================================================================
    // NEW: FormatType from_magic tests
    // ========================================================================

    #[test]
    fn test_format_type_from_magic_gguf() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        // GGUF magic: "GGUF" + version bytes
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[3, 0, 0, 0]); // version 3
        std::fs::write(&file_path, &data).expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Gguf);
    }
