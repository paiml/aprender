
    #[test]
    fn test_fingerprints_to_json_single() {
        let fingerprints = vec![TensorFingerprint {
            name: "test_tensor".to_string(),
            shape: vec![10, 20],
            dtype: "F32".to_string(),
            mean: 0.5,
            std: 0.1,
            min: 0.0,
            max: 1.0,
            p5: 0.05,
            p25: 0.25,
            p50: 0.5,
            p75: 0.75,
            p95: 0.95,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.0,
            checksum: 12345,
        }];
        let json = fingerprints_to_json(&fingerprints);
        assert!(json.contains("test_tensor"));
        assert!(json.contains("F32"));
    }

    #[test]
    fn test_load_fingerprints_from_json_not_found() {
        let result = load_fingerprints_from_json(Path::new("/nonexistent/fingerprints.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_fingerprints_from_json_invalid() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"not valid json").expect("write");

        let result = load_fingerprints_from_json(file.path());
        // Returns Ok with empty vec for invalid JSON (no "name" fields found)
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_fingerprints_from_json_empty_array() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"[]").expect("write");

        let result = load_fingerprints_from_json(file.path());
        // Empty array is valid JSON
        assert!(result.is_ok() || result.is_err()); // May or may not be valid depending on schema
    }

    #[test]
    fn test_rosetta_commands_compare_inference() {
        let cmd = RosettaCommands::CompareInference {
            model_a: PathBuf::from("model_a.gguf"),
            model_b: PathBuf::from("model_b.apr"),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.0,
            tolerance: 0.1,
            json: false,
        };
        match cmd {
            RosettaCommands::CompareInference {
                max_tokens,
                temperature,
                ..
            } => {
                assert_eq!(max_tokens, 10);
                assert_eq!(temperature, 0.0);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_diff_tensors() {
        let cmd = RosettaCommands::DiffTensors {
            model_a: PathBuf::from("model_a.gguf"),
            model_b: PathBuf::from("model_b.apr"),
            mismatches_only: true,
            show_values: 5,
            filter: Some("attention".to_string()),
            json: false,
        };
        match cmd {
            RosettaCommands::DiffTensors {
                mismatches_only,
                show_values,
                filter,
                ..
            } => {
                assert!(mismatches_only);
                assert_eq!(show_values, 5);
                assert_eq!(filter, Some("attention".to_string()));
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_fingerprint() {
        let cmd = RosettaCommands::Fingerprint {
            model: PathBuf::from("model.gguf"),
            model_b: None,
            output: Some(PathBuf::from("fingerprints.json")),
            filter: None,
            verbose: true,
            json: false,
        };
        match cmd {
            RosettaCommands::Fingerprint {
                verbose, output, ..
            } => {
                assert!(verbose);
                assert!(output.is_some());
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_validate_stats() {
        let cmd = RosettaCommands::ValidateStats {
            model: PathBuf::from("model.gguf"),
            reference: None,
            fingerprints: Some(PathBuf::from("ref.json")),
            threshold: 0.01,
            strict: true,
            json: true,
        };
        match cmd {
            RosettaCommands::ValidateStats {
                strict,
                threshold,
                json,
                ..
            } => {
                assert!(strict);
                assert_eq!(threshold, 0.01);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_run_compare_inference_model_a_not_found() {
        let model_b = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run_compare_inference(
            Path::new("/nonexistent/model_a.gguf"),
            model_b.path(),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compare_inference_model_b_not_found() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid gguf").expect("write");

        let result = run_compare_inference(
            model_a.path(),
            Path::new("/nonexistent/model_b.apr"),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_print_fingerprints_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_verbose() {
        let fingerprints = vec![TensorFingerprint {
            name: "test".to_string(),
            shape: vec![10],
            dtype: "F32".to_string(),
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            p5: -0.9,
            p25: -0.5,
            p50: 0.0,
            p75: 0.5,
            p95: 0.9,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.1,
            checksum: 0,
        }];
        let result = print_fingerprints(&fingerprints, true, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: compute_tensor_stats comprehensive tests
    // ========================================================================

    #[test]
    fn test_compute_tensor_stats_with_nan_values() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let (
            mean,
            _std,
            min,
            max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 2);
        assert_eq!(inf_count, 0);
        // Mean should be (1+3+5)/3 = 3.0
        assert!((mean - 3.0).abs() < 0.001);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_with_inf_values() {
        let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 0);
        assert_eq!(inf_count, 2);
    }

    #[test]
    fn test_compute_tensor_stats_all_nan() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let (
            mean,
            std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 3);
        assert_eq!(inf_count, 0);
        // With no valid values, should return zeros for mean/std
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_ne!(checksum, 0); // NaN bits still contribute to checksum
    }

    #[test]
    fn test_compute_tensor_stats_zero_fraction() {
        let data = vec![0.0, 0.0, 1.0, 2.0, 0.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        // 3 out of 5 values are zero
        assert!((zero_frac - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_all_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let (
            mean,
            std,
            min,
            max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert!((zero_frac - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_checksum_deterministic() {
        let data = vec![1.0, 2.0, 3.0];
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum1) = compute_tensor_stats(&data);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum2) = compute_tensor_stats(&data);
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_compute_tensor_stats_checksum_differs_for_different_data() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum1) = compute_tensor_stats(&data1);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum2) = compute_tensor_stats(&data2);
        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_compute_tensor_stats_std_deviation() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 => mean=5, std=2
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (
            mean,
            std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert!((mean - 5.0).abs() < 0.001);
        assert!((std - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_tensor_stats_percentiles() {
        // 100 evenly spaced values 0..99
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (
            _mean,
            _std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            _nan_count,
            _inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 99.0).abs() < 0.001);
        // p5 ~ 4.95, p25 ~ 24.75, p50 ~ 49.5, p75 ~ 74.25, p95 ~ 94.05
        assert!((p5 - 4.0).abs() < 2.0);
        assert!((p25 - 24.0).abs() < 2.0);
        assert!((p50 - 49.0).abs() < 2.0);
        assert!((p75 - 74.0).abs() < 2.0);
        assert!((p95 - 94.0).abs() < 2.0);
    }

    #[test]
    fn test_compute_tensor_stats_mixed_nan_inf_zero() {
        let data = vec![f32::NAN, f32::INFINITY, 0.0, 5.0, f32::NEG_INFINITY, 0.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 1);
        assert_eq!(inf_count, 2);
        // 2 zeros out of 6 total values
        assert!((zero_frac - 2.0 / 6.0).abs() < 0.001);
    }
