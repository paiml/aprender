
    #[test]
    fn test_tensor_stats_empty() {
        let stats = TensorStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_tensor_stats_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.nan_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_anomaly_detection() {
        let stats = TensorStats {
            count: 100,
            mean: 15.0, // Large mean
            std: 1.0,
            l2_norm: 100.0,
            min: 0.0,
            max: 20.0,
            max_abs: 20.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test_layer");
        assert!(anomalies.iter().any(|a| a.contains("large mean")));
    }

    #[test]
    fn test_anomaly_detection_nan() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 10.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 5,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("NaN")));
    }

    // ========================================================================
    // Additional TensorStats Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.inf_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_tensor_stats_single_value() {
        let data = vec![5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 5.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_tensor_stats_negative_values() {
        let data = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 0.0).abs() < 1e-5);
        assert_eq!(stats.min, -5.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.max_abs, 5.0);
    }

    #[test]
    fn test_tensor_stats_l2_norm() {
        let data = vec![3.0, 4.0]; // 3^2 + 4^2 = 25, sqrt(25) = 5
        let stats = TensorStats::from_slice(&data);

        assert!((stats.l2_norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_clone() {
        let stats = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        let cloned = stats.clone();
        assert_eq!(cloned.count, stats.count);
        assert_eq!(cloned.mean, stats.mean);
    }

    // ========================================================================
    // LayerTrace Tests
    // ========================================================================

    #[test]
    fn test_layer_trace_basic() {
        let trace = LayerTrace {
            name: "attention".to_string(),
            index: Some(0),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        assert_eq!(trace.name, "attention");
        assert_eq!(trace.index, Some(0));
        assert!(trace.anomalies.is_empty());
    }

    #[test]
    fn test_layer_trace_with_anomalies() {
        let trace = LayerTrace {
            name: "ffn".to_string(),
            index: Some(5),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec!["NaN detected".to_string(), "Large mean".to_string()],
        };
        assert_eq!(trace.anomalies.len(), 2);
    }

    #[test]
    fn test_layer_trace_clone() {
        let trace = LayerTrace {
            name: "mlp".to_string(),
            index: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        let cloned = trace.clone();
        assert_eq!(cloned.name, trace.name);
    }

    #[test]
    fn test_layer_trace_serialize() {
        let trace = LayerTrace {
            name: "layer_0".to_string(),
            index: Some(0),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        let json = serde_json::to_string(&trace).expect("serialize");
        assert!(json.contains("layer_0"));
    }

    // ========================================================================
    // Anomaly Detection Tests
    // ========================================================================

    #[test]
    fn test_anomaly_detection_inf() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 10.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 3,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("Inf")));
    }

    #[test]
    fn test_anomaly_detection_zero_std() {
        let stats = TensorStats {
            count: 100,
            mean: 1.0,
            std: 0.0, // Zero variance
            l2_norm: 10.0,
            min: 1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies
            .iter()
            .any(|a| a.contains("zero std") || a.contains("variance")));
    }

    #[test]
    fn test_anomaly_detection_no_anomalies() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 0.5,
            l2_norm: 5.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.is_empty());
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid APR)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, None, false, false, false, false, false);
        // Should fail (is a directory)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            Some("encoder"),
            None,
            false,
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid file) but tests filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_reference() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".apr").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result = run(
            file.path(),
            None,
            Some(ref_file.path()),
            false,
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            true, // json output
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

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
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_payload() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            false,
            false,
            true, // payload
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_diff() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".apr").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result = run(
            file.path(),
            None,
            Some(ref_file.path()),
            false,
            false,
            false,
            true, // diff
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_format_invalid() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid GGUF)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_format_invalid() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid SafeTensors)
        assert!(result.is_err());
    }
