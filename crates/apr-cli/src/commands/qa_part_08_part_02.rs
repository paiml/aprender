
    // ========================================================================
    // QaConfig Tests
    // ========================================================================

    #[test]
    fn test_qa_config_default() {
        let config = QaConfig::default();
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 0.2).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 2.0).abs() < f64::EPSILON);
        assert!(!config.skip_golden);
        assert!(!config.skip_throughput);
        assert!(!config.skip_ollama);
        assert!(!config.skip_gpu_speedup);
        assert!(!config.skip_format_parity);
        assert!(config.safetensors_path.is_none());
    }

    #[test]
    fn test_qa_config_default_iterations() {
        let config = QaConfig::default();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.warmup, 3);
        assert_eq!(config.max_tokens, 32);
    }

    #[test]
    fn test_qa_config_default_output_flags() {
        let config = QaConfig::default();
        assert!(!config.json);
        assert!(!config.verbose);
    }

    #[test]
    fn test_qa_config_clone() {
        let config = QaConfig {
            min_tps: 50.0,
            skip_golden: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert!((cloned.min_tps - 50.0).abs() < f64::EPSILON);
        assert!(cloned.skip_golden);
    }

    #[test]
    fn test_qa_config_debug() {
        let config = QaConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("QaConfig"));
        assert!(debug.contains("min_tps"));
    }

    // ========================================================================
    // GateResult Tests
    // ========================================================================

    #[test]
    fn test_gate_result_passed() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            Some(150.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.passed);
        assert!(!result.skipped);
        assert_eq!(result.name, "test_gate");
    }

    #[test]
    fn test_gate_result_passed_duration() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            Some(150.0),
            Some(100.0),
            Duration::from_millis(1500),
        );
        assert_eq!(result.duration_ms, 1500);
    }

    #[test]
    fn test_gate_result_passed_no_value() {
        let result = GateResult::passed(
            "test_gate",
            "Test passed",
            None,
            None,
            Duration::from_secs(1),
        );
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    #[test]
    fn test_gate_result_failed() {
        let result = GateResult::failed(
            "test_gate",
            "Test failed",
            Some(50.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(!result.passed);
        assert!(!result.skipped);
    }

    #[test]
    fn test_gate_result_failed_message() {
        let result = GateResult::failed(
            "throughput",
            "50 tok/s < 100 tok/s",
            Some(50.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.message.contains("50"));
        assert!(result.message.contains("100"));
    }

    #[test]
    fn test_gate_result_skipped() {
        let result = GateResult::skipped("test_gate", "No GPU available");
        assert!(result.passed); // Skipped doesn't fail
        assert!(result.skipped);
    }

    #[test]
    fn test_gate_result_skipped_message() {
        let result = GateResult::skipped("gpu_speedup", "GPU not available");
        assert!(result.message.contains("Skipped"));
        assert!(result.message.contains("GPU not available"));
    }

    #[test]
    fn test_gate_result_skipped_no_duration() {
        let result = GateResult::skipped("test", "reason");
        assert_eq!(result.duration_ms, 0);
    }

    #[test]
    fn test_gate_result_clone() {
        let result = GateResult::passed("test", "ok", Some(100.0), None, Duration::from_secs(1));
        let cloned = result.clone();
        assert_eq!(cloned.name, result.name);
        assert_eq!(cloned.passed, result.passed);
    }

    #[test]
    fn test_gate_result_debug() {
        let result = GateResult::passed("test", "ok", None, None, Duration::from_secs(0));
        let debug = format!("{result:?}");
        assert!(debug.contains("GateResult"));
    }

    #[test]
    fn test_gate_result_serialize() {
        let result = GateResult::passed(
            "throughput",
            "100 tok/s",
            Some(100.0),
            Some(60.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("throughput"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_gate_result_deserialize() {
        let json =
            r#"{"name":"test","passed":true,"message":"ok","duration_ms":1000,"skipped":false}"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.name, "test");
        assert!(result.passed);
    }

    // ========================================================================
    // QaReport Tests
    // ========================================================================

    #[test]
    fn test_qa_report_serialization() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![GateResult::passed(
                "throughput",
                "100 tok/s",
                Some(100.0),
                Some(60.0),
                Duration::from_secs(5),
            )],
            total_duration_ms: 5000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "All gates passed".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };

        let json = serde_json::to_string(&report).expect("serialization failed");
        assert!(json.contains("throughput"));
        assert!(json.contains("passed"));
    }

    #[test]
    fn test_qa_report_deserialization() {
        let json = r#"{
            "model": "test.gguf",
            "passed": true,
            "gates": [],
            "total_duration_ms": 1000,
            "timestamp": "2026-01-01T00:00:00Z",
            "summary": "All passed"
        }"#;
        let report: QaReport = serde_json::from_str(json).expect("deserialize");
        assert_eq!(report.model, "test.gguf");
        assert!(report.passed);
    }

    #[test]
    fn test_qa_report_failed() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: false,
            gates: vec![GateResult::failed(
                "throughput",
                "50 tok/s < 100 tok/s",
                Some(50.0),
                Some(100.0),
                Duration::from_secs(5),
            )],
            total_duration_ms: 5000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "1 gate failed".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        assert!(!report.passed);
        assert_eq!(report.gates.len(), 1);
    }

    #[test]
    fn test_qa_report_multiple_gates() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![
                GateResult::passed("golden", "ok", None, None, Duration::from_secs(1)),
                GateResult::passed(
                    "throughput",
                    "ok",
                    Some(100.0),
                    Some(60.0),
                    Duration::from_secs(2),
                ),
                GateResult::skipped("ollama", "skipped"),
            ],
            total_duration_ms: 3000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        assert_eq!(report.gates.len(), 3);
    }

    #[test]
    fn test_qa_report_clone() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 1000,
            timestamp: "2026-01-15T00:00:00Z".to_string(),
            summary: "ok".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let cloned = report.clone();
        assert_eq!(cloned.model, report.model);
    }

    #[test]
    fn test_qa_report_debug() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 1000,
            timestamp: "now".to_string(),
            summary: "ok".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("QaReport"));
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false, // skip_capability
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_model() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false, // skip_capability
        );
        // Should fail (invalid GGUF)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_custom_thresholds() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            Some(50.0), // min_tps
            Some(1.5),  // min_speedup
            Some(3.0),  // min_gpu_speedup
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            5,
            2,
            16,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false, // skip_capability
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_all_skips() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            true, // skip_golden
            true, // skip_throughput
            true, // skip_ollama
            true, // skip_gpu_speedup
            true, // skip_contract
            true, // skip_format_parity
            true, // skip_ptx_parity
            None,
            10,
            3,
            32,
            false,
            false,
            None,
            None,
            None,
            true, // skip_gpu_state
            true, // skip_metadata
            true, // skip_capability
        );
        // When all gates are skipped, the QA passes (skipped gates don't fail)
        assert!(result.is_ok());
    }
