
    /// Simulate how run() builds QaConfig with Some parameters (overrides).
    #[test]
    fn run_config_building_some_overrides_defaults() {
        let min_tps: Option<f64> = Some(50.0);
        let min_speedup: Option<f64> = Some(1.5);
        let min_gpu_speedup: Option<f64> = Some(3.0);
        let config = QaConfig {
            min_tps: min_tps.unwrap_or(100.0),
            min_speedup: min_speedup.unwrap_or(0.2),
            min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0),
            ..Default::default()
        };
        assert!((config.min_tps - 50.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 1.5).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 3.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // NEW: print_gate_result duration formatting (line 1490-1491)
    // ========================================================================

    /// Verify print_gate_result handles zero duration without division errors.
    #[test]
    fn print_gate_result_zero_duration_formatting() {
        let result = GateResult::passed(
            "tensor_contract",
            "0 tensors",
            Some(0.0),
            Some(0.0),
            Duration::from_millis(0),
        );
        // Should print "Duration: 0.00s" without panic
        print_gate_result(&result);
    }

    /// Verify print_gate_result handles large duration values.
    #[test]
    fn print_gate_result_large_duration_formatting() {
        let result = GateResult::passed(
            "throughput",
            "ok",
            Some(100.0),
            Some(50.0),
            Duration::from_secs(3600),
        );
        // duration_ms = 3600000, format as 3600000.0/1000.0 = 3600.00s
        assert_eq!(result.duration_ms, 3_600_000);
        print_gate_result(&result);
    }

    /// Verify print_gate_result formats duration_ms correctly for sub-second durations.
    #[test]
    fn print_gate_result_subsecond_duration_formatting() {
        let result = GateResult::passed(
            "golden_output",
            "2 cases passed",
            Some(2.0),
            Some(2.0),
            Duration::from_millis(250),
        );
        assert_eq!(result.duration_ms, 250);
        // 250ms / 1000.0 = 0.25s -> should print "Duration: 0.25s"
        print_gate_result(&result);
    }

    // ========================================================================
    // NEW: QaReport with all 6 canonical gates
    // ========================================================================

    /// Verify a report with all 6 canonical gates can be serialized/deserialized.
    #[test]
    fn qa_report_all_six_canonical_gates_roundtrip() {
        let report = QaReport {
            model: "/models/qwen2-0.5b-q4_k.gguf".to_string(),
            passed: false,
            gates: vec![
                GateResult::passed(
                    "tensor_contract",
                    "50 tensors ok",
                    Some(50.0),
                    Some(0.0),
                    Duration::from_millis(100),
                ),
                GateResult::passed(
                    "golden_output",
                    "2 test cases passed",
                    Some(2.0),
                    Some(2.0),
                    Duration::from_millis(5000),
                ),
                GateResult::failed(
                    "throughput",
                    "5 tok/s < 100 tok/s",
                    Some(5.0),
                    Some(100.0),
                    Duration::from_millis(10000),
                ),
                GateResult::skipped("ollama_parity", "Ollama not available"),
                GateResult::skipped("gpu_speedup", "CUDA not available"),
                GateResult::skipped("format_parity", "No --safetensors-path provided"),
            ],
            total_duration_ms: 15100,
            timestamp: "2026-02-07T12:00:00Z".to_string(),
            summary: "Failed gates: throughput".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.gates.len(), 6);
        assert!(!restored.passed);
        // Verify each gate type
        assert!(restored.gates[0].passed && !restored.gates[0].skipped);
        assert!(restored.gates[1].passed && !restored.gates[1].skipped);
        assert!(!restored.gates[2].passed && !restored.gates[2].skipped);
        assert!(restored.gates[3].skipped);
        assert!(restored.gates[4].skipped);
        assert!(restored.gates[5].skipped);
    }

    // ========================================================================
    // NEW: GateResult message content validation
    // ========================================================================

    /// Passed gate message should be stored verbatim.
    #[test]
    fn gate_result_passed_message_stored_verbatim() {
        let msg = "150.0 tok/s >= 100.0 tok/s threshold";
        let result = GateResult::passed(
            "throughput",
            msg,
            Some(150.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.message, msg);
    }

    /// Failed gate message should be stored verbatim.
    #[test]
    fn gate_result_failed_message_stored_verbatim() {
        let msg = "5.0 tok/s < 100.0 tok/s threshold";
        let result = GateResult::failed(
            "throughput",
            msg,
            Some(5.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.message, msg);
    }

    /// Skipped gate message format: "Skipped: {reason}".
    #[test]
    fn gate_result_skipped_message_exact_format() {
        let result = GateResult::skipped("gpu_speedup", "CUDA not available");
        assert_eq!(result.message, "Skipped: CUDA not available");
    }

    /// Empty reason for skipped gate should produce "Skipped: ".
    #[test]
    fn gate_result_skipped_empty_reason() {
        let result = GateResult::skipped("test", "");
        assert_eq!(result.message, "Skipped: ");
        assert!(result.skipped);
    }

    /// Empty name for gate result should be stored as empty string.
    #[test]
    fn gate_result_empty_name() {
        let result = GateResult::passed("", "ok", None, None, Duration::from_secs(0));
        assert_eq!(result.name, "");
        assert!(result.passed);
    }

    // ========================================================================
    // NEW: GateResult negative and zero values
    // ========================================================================

    /// Negative value in a gate result (e.g., from subtraction error).
    #[test]
    fn gate_result_negative_value() {
        let result = GateResult::failed(
            "gpu_speedup",
            "-0.5x slower",
            Some(-0.5),
            Some(2.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(-0.5));
        assert!(!result.passed);
    }

    /// Zero value should be representable.
    #[test]
    fn gate_result_zero_value() {
        let result = GateResult::failed(
            "throughput",
            "0 tok/s",
            Some(0.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(0.0));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Very small positive value (epsilon-level).
    #[test]
    fn gate_result_epsilon_value() {
        let result = GateResult::passed(
            "throughput",
            "barely passing",
            Some(f64::MIN_POSITIVE),
            Some(0.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(f64::MIN_POSITIVE));
        assert!(result.passed);
    }

    // ========================================================================
    // NEW: QaReport timestamp and model path edge cases
    // ========================================================================

    /// Report with Unicode characters in model path.
    #[test]
    fn qa_report_unicode_model_path() {
        let report = QaReport {
            model: "/modelos/modelo_espa\u{00f1}ol.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string(&report).expect("serialize unicode path");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize unicode path");
        assert!(restored.model.contains("espa\u{00f1}ol"));
    }

    /// Report with very long model path.
    #[test]
    fn qa_report_long_model_path() {
        let long_path = format!("/very/{}/model.gguf", "deep/".repeat(100));
        let report = QaReport {
            model: long_path.clone(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string(&report).expect("serialize long path");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize long path");
        assert_eq!(restored.model, long_path);
    }

    /// Report with empty model path.
    #[test]
    fn qa_report_empty_model_path() {
        let report = QaReport {
            model: String::new(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "ok".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string(&report).expect("serialize empty model");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize empty model");
        assert!(restored.model.is_empty());
    }

    // ========================================================================
    // NEW: QaReport aggregate pass/fail with mixed states
    // ========================================================================

    /// All gates failed: report.passed should be false and all gates listed.
    #[test]
    fn qa_report_all_gates_failed() {
        let gates = vec![
            GateResult::failed(
                "tensor_contract",
                "violations",
                Some(5.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::failed(
                "golden_output",
                "wrong output",
                None,
                None,
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "throughput",
                "too slow",
                Some(1.0),
                Some(100.0),
                Duration::from_secs(3),
            ),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(!passed);
        let failed_names: Vec<&str> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        assert_eq!(failed_names.len(), 3);
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert_eq!(
            summary,
            "Failed gates: tensor_contract, golden_output, throughput"
        );
    }

    /// Single passed gate among skipped: overall pass.
    #[test]
    fn qa_report_single_pass_rest_skipped() {
        let gates = vec![
            GateResult::passed(
                "tensor_contract",
                "ok",
                Some(10.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("golden_output", "no engine"),
            GateResult::skipped("throughput", "no engine"),
            GateResult::skipped("ollama_parity", "not available"),
            GateResult::skipped("gpu_speedup", "no GPU"),
            GateResult::skipped("format_parity", "no path"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed);
    }

    // ========================================================================
    // NEW: print_gate_result exercises for each known gate display name
    // ========================================================================

    /// Exercise print_gate_result with "format_parity" gate -- one of the display names.
    #[test]
    fn print_gate_result_format_parity_display_name() {
        let result = GateResult::passed(
            "format_parity",
            "GGUF argmax=42 == SafeTensors argmax=42",
            Some(42.0),
            Some(42.0),
            Duration::from_millis(8000),
        );
        print_gate_result(&result);
    }

    /// Exercise print_gate_result with "gpu_speedup" gate in failed state.
    #[test]
    fn print_gate_result_gpu_speedup_failed() {
        let result = GateResult::failed(
            "gpu_speedup",
            "GPU 1.2x faster than CPU < 2.0x threshold",
            Some(1.2),
            Some(2.0),
            Duration::from_millis(15000),
        );
        print_gate_result(&result);
    }

    // ========================================================================
    // NEW: QaConfig with zero and extreme iteration/token values
    // ========================================================================

    /// Zero iterations and warmup should be representable.
    #[test]
    fn qa_config_zero_iterations_and_warmup() {
        let config = QaConfig {
            iterations: 0,
            warmup: 0,
            max_tokens: 0,
            ..Default::default()
        };
        assert_eq!(config.iterations, 0);
        assert_eq!(config.warmup, 0);
        assert_eq!(config.max_tokens, 0);
    }

    /// Large max_tokens value.
    #[test]
    fn qa_config_large_max_tokens() {
        let config = QaConfig {
            max_tokens: 1_000_000,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, 1_000_000);
    }

    // ========================================================================
    // NEW: GateResult serialization with special f64 values
    // ========================================================================

    /// Serialize gate with very large value.
    #[test]
    fn gate_result_serialize_large_value() {
        let result = GateResult::passed(
            "throughput",
            "very fast",
            Some(999_999.99),
            Some(100.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize large value");
        assert!(json.contains("999999.99"));
    }

    /// Serialize gate with very small (near-zero) positive value.
    #[test]
    fn gate_result_serialize_tiny_value() {
        let result = GateResult::failed(
            "throughput",
            "basically zero",
            Some(0.000_001),
            Some(100.0),
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize tiny value");
        // serde_json will serialize this as something like 1e-6 or 0.000001
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize tiny value");
        assert!((restored.value.expect("has value") - 0.000_001).abs() < 1e-10);
    }
