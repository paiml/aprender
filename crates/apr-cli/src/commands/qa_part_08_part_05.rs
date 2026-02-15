
    // ========================================================================
    // QaReport: summary generation logic (mirrors run_qa's summary builder)
    // ========================================================================

    /// Summary for all-passed report should be the standard success message.
    #[test]
    fn qa_report_summary_all_passed_message() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::passed(
                "throughput",
                "150 tok/s",
                Some(150.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
        ];
        let passed = gates.iter().all(|g| g.passed);
        let summary = if passed {
            "All QA gates passed".to_string()
        } else {
            let failed: Vec<_> = gates
                .iter()
                .filter(|g| !g.passed && !g.skipped)
                .map(|g| g.name.as_str())
                .collect();
            format!("Failed gates: {}", failed.join(", "))
        };
        assert_eq!(summary, "All QA gates passed");
    }

    /// Summary for a failed report should list the failed gate names.
    #[test]
    fn qa_report_summary_lists_failed_gate_names() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::failed(
                "throughput",
                "too slow",
                Some(5.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "ollama_parity",
                "too slow vs ollama",
                Some(0.1),
                Some(0.2),
                Duration::from_secs(3),
            ),
            GateResult::skipped("gpu_speedup", "no GPU"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(!passed);
        let failed_names: Vec<_> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert_eq!(summary, "Failed gates: throughput, ollama_parity");
    }

    /// Summary for a report where only skipped gates exist (no real failures).
    #[test]
    fn qa_report_summary_skipped_only_is_passed() {
        let gates = vec![
            GateResult::skipped("golden_output", "no model"),
            GateResult::skipped("throughput", "no engine"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed, "All-skipped should be passed");
    }

    // ========================================================================
    // QaConfig: safetensors_path and combined flag states
    // ========================================================================

    /// QaConfig with safetensors_path set to Some should preserve the path.
    #[test]
    fn qa_config_with_safetensors_path() {
        let config = QaConfig {
            safetensors_path: Some(std::path::PathBuf::from("/models/qwen.safetensors")),
            ..Default::default()
        };
        assert_eq!(
            config.safetensors_path.as_deref(),
            Some(std::path::Path::new("/models/qwen.safetensors"))
        );
    }

    /// QaConfig default has skip_contract = false.
    /// Bug class: new skip flag defaulting to true, silently disabling a gate.
    #[test]
    fn qa_config_default_skip_contract_is_false() {
        let config = QaConfig::default();
        assert!(
            !config.skip_contract,
            "skip_contract must default to false to ensure tensor validation runs"
        );
    }

    /// All skip flags set to true simultaneously.
    /// Bug class: skip flag interaction causing unexpected behavior.
    #[test]
    fn qa_config_all_skips_enabled() {
        let config = QaConfig {
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            ..Default::default()
        };
        assert!(config.skip_golden);
        assert!(config.skip_throughput);
        assert!(config.skip_ollama);
        assert!(config.skip_gpu_speedup);
        assert!(config.skip_contract);
        assert!(config.skip_format_parity);
        // Non-skip fields should be default
        assert_eq!(config.iterations, 10);
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
    }

    /// QaConfig with json=true and verbose=true simultaneously.
    /// Bug class: mutually exclusive flags not being properly independent.
    #[test]
    fn qa_config_json_and_verbose_independent() {
        let config = QaConfig {
            json: true,
            verbose: true,
            ..Default::default()
        };
        assert!(config.json);
        assert!(config.verbose);
    }

    /// QaConfig with extreme numeric values should not panic.
    #[test]
    fn qa_config_extreme_thresholds() {
        let config = QaConfig {
            min_tps: f64::MAX,
            min_speedup: 0.0,
            min_gpu_speedup: f64::MIN_POSITIVE,
            iterations: usize::MAX,
            warmup: 0,
            max_tokens: 1,
            ..Default::default()
        };
        assert_eq!(config.min_tps, f64::MAX);
        assert!((config.min_speedup).abs() < f64::EPSILON);
        assert_eq!(config.iterations, usize::MAX);
        assert_eq!(config.warmup, 0);
        assert_eq!(config.max_tokens, 1);
    }

    // ========================================================================
    // GateResult: duration conversion edge cases
    // ========================================================================

    /// Sub-millisecond durations should truncate to 0ms (not round up).
    /// Bug class: using as_millis() which truncates, vs round() which would round.
    #[test]
    fn gate_result_submillisecond_duration_truncates_to_zero() {
        let result = GateResult::passed(
            "fast",
            "blazing fast",
            None,
            None,
            Duration::from_micros(999),
        );
        assert_eq!(
            result.duration_ms, 0,
            "999 microseconds should truncate to 0ms"
        );
    }

    /// Duration at exactly 1ms boundary.
    #[test]
    fn gate_result_exact_one_millisecond() {
        let result = GateResult::passed("gate", "msg", None, None, Duration::from_millis(1));
        assert_eq!(result.duration_ms, 1);
    }

    /// Duration from nanoseconds: 1_500_000 ns = 1ms (truncated from 1.5ms).
    #[test]
    fn gate_result_nanos_to_millis_truncation() {
        let result = GateResult::failed("gate", "msg", None, None, Duration::from_nanos(1_500_000));
        assert_eq!(
            result.duration_ms, 1,
            "1.5ms in nanos should truncate to 1ms"
        );
    }

    // ========================================================================
    // GateResult: message format contracts
    // ========================================================================

    /// Skipped gate message must always be prefixed with "Skipped: ".
    /// Bug class: changing the format string and breaking downstream parsers.
    #[test]
    fn gate_result_skipped_message_format_contract() {
        let reasons = [
            "No GPU available",
            "Ollama not available (start with: ollama serve)",
            "Requires 'inference' feature",
            "Non-GGUF format (F32/F16 lacks fused kernels for Ollama parity)",
            "Skipped by --skip-format-parity",
            "Skipped by --skip-golden",
        ];
        for reason in &reasons {
            let result = GateResult::skipped("test", reason);
            assert!(
                result.message.starts_with("Skipped: "),
                "Skipped message must start with 'Skipped: ', got: '{}'",
                result.message
            );
            assert!(
                result.message.ends_with(reason),
                "Skipped message must end with reason"
            );
        }
    }

    /// Passed gate with value and threshold: values should appear in the struct.
    #[test]
    fn gate_result_passed_preserves_value_and_threshold() {
        let result = GateResult::passed(
            "throughput",
            "150.0 tok/s >= 100.0 tok/s",
            Some(150.5),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert_eq!(result.value, Some(150.5));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Failed gate with value and threshold: values should appear in the struct.
    #[test]
    fn gate_result_failed_preserves_value_and_threshold() {
        let result = GateResult::failed(
            "ollama_parity",
            "0.15x < 0.2x",
            Some(0.15),
            Some(0.2),
            Duration::from_secs(5),
        );
        assert_eq!(result.value, Some(0.15));
        assert_eq!(result.threshold, Some(0.2));
        assert!(!result.passed);
    }

    // ========================================================================
    // GateResult: JSON deserialization edge cases
    // ========================================================================

    /// Deserializing JSON with explicit null for value/threshold should produce None.
    /// Bug class: serde treating null as missing vs explicit null differently.
    #[test]
    fn gate_result_deserialize_explicit_null_values() {
        let json = r#"{
            "name": "throughput",
            "passed": true,
            "message": "ok",
            "value": null,
            "threshold": null,
            "duration_ms": 100,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize with nulls");
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    /// Deserializing JSON with missing optional fields (value/threshold omitted).
    #[test]
    fn gate_result_deserialize_missing_optional_fields() {
        let json = r#"{
            "name": "contract",
            "passed": false,
            "message": "validation error",
            "duration_ms": 50,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize missing optionals");
        assert_eq!(result.name, "contract");
        assert!(!result.passed);
        assert!(result.value.is_none());
        assert!(result.threshold.is_none());
    }

    // ========================================================================
    // QaReport: empty gates edge case
    // ========================================================================

    /// A report with zero gates should still be valid and serializable.
    /// Bug class: division by zero or index-out-of-bounds on empty gate list.
    #[test]
    fn qa_report_empty_gates_is_valid() {
        let report = QaReport {
            model: "empty.gguf".to_string(),
            passed: true,
            gates: vec![],
            total_duration_ms: 0,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "No gates run".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        assert!(report.passed);
        assert!(report.gates.is_empty());
        let json = serde_json::to_string(&report).expect("serialize empty report");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize empty report");
        assert!(restored.gates.is_empty());
    }

    /// A report with many gates (stress test for serialization).
    #[test]
    fn qa_report_many_gates_serialization() {
        let gates: Vec<GateResult> = (0..100)
            .map(|i| {
                GateResult::passed(
                    &format!("gate_{i}"),
                    &format!("Gate {i} passed"),
                    Some(i as f64),
                    Some(0.0),
                    Duration::from_millis(i as u64),
                )
            })
            .collect();
        let report = QaReport {
            model: "stress.gguf".to_string(),
            passed: true,
            gates,
            total_duration_ms: 4950,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string(&report).expect("serialize many gates");
        let restored: QaReport = serde_json::from_str(&json).expect("deserialize many gates");
        assert_eq!(restored.gates.len(), 100);
    }

    // ========================================================================
    // detect_ollama_model_from_path: format string contract
    // ========================================================================

    /// Output format must always be "qwen2.5-coder:{size}".
    /// Bug class: format string mismatch breaking Ollama API calls.
    #[test]
    fn detect_ollama_model_output_format_contract() {
        let test_paths = [
            "/tmp/model-0.5b.gguf",
            "/tmp/model-1.5b.gguf",
            "/tmp/model-3b.gguf",
            "/tmp/model-7b.gguf",
            "/tmp/model-14b.gguf",
            "/tmp/model-32b.gguf",
        ];
        for path in &test_paths {
            let model = detect_ollama_model_from_path(Path::new(path));
            assert!(
                model.starts_with("qwen2.5-coder:"),
                "Model tag must start with 'qwen2.5-coder:', got: {model}"
            );
            let size = model.strip_prefix("qwen2.5-coder:").expect("strip prefix");
            assert!(
                ["0.5b", "1.5b", "3b", "7b", "14b", "32b"].contains(&size),
                "Size must be one of the known sizes, got: {size}"
            );
        }
    }

    /// Empty filename (just a directory path) should not panic.
    #[test]
    fn detect_ollama_model_directory_path() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/models/"));
        // No filename -> empty string -> falls to file size heuristic -> metadata fails -> "7b"
        assert!(
            model.starts_with("qwen2.5-coder:"),
            "Directory path should produce valid tag: {model}"
        );
    }

    // ========================================================================
    // run_qa summary builder: failed_gates name collection
    // ========================================================================

    /// Multiple failed gates should all appear in the summary, comma-separated.
    #[test]
    fn failed_gates_summary_multiple_failures() {
        let gates = vec![
            GateResult::failed("golden_output", "wrong", None, None, Duration::from_secs(1)),
            GateResult::failed(
                "throughput",
                "slow",
                Some(1.0),
                Some(100.0),
                Duration::from_secs(2),
            ),
            GateResult::failed(
                "tensor_contract",
                "violations",
                Some(5.0),
                Some(0.0),
                Duration::from_secs(1),
            ),
            GateResult::skipped("ollama_parity", "not available"),
            GateResult::passed(
                "gpu_speedup",
                "ok",
                Some(3.0),
                Some(2.0),
                Duration::from_secs(4),
            ),
        ];
        let failed_names: Vec<&str> = gates
            .iter()
            .filter(|g| !g.passed && !g.skipped)
            .map(|g| g.name.as_str())
            .collect();
        assert_eq!(failed_names.len(), 3);
        let summary = format!("Failed gates: {}", failed_names.join(", "));
        assert!(summary.contains("golden_output"));
        assert!(summary.contains("throughput"));
        assert!(summary.contains("tensor_contract"));
        assert!(
            !summary.contains("ollama_parity"),
            "Skipped gate should not appear in failures"
        );
        assert!(
            !summary.contains("gpu_speedup"),
            "Passed gate should not appear in failures"
        );
    }
