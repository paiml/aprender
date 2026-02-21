
    // ========================================================================
    // format_report_as_json: multiple brick scores
    // ========================================================================

    #[test]
    fn test_json_output_multiple_brick_scores() {
        let report = HeadlessReport {
            model: "multi-brick".to_string(),
            timestamp: "2026-01-12T00:00:00Z".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.0,
                cv_percent: 2.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![
                BrickScore {
                    name: "RmsNorm".to_string(),
                    score: 100,
                    grade: "A".to_string(),
                    budget_us: 1.5,
                    actual_us: 1.0,
                    gap_factor: 0.67,
                },
                BrickScore {
                    name: "Attention".to_string(),
                    score: 85,
                    grade: "B".to_string(),
                    budget_us: 10.0,
                    actual_us: 11.5,
                    gap_factor: 1.15,
                },
                BrickScore {
                    name: "FfnBrick".to_string(),
                    score: 50,
                    grade: "F".to_string(),
                    budget_us: 12.2,
                    actual_us: 18.3,
                    gap_factor: 1.5,
                },
            ],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 78,
                grade: "C".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 137,
                passed: 130,
                failed: 7,
                blocked: 0,
            },
            status: "FAIL".to_string(),
            ci_result: "red".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.contains("\"name\": \"RmsNorm\""));
        assert!(json.contains("\"name\": \"Attention\""));
        assert!(json.contains("\"name\": \"FfnBrick\""));
        assert!(json.contains("\"score\": 100"));
        assert!(json.contains("\"score\": 85"));
        assert!(json.contains("\"score\": 50"));
        assert!(json.contains("\"grade\": \"A\""));
        assert!(json.contains("\"grade\": \"B\""));
        assert!(json.contains("\"grade\": \"F\""));
    }

    // ========================================================================
    // print_report_text: smoke test (no crash)
    // ========================================================================

    #[test]
    fn test_print_report_text_no_panic() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![
                BrickScore {
                    name: "pass_brick".to_string(),
                    score: 100,
                    grade: "A".to_string(),
                    budget_us: 1.0,
                    actual_us: 0.8,
                    gap_factor: 0.8,
                },
                BrickScore {
                    name: "fail_brick".to_string(),
                    score: 50,
                    grade: "F".to_string(),
                    budget_us: 1.0,
                    actual_us: 2.0,
                    gap_factor: 2.0,
                },
            ],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 75,
                grade: "C".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 137,
                passed: 137,
                failed: 0,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        // Should not panic
        print_report_text(&report);
    }

    #[test]
    fn test_print_report_text_empty_bricks_no_panic() {
        let report = HeadlessReport {
            model: "empty".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "none".to_string(),
                cpu: "none".to_string(),
                memory_gb: 0,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 0.0,
                ttft_ms: 0.0,
                cv_percent: 0.0,
                p50_us: 0.0,
                p99_us: 0.0,
            },
            brick_scores: vec![],
            pmat_scores: PmatScores {
                rust_project_score: 0.0,
                tdg_score: 0.0,
                cuda_tdg_score: 0.0,
                brick_score: 0,
                grade: "F".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 0,
                passed: 0,
                failed: 0,
                blocked: 0,
            },
            status: "FAIL".to_string(),
            ci_result: "red".to_string(),
        };

        // Should not panic even with empty bricks
        print_report_text(&report);
    }

    // ========================================================================
    // ModelFormat: Clone/Copy/Debug/PartialEq
    // ========================================================================

    #[test]
    fn test_model_format_clone_copy() {
        let fmt = ModelFormat::Gguf;
        let cloned = fmt;
        assert_eq!(fmt, cloned);

        let fmt2 = ModelFormat::SafeTensors;
        let fmt3 = fmt2;
        assert_eq!(fmt2, fmt3);
    }

    #[test]
    fn test_model_format_debug() {
        let fmt = ModelFormat::Apr;
        let debug_str = format!("{:?}", fmt);
        assert_eq!(debug_str, "Apr");
    }

    #[test]
    fn test_model_format_ne() {
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::SafeTensors, ModelFormat::Apr);
        assert_ne!(ModelFormat::Gguf, ModelFormat::Apr);
    }

    #[test]
    fn test_model_format_from_path_case_insensitive() {
        use std::path::Path;
        assert_eq!(
            ModelFormat::from_path(Path::new("model.SAFETENSORS")),
            Some(ModelFormat::SafeTensors)
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("model.Apr")),
            Some(ModelFormat::Apr)
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("model.GgUf")),
            Some(ModelFormat::Gguf)
        );
    }

    // ========================================================================
    // generate_headless_report_simulated: edge cases
    // ========================================================================

    #[test]
    fn test_headless_report_simulated_no_samples() {
        let pipeline = PipelineState::new(); // No samples added
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("no-samples", &pipeline, &config);

        assert_eq!(report.model, "no-samples");
        assert_eq!(report.brick_scores.len(), 7);
        // With no samples, all actual_us = 0, so all bricks score 100
        for brick in &report.brick_scores {
            assert_eq!(brick.score, 100);
        }
    }

    #[test]
    fn test_headless_report_simulated_status_pass() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 0.8; // 80% of budget
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("passing", &pipeline, &config);
        assert_eq!(report.status, "PASS");
    }

    #[test]
    fn test_headless_report_simulated_status_fail() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 2.0; // 200% of budget
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("failing", &pipeline, &config);
        assert_eq!(report.status, "FAIL");
    }

    #[test]
    fn test_headless_report_simulated_cv_calculation() {
        let mut pipeline = PipelineState::new();
        // Add identical samples to all bricks so CV is well-defined
        for brick in &mut pipeline.bricks {
            brick.add_sample(5.0);
            brick.add_sample(5.0);
            brick.add_sample(5.0);
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("cv-test", &pipeline, &config);
        // With identical samples, CV should be 0
        assert!((report.throughput.cv_percent - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // CbtopConfig: custom construction
    // ========================================================================

    #[test]
    fn test_cbtop_config_custom_values() {
        let config = CbtopConfig {
            model: Some("my-model".to_string()),
            attach: Some("realizar".to_string()),
            model_path: Some(PathBuf::from("/tmp/model.gguf")),
            headless: true,
            json: true,
            output: Some(PathBuf::from("/tmp/out.json")),
            ci: true,
            throughput_threshold: Some(500.0),
            brick_score_threshold: Some(95),
            warmup: 20,
            iterations: 200,
            speculative: true,
            speculation_k: 8,
            draft_model_path: Some(PathBuf::from("/tmp/draft.gguf")),
            concurrent: 4,
            simulated: true,
        };

        assert_eq!(config.model.as_deref(), Some("my-model"));
        assert_eq!(config.attach.as_deref(), Some("realizar"));
        assert!(config.headless);
        assert!(config.json);
        assert!(config.ci);
        assert_eq!(config.warmup, 20);
        assert_eq!(config.iterations, 200);
        assert!(config.speculative);
        assert_eq!(config.speculation_k, 8);
        assert_eq!(config.concurrent, 4);
        assert!(config.simulated);
    }

    // ========================================================================
    // HeadlessReport: Clone trait
    // ========================================================================

    #[test]
    fn test_headless_report_clone() {
        let report = HeadlessReport {
            model: "clone-test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 95,
                grade: "A".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 120,
                passed: 100,
                failed: 20,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let cloned = report.clone();
        assert_eq!(cloned.model, report.model);
        assert_eq!(cloned.status, report.status);
        assert!(
            (cloned.throughput.tokens_per_sec - report.throughput.tokens_per_sec).abs() < 0.001
        );
    }

    // ========================================================================
    // View: Debug/Clone/Copy/PartialEq
    // ========================================================================

    #[test]
    fn test_view_debug() {
        assert_eq!(format!("{:?}", View::Pipeline), "Pipeline");
        assert_eq!(format!("{:?}", View::Budget), "Budget");
        assert_eq!(format!("{:?}", View::Histogram), "Histogram");
        assert_eq!(format!("{:?}", View::Gpu), "Gpu");
        assert_eq!(format!("{:?}", View::Memory), "Memory");
    }

    #[test]
    fn test_view_clone_copy_eq() {
        let v = View::Pipeline;
        let v2 = v;
        assert_eq!(v, v2);
    }

    #[test]
    fn test_view_ne() {
        assert_ne!(View::Pipeline, View::Budget);
        assert_ne!(View::Histogram, View::Gpu);
    }

    // ========================================================================
    // JSON format: pmat_scores and falsification fields
    // ========================================================================

    #[test]
    fn test_json_output_pmat_scores_fields() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 0.0,
                ttft_ms: 0.0,
                cv_percent: 0.0,
                p50_us: 0.0,
                p99_us: 0.0,
            },
            brick_scores: vec![],
            pmat_scores: PmatScores {
                rust_project_score: 173.9,
                tdg_score: 98.1,
                cuda_tdg_score: 95.2,
                brick_score: 100,
                grade: "A+".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 137,
                passed: 135,
                failed: 2,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.contains("\"rust_project_score\": 173.9"));
        assert!(json.contains("\"tdg_score\": 98.1"));
        assert!(json.contains("\"cuda_tdg_score\": 95.2"));
        assert!(json.contains("\"brick_score\": 100"));
        assert!(json.contains("\"grade\": \"A+\""));
        assert!(json.contains("\"total_points\": 137"));
        assert!(json.contains("\"passed\": 135"));
        assert!(json.contains("\"failed\": 2"));
        assert!(json.contains("\"blocked\": 0"));
    }

    // ========================================================================
    // BrickTiming: new constructor field defaults
    // ========================================================================

    #[test]
    fn test_brick_timing_new_initializes_empty_samples() {
        let brick = BrickTiming::new("test", 5.0);
        assert!(brick.samples.is_empty());
    }
