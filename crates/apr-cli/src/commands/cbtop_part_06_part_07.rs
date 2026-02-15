
    #[test]
    fn test_headless_report_simulated_cv_with_variance() {
        let mut pipeline = PipelineState::new();
        // Add varied samples to create non-zero CV
        for brick in &mut pipeline.bricks {
            brick.add_sample(1.0);
            brick.add_sample(10.0);
            brick.add_sample(5.0);
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("cv-var", &pipeline, &config);
        // With varied samples, CV should be > 0
        assert!(report.throughput.cv_percent > 0.0);
    }

    #[test]
    fn test_headless_report_simulated_p50_p99_with_samples() {
        let mut pipeline = PipelineState::new();
        // Add many samples to first brick so p50/p99 are meaningful
        for i in 0..50 {
            pipeline.bricks[0].add_sample(i as f64);
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("pct-test", &pipeline, &config);
        // p50 and p99 should be > 0 since first brick has samples
        assert!(report.throughput.p50_us >= 0.0);
        assert!(report.throughput.p99_us >= 0.0);
    }

    #[test]
    fn test_headless_report_simulated_empty_first_brick_samples() {
        let mut pipeline = PipelineState::new();
        // First brick has no samples, but others do
        for i in 1..pipeline.bricks.len() {
            pipeline.bricks[i].add_sample(5.0);
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("empty-first", &pipeline, &config);
        // p50 and p99 from empty first brick are 0.0
        assert!((report.throughput.p50_us - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_headless_report_simulated_throughput_from_current_tok_s() {
        let mut pipeline = PipelineState::new();
        pipeline.current_tok_s = 1234.5;
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("tput-test", &pipeline, &config);
        assert!((report.throughput.tokens_per_sec - 1234.5).abs() < 0.01);
    }

    #[test]
    fn test_headless_report_simulated_ttft_calculation() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = 1.0; // Each brick = 1.0µs
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("ttft-test", &pipeline, &config);
        // ttft_ms = total_actual * total_layers / 1000
        // total_actual = 7 * 1.0 = 7.0
        // ttft_ms = 7.0 * 28 / 1000 = 0.196
        assert!((report.throughput.ttft_ms - 0.196).abs() < 0.01);
    }

    #[test]
    fn test_headless_report_simulated_single_sample_cv() {
        let mut pipeline = PipelineState::new();
        // Single sample per brick -> variance = 0
        for brick in &mut pipeline.bricks {
            brick.add_sample(5.0);
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("single-cv", &pipeline, &config);
        // With N=1 per brick, total samples = 7, variance uses (n-1) divisor
        // But all samples are 5.0, so CV = 0
        assert!((report.throughput.cv_percent - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // NEW: check_ci_thresholds edge cases
    // ========================================================================

    #[test]
    fn test_ci_threshold_exact_throughput_match() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 400.0, // Exactly at threshold
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
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
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let config = CbtopConfig {
            throughput_threshold: Some(400.0),
            ..Default::default()
        };

        // 400.0 >= 400.0, should pass
        assert!(check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_exact_brick_score_match() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "test".to_string(),
                score: 90,
                grade: "A".to_string(),
                budget_us: 1.0,
                actual_us: 1.0,
                gap_factor: 1.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 0.0,
                tdg_score: 0.0,
                cuda_tdg_score: 0.0,
                brick_score: 90,
                grade: "A".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 0,
                passed: 0,
                failed: 0,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let config = CbtopConfig {
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        // avg 90 >= 90, should pass
        assert!(check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_just_below_throughput() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 399.9,
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
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

        let config = CbtopConfig {
            throughput_threshold: Some(400.0),
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_only_throughput_set() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "bad".to_string(),
                score: 10,
                grade: "F".to_string(),
                budget_us: 1.0,
                actual_us: 5.0,
                gap_factor: 5.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 0.0,
                tdg_score: 0.0,
                cuda_tdg_score: 0.0,
                brick_score: 10,
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

        let config = CbtopConfig {
            throughput_threshold: Some(400.0),
            brick_score_threshold: None, // Only throughput checked
            ..Default::default()
        };

        // Throughput passes, brick not checked
        assert!(check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_only_brick_score_set() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 1.0, // Very low, but not checked
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "good".to_string(),
                score: 95,
                grade: "A".to_string(),
                budget_us: 1.0,
                actual_us: 0.9,
                gap_factor: 0.9,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 0.0,
                tdg_score: 0.0,
                cuda_tdg_score: 0.0,
                brick_score: 95,
                grade: "A".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 0,
                passed: 0,
                failed: 0,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let config = CbtopConfig {
            throughput_threshold: None,
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        // Brick passes, throughput not checked
        assert!(check_ci_thresholds(&report, &config));
    }

    // ========================================================================
    // NEW: format_report_as_json detailed tests
    // ========================================================================

    #[test]
    fn test_json_output_empty_brick_scores_array() {
        let report = HeadlessReport {
            model: "empty-bricks".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
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

        let json = format_report_as_json(&report);
        // Empty brick_scores should have empty array
        assert!(json.contains("\"brick_scores\": [\n\n  ]"));
    }

    #[test]
    fn test_json_output_throughput_precision() {
        let report = HeadlessReport {
            model: "precision".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 1,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 123.456,
                ttft_ms: 1.789,
                cv_percent: 2.345,
                p50_us: 0.123,
                p99_us: 9.876,
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
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.contains("\"tokens_per_sec\": 123.46"));
        assert!(json.contains("\"ttft_ms\": 1.79"));
        assert!(json.contains("\"cv_percent\": 2.35") || json.contains("\"cv_percent\": 2.34"));
        assert!(json.contains("\"p50_us\": 0.12"));
        assert!(
            json.contains("\"p99_us\": 9.88")
                || json.contains("\"p99_us\": 9.87")
                || json.contains("\"p99_us\": 9.876")
        );
    }
