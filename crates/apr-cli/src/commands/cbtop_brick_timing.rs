
    #[test]
    fn test_brick_timing_new() {
        let brick = BrickTiming::new("test", 5.0);
        assert_eq!(brick.name, "test");
        assert_eq!(brick.budget_us, 5.0);
        assert_eq!(brick.actual_us, 0.0);
    }

    #[test]
    fn test_brick_timing_gap_factor() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 7.5;
        assert!((brick.gap_factor() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_status() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 4.0;
        assert_eq!(brick.status(), "✅");

        brick.actual_us = 6.0;
        assert_eq!(brick.status(), "❌");
    }

    #[test]
    fn test_brick_timing_add_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(4.0);
        brick.add_sample(6.0);
        assert_eq!(brick.samples.len(), 2);
        assert!((brick.actual_us - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_state_new() {
        let state = PipelineState::new();
        assert_eq!(state.bricks.len(), 7);
        assert_eq!(state.total_layers, 28);
    }

    #[test]
    fn test_pipeline_total_budget() {
        let state = PipelineState::new();
        let total = state.total_budget();
        // Sum: 1.5 + 6.0 + 1.0 + 10.0 + 3.5 + 1.5 + 12.2 = 35.7
        assert!((total - 35.7).abs() < 0.001);
    }

    #[test]
    fn test_view_titles() {
        let titles = View::titles();
        assert_eq!(titles.len(), 5);
        assert!(titles[0].contains("Pipeline"));
    }

    #[test]
    fn test_view_index() {
        assert_eq!(View::Pipeline.index(), 0);
        assert_eq!(View::Budget.index(), 1);
        assert_eq!(View::Histogram.index(), 2);
        assert_eq!(View::Gpu.index(), 3);
        assert_eq!(View::Memory.index(), 4);
    }

    #[test]
    fn test_app_new() {
        let app = App::new(Some("test-model"));
        assert_eq!(app.model_name, "test-model");
        assert_eq!(app.current_view, View::Pipeline);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_app_navigation() {
        let mut app = App::new(None);
        assert_eq!(app.selected_brick, 0);

        app.next_brick();
        assert_eq!(app.selected_brick, 1);

        app.prev_brick();
        assert_eq!(app.selected_brick, 0);

        // Wrap around
        app.prev_brick();
        assert_eq!(app.selected_brick, 6); // 7 bricks, wraps to last
    }

    // === Headless Mode Tests (M001-M010) ===

    #[test]
    fn test_cbtop_config_default() {
        let config = CbtopConfig::default();
        assert!(!config.headless);
        assert!(!config.json);
        assert!(!config.ci);
        assert_eq!(config.warmup, 10);
        assert_eq!(config.iterations, 100);
    }

    #[test]
    fn test_headless_report_generation() {
        let mut pipeline = PipelineState::new();
        // Run some iterations
        for _ in 0..50 {
            pipeline.update_demo();
        }

        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("test-model", &pipeline, &config);

        assert_eq!(report.model, "test-model");
        assert!(!report.timestamp.is_empty());
        assert_eq!(report.brick_scores.len(), 7);
        assert!(report.throughput.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_brick_score_calculation() {
        let mut pipeline = PipelineState::new();
        // Set specific values for testing
        pipeline.bricks[0].actual_us = 1.5; // Exactly at budget
        pipeline.bricks[1].actual_us = 7.2; // 20% over budget (6.0 * 1.2)

        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("test", &pipeline, &config);

        // First brick at budget should score 100
        assert_eq!(report.brick_scores[0].score, 100);
        // Second brick at 1.2x should score ~90
        assert!(report.brick_scores[1].score >= 85 && report.brick_scores[1].score <= 95);
    }

    #[test]
    fn test_ci_threshold_pass() {
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
                score: 95,
                grade: "A".to_string(),
                budget_us: 1.0,
                actual_us: 0.9,
                gap_factor: 0.9,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 95,
                grade: "A+".to_string(),
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

        let config = CbtopConfig {
            ci: true,
            throughput_threshold: Some(400.0),
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        assert!(check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_fail_throughput() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 300.0, // Below 400 threshold
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
                grade: "A+".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 120,
                passed: 100,
                failed: 20,
                blocked: 0,
            },
            status: "FAIL".to_string(),
            ci_result: "red".to_string(),
        };

        let config = CbtopConfig {
            ci: true,
            throughput_threshold: Some(400.0),
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_json_output_format() {
        let report = HeadlessReport {
            model: "test-model".to_string(),
            timestamp: "2026-01-11T00:00:00Z".to_string(),
            hardware: HardwareInfo {
                gpu: "RTX 4090".to_string(),
                cpu: "Ryzen 9".to_string(),
                memory_gb: 64,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0,
                ttft_ms: 1.5,
                cv_percent: 3.2,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "RmsNorm".to_string(),
                score: 100,
                grade: "A".to_string(),
                budget_us: 1.5,
                actual_us: 1.4,
                gap_factor: 0.93,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 95,
                grade: "A+".to_string(),
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

        let json = format_report_as_json(&report);

        // Verify JSON structure
        assert!(json.contains(r#""model": "test-model""#));
        assert!(json.contains(r#""tokens_per_sec": 500.00"#));
        assert!(json.contains(r#""name": "RmsNorm""#));
        assert!(json.contains(r#""score": 100"#));
        assert!(json.contains(r#""ci_result": "green""#));
    }

    #[test]
    fn test_grade_assignment() {
        // Test that grades are assigned correctly based on score
        let mut pipeline = PipelineState::new();
        for _ in 0..10 {
            pipeline.update_demo();
        }

        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("test", &pipeline, &config);

        for brick in &report.brick_scores {
            let expected_grade = match brick.score {
                90..=100 => "A",
                80..=89 => "B",
                70..=79 => "C",
                60..=69 => "D",
                _ => "F",
            };
            assert_eq!(
                brick.grade, expected_grade,
                "Grade mismatch for score {}",
                brick.score
            );
        }
    }

    // === ModelFormat Tests ===

    #[test]
    fn test_model_format_from_path_gguf() {
        use std::path::Path;
        let path = Path::new("model.gguf");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_model_format_from_path_safetensors() {
        use std::path::Path;
        let path = Path::new("model.safetensors");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_model_format_from_path_apr() {
        use std::path::Path;
        let path = Path::new("model.apr");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::Apr));
    }

    #[test]
    fn test_model_format_from_path_unknown() {
        use std::path::Path;
        let path = Path::new("model.bin");
        assert_eq!(ModelFormat::from_path(path), None);
    }

    #[test]
    fn test_model_format_from_path_no_extension() {
        use std::path::Path;
        let path = Path::new("model");
        assert_eq!(ModelFormat::from_path(path), None);
    }

    #[test]
    fn test_model_format_from_path_uppercase() {
        use std::path::Path;
        let path = Path::new("model.GGUF");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_model_format_brick_prefix_gguf() {
        assert_eq!(ModelFormat::Gguf.brick_prefix(), "gguf");
    }

    #[test]
    fn test_model_format_brick_prefix_safetensors() {
        assert_eq!(ModelFormat::SafeTensors.brick_prefix(), "st");
    }

    #[test]
    fn test_model_format_brick_prefix_apr() {
        assert_eq!(ModelFormat::Apr.brick_prefix(), "apr");
    }

    // === compute_brick_score Tests ===

    #[test]
    fn test_compute_brick_score_at_budget() {
        // Actual equals budget, gap = 1.0, score = 100
        assert_eq!(compute_brick_score(5.0, 5.0), 100);
    }

    #[test]
    fn test_compute_brick_score_under_budget() {
        // Actual under budget, gap < 1.0, score = 100
        assert_eq!(compute_brick_score(4.0, 5.0), 100);
    }

    #[test]
    fn test_compute_brick_score_10_percent_over() {
        // gap = 1.1, in 1.0-1.2 range: 100 - (0.1 * 50) = 95
        assert_eq!(compute_brick_score(5.5, 5.0), 95);
    }

    #[test]
    fn test_compute_brick_score_20_percent_over() {
        // gap = 1.2, in 1.0-1.2 range: 100 - (0.2 * 50) = 90
        assert_eq!(compute_brick_score(6.0, 5.0), 90);
    }

    #[test]
    fn test_compute_brick_score_50_percent_over() {
        // gap = 1.5, beyond 1.2: 100 - (0.5 * 100) = 50
        assert_eq!(compute_brick_score(7.5, 5.0), 50);
    }

    #[test]
    fn test_compute_brick_score_double_budget() {
        // gap = 2.0, beyond 1.2: 100 - (1.0 * 100) = 0
        assert_eq!(compute_brick_score(10.0, 5.0), 0);
    }

    #[test]
    fn test_compute_brick_score_extreme_over() {
        // gap = 3.0, beyond 1.2: 100 - (2.0 * 100) = -100, clamped to 0
        assert_eq!(compute_brick_score(15.0, 5.0), 0);
    }

    // === score_to_grade Tests ===

    #[test]
    fn test_score_to_grade_a() {
        assert_eq!(score_to_grade(100), "A");
        assert_eq!(score_to_grade(95), "A");
        assert_eq!(score_to_grade(90), "A");
    }

    #[test]
    fn test_score_to_grade_b() {
        assert_eq!(score_to_grade(89), "B");
        assert_eq!(score_to_grade(85), "B");
        assert_eq!(score_to_grade(80), "B");
    }

    #[test]
    fn test_score_to_grade_c() {
        assert_eq!(score_to_grade(79), "C");
        assert_eq!(score_to_grade(75), "C");
        assert_eq!(score_to_grade(70), "C");
    }

    #[test]
    fn test_score_to_grade_d() {
        assert_eq!(score_to_grade(69), "D");
        assert_eq!(score_to_grade(65), "D");
        assert_eq!(score_to_grade(60), "D");
    }

    #[test]
    fn test_score_to_grade_f() {
        assert_eq!(score_to_grade(59), "F");
        assert_eq!(score_to_grade(50), "F");
        assert_eq!(score_to_grade(0), "F");
    }
