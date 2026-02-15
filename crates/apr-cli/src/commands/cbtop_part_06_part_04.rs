
    #[test]
    fn test_app_tick_updates_demo() {
        let mut app = App::new(None);
        let initial_tokens = app.pipeline.tokens_generated;
        app.tick();
        assert_eq!(app.pipeline.tokens_generated, initial_tokens + 1);
    }

    #[test]
    fn test_app_model_name_default() {
        let app = App::new(None);
        assert_eq!(app.model_name, "qwen2.5-coder-1.5b");
    }

    #[test]
    fn test_app_model_name_custom() {
        let app = App::new(Some("custom-model"));
        assert_eq!(app.model_name, "custom-model");
    }

    // ========================================================================
    // BrickTiming::percent_of_budget comprehensive tests
    // ========================================================================

    #[test]
    fn test_brick_percent_of_budget_under() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 5.0;
        assert_eq!(brick.percent_of_budget(), 50);
    }

    #[test]
    fn test_brick_percent_of_budget_at() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 10.0;
        assert_eq!(brick.percent_of_budget(), 100);
    }

    #[test]
    fn test_brick_percent_of_budget_over() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 15.0;
        assert_eq!(brick.percent_of_budget(), 150);
    }

    #[test]
    fn test_brick_percent_of_budget_capped_at_200() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 30.0; // 300%, but capped at 200
        assert_eq!(brick.percent_of_budget(), 200);
    }

    #[test]
    fn test_brick_percent_of_budget_zero_budget() {
        let mut brick = BrickTiming::new("test", 0.0);
        brick.actual_us = 5.0;
        // Zero budget returns 100 as defensive guard
        assert_eq!(brick.percent_of_budget(), 100);
    }

    #[test]
    fn test_brick_percent_of_budget_zero_actual() {
        let brick = BrickTiming::new("test", 10.0);
        // actual_us is 0.0 by default
        assert_eq!(brick.percent_of_budget(), 0);
    }

    // ========================================================================
    // BrickTiming::sparkline_data tests
    // ========================================================================

    #[test]
    fn test_sparkline_data_empty() {
        let brick = BrickTiming::new("test", 5.0);
        let data = brick.sparkline_data();
        assert!(data.is_empty());
    }

    #[test]
    fn test_sparkline_data_single_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(10.0);
        let data = brick.sparkline_data();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 100); // 10.0 * 10.0 = 100
    }

    #[test]
    fn test_sparkline_data_overflow_capped() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(30.0); // 30.0 * 10.0 = 300, capped at 255
        let data = brick.sparkline_data();
        assert_eq!(data[0], 255);
    }

    #[test]
    fn test_sparkline_data_zero_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(0.0);
        let data = brick.sparkline_data();
        assert_eq!(data[0], 0);
    }

    #[test]
    fn test_sparkline_data_multiple_samples() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(1.0);
        brick.add_sample(2.0);
        brick.add_sample(3.0);
        let data = brick.sparkline_data();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], 10); // 1.0 * 10.0
        assert_eq!(data[1], 20); // 2.0 * 10.0
        assert_eq!(data[2], 30); // 3.0 * 10.0
    }

    // ========================================================================
    // BrickTiming::add_sample ring buffer tests
    // ========================================================================

    #[test]
    fn test_add_sample_ring_buffer_overflow() {
        let mut brick = BrickTiming::new("test", 5.0);
        // Add 105 samples - should only keep last 100
        for i in 0..105 {
            brick.add_sample(i as f64);
        }
        assert_eq!(brick.samples.len(), 100);
        // First sample should be 5 (the oldest removed were 0..5)
        assert!((brick.samples[0] - 5.0).abs() < 0.001);
        // Last sample should be 104
        assert!((brick.samples[99] - 104.0).abs() < 0.001);
    }

    #[test]
    fn test_add_sample_updates_moving_average() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(10.0);
        assert!((brick.actual_us - 10.0).abs() < 0.001);
        brick.add_sample(20.0);
        assert!((brick.actual_us - 15.0).abs() < 0.001); // (10+20)/2
        brick.add_sample(30.0);
        assert!((brick.actual_us - 20.0).abs() < 0.001); // (10+20+30)/3
    }

    // ========================================================================
    // BrickTiming::status edge cases
    // ========================================================================

    #[test]
    fn test_brick_status_exact_budget() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 5.0; // Exactly at budget
        assert_eq!(brick.status(), "✅"); // <= is pass
    }

    #[test]
    fn test_brick_status_slightly_over() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.actual_us = 5.001;
        assert_eq!(brick.status(), "❌");
    }

    // ========================================================================
    // BrickTiming::gap_factor edge cases
    // ========================================================================

    #[test]
    fn test_gap_factor_under_budget() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 5.0;
        assert!((brick.gap_factor() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_gap_factor_exact_budget() {
        let mut brick = BrickTiming::new("test", 10.0);
        brick.actual_us = 10.0;
        assert!((brick.gap_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gap_factor_zero_actual_zero_budget() {
        let brick = BrickTiming::new("test", 0.0);
        // Both zero: returns 1.0 (budget is 0, defensive)
        assert!((brick.gap_factor() - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // PipelineState::bottleneck tests
    // ========================================================================

    #[test]
    fn test_pipeline_bottleneck_returns_some() {
        let mut pipeline = PipelineState::new();
        // Make one brick much worse than others
        pipeline.bricks[3].actual_us = 100.0; // Attention brick
        let bottleneck = pipeline.bottleneck();
        assert!(bottleneck.is_some());
        assert_eq!(
            bottleneck.expect("should have bottleneck").name,
            "Attention"
        );
    }

    #[test]
    fn test_pipeline_bottleneck_all_zero_actual() {
        let pipeline = PipelineState::new();
        // All actual_us are 0, all gap_factors are 0.0
        let bottleneck = pipeline.bottleneck();
        assert!(bottleneck.is_some()); // Returns the first brick with gap=0
    }

    #[test]
    fn test_pipeline_bottleneck_all_at_budget() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us;
        }
        // All at gap 1.0, bottleneck returns one of them
        let bottleneck = pipeline.bottleneck();
        assert!(bottleneck.is_some());
    }

    // ========================================================================
    // PipelineState::update_demo tok/s and total_us tests
    // ========================================================================

    #[test]
    fn test_pipeline_update_demo_sets_current_tok_s() {
        let mut pipeline = PipelineState::new();
        // After multiple updates, current_tok_s should be > 0
        for _ in 0..10 {
            pipeline.update_demo();
        }
        assert!(pipeline.current_tok_s > 0.0);
    }

    #[test]
    fn test_pipeline_update_demo_total_us_calculated() {
        let mut pipeline = PipelineState::new();
        pipeline.update_demo();
        // total_us should be total_actual * total_layers
        let expected = pipeline.total_actual() * pipeline.total_layers as f64;
        assert!((pipeline.total_us - expected).abs() < 0.001);
    }

    // ========================================================================
    // compute_brick_score boundary tests
    // ========================================================================

    #[test]
    fn test_compute_brick_score_zero_actual() {
        // 0 / 5.0 = 0.0 gap, which is <= 1.0 → 100
        assert_eq!(compute_brick_score(0.0, 5.0), 100);
    }

    #[test]
    fn test_compute_brick_score_exactly_120_percent() {
        // gap = 1.2 exactly → 100 - (0.2 * 50) = 90
        assert_eq!(compute_brick_score(6.0, 5.0), 90);
    }

    #[test]
    fn test_compute_brick_score_just_over_120_percent() {
        // gap = 1.21 → beyond 1.2 range: 100 - (0.21 * 100) = 79
        assert_eq!(compute_brick_score(6.05, 5.0), 79);
    }

    // ========================================================================
    // check_ci_thresholds: both failing simultaneously
    // ========================================================================

    #[test]
    fn test_ci_threshold_both_fail() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 100.0, // Below 400
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "test".to_string(),
                score: 50, // Below 90
                grade: "F".to_string(),
                budget_us: 1.0,
                actual_us: 2.0,
                gap_factor: 2.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 50,
                grade: "F".to_string(),
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
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_throughput_passes_brick_fails() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 500.0, // Above 400
                ttft_ms: 1.0,
                cv_percent: 3.0,
                p50_us: 1.0,
                p99_us: 2.0,
            },
            brick_scores: vec![BrickScore {
                name: "test".to_string(),
                score: 50, // Below 90
                grade: "F".to_string(),
                budget_us: 1.0,
                actual_us: 2.0,
                gap_factor: 2.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 50,
                grade: "F".to_string(),
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
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_multiple_brick_scores_average() {
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
            brick_scores: vec![
                BrickScore {
                    name: "brick1".to_string(),
                    score: 100,
                    grade: "A".to_string(),
                    budget_us: 1.0,
                    actual_us: 0.5,
                    gap_factor: 0.5,
                },
                BrickScore {
                    name: "brick2".to_string(),
                    score: 80,
                    grade: "B".to_string(),
                    budget_us: 1.0,
                    actual_us: 1.2,
                    gap_factor: 1.2,
                },
            ],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 90,
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

        let config = CbtopConfig {
            ci: true,
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        // Average = (100 + 80) / 2 = 90, which meets threshold
        assert!(check_ci_thresholds(&report, &config));
    }
