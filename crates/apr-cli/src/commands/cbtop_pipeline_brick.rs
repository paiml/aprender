
    // ========================================================================
    // PipelineState: brick default names and budgets
    // ========================================================================

    #[test]
    fn test_pipeline_brick_names() {
        let pipeline = PipelineState::new();
        assert_eq!(pipeline.bricks[0].name, "RmsNorm");
        assert_eq!(pipeline.bricks[1].name, "QkvBrick");
        assert_eq!(pipeline.bricks[2].name, "RoPE");
        assert_eq!(pipeline.bricks[3].name, "Attention");
        assert_eq!(pipeline.bricks[4].name, "OProj");
        assert_eq!(pipeline.bricks[5].name, "RmsNorm");
        assert_eq!(pipeline.bricks[6].name, "FfnBrick");
    }

    #[test]
    fn test_pipeline_brick_budgets() {
        let pipeline = PipelineState::new();
        assert!((pipeline.bricks[0].budget_us - 1.5).abs() < 0.001);
        assert!((pipeline.bricks[1].budget_us - 6.0).abs() < 0.001);
        assert!((pipeline.bricks[2].budget_us - 1.0).abs() < 0.001);
        assert!((pipeline.bricks[3].budget_us - 10.0).abs() < 0.001);
        assert!((pipeline.bricks[4].budget_us - 3.5).abs() < 0.001);
        assert!((pipeline.bricks[5].budget_us - 1.5).abs() < 0.001);
        assert!((pipeline.bricks[6].budget_us - 12.2).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_defaults() {
        let pipeline = PipelineState::new();
        assert_eq!(pipeline.layer_idx, 0);
        assert_eq!(pipeline.total_layers, 28);
        assert_eq!(pipeline.tokens_generated, 0);
        assert!((pipeline.total_us - 0.0).abs() < 0.001);
        assert!((pipeline.target_tok_s - 976.0).abs() < 0.001);
        assert!((pipeline.current_tok_s - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // App: demo_mode and should_quit defaults
    // ========================================================================

    #[test]
    fn test_app_defaults() {
        let app = App::new(None);
        assert!(app.demo_mode);
        assert!(!app.should_quit);
        assert_eq!(app.selected_brick, 0);
        assert_eq!(app.current_view, View::Pipeline);
    }

    #[test]
    fn test_app_tick_no_demo_mode() {
        let mut app = App::new(None);
        app.demo_mode = false;
        let initial_tokens = app.pipeline.tokens_generated;
        app.tick();
        // In non-demo mode, tick should NOT update pipeline
        assert_eq!(app.pipeline.tokens_generated, initial_tokens);
    }

    // ========================================================================
    // Navigation with empty bricks (defensive edge case)
    // ========================================================================

    #[test]
    fn test_app_next_brick_empty_bricks() {
        let mut app = App::new(None);
        app.pipeline.bricks.clear();
        app.next_brick(); // Should not panic
        assert_eq!(app.selected_brick, 0);
    }

    #[test]
    fn test_app_prev_brick_empty_bricks() {
        let mut app = App::new(None);
        app.pipeline.bricks.clear();
        app.prev_brick(); // Should not panic
        assert_eq!(app.selected_brick, 0);
    }

    // ========================================================================
    // NEW: compute_brick_score edge cases
    // ========================================================================

    #[test]
    fn test_compute_brick_score_very_small_budget() {
        // Very small budget, actual slightly over
        assert_eq!(compute_brick_score(0.002, 0.001), 0);
    }

    #[test]
    fn test_compute_brick_score_very_large_values() {
        // Large numbers, at budget
        assert_eq!(compute_brick_score(1_000_000.0, 1_000_000.0), 100);
    }

    #[test]
    fn test_compute_brick_score_just_under_budget() {
        // gap = 0.999... -> 100
        assert_eq!(compute_brick_score(4.999, 5.0), 100);
    }

    #[test]
    fn test_compute_brick_score_just_over_budget() {
        // gap = 1.001, in 1.0-1.2 range: 100 - (0.001 * 50) = 99.95 -> 99
        assert_eq!(compute_brick_score(5.005, 5.0), 99);
    }

    #[test]
    fn test_compute_brick_score_at_boundary_1_2() {
        // gap = 1.2 exactly: 100 - (0.2 * 50) = 90
        assert_eq!(compute_brick_score(12.0, 10.0), 90);
    }

    #[test]
    fn test_compute_brick_score_slightly_past_1_2() {
        // gap = 1.201, beyond 1.2: 100 - (0.201 * 100) = 79.9 -> 79
        assert_eq!(compute_brick_score(12.01, 10.0), 79);
    }

    #[test]
    fn test_compute_brick_score_gap_1_5() {
        // gap = 1.5, 100 - (0.5 * 100) = 50
        assert_eq!(compute_brick_score(15.0, 10.0), 50);
    }

    #[test]
    fn test_compute_brick_score_gap_2_0_gives_zero() {
        // gap = 2.0, 100 - (1.0 * 100) = 0
        assert_eq!(compute_brick_score(20.0, 10.0), 0);
    }

    #[test]
    fn test_compute_brick_score_gap_beyond_2_clamps_zero() {
        // gap = 5.0, 100 - (4.0 * 100) = -300, clamped to 0
        assert_eq!(compute_brick_score(50.0, 10.0), 0);
    }

    // ========================================================================
    // NEW: score_to_grade boundary tests
    // ========================================================================

    #[test]
    fn test_score_to_grade_boundary_89_90() {
        assert_eq!(score_to_grade(89), "B");
        assert_eq!(score_to_grade(90), "A");
    }

    #[test]
    fn test_score_to_grade_boundary_79_80() {
        assert_eq!(score_to_grade(79), "C");
        assert_eq!(score_to_grade(80), "B");
    }

    #[test]
    fn test_score_to_grade_boundary_69_70() {
        assert_eq!(score_to_grade(69), "D");
        assert_eq!(score_to_grade(70), "C");
    }

    #[test]
    fn test_score_to_grade_boundary_59_60() {
        assert_eq!(score_to_grade(59), "F");
        assert_eq!(score_to_grade(60), "D");
    }

    #[test]
    fn test_score_to_grade_zero() {
        assert_eq!(score_to_grade(0), "F");
    }

    #[test]
    fn test_score_to_grade_max() {
        assert_eq!(score_to_grade(100), "A");
    }

    #[test]
    fn test_score_to_grade_above_100() {
        // Scores > 100 fall into the catch-all (not matched by 90..=100)
        assert_eq!(score_to_grade(101), "F");
        assert_eq!(score_to_grade(200), "F");
    }

    // ========================================================================
    // NEW: chrono_timestamp additional tests
    // ========================================================================

    #[test]
    fn test_chrono_timestamp_not_empty() {
        let ts = chrono_timestamp();
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_chrono_timestamp_contains_t_separator() {
        let ts = chrono_timestamp();
        if ts != "unknown" {
            assert!(ts.contains('T'));
        }
    }

    // ========================================================================
    // NEW: BrickTiming edge cases with negative and extreme values
    // ========================================================================

    #[test]
    fn test_brick_timing_negative_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(-3.0);
        // actual_us becomes the average of samples (just -3.0)
        assert!((brick.actual_us - (-3.0)).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_very_large_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(1e12);
        assert!((brick.actual_us - 1e12).abs() < 1.0);
    }

    #[test]
    fn test_brick_timing_sparkline_negative_sample() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.samples.push(-5.0);
        let data = brick.sparkline_data();
        // -5.0 * 10.0 = -50.0, min(255.0) = -50.0, as u64 wraps
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_brick_timing_gap_factor_negative_budget() {
        let mut brick = BrickTiming::new("test", -5.0);
        brick.actual_us = 10.0;
        // budget < 0, not > 0, so returns 1.0
        assert!((brick.gap_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_percent_of_budget_negative_budget() {
        let mut brick = BrickTiming::new("test", -5.0);
        brick.actual_us = 10.0;
        // budget not > 0, returns 100
        assert_eq!(brick.percent_of_budget(), 100);
    }

    #[test]
    fn test_brick_timing_status_zero_actual_zero_budget() {
        let brick = BrickTiming::new("test", 0.0);
        // actual (0) <= budget (0), so pass
        assert_eq!(brick.status(), "\u{2705}");
    }

    #[test]
    fn test_brick_timing_add_sample_exactly_100() {
        let mut brick = BrickTiming::new("test", 5.0);
        for i in 0..100 {
            brick.add_sample(i as f64);
        }
        assert_eq!(brick.samples.len(), 100);
        // Add one more, should drop oldest
        brick.add_sample(999.0);
        assert_eq!(brick.samples.len(), 100);
        assert!((brick.samples[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_sparkline_exactly_25_5() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.samples.push(25.5);
        let data = brick.sparkline_data();
        // 25.5 * 10.0 = 255.0, min(255.0) = 255
        assert_eq!(data[0], 255);
    }

    #[test]
    fn test_brick_timing_sparkline_just_under_cap() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.samples.push(25.4);
        let data = brick.sparkline_data();
        // 25.4 * 10.0 = 254.0
        assert_eq!(data[0], 254);
    }

    // ========================================================================
    // NEW: PipelineState detailed tests
    // ========================================================================

    #[test]
    fn test_pipeline_total_actual_with_samples() {
        let mut pipeline = PipelineState::new();
        pipeline.bricks[0].add_sample(2.0);
        pipeline.bricks[1].add_sample(8.0);
        // Others have actual_us = 0
        let total = pipeline.total_actual();
        assert!((total - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_bottleneck_single_bad_brick() {
        let mut pipeline = PipelineState::new();
        // Set all bricks at budget except one
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us;
        }
        // Make FfnBrick much worse
        pipeline.bricks[6].actual_us = 100.0;
        let bottleneck = pipeline.bottleneck().expect("should find bottleneck");
        assert_eq!(bottleneck.name, "FfnBrick");
    }

    #[test]
    fn test_pipeline_update_demo_multiple_iterations() {
        let mut pipeline = PipelineState::new();
        for _ in 0..50 {
            pipeline.update_demo();
        }
        assert_eq!(pipeline.tokens_generated, 50);
        // All bricks should have samples
        for brick in &pipeline.bricks {
            assert!(!brick.samples.is_empty());
            assert!(brick.actual_us > 0.0);
        }
    }

    #[test]
    fn test_pipeline_current_tok_s_positive_after_updates() {
        let mut pipeline = PipelineState::new();
        for _ in 0..5 {
            pipeline.update_demo();
        }
        assert!(pipeline.current_tok_s > 0.0);
    }

    #[test]
    fn test_pipeline_total_us_equals_actual_times_layers() {
        let mut pipeline = PipelineState::new();
        for _ in 0..10 {
            pipeline.update_demo();
        }
        let expected = pipeline.total_actual() * pipeline.total_layers as f64;
        assert!((pipeline.total_us - expected).abs() < 0.01);
    }

    // ========================================================================
    // NEW: generate_headless_report_simulated detailed tests
    // ========================================================================

    #[test]
    fn test_headless_report_simulated_brick_gap_factors() {
        let mut pipeline = PipelineState::new();
        // Set specific actual values
        pipeline.bricks[0].actual_us = 1.5; // gap = 1.0
        pipeline.bricks[1].actual_us = 9.0; // gap = 1.5
        pipeline.bricks[2].actual_us = 0.5; // gap = 0.5
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("gap-test", &pipeline, &config);

        assert!((report.brick_scores[0].gap_factor - 1.0).abs() < 0.01);
        assert!((report.brick_scores[1].gap_factor - 1.5).abs() < 0.01);
        assert!((report.brick_scores[2].gap_factor - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_headless_report_simulated_pmat_brick_score_weighting() {
        let mut pipeline = PipelineState::new();
        // All at budget -> all score 100 -> weighted avg ~100 (may truncate to 99 due to f64)
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 0.5; // All under budget
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("weight-test", &pipeline, &config);
        // All bricks score 100, but (100.0 * total_weight) / total_weight can truncate
        assert!(report.pmat_scores.brick_score >= 99);
    }

    #[test]
    fn test_headless_report_simulated_ci_result_green() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 0.5; // All pass
        }
        pipeline.current_tok_s = 1000.0; // Above target 976
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("green-test", &pipeline, &config);
        assert_eq!(report.ci_result, "green");
        assert_eq!(report.status, "PASS");
    }

    #[test]
    fn test_headless_report_simulated_ci_result_red_low_throughput() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 0.5; // All pass
        }
        pipeline.current_tok_s = 500.0; // Below target 976
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("red-test", &pipeline, &config);
        assert_eq!(report.ci_result, "red");
    }

    #[test]
    fn test_headless_report_simulated_ci_result_red_bricks_fail() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 2.0; // All over budget
        }
        pipeline.current_tok_s = 1000.0; // Above target
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("fail-bricks", &pipeline, &config);
        assert_eq!(report.status, "FAIL");
        assert_eq!(report.ci_result, "red");
    }

    #[test]
    fn test_headless_report_simulated_hardware_is_simulated() {
        let pipeline = PipelineState::new();
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("hw-test", &pipeline, &config);
        assert!(report.hardware.gpu.contains("simulated"));
        assert!(report.hardware.cpu.contains("simulated"));
        assert_eq!(report.hardware.memory_gb, 64);
    }

    #[test]
    fn test_headless_report_simulated_falsification_defaults() {
        let pipeline = PipelineState::new();
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("fals-test", &pipeline, &config);
        assert_eq!(report.falsification.total_points, 137);
        assert_eq!(report.falsification.passed, 137);
        assert_eq!(report.falsification.failed, 0);
        assert_eq!(report.falsification.blocked, 0);
    }

    #[test]
    fn test_headless_report_simulated_timestamp_not_empty() {
        let pipeline = PipelineState::new();
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("ts-test", &pipeline, &config);
        assert!(!report.timestamp.is_empty());
    }
