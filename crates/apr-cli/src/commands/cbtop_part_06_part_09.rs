
    #[test]
    fn test_falsification_summary_debug() {
        let fs = FalsificationSummary {
            total_points: 0,
            passed: 0,
            failed: 0,
            blocked: 0,
        };
        let debug_str = format!("{fs:?}");
        assert!(debug_str.contains("FalsificationSummary"));
    }

    // ========================================================================
    // NEW: Clone trait tests
    // ========================================================================

    #[test]
    fn test_hardware_info_clone() {
        let hw = HardwareInfo {
            gpu: "RTX".to_string(),
            cpu: "Ryzen".to_string(),
            memory_gb: 64,
        };
        let cloned = hw.clone();
        assert_eq!(cloned.gpu, "RTX");
        assert_eq!(cloned.cpu, "Ryzen");
        assert_eq!(cloned.memory_gb, 64);
    }

    #[test]
    fn test_throughput_metrics_clone() {
        let tm = ThroughputMetrics {
            tokens_per_sec: 123.4,
            ttft_ms: 5.6,
            cv_percent: 7.8,
            p50_us: 9.0,
            p99_us: 11.2,
        };
        let cloned = tm.clone();
        assert!((cloned.tokens_per_sec - 123.4).abs() < 0.001);
        assert!((cloned.p99_us - 11.2).abs() < 0.001);
    }

    #[test]
    fn test_pmat_scores_clone() {
        let ps = PmatScores {
            rust_project_score: 173.9,
            tdg_score: 98.1,
            cuda_tdg_score: 95.2,
            brick_score: 99,
            grade: "A+".to_string(),
        };
        let cloned = ps.clone();
        assert_eq!(cloned.brick_score, 99);
        assert_eq!(cloned.grade, "A+");
    }

    #[test]
    fn test_brick_score_clone() {
        let bs = BrickScore {
            name: "Attention".to_string(),
            score: 85,
            grade: "B".to_string(),
            budget_us: 10.0,
            actual_us: 11.5,
            gap_factor: 1.15,
        };
        let cloned = bs.clone();
        assert_eq!(cloned.name, "Attention");
        assert_eq!(cloned.score, 85);
    }

    #[test]
    fn test_falsification_summary_clone() {
        let fs = FalsificationSummary {
            total_points: 137,
            passed: 130,
            failed: 5,
            blocked: 2,
        };
        let cloned = fs.clone();
        assert_eq!(cloned.total_points, 137);
        assert_eq!(cloned.failed, 5);
    }

    // ========================================================================
    // NEW: generate_headless_report grade mapping from inline score
    // ========================================================================

    #[test]
    fn test_headless_report_simulated_grade_a_for_perfect_bricks() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = 0.0; // All 0 -> gap_factor = 0 -> score 100
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("grade-a", &pipeline, &config);
        for brick_score in &report.brick_scores {
            assert_eq!(brick_score.grade, "A");
            assert_eq!(brick_score.score, 100);
        }
    }

    #[test]
    fn test_headless_report_simulated_grade_f_for_bad_bricks() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 3.0; // gap = 3.0, score = 0
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("grade-f", &pipeline, &config);
        for brick_score in &report.brick_scores {
            assert_eq!(brick_score.grade, "F");
            assert_eq!(brick_score.score, 0);
        }
    }

    #[test]
    fn test_headless_report_simulated_pmat_grade_f_for_all_bad() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us * 3.0;
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("pmat-f", &pipeline, &config);
        assert_eq!(report.pmat_scores.grade, "F");
        assert_eq!(report.pmat_scores.brick_score, 0);
    }

    #[test]
    fn test_headless_report_simulated_pmat_grade_a_for_all_good() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = 0.0;
        }
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("pmat-a", &pipeline, &config);
        // All bricks score 100, weighted average may truncate to 99 due to f64
        assert!(report.pmat_scores.grade == "A");
        assert!(report.pmat_scores.brick_score >= 99);
    }

    // ========================================================================
    // NEW: BrickTiming::add_sample moving average accuracy
    // ========================================================================

    #[test]
    fn test_add_sample_moving_average_with_overflow() {
        let mut brick = BrickTiming::new("test", 5.0);
        // Fill with 100 samples of value 10.0
        for _ in 0..100 {
            brick.add_sample(10.0);
        }
        assert!((brick.actual_us - 10.0).abs() < 0.001);
        // Now add 100.0 - oldest (10.0) removed, average shifts
        brick.add_sample(100.0);
        // (99 * 10.0 + 100.0) / 100 = 10.9
        assert!((brick.actual_us - 10.9).abs() < 0.01);
    }

    #[test]
    fn test_add_sample_single_sample_is_exact() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(42.0);
        assert!((brick.actual_us - 42.0).abs() < 0.001);
    }

    // ========================================================================
    // NEW: Pipeline state after warmup+measurement pattern
    // ========================================================================

    #[test]
    fn test_pipeline_warmup_clear_measurement_pattern() {
        let mut pipeline = PipelineState::new();
        // Warmup
        for _ in 0..10 {
            pipeline.update_demo();
        }
        // Clear samples (like run_headless_simulated does)
        for brick in &mut pipeline.bricks {
            brick.samples.clear();
            brick.actual_us = 0.0;
        }
        // Verify cleared
        for brick in &pipeline.bricks {
            assert!(brick.samples.is_empty());
            assert!((brick.actual_us - 0.0).abs() < 0.001);
        }
        // Measurement
        for _ in 0..50 {
            pipeline.update_demo();
        }
        // Verify we have fresh data
        for brick in &pipeline.bricks {
            assert_eq!(brick.samples.len(), 50);
            assert!(brick.actual_us > 0.0);
        }
    }

    // ========================================================================
    // NEW: Comprehensive JSON round-trip verification
    // ========================================================================

    #[test]
    fn test_json_output_well_formed_braces() {
        let report = HeadlessReport {
            model: "brace-test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "G".to_string(),
                cpu: "C".to_string(),
                memory_gb: 1,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 1.0,
                ttft_ms: 1.0,
                cv_percent: 1.0,
                p50_us: 1.0,
                p99_us: 1.0,
            },
            brick_scores: vec![BrickScore {
                name: "b".to_string(),
                score: 50,
                grade: "F".to_string(),
                budget_us: 1.0,
                actual_us: 2.0,
                gap_factor: 2.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 1.0,
                tdg_score: 1.0,
                cuda_tdg_score: 1.0,
                brick_score: 1,
                grade: "F".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 1,
                passed: 1,
                failed: 0,
                blocked: 0,
            },
            status: "X".to_string(),
            ci_result: "X".to_string(),
        };

        let json = format_report_as_json(&report);
        // Count opening and closing braces - should match
        let open_braces = json.chars().filter(|&c| c == '{').count();
        let close_braces = json.chars().filter(|&c| c == '}').count();
        assert_eq!(open_braces, close_braces);

        let open_brackets = json.chars().filter(|&c| c == '[').count();
        let close_brackets = json.chars().filter(|&c| c == ']').count();
        assert_eq!(open_brackets, close_brackets);
    }

    #[test]
    fn test_json_output_starts_and_ends_correctly() {
        let report = HeadlessReport {
            model: "form-test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "G".to_string(),
                cpu: "C".to_string(),
                memory_gb: 1,
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
            status: "X".to_string(),
            ci_result: "X".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    // ========================================================================
    // NEW: generate_headless_report_simulated p50/p99 with empty pipeline
    // ========================================================================

    #[test]
    fn test_headless_report_simulated_no_bricks() {
        let mut pipeline = PipelineState::new();
        pipeline.bricks.clear(); // Remove all bricks
        let config = CbtopConfig::default();
        let report = generate_headless_report_simulated("no-bricks", &pipeline, &config);
        assert_eq!(report.brick_scores.len(), 0);
        assert!((report.throughput.p50_us - 0.0).abs() < 0.001);
        assert!((report.throughput.p99_us - 0.0).abs() < 0.001);
        assert!((report.throughput.cv_percent - 0.0).abs() < 0.001);
        assert_eq!(report.status, "PASS"); // Empty bricks -> all pass vacuously
    }

    // ========================================================================
    // NEW: PipelineState totals with zeroed bricks
    // ========================================================================

    #[test]
    fn test_pipeline_total_budget_sum_precision() {
        let pipeline = PipelineState::new();
        // 1.5 + 6.0 + 1.0 + 10.0 + 3.5 + 1.5 + 12.2 = 35.7
        let budget = pipeline.total_budget();
        assert!((budget - 35.7).abs() < 0.0001);
    }

    #[test]
    fn test_pipeline_total_actual_all_zero() {
        let pipeline = PipelineState::new();
        assert!((pipeline.total_actual() - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // NEW: compute_brick_score score ranges for each grade band
    // ========================================================================

    #[test]
    fn test_compute_brick_score_produces_grade_b_range() {
        // gap = 1.15 -> 100 - (0.15 * 50) = 92.5 -> 92 (A)
        // gap = 1.19 -> 100 - (0.19 * 50) = 90.5 -> 90 (A)
        // Need gap > 1.2 for below 90
        // gap = 1.3 -> 100 - (0.3 * 100) = 70 (C)
        // gap = 1.11 -> still in 1.0-1.2: 100 - (0.11*50) = 94.5 -> 94 (A)
        // For B (80-89), we need: 100 - (gap-1.0)*100 in [80,89] for gap > 1.2
        // 100 - (gap-1.0)*100 = 80 => gap = 1.2 -> 90 (1.0-1.2 formula)
        // Actually for gap > 1.2: score = 100 - (gap-1.0)*100
        // score=89 => gap = 1.11 (but 1.11 < 1.2 so different formula)
        // For gap in (1.2, x]: 100 - (gap-1.0)*100
        // score=89: gap=1.11 -> in 1.0-1.2 range -> 100 - (0.11*50) = 94.5 -> 94
        // For score=80: 100-(gap-1)*100=80 -> gap=1.2 -> on boundary
        // For score=85: 100-(gap-1)*100=85 -> gap=1.15 (in 1.0-1.2)
        // Actually at gap=1.2 in 1.0-1.2 range: 100-(0.2*50)=90
        // At gap=1.21: 100-(0.21*100)=79 -> C
        // So B range (80-89) is actually gap in [1.2, 1.2] = just 90
        // The only way to get B is score 80-89, which means gap 1.11-1.2 in first formula
        // gap=1.12: 100-(0.12*50)=94 -> A
        // gap=1.20: 100-(0.20*50)=90 -> A
        // So the 1.0-1.2 range yields scores 90-100 (all A)
        // And gap > 1.2 yields: 100-(gap-1)*100, where gap=1.21 -> 79 (C)
        // There's a discontinuity at gap=1.2 (90 with first formula, then gap=1.201 -> 79.9)
        // So actually scores 80-89 (B) are never produced! Let's verify:
        let score_at_gap_1_2 = compute_brick_score(12.0, 10.0); // gap=1.2
        assert_eq!(score_at_gap_1_2, 90); // A

        let score_just_over = compute_brick_score(12.01, 10.0); // gap=1.201
        assert_eq!(score_just_over, 79); // C -- skips B entirely
    }

    // ========================================================================
    // NEW: Integration-style test for full simulated flow
    // ========================================================================

    #[test]
    fn test_full_simulated_report_flow() {
        let mut pipeline = PipelineState::new();
        // Simulate warmup
        for _ in 0..10 {
            pipeline.update_demo();
        }
        // Clear
        for brick in &mut pipeline.bricks {
            brick.samples.clear();
            brick.actual_us = 0.0;
        }
        // Simulate measurement
        for _ in 0..100 {
            pipeline.update_demo();
        }

        let config = CbtopConfig {
            warmup: 10,
            iterations: 100,
            ..Default::default()
        };
        let report = generate_headless_report_simulated("full-flow", &pipeline, &config);

        // Basic sanity
        assert_eq!(report.model, "full-flow");
        assert_eq!(report.brick_scores.len(), 7);
        assert!(report.throughput.tokens_per_sec >= 0.0);
        assert!(!report.timestamp.is_empty());
        assert!(report.status == "PASS" || report.status == "FAIL");
        assert!(report.ci_result == "green" || report.ci_result == "red");

        // JSON formatting
        let json = format_report_as_json(&report);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));

        // CI thresholds with no thresholds set
        assert!(check_ci_thresholds(&report, &config));

        // Text output should not panic
        print_report_text(&report);
    }
