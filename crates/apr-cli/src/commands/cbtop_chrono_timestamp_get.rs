
    // === chrono_timestamp Tests ===

    #[test]
    fn test_chrono_timestamp_format() {
        let ts = chrono_timestamp();
        // Should be in ISO 8601-like format: 2026-01-12THH:MM:SSZ
        assert!(ts.starts_with("2026-01-12T") || ts == "unknown");
        if ts != "unknown" {
            assert!(ts.ends_with('Z'));
            assert_eq!(ts.len(), 20); // "2026-01-12THH:MM:SSZ"
        }
    }

    // === get_cpu_info Tests ===

    #[test]
    fn test_get_cpu_info_returns_string() {
        let info = get_cpu_info();
        // Should return either a real CPU name or "Unknown CPU"
        assert!(!info.is_empty());
    }

    // === get_memory_gb Tests ===

    #[test]
    fn test_get_memory_gb_returns_value() {
        let mem = get_memory_gb();
        // Should return either real memory or 64 (fallback)
        assert!(mem > 0);
    }

    // === CbtopConfig Tests ===

    #[test]
    fn test_cbtop_config_default_speculative() {
        let config = CbtopConfig::default();
        assert!(!config.speculative);
        assert_eq!(config.speculation_k, 4);
    }

    #[test]
    fn test_cbtop_config_default_concurrent() {
        let config = CbtopConfig::default();
        assert_eq!(config.concurrent, 1);
    }

    #[test]
    fn test_cbtop_config_default_simulated() {
        let config = CbtopConfig::default();
        assert!(!config.simulated);
    }

    // === Struct Construction Tests ===

    #[test]
    fn test_hardware_info_construction() {
        let hw = HardwareInfo {
            gpu: "RTX 4090".to_string(),
            cpu: "AMD Ryzen 9".to_string(),
            memory_gb: 64,
        };
        assert_eq!(hw.gpu, "RTX 4090");
        assert_eq!(hw.cpu, "AMD Ryzen 9");
        assert_eq!(hw.memory_gb, 64);
    }

    #[test]
    fn test_throughput_metrics_construction() {
        let tm = ThroughputMetrics {
            tokens_per_sec: 500.5,
            ttft_ms: 1.25,
            cv_percent: 3.5,
            p50_us: 1.0,
            p99_us: 2.5,
        };
        assert!((tm.tokens_per_sec - 500.5).abs() < 0.001);
        assert!((tm.ttft_ms - 1.25).abs() < 0.001);
        assert!((tm.cv_percent - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_brick_score_construction() {
        let bs = BrickScore {
            name: "Attention".to_string(),
            score: 95,
            grade: "A".to_string(),
            budget_us: 10.0,
            actual_us: 9.5,
            gap_factor: 0.95,
        };
        assert_eq!(bs.name, "Attention");
        assert_eq!(bs.score, 95);
        assert_eq!(bs.grade, "A");
    }

    #[test]
    fn test_pmat_scores_construction() {
        let ps = PmatScores {
            rust_project_score: 92.5,
            tdg_score: 95.2,
            cuda_tdg_score: 88.0,
            brick_score: 95,
            grade: "A+".to_string(),
        };
        assert!((ps.rust_project_score - 92.5).abs() < 0.001);
        assert_eq!(ps.grade, "A+");
    }

    #[test]
    fn test_falsification_summary_construction() {
        let fs = FalsificationSummary {
            total_points: 120,
            passed: 100,
            failed: 15,
            blocked: 5,
        };
        assert_eq!(fs.total_points, 120);
        assert_eq!(fs.passed, 100);
        assert_eq!(fs.failed, 15);
        assert_eq!(fs.blocked, 5);
    }

    // === CI Threshold Edge Cases ===

    #[test]
    fn test_ci_threshold_fail_brick_score() {
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
                score: 70, // Below 90 threshold
                grade: "C".to_string(),
                budget_us: 1.0,
                actual_us: 1.4,
                gap_factor: 1.4,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 70,
                grade: "C".to_string(),
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
            brick_score_threshold: Some(90),
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_no_thresholds() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "test".to_string(),
                cpu: "test".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 100.0, // Low, but no threshold set
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
                brick_score: 50,
                grade: "F".to_string(),
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

        // No thresholds set, should pass
        let config = CbtopConfig::default();
        assert!(check_ci_thresholds(&report, &config));
    }

    #[test]
    fn test_ci_threshold_empty_brick_scores() {
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
            brick_scores: vec![], // Empty scores
            pmat_scores: PmatScores {
                rust_project_score: 92.5,
                tdg_score: 95.2,
                cuda_tdg_score: 88.0,
                brick_score: 0,
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
            brick_score_threshold: Some(90), // avg of empty is 0
            ..Default::default()
        };

        assert!(!check_ci_thresholds(&report, &config));
    }

    // === View Tests ===

    #[test]
    fn test_view_pipeline_index() {
        assert_eq!(View::Pipeline.index(), 0);
    }

    #[test]
    fn test_view_budget_index() {
        assert_eq!(View::Budget.index(), 1);
    }

    #[test]
    fn test_view_histogram_index() {
        assert_eq!(View::Histogram.index(), 2);
    }

    #[test]
    fn test_view_gpu_index() {
        assert_eq!(View::Gpu.index(), 3);
    }

    #[test]
    fn test_view_memory_index() {
        assert_eq!(View::Memory.index(), 4);
    }

    // === Pipeline State Tests ===

    #[test]
    fn test_pipeline_total_actual() {
        let mut pipeline = PipelineState::new();
        for brick in &mut pipeline.bricks {
            brick.actual_us = brick.budget_us; // Set actual to budget
        }
        let total_actual = pipeline.total_actual();
        let total_budget = pipeline.total_budget();
        assert!((total_actual - total_budget).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_update_increments_tokens() {
        let mut pipeline = PipelineState::new();
        let initial = pipeline.tokens_generated;
        pipeline.update_demo();
        assert_eq!(pipeline.tokens_generated, initial + 1);
    }

    // === BrickTiming Edge Cases ===

    #[test]
    fn test_brick_timing_gap_factor_zero_budget() {
        let mut brick = BrickTiming::new("test", 0.0);
        brick.actual_us = 5.0;
        // gap_factor with zero budget returns 1.0 as a defensive guard
        let gap = brick.gap_factor();
        assert!((gap - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_timing_samples_statistics() {
        let mut brick = BrickTiming::new("test", 5.0);
        brick.add_sample(4.0);
        brick.add_sample(5.0);
        brick.add_sample(6.0);
        // Mean should be 5.0
        assert!((brick.actual_us - 5.0).abs() < 0.001);
    }

    // === JSON Format Tests ===

    #[test]
    fn test_json_output_includes_all_fields() {
        let report = HeadlessReport {
            model: "test-model".to_string(),
            timestamp: "2026-01-12T00:00:00Z".to_string(),
            hardware: HardwareInfo {
                gpu: "Test GPU".to_string(),
                cpu: "Test CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 250.0,
                ttft_ms: 2.5,
                cv_percent: 5.0,
                p50_us: 1.5,
                p99_us: 3.0,
            },
            brick_scores: vec![],
            pmat_scores: PmatScores {
                rust_project_score: 90.0,
                tdg_score: 92.0,
                cuda_tdg_score: 85.0,
                brick_score: 88,
                grade: "B".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 100,
                passed: 95,
                failed: 5,
                blocked: 0,
            },
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let json = format_report_as_json(&report);

        // Check all top-level fields
        assert!(json.contains("\"model\":"));
        assert!(json.contains("\"timestamp\":"));
        assert!(json.contains("\"hardware\":"));
        assert!(json.contains("\"throughput\":"));
        assert!(json.contains("\"brick_scores\":"));
        assert!(json.contains("\"pmat_scores\":"));
        assert!(json.contains("\"falsification\":"));
        assert!(json.contains("\"status\":"));
        assert!(json.contains("\"ci_result\":"));
    }

    #[test]
    fn test_json_output_hardware_fields() {
        let report = HeadlessReport {
            model: "test".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "NVIDIA RTX 4090".to_string(),
                cpu: "AMD Ryzen 9 7950X".to_string(),
                memory_gb: 128,
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
            status: "".to_string(),
            ci_result: "".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.contains("NVIDIA RTX 4090"));
        assert!(json.contains("AMD Ryzen 9 7950X"));
        assert!(json.contains("\"memory_gb\": 128"));
    }

    // === App Navigation Edge Cases ===

    #[test]
    fn test_app_navigation_wrap_forward() {
        let mut app = App::new(None);
        // Navigate to last brick
        for _ in 0..6 {
            app.next_brick();
        }
        assert_eq!(app.selected_brick, 6);
        // One more should wrap to 0
        app.next_brick();
        assert_eq!(app.selected_brick, 0);
    }

    #[test]
    fn test_app_navigation_wrap_backward() {
        let mut app = App::new(None);
        assert_eq!(app.selected_brick, 0);
        // Going backward from 0 should wrap to last
        app.prev_brick();
        assert_eq!(app.selected_brick, 6);
    }
