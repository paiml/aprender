
    #[test]
    fn test_json_output_brick_gap_factor_precision() {
        let report = HeadlessReport {
            model: "gap".to_string(),
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
            brick_scores: vec![BrickScore {
                name: "precise".to_string(),
                score: 75,
                grade: "C".to_string(),
                budget_us: 3.14,
                actual_us: 4.56,
                gap_factor: 1.452,
            }],
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
        assert!(json.contains("\"budget_us\": 3.14"));
        assert!(json.contains("\"actual_us\": 4.56"));
        assert!(json.contains("\"gap_factor\": 1.452"));
    }

    #[test]
    fn test_json_output_status_and_ci_result() {
        let report = HeadlessReport {
            model: "status".to_string(),
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
            status: "PASS".to_string(),
            ci_result: "green".to_string(),
        };

        let json = format_report_as_json(&report);
        assert!(json.contains("\"status\": \"PASS\""));
        assert!(json.contains("\"ci_result\": \"green\""));
    }

    // ========================================================================
    // NEW: print_report_text branch coverage
    // ========================================================================

    #[test]
    fn test_print_report_text_with_passing_bricks() {
        let report = HeadlessReport {
            model: "pass-model".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 1000.0,
                ttft_ms: 0.5,
                cv_percent: 1.0,
                p50_us: 0.5,
                p99_us: 1.0,
            },
            brick_scores: vec![
                BrickScore {
                    name: "good".to_string(),
                    score: 100,
                    grade: "A".to_string(),
                    budget_us: 5.0,
                    actual_us: 3.0,
                    gap_factor: 0.6,
                },
                BrickScore {
                    name: "also_good".to_string(),
                    score: 95,
                    grade: "A".to_string(),
                    budget_us: 10.0,
                    actual_us: 9.5,
                    gap_factor: 0.95,
                },
            ],
            pmat_scores: PmatScores {
                rust_project_score: 100.0,
                tdg_score: 100.0,
                cuda_tdg_score: 100.0,
                brick_score: 100,
                grade: "A+".to_string(),
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

        // Should not panic, exercises the pass branch
        print_report_text(&report);
    }

    #[test]
    fn test_print_report_text_with_failing_bricks() {
        let report = HeadlessReport {
            model: "fail-model".to_string(),
            timestamp: "now".to_string(),
            hardware: HardwareInfo {
                gpu: "GPU".to_string(),
                cpu: "CPU".to_string(),
                memory_gb: 32,
            },
            throughput: ThroughputMetrics {
                tokens_per_sec: 100.0,
                ttft_ms: 10.0,
                cv_percent: 50.0,
                p50_us: 5.0,
                p99_us: 50.0,
            },
            brick_scores: vec![BrickScore {
                name: "terrible".to_string(),
                score: 0,
                grade: "F".to_string(),
                budget_us: 1.0,
                actual_us: 100.0,
                gap_factor: 100.0,
            }],
            pmat_scores: PmatScores {
                rust_project_score: 0.0,
                tdg_score: 0.0,
                cuda_tdg_score: 0.0,
                brick_score: 0,
                grade: "F".to_string(),
            },
            falsification: FalsificationSummary {
                total_points: 100,
                passed: 10,
                failed: 80,
                blocked: 10,
            },
            status: "FAIL".to_string(),
            ci_result: "red".to_string(),
        };

        // Should not panic, exercises the fail branch
        print_report_text(&report);
    }

    // ========================================================================
    // NEW: ModelFormat from_path with unusual paths
    // ========================================================================

    #[test]
    fn test_model_format_from_path_with_directory() {
        use std::path::Path;
        let path = Path::new("/some/dir/model.gguf");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_model_format_from_path_multiple_dots() {
        use std::path::Path;
        let path = Path::new("model.v2.gguf");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_model_format_from_path_dot_only() {
        use std::path::Path;
        let path = Path::new(".");
        assert_eq!(ModelFormat::from_path(path), None);
    }

    #[test]
    fn test_model_format_from_path_hidden_file() {
        use std::path::Path;
        let path = Path::new(".hidden_model.safetensors");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_model_format_from_path_empty_extension() {
        use std::path::Path;
        let path = Path::new("model.");
        // Path::extension() returns None for "model." on most platforms
        // (the part after the dot is empty)
        // Actually, on std: "model." -> extension is Some("")
        assert_eq!(ModelFormat::from_path(path), None);
    }

    #[test]
    fn test_model_format_from_path_mixed_case_safetensors() {
        use std::path::Path;
        let path = Path::new("model.SafeTensors");
        assert_eq!(ModelFormat::from_path(path), Some(ModelFormat::SafeTensors));
    }

    // ========================================================================
    // NEW: App and View integration
    // ========================================================================

    #[test]
    fn test_app_multiple_next_prev_cycles() {
        let mut app = App::new(None);
        // Go forward 3, back 2, should be at 1
        app.next_brick();
        app.next_brick();
        app.next_brick();
        app.prev_brick();
        app.prev_brick();
        assert_eq!(app.selected_brick, 1);
    }

    #[test]
    fn test_app_full_cycle_forward() {
        let mut app = App::new(None);
        let num_bricks = app.pipeline.bricks.len();
        for _ in 0..num_bricks {
            app.next_brick();
        }
        // Should wrap back to 0
        assert_eq!(app.selected_brick, 0);
    }

    #[test]
    fn test_app_full_cycle_backward() {
        let mut app = App::new(None);
        let num_bricks = app.pipeline.bricks.len();
        for _ in 0..num_bricks {
            app.prev_brick();
        }
        // Should wrap back to 0
        assert_eq!(app.selected_brick, 0);
    }

    #[test]
    fn test_view_titles_count_matches_variants() {
        let titles = View::titles();
        // 5 variants: Pipeline, Budget, Histogram, Gpu, Memory
        assert_eq!(titles.len(), 5);
    }

    #[test]
    fn test_view_titles_all_contain_brackets() {
        let titles = View::titles();
        for title in titles {
            assert!(
                title.contains('['),
                "Title should contain key hint: {title}"
            );
        }
    }

    #[test]
    fn test_view_index_is_sequential() {
        assert_eq!(View::Pipeline.index(), 0);
        assert_eq!(View::Budget.index(), 1);
        assert_eq!(View::Histogram.index(), 2);
        assert_eq!(View::Gpu.index(), 3);
        assert_eq!(View::Memory.index(), 4);
    }

    // ========================================================================
    // NEW: CbtopConfig field combinations
    // ========================================================================

    #[test]
    fn test_cbtop_config_default_model_path_none() {
        let config = CbtopConfig::default();
        assert!(config.model_path.is_none());
        assert!(config.model.is_none());
        assert!(config.attach.is_none());
        assert!(config.output.is_none());
        assert!(config.draft_model_path.is_none());
        assert!(config.throughput_threshold.is_none());
        assert!(config.brick_score_threshold.is_none());
    }

    #[test]
    fn test_cbtop_config_clone() {
        let config = CbtopConfig {
            model: Some("test".to_string()),
            headless: true,
            json: true,
            warmup: 5,
            iterations: 50,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.model, Some("test".to_string()));
        assert!(cloned.headless);
        assert!(cloned.json);
        assert_eq!(cloned.warmup, 5);
        assert_eq!(cloned.iterations, 50);
    }

    // ========================================================================
    // NEW: Struct Debug trait tests
    // ========================================================================

    #[test]
    fn test_cbtop_config_debug() {
        let config = CbtopConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("CbtopConfig"));
    }

    #[test]
    fn test_headless_report_debug() {
        let report = HeadlessReport {
            model: "debug".to_string(),
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
        let debug_str = format!("{report:?}");
        assert!(debug_str.contains("HeadlessReport"));
    }

    #[test]
    fn test_pmat_scores_debug() {
        let ps = PmatScores {
            rust_project_score: 0.0,
            tdg_score: 0.0,
            cuda_tdg_score: 0.0,
            brick_score: 0,
            grade: "F".to_string(),
        };
        let debug_str = format!("{ps:?}");
        assert!(debug_str.contains("PmatScores"));
    }

    #[test]
    fn test_hardware_info_debug() {
        let hw = HardwareInfo {
            gpu: "G".to_string(),
            cpu: "C".to_string(),
            memory_gb: 1,
        };
        let debug_str = format!("{hw:?}");
        assert!(debug_str.contains("HardwareInfo"));
    }

    #[test]
    fn test_throughput_metrics_debug() {
        let tm = ThroughputMetrics {
            tokens_per_sec: 0.0,
            ttft_ms: 0.0,
            cv_percent: 0.0,
            p50_us: 0.0,
            p99_us: 0.0,
        };
        let debug_str = format!("{tm:?}");
        assert!(debug_str.contains("ThroughputMetrics"));
    }

    #[test]
    fn test_brick_score_debug() {
        let bs = BrickScore {
            name: "test".to_string(),
            score: 50,
            grade: "F".to_string(),
            budget_us: 1.0,
            actual_us: 2.0,
            gap_factor: 2.0,
        };
        let debug_str = format!("{bs:?}");
        assert!(debug_str.contains("BrickScore"));
    }
