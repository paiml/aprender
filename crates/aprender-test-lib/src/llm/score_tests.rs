    use super::*;

    #[test]
    fn test_higher_is_better_at_excellent() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(160.0, &t), 100);
        assert_eq!(compute_metric_score(200.0, &t), 100); // capped
    }

    #[test]
    fn test_higher_is_better_at_good() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(120.0, &t), 75);
    }

    #[test]
    fn test_higher_is_better_below_good() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        let score = compute_metric_score(60.0, &t);
        assert_eq!(score, 38); // 75 * 60/120 = 37.5 → 38
    }

    #[test]
    fn test_higher_is_better_zero() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(0.0, &t), 0);
    }

    #[test]
    fn test_lower_is_better_at_excellent() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(12.0, &t), 100);
        assert_eq!(compute_metric_score(5.0, &t), 100); // better than excellent
    }

    #[test]
    fn test_lower_is_better_at_good() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(50.0, &t), 75);
    }

    #[test]
    fn test_lower_is_better_above_good() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        let score = compute_metric_score(100.0, &t);
        assert_eq!(score, 38); // 75 * 50/100 = 37.5 → 38
    }

    #[test]
    fn test_error_rate_zero_is_perfect() {
        let t = MetricThreshold {
            excellent: 0.0,
            good: 0.01,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(0.0, &t), 100);
    }

    #[test]
    fn test_error_rate_low_still_high_score() {
        // F-SCORE-007: 0.7% error should score >= 80
        let t = MetricThreshold {
            excellent: 0.0,
            good: 0.01,
            higher_is_better: false,
        };
        let score = compute_metric_score(0.007, &t);
        assert!(
            score >= 80,
            "0.7% error rate scored {score}, expected >= 80"
        );
    }

    #[test]
    fn test_jitter_penalty_clean() {
        let tail = TailAnalysis {
            itl_p999_ms: 7.0,
            itl_p9999_ms: 7.0,
            ttft_p999_ms: 15.0,
            ttft_p9999_ms: 15.0,
            latency_p999_ms: 250.0,
            latency_p9999_ms: 250.0,
            tail_ratio_itl: 1.0,
            tail_ratio_ttft: 1.0,
            tail_ratio_latency: 1.0,
            jitter: super::super::loadtest::JitterAnalysis {
                itl_cv: 0.01,
                itl_iqr_ms: 0.1,
                spike_count: 0,
                spike_threshold_ms: 35.0,
                spikes: vec![],
            },
            drift: super::super::loadtest::DriftAnalysis {
                itl_slope_ms_per_min: 0.0,
                ttft_slope_ms_per_min: 0.0,
                degradation_detected: false,
            },
        };
        assert_eq!(compute_jitter_penalty(&tail), 1); // just 0.01*100 = 1
    }

    #[test]
    fn test_jitter_penalty_spiky() {
        // F-SCORE-003: spiky runtime should get significant penalty
        let tail = TailAnalysis {
            itl_p999_ms: 50.0,
            itl_p9999_ms: 100.0,
            ttft_p999_ms: 15.0,
            ttft_p9999_ms: 15.0,
            latency_p999_ms: 300.0,
            latency_p9999_ms: 350.0,
            tail_ratio_itl: 7.0,
            tail_ratio_ttft: 1.0,
            tail_ratio_latency: 1.2,
            jitter: super::super::loadtest::JitterAnalysis {
                itl_cv: 0.15,
                itl_iqr_ms: 5.0,
                spike_count: 10,
                spike_threshold_ms: 35.0,
                spikes: vec![],
            },
            drift: super::super::loadtest::DriftAnalysis {
                itl_slope_ms_per_min: 0.0,
                ttft_slope_ms_per_min: 0.0,
                degradation_detected: false,
            },
        };
        let penalty = compute_jitter_penalty(&tail);
        assert!(penalty >= 25, "spiky penalty={penalty}, expected >= 25");
        assert!(penalty <= 30, "spiky penalty={penalty}, expected <= 30");
    }

    #[test]
    fn test_grade_assignment() {
        let grades = ScoringContract::default().grades;
        assert_eq!(assign_grade(97.0, &grades), "A+");
        assert_eq!(assign_grade(92.0, &grades), "A");
        assert_eq!(assign_grade(85.0, &grades), "A-");
        assert_eq!(assign_grade(80.0, &grades), "B+");
        assert_eq!(assign_grade(75.0, &grades), "B");
        assert_eq!(assign_grade(60.0, &grades), "C+");
        assert_eq!(assign_grade(50.0, &grades), "C");
        assert_eq!(assign_grade(40.0, &grades), "D");
        assert_eq!(assign_grade(30.0, &grades), "D-");
        assert_eq!(assign_grade(10.0, &grades), "F");
    }

    #[test]
    fn test_no_single_metric_dominates() {
        // F-SCORE-002: zeroing any one metric cannot drop composite below 40
        let contract = ScoringContract::default();
        for (zeroed_metric, _) in &contract.interactive_weights {
            let mut weighted_sum = 0.0;
            for (metric, weight) in &contract.interactive_weights {
                let score = if metric == zeroed_metric { 0.0 } else { 100.0 };
                weighted_sum += weight * score;
            }
            assert!(
                weighted_sum >= 40.0,
                "Zeroing {zeroed_metric} drops composite to {weighted_sum}"
            );
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let contract = ScoringContract::default();
        let interactive_sum: f64 = contract.interactive_weights.values().sum();
        assert!(
            (interactive_sum - 1.0).abs() < 0.001,
            "Interactive weights sum to {interactive_sum}"
        );
        let throughput_sum: f64 = contract.throughput_weights.values().sum();
        assert!(
            (throughput_sum - 1.0).abs() < 0.001,
            "Throughput weights sum to {throughput_sum}"
        );
    }

    #[test]
    fn test_score_independence_from_field() {
        // F-SCORE-001: Adding/removing a runtime changes scores by at most the bonus amount
        let contract = ScoringContract::default();

        // Create two fake results
        let result_a = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        let result_b = make_test_result("runtime_b", 130.0, 30.0, 8.0, 40.0, 0.0, 1);
        let result_c = make_test_result("runtime_c", 100.0, 60.0, 12.0, 80.0, 0.01, 1);

        let card_abc = compute_scorecard(
            &[
                (result_a.clone(), "a.json".into()),
                (result_b.clone(), "b.json".into()),
                (result_c.clone(), "c.json".into()),
            ],
            None,
            &contract,
        );

        let card_ab = compute_scorecard(
            &[
                (result_a.clone(), "a.json".into()),
                (result_b.clone(), "b.json".into()),
            ],
            None,
            &contract,
        );

        let score_a_with_bc = card_abc
            .runtimes
            .iter()
            .find(|r| r.name == "runtime_a")
            .unwrap()
            .composite;
        let score_a_with_b = card_ab
            .runtimes
            .iter()
            .find(|r| r.name == "runtime_a")
            .unwrap()
            .composite;

        let diff = (score_a_with_bc - score_a_with_b).abs();
        assert!(
            diff <= f64::from(contract.best_in_class_bonus),
            "Score changed by {diff} when removing runtime_c (max allowed: {})",
            contract.best_in_class_bonus
        );
    }

    fn make_test_result(
        name: &str,
        decode: f64,
        ttft: f64,
        itl: f64,
        ttft_p99: f64,
        error_rate: f64,
        concurrency: usize,
    ) -> LoadTestResult {
        LoadTestResult {
            total_requests: 100,
            successful: (100.0 * (1.0 - error_rate)) as u64,
            failed: (100.0 * error_rate) as u64,
            throughput_rps: decode / 32.0,
            latency_p50_ms: ttft + itl * 31.0,
            latency_p95_ms: ttft + itl * 31.0 * 1.1,
            latency_p99_ms: ttft + itl * 31.0 * 1.2,
            ttft_p50_ms: ttft,
            tokens_per_sec: decode * concurrency as f64,
            avg_tok_per_req: 32.0,
            itl_p50_ms: itl,
            decode_tok_per_sec: decode,
            prefill_tok_per_sec: 1000.0 / ttft * 23.0,
            timestamp: "2026-03-11T00:00:00Z".into(),
            runtime_name: name.into(),
            elapsed_secs: 60.0,
            concurrency,
            ttft_p90_ms: ttft * 1.1,
            ttft_p95_ms: ttft * 1.2,
            ttft_p99_ms: ttft_p99,
            tpot_p50_ms: itl,
            tpot_p90_ms: itl * 1.1,
            tpot_p95_ms: itl * 1.2,
            tpot_p99_ms: itl * 1.3,
            latency_min_ms: ttft + itl * 30.0,
            latency_max_ms: ttft + itl * 35.0,
            latency_stddev_ms: itl * 0.5,
            error_rate,
            prompt_tokens_total: 2300,
            completion_tokens_total: 3200,
            truncated_pct: 0.0,
            sse_batch_ratio: 1.0,
            goodput_pct: 100.0,
            output_tokens_dist: None,
            decode_us_per_layer: None,
            num_layers: Some(28),
            brick_trace_summary: None,
            request_details: vec![],
            quality: None,
            tail_analysis: None,
            gpu_telemetry: None,
            dataset_stats: None,
            cold_start_ms: None,
        }
    }

    fn make_test_result_with_layers(
        name: &str,
        decode: f64,
        ttft: f64,
        us_per_layer: f64,
        prompt_tokens: u64,
    ) -> LoadTestResult {
        let mut r = make_test_result(name, decode, ttft, 7.0, 20.0, 0.0, 1);
        r.decode_us_per_layer = Some(us_per_layer);
        r.prompt_tokens_total = prompt_tokens;
        r
    }

    #[test]
    fn test_layer_scoring_best_first() {
        let contract = ScoringContract::default();
        let results = vec![
            (
                make_test_result_with_layers("fast", 160.0, 12.0, 220.0, 2300),
                "a.json".into(),
            ),
            (
                make_test_result_with_layers("slow", 100.0, 50.0, 350.0, 2300),
                "b.json".into(),
            ),
        ];
        let card = compute_layer_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 2);
        assert_eq!(card.runtimes[0].name, "fast");
        assert!(card.runtimes[0].best);
        assert!(card.runtimes[0].score > card.runtimes[1].score);
    }

    #[test]
    fn test_layer_scoring_excellent_threshold() {
        let contract = ScoringContract::default();
        let results = vec![(
            make_test_result_with_layers("vllm", 160.0, 12.0, 220.0, 2300),
            "a.json".into(),
        )];
        let card = compute_layer_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes[0].score, 100);
    }

    #[test]
    fn test_prompt_category_classification() {
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(10.0),
            PromptCategory::Micro
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(23.0),
            PromptCategory::Short
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(102.0),
            PromptCategory::Medium
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(512.0),
            PromptCategory::Long
        );
    }

    #[test]
    fn test_profile_consistency_perfect() {
        let contract = ScoringContract::default();
        // Same runtime, same metrics, different prompt lengths
        let r_short = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 2300);
        let mut r_medium = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 10200);
        r_medium.prompt_tokens_total = 10200; // 102 avg prompt tokens
        let results = vec![
            (r_short, "short.json".into()),
            (r_medium, "medium.json".into()),
        ];
        let card = compute_profile_scorecard(&results, &contract);
        assert!(card.entries.len() >= 2);
        // Same metrics → consistency should be 100%
        if let Some(cs) = card.consistency.first() {
            assert_eq!(cs.consistency, 100.0);
        }
    }

    #[test]
    fn test_correctness_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.quality = Some(super::super::loadtest::QualityResult {
            validation_level: "basic".into(),
            total_validated: 100,
            passed: 100,
            failed: 0,
            pass_rate: 1.0,
            failures: vec![],
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_correctness_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        assert_eq!(card.runtimes[0].score, 100);
    }

    #[test]
    fn test_correctness_partial() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.quality = Some(super::super::loadtest::QualityResult {
            validation_level: "basic".into(),
            total_validated: 100,
            passed: 90,
            failed: 10,
            pass_rate: 0.9,
            failures: vec![],
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_correctness_scorecard(&results, &contract.grades);
        assert!(
            card.runtimes[0].score < 75,
            "90% pass rate should score below good"
        );
    }

    #[test]
    fn test_output_length_classification() {
        assert_eq!(
            OutputLengthCategory::from_tokens(10),
            OutputLengthCategory::Short
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(32),
            OutputLengthCategory::Medium
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(128),
            OutputLengthCategory::Medium
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(200),
            OutputLengthCategory::Long
        );
    }

    #[test]
    fn test_memory_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.gpu_telemetry = Some(super::super::loadtest::GpuTelemetry {
            samples: 10,
            gpu_utilization_pct: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 95.0,
                min: 60.0,
            },
            memory_used_mb: super::super::loadtest::TelemetryStat {
                mean: 3200.0,
                max: 3500.0,
                min: 3000.0,
            },
            memory_total_mb: 8192.0,
            power_draw_w: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 100.0,
                min: 60.0,
            },
            temperature_c: super::super::loadtest::TelemetryStat {
                mean: 70.0,
                max: 80.0,
                min: 50.0,
            },
            clock_gpu_mhz: super::super::loadtest::TelemetryStat {
                mean: 1500.0,
                max: 1500.0,
                min: 1500.0,
            },
            throttle_events: 0,
            energy_total_wh: 1.0,
            energy_per_token_mj: 5.0,
            energy_per_request_mj: 160.0,
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_memory_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 140 tok/s / 3.42 GB = ~40.9 tok/s/GB → excellent
        assert!(
            card.runtimes[0].score >= 95,
            "High efficiency should score well: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_cold_start_scoring() {
        let contract = ScoringContract::default();
        let mut r_fast = make_test_result("realizr", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r_fast.cold_start_ms = Some(300.0);
        let mut r_slow = make_test_result("vllm", 160.0, 12.0, 6.0, 15.0, 0.0, 1);
        r_slow.cold_start_ms = Some(15000.0);
        let results = vec![(r_fast, "a.json".into()), (r_slow, "b.json".into())];
        let card = compute_cold_start_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 2);
        assert_eq!(card.runtimes[0].name, "realizr"); // fastest first
        assert!(card.runtimes[0].score > card.runtimes[1].score);
    }

    #[test]
    fn test_power_efficiency_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.gpu_telemetry = Some(super::super::loadtest::GpuTelemetry {
            samples: 10,
            gpu_utilization_pct: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 95.0,
                min: 60.0,
            },
            memory_used_mb: super::super::loadtest::TelemetryStat {
                mean: 3200.0,
                max: 3500.0,
                min: 3000.0,
            },
            memory_total_mb: 8192.0,
            power_draw_w: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 100.0,
                min: 60.0,
            },
            temperature_c: super::super::loadtest::TelemetryStat {
                mean: 70.0,
                max: 80.0,
                min: 50.0,
            },
            clock_gpu_mhz: super::super::loadtest::TelemetryStat {
                mean: 1500.0,
                max: 1500.0,
                min: 1500.0,
            },
            throttle_events: 0,
            energy_total_wh: 1.0,
            energy_per_token_mj: 5.0,
            energy_per_request_mj: 160.0,
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_power_efficiency_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 140 tok/s / 80W = 1.75 tok/s/W → above good
        assert!(
            card.runtimes[0].score >= 75,
            "1.75 tok/s/W should be above good: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_concurrency_scaling() {
        let contract = ScoringContract::default();
        let r_c1 = make_test_result("runtime_a-c1", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        let mut r_c4 = make_test_result("runtime_a-c4", 140.0, 30.0, 8.0, 40.0, 0.0, 4);
        r_c4.tokens_per_sec = 540.0; // aggregate = 540
        let results = vec![(r_c1, "c1.json".into()), (r_c4, "c4.json".into())];
        let card = compute_concurrency_scaling_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 540 / (150 * 4) = 0.90 → excellent
        assert!(card.runtimes[0].scaling_efficiency > 0.85);
        assert!(
            card.runtimes[0].score >= 90,
            "Near-linear scaling: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_profile_consistency_degradation() {
        let contract = ScoringContract::default();
        // Good on short, bad on medium (TTFT degrades)
        let r_short = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 2300);
        let mut r_medium = make_test_result_with_layers("runtime_a", 140.0, 80.0, 240.0, 10200);
        r_medium.prompt_tokens_total = 10200;
        let results = vec![
            (r_short, "short.json".into()),
            (r_medium, "medium.json".into()),
        ];
        let card = compute_profile_scorecard(&results, &contract);
        if let Some(cs) = card.consistency.first() {
            assert!(
                cs.consistency < 90.0,
                "Expected degradation, got {}%",
                cs.consistency
            );
            assert!(cs.worst_score < cs.best_score);
        }
    }
