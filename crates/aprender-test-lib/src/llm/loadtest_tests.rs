    use super::*;

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 0.5), 42.0);
        assert_eq!(percentile(&[42.0], 0.99), 42.0);
    }

    #[test]
    fn test_percentile_multiple() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        // Linear interpolation: idx = 99 * p, lerp between floor and ceil
        // p50: idx=49.5, lerp(50, 51, 0.5) = 50.5
        assert!((percentile(&data, 0.50) - 50.5).abs() < 0.01);
        // p95: idx=94.05, lerp(95, 96, 0.05) = 95.05
        assert!((percentile(&data, 0.95) - 95.05).abs() < 0.01);
        // p99: idx=98.01, lerp(99, 100, 0.01) = 99.01
        assert!((percentile(&data, 0.99) - 99.01).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_empty() {
        let result = aggregate_results(&[], 10.0, "test", 1, None, None, None, None);
        assert_eq!(result.total_requests, 0);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(result.throughput_rps, 0.0);
        assert_eq!(result.latency_p50_ms, 0.0);
        assert_eq!(result.error_rate, 0.0);
        assert_eq!(result.prompt_tokens_total, 0);
        assert_eq!(result.completion_tokens_total, 0);
    }

    #[test]
    fn test_aggregate_all_success() {
        let records: Vec<RequestRecord> = (0..10)
            .map(|i| RequestRecord {
                latency: Duration::from_millis(100 + i * 10),
                ttfb: Duration::from_millis(50 + i * 5),
                tokens: 20,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            })
            .collect();
        let result = aggregate_results(&records, 10.0, "realizar", 2, None, None, None, None);
        assert_eq!(result.total_requests, 10);
        assert_eq!(result.successful, 10);
        assert_eq!(result.failed, 0);
        assert!((result.throughput_rps - 1.0).abs() < f64::EPSILON);
        assert!(result.latency_p50_ms > 0.0);
        assert!(result.tokens_per_sec > 0.0);
        // GH-23: normalized metrics
        assert!((result.avg_tok_per_req - 20.0).abs() < f64::EPSILON);
        assert!(result.itl_p50_ms > 0.0);
        assert!(result.decode_tok_per_sec > 0.0);
        assert_eq!(result.runtime_name, "realizar");
        assert_eq!(result.concurrency, 2);
        // Extended percentiles
        assert!(result.ttft_p90_ms > 0.0);
        assert!(result.ttft_p95_ms > 0.0);
        assert!(result.ttft_p99_ms > 0.0);
        assert!(result.tpot_p50_ms > 0.0);
        assert!(result.latency_min_ms > 0.0);
        assert!(result.latency_max_ms >= result.latency_min_ms);
        assert!(result.latency_stddev_ms >= 0.0);
        assert!((result.error_rate).abs() < f64::EPSILON);
        assert_eq!(result.prompt_tokens_total, 100);
        assert_eq!(result.completion_tokens_total, 200);
    }

    #[test]
    fn test_aggregate_mixed() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(0),
                ttfb: Duration::from_millis(0),
                tokens: 0,
                prompt_tokens: 0,
                success: false,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 5.0, "ollama", 1, None, None, None, None);
        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 1);
        assert!((result.error_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_config() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.prompts.len(), 1);
        assert_eq!(config.warmup_duration, Duration::ZERO);
    }

    #[test]
    fn test_default_prompt() {
        let p = default_prompt();
        assert_eq!(p.messages.len(), 1);
        assert_eq!(p.messages[0].role, Role::User);
        assert_eq!(p.temperature, Some(0.0));
    }

    #[test]
    fn test_load_test_result_serialization() {
        let result = LoadTestResult {
            total_requests: 100,
            successful: 95,
            failed: 5,
            throughput_rps: 10.0,
            latency_p50_ms: 150.0,
            latency_p95_ms: 300.0,
            latency_p99_ms: 500.0,
            ttft_p50_ms: 80.0,
            tokens_per_sec: 200.0,
            avg_tok_per_req: 15.0,
            itl_p50_ms: 5.0,
            decode_tok_per_sec: 200.0,
            prefill_tok_per_sec: 0.0,
            timestamp: "2026-03-01T00:00:00Z".to_string(),
            runtime_name: "realizar".to_string(),
            elapsed_secs: 10.0,
            concurrency: 4,
            ttft_p90_ms: 90.0,
            ttft_p95_ms: 95.0,
            ttft_p99_ms: 99.0,
            tpot_p50_ms: 6.0,
            tpot_p90_ms: 8.0,
            tpot_p95_ms: 9.0,
            tpot_p99_ms: 12.0,
            latency_min_ms: 50.0,
            latency_max_ms: 800.0,
            latency_stddev_ms: 120.0,
            error_rate: 0.05,
            prompt_tokens_total: 950,
            completion_tokens_total: 1425,
            truncated_pct: 0.0,
            sse_batch_ratio: 0.0,
            goodput_pct: 0.0,
            decode_us_per_layer: None,
            num_layers: None,
            output_tokens_dist: None,
            brick_trace_summary: None,
            request_details: Vec::new(),
            quality: None,
            tail_analysis: None,
            gpu_telemetry: None,
            dataset_stats: None,
            cold_start_ms: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: LoadTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_requests, 100);
        assert_eq!(back.runtime_name, "realizar");
        assert!((back.avg_tok_per_req - 15.0).abs() < f64::EPSILON);
        assert!((back.itl_p50_ms - 5.0).abs() < f64::EPSILON);
        assert!((back.decode_tok_per_sec - 200.0).abs() < f64::EPSILON);
        assert!((back.tpot_p50_ms - 6.0).abs() < f64::EPSILON);
        assert!((back.error_rate - 0.05).abs() < f64::EPSILON);
        assert_eq!(back.prompt_tokens_total, 950);
        assert_eq!(back.completion_tokens_total, 1425);
    }

    #[test]
    fn test_load_test_result_backwards_compat() {
        // Old JSON without new fields should deserialize with defaults
        let json = r#"{
            "total_requests": 50,
            "successful": 50,
            "failed": 0,
            "throughput_rps": 5.0,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 200.0,
            "latency_p99_ms": 300.0,
            "ttft_p50_ms": 50.0,
            "tokens_per_sec": 100.0,
            "timestamp": "2026-01-01T00:00:00Z",
            "runtime_name": "old",
            "elapsed_secs": 10.0,
            "concurrency": 1
        }"#;
        let result: LoadTestResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.total_requests, 50);
        assert_eq!(result.tpot_p50_ms, 0.0);
        assert_eq!(result.error_rate, 0.0);
        assert_eq!(result.prompt_tokens_total, 0);
    }

    #[test]
    fn test_percentile_boundary() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 1.0), 3.0);
    }

    #[test]
    fn test_itl_streaming() {
        // GH-23: Streaming mode — ITL = (latency - ttfb) / (tokens - 1)
        // Request: 200ms latency, 50ms ttfb, 16 tokens
        // ttfb/latency = 0.25 < 0.95 → streaming detected
        // Decode time = 200 - 50 = 150ms, ITL = 150 / 15 = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.itl_p50_ms - 10.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 100.0).abs() < 1.0);
        assert!((result.avg_tok_per_req - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_itl_non_streaming() {
        // GH-23: Non-streaming — ttfb ≈ latency, fallback to latency/tokens
        // Request: 1600ms latency, 1599ms ttfb, 16 tokens
        // ttfb/latency = 0.999 > 0.95 → non-streaming detected
        // ITL proxy = 1600 / 16 = 100ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(1600),
            ttfb: Duration::from_millis(1599),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.itl_p50_ms - 100.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_itl_single_token_excluded() {
        // GH-23: Requests with < 2 tokens should be excluded from ITL
        // (can't compute inter-token latency with 0 or 1 token)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(100),
            tokens: 1,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.itl_p50_ms, 0.0);
        assert_eq!(result.decode_tok_per_sec, 0.0);
        assert!((result.avg_tok_per_req - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_zero_elapsed() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 0.0, "test", 1, None, None, None, None);
        assert_eq!(result.throughput_rps, 0.0);
        assert_eq!(result.tokens_per_sec, 0.0);
    }

    #[test]
    fn test_stddev() {
        assert_eq!(stddev(&[]), 0.0);
        assert_eq!(stddev(&[5.0]), 0.0);
        // [10, 20, 30]: mean=20, var=((100+0+100)/2)=100, stddev=10
        let sd = stddev(&[10.0, 20.0, 30.0]);
        assert!((sd - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_tpot_computation() {
        // TPOT = (latency - ttfb) / (tokens - 1)
        // Streaming: 200ms latency, 50ms ttfb, 16 tokens
        // TPOT = (200 - 50) / 15 = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.tpot_p50_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_latency_min_max_stddev() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(300),
                ttfb: Duration::from_millis(100),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert!((result.latency_min_ms - 100.0).abs() < 0.1);
        assert!((result.latency_max_ms - 300.0).abs() < 0.1);
        assert!(result.latency_stddev_ms > 0.0);
    }

    #[test]
    fn test_prompt_tokens_tracking() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 20,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 15,
                prompt_tokens: 25,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.prompt_tokens_total, 45);
        assert_eq!(result.completion_tokens_total, 25);
    }

    #[test]
    fn test_tpot_from_streaming_timestamps() {
        // GH-24: When token_timestamps are available, TPOT uses real per-token deltas.
        // 5 tokens arriving at 50ms, 60ms, 70ms, 80ms, 90ms
        // Inter-token deltas: 10ms, 10ms, 10ms, 10ms → mean TPOT = 10ms
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 5,
            prompt_tokens: 10,
            success: true,
            token_timestamps: vec![
                Duration::from_millis(50),
                Duration::from_millis(60),
                Duration::from_millis(70),
                Duration::from_millis(80),
                Duration::from_millis(90),
            ],
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Real TPOT from timestamps: mean of [10, 10, 10, 10] = 10ms
        assert!((result.tpot_p50_ms - 10.0).abs() < 0.1);
        // ITL also uses real timestamps
        assert!((result.itl_p50_ms - 10.0).abs() < 0.1);
        assert!((result.decode_tok_per_sec - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_tpot_mixed_streaming_and_non_streaming() {
        // GH-24: When some records have timestamps and some don't,
        // only records with timestamps >= 2 are used for streaming TPOT.
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(200),
                ttfb: Duration::from_millis(50),
                tokens: 4,
                prompt_tokens: 10,
                success: true,
                token_timestamps: vec![
                    Duration::from_millis(50),
                    Duration::from_millis(70),
                    Duration::from_millis(90),
                    Duration::from_millis(110),
                ],
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 5,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(), // non-streaming request
                brick_trace: None,
                finish_reason: None,
                response_content: None,
            },
        ];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Only the first record with timestamps is used for TPOT
        // Deltas: [20, 20, 20] → mean TPOT = 20ms
        assert!((result.tpot_p50_ms - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_stream_config_default() {
        let config = LoadTestConfig::default();
        assert!(!config.stream);
    }

    #[test]
    fn test_tpot_non_streaming_uses_latency_per_token() {
        // Non-streaming: ttfb ≈ latency → TPOT should use latency/tokens (not near-zero).
        // Before fix: TPOT = (latency - ttfb)/(tokens-1) = (1600-1599)/15 = 0.067ms (WRONG)
        // After fix: TPOT = latency/tokens = 1600/16 = 100ms (correct, matches ITL)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(1600),
            ttfb: Duration::from_millis(1599),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Both TPOT and ITL should be latency/tokens = 100ms
        assert!(
            (result.tpot_p50_ms - 100.0).abs() < 0.1,
            "tpot={}",
            result.tpot_p50_ms
        );
        assert!(
            (result.itl_p50_ms - 100.0).abs() < 0.1,
            "itl={}",
            result.itl_p50_ms
        );
    }

    #[test]
    fn test_itl_robust_to_token_batching() {
        // Server sends tokens in pairs (batch=2): timestamps are [100, 100, 200, 200, 300]
        // Old code (flat_map): deltas = [0, 100, 0, 100] → P50 = 50ms (bimodal, fragile)
        // New code (per-request mean): (300-100)/4 = 50ms (robust)
        // With batch=3: timestamps = [100, 100, 100, 300, 300, 300]
        // Old code: deltas = [0, 0, 200, 0, 0] → P50 = 0ms (WRONG)
        // New code: (300-100)/5 = 40ms (correct)
        let records = vec![RequestRecord {
            latency: Duration::from_millis(350),
            ttfb: Duration::from_millis(100),
            tokens: 6,
            prompt_tokens: 10,
            success: true,
            token_timestamps: vec![
                Duration::from_millis(100), // batch 1
                Duration::from_millis(100),
                Duration::from_millis(100),
                Duration::from_millis(300), // batch 2
                Duration::from_millis(300),
                Duration::from_millis(300),
            ],
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        // Per-request mean: (300-100)/5 = 40ms
        assert!(
            (result.itl_p50_ms - 40.0).abs() < 0.1,
            "itl={}",
            result.itl_p50_ms
        );
        assert!(
            (result.tpot_p50_ms - 40.0).abs() < 0.1,
            "tpot={}",
            result.tpot_p50_ms
        );
        assert!(
            (result.decode_tok_per_sec - 25.0).abs() < 0.5,
            "decode={}",
            result.decode_tok_per_sec
        );
    }

    #[test]
    fn test_request_details_populated() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(200),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let result = aggregate_results(&records, 1.0, "test", 1, None, None, None, None);
        assert_eq!(result.request_details.len(), 1);
        let detail = &result.request_details[0];
        assert!((detail.latency_ms - 200.0).abs() < 0.1);
        assert!((detail.ttft_ms - 50.0).abs() < 0.1);
        assert_eq!(detail.completion_tokens, 16);
        assert_eq!(detail.prompt_tokens, 10);
        assert!(detail.itl_ms > 0.0);
    }

    // =========================================================================
    // Feature 5: Quality validation tests
    // =========================================================================

    #[test]
    fn test_quality_basic_all_pass() {
        let records = vec![
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
            RequestRecord {
                latency: Duration::from_millis(120),
                ttfb: Duration::from_millis(60),
                tokens: 8,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
        ];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.total_validated, 2);
        assert_eq!(quality.passed, 2);
        assert_eq!(quality.failed, 0);
        assert!((quality.pass_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_basic_zero_tokens() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(100),
            tokens: 0,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: None,
        }];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.failed, 1);
        assert_eq!(quality.failures[0].reason, "zero_tokens");
    }

    #[test]
    fn test_quality_basic_no_finish_reason() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        assert_eq!(quality.failed, 1);
        assert_eq!(quality.failures[0].reason, "no_finish_reason");
    }

    #[test]
    fn test_quality_contains_match() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: Some("hello world".to_string()),
        }];
        let quality = compute_quality(&records, &ValidationMode::Contains("hello".to_string()));
        assert_eq!(quality.passed, 1);
        assert_eq!(quality.failed, 0);
    }

    #[test]
    fn test_quality_contains_mismatch() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 10,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: Some("goodbye world".to_string()),
        }];
        let quality = compute_quality(&records, &ValidationMode::Contains("hello".to_string()));
        assert_eq!(quality.failed, 1);
        assert!(quality.failures[0].reason.starts_with("missing_substring:"));
    }

    #[test]
    fn test_quality_none_skipped() {
        let records = vec![RequestRecord {
            latency: Duration::from_millis(100),
            ttfb: Duration::from_millis(50),
            tokens: 0,
            prompt_tokens: 5,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: None,
            response_content: None,
        }];
        // ValidationMode::None should still return results if called directly
        let quality = compute_quality(&records, &ValidationMode::None);
        // But in practice, LoadTest::run() skips calling compute_quality when mode is None
        assert_eq!(quality.validation_level, "none");
    }

    #[test]
    fn test_quality_skips_failed_requests() {
        let records = vec![
            failed_record(), // success: false
            RequestRecord {
                latency: Duration::from_millis(100),
                ttfb: Duration::from_millis(50),
                tokens: 10,
                prompt_tokens: 5,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            },
        ];
        let quality = compute_quality(&records, &ValidationMode::Basic);
        // Only the successful request should be validated
        assert_eq!(quality.total_validated, 1);
        assert_eq!(quality.passed, 1);
    }

    // =========================================================================
    // Feature 3: Tail latency analysis tests
    // =========================================================================

    #[test]
    fn test_tail_analysis_basic() {
        let records: Vec<RequestRecord> = (0..100)
            .map(|i| RequestRecord {
                latency: Duration::from_millis(100 + i),
                ttfb: Duration::from_millis(50 + i / 2),
                tokens: 20,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            })
            .collect();
        let tail = compute_tail_analysis(&records, 5.0);
        // P99.9 should be near the max
        assert!(tail.latency_p999_ms > 0.0);
        assert!(tail.ttft_p999_ms > 0.0);
        // Tail ratios should be computed
        assert!(tail.tail_ratio_latency > 0.0);
    }

    #[test]
    fn test_spike_detection() {
        // Create records with one outlier
        let mut records: Vec<RequestRecord> = (0..50)
            .map(|_| RequestRecord {
                latency: Duration::from_millis(200),
                ttfb: Duration::from_millis(50),
                tokens: 16,
                prompt_tokens: 10,
                success: true,
                token_timestamps: Vec::new(),
                brick_trace: None,
                finish_reason: Some("stop".to_string()),
                response_content: None,
            })
            .collect();
        // Add a spike (10x normal latency)
        records.push(RequestRecord {
            latency: Duration::from_millis(2000),
            ttfb: Duration::from_millis(50),
            tokens: 16,
            prompt_tokens: 10,
            success: true,
            token_timestamps: Vec::new(),
            brick_trace: None,
            finish_reason: Some("stop".to_string()),
            response_content: None,
        });
        let tail = compute_tail_analysis(&records, 5.0);
        // The spike should be detected (its ITL is much higher than median)
        assert!(tail.jitter.spike_threshold_ms > 0.0);
    }

    #[test]
    fn test_linear_regression() {
        // Perfect positive slope: y = 2x
        let values: Vec<f64> = (0..10).map(|x| 2.0 * x as f64).collect();
        let (slope, r2) = linear_regression(&values);
        assert!((slope - 2.0).abs() < 0.01);
        assert!((r2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_flat() {
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (slope, _r2) = linear_regression(&values);
        assert!(slope.abs() < 0.01);
    }

    #[test]
    fn test_validation_mode_parse() {
        assert!(matches!(
            ValidationMode::parse("none"),
            ValidationMode::None
        ));
        assert!(matches!(
            ValidationMode::parse("basic"),
            ValidationMode::Basic
        ));
        if let ValidationMode::Contains(s) = ValidationMode::parse("contains:hello") {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected Contains");
        }
        if let ValidationMode::Pattern(p) = ValidationMode::parse("pattern:\\d+") {
            assert_eq!(p, "\\d+");
        } else {
            panic!("Expected Pattern");
        }
    }

    #[test]
    fn test_tail_analysis_empty() {
        let records: Vec<RequestRecord> = Vec::new();
        let tail = compute_tail_analysis(&records, 5.0);
        assert_eq!(tail.itl_p999_ms, 0.0);
        assert_eq!(tail.jitter.spike_count, 0);
        assert!(!tail.drift.degradation_detected);
    }
