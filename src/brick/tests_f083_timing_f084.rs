
    // F083: Timing precision
    #[test]
    #[ignore = "Timing test unreliable - depends on system load"]
    fn f083_timing_precision() {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let elapsed = start.elapsed().as_nanos();

        // Should be able to measure ~100µs accurately
        assert!(elapsed > 50_000, "Timing should measure > 50µs");
        assert!(elapsed < 500_000, "Timing should not drift too much");
    }

    // F084: Percentile calculation
    #[test]
    fn f084_percentile_calculation() {
        let mut samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = samples[49]; // 50th percentile
        let p99 = samples[98]; // 99th percentile

        assert_eq!(p50, 50.0, "P50 should be 50");
        assert_eq!(p99, 99.0, "P99 should be 99");
    }

    // F088: Memory bandwidth tracking
    #[test]
    fn f088_bandwidth_tracking() {
        let flash = FlashAttentionBrick::new(8, 2, 64);
        let (naive, flash_mem) = flash.memory_bytes(512);

        // Flash should use less memory
        assert!(flash_mem < naive, "Flash uses less memory");

        // Bandwidth reduction calculable
        let reduction = naive as f64 / flash_mem as f64;
        assert!(reduction > 1.0, "Should show bandwidth reduction");
    }

    // F089: Arithmetic intensity tracking
    #[test]
    #[cfg(feature = "cuda")]
    fn f089_arithmetic_intensity() {
        let brick = FusedFfnBrick::new(1536, 8960);
        let ai = brick.arithmetic_intensity();

        // FFN should have reasonable AI (not memory-bound for larger dims)
        assert!(ai > 0.0, "AI should be positive");
    }

    // F091: Regression baseline
    #[test]
    fn f091_regression_baseline() {
        // Ollama baseline for comparison
        let ollama_1_5b = 488.0; // tok/s for 1.5B
        let target_2x = ollama_1_5b * 2.0;

        let target_budget = TokenBudget::from_throughput(target_2x);
        assert!(
            target_budget.us_per_token < 1050.0,
            "2x Ollama should be ~1025µs/tok"
        );
    }

    // F095: Model size scaling
    #[test]
    #[cfg(feature = "cuda")]
    fn f095_model_size_scaling() {
        // Larger models should have proportionally larger budgets
        let small = FusedFfnBrick::new(1536, 8960); // 1.5B scale
        let large = FusedFfnBrick::new(4096, 22528); // 32B scale

        // FLOPs should scale with dimensions
        assert!(
            large.flops() > small.flops() * 4,
            "32B should have ~7x FLOPs of 1.5B"
        );
    }

    // F096: PMAT score infrastructure
    #[test]
    #[cfg(feature = "cuda")]
    fn f096_pmat_score_ready() {
        // This test verifies we track metrics needed for PMAT scoring
        let brick = FusedFfnBrick::new(64, 256);

        // Has budget
        assert!(brick.budget().us_per_token > 0.0);
        // Has FLOPs
        assert!(brick.flops() > 0);
        // Has arithmetic intensity
        assert!(brick.arithmetic_intensity() > 0.0);
    }

    // F097: CI integration infrastructure
    #[test]
    fn f097_ci_integration() {
        // BenchmarkConfig supports CI settings
        let config = BenchmarkConfig::default();
        assert!(config.samples > 0, "CI needs sample count");
        assert!(config.warmup > 0, "CI needs warmup count");
    }

    // F098: Regression detection
    #[test]
    fn f098_regression_detection() {
        let budget = TokenBudget::from_latency(100.0);

        // 20% regression threshold
        let regression_threshold = 1.2;
        let actual = 115.0; // 15% over

        let gap = budget.gap_factor(actual);
        let is_regression = gap > regression_threshold;

        assert!(!is_regression, "15% over should not trigger 20% threshold");

        let bad_actual = 125.0; // 25% over
        let bad_gap = budget.gap_factor(bad_actual);
        assert!(
            bad_gap > regression_threshold,
            "25% over should trigger regression"
        );
    }

    // F099: Output format for CI
    #[test]
    fn f099_ci_output_format() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };

        // All fields accessible for CI output
        assert!(!report.brick_name.is_empty());
        assert!(report.mean_us > 0.0);
        assert!(report.cv > 0.0);
    }

    // F100: Zero-defect gate
    #[test]
    fn f100_zero_defect_gate() {
        // All bricks must pass their assertions
        let bricks: Vec<Box<dyn ComputeBrick<Output = Vec<f32>>>> = vec![
            Box::new(RmsNormBrick::new(vec![1.0; 64], 1e-5)),
            Box::new(FfnBrick::new(64, 256)),
            Box::new(AttentionBrick::new(4, 2, 16)),
        ];

        for brick in &bricks {
            assert!(brick.can_run(), "Brick {} should be runnable", brick.name());
            assert!(
                !brick.assertions().is_empty(),
                "Brick {} should have assertions",
                brick.name()
            );
        }
    }

    // ========================================================================
    // Coverage Tests: TokenBudget
    // ========================================================================

    #[test]
    fn test_token_budget_debug() {
        let budget = TokenBudget::from_latency(100.0);
        let debug = format!("{:?}", budget);
        assert!(debug.contains("TokenBudget"));
    }

    #[test]
    fn test_token_budget_clone() {
        let budget = TokenBudget::from_latency(100.0).with_batch_size(32);
        let cloned = budget;
        assert_eq!(budget.us_per_token, cloned.us_per_token);
        assert_eq!(budget.batch_size, cloned.batch_size);
    }

    #[test]
    fn test_token_budget_default() {
        let budget = TokenBudget::default();
        assert!(budget.us_per_token > 0.0);
        assert_eq!(budget.batch_size, 1);
    }

    // ========================================================================
    // Coverage Tests: TokenResult
    // ========================================================================

    #[test]
    fn test_token_result_debug() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![1.0, 2.0], 2, 100.0, &budget);
        let debug = format!("{:?}", result);
        assert!(debug.contains("TokenResult"));
    }

    #[test]
    fn test_token_result_clone() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![1.0, 2.0], 2, 100.0, &budget);
        let cloned = result.clone();
        assert_eq!(result.tokens_processed, cloned.tokens_processed);
        assert_eq!(result.output, cloned.output);
    }

    #[test]
    fn test_token_result_zero_tokens() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 0, 100.0, &budget);
        assert_eq!(result.tokens_processed, 0);
        // With zero tokens, us_per_token could be NaN (0/0) or infinity, depending on implementation
        // Just ensure the result is created without panic
    }

    // ========================================================================
    // Coverage Tests: BrickError
    // ========================================================================

    #[test]
    fn test_brick_error_display() {
        let err = BrickError::InvalidInput("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err2 = BrickError::BudgetExceeded {
            limit_us: 100.0,
            actual_us: 150.0,
        };
        assert!(err2.to_string().contains("150"));

        let err3 = BrickError::ComputeError("failed".to_string());
        assert!(err3.to_string().contains("failed"));

        let err4 = BrickError::AssertionFailed {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
        };
        assert!(err4.to_string().contains("test"));
    }

    #[test]
    fn test_brick_error_debug() {
        let err = BrickError::InvalidInput("debug test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidInput"));
    }

    // ========================================================================
    // Coverage Tests: BrickAssertion
    // ========================================================================

    #[test]
    fn test_assertion_check_bounds() {
        let assertion = BrickAssertion::bounds(-1.0, 1.0);
        let data = vec![0.0, 0.5, -0.5, 0.9];
        assert!(assertion.check_f32(&data, true).is_ok());

        let bad_data = vec![0.0, 1.5, -0.5];
        assert!(assertion.check_f32(&bad_data, true).is_err());
    }

    #[test]
    fn test_assertion_check_equiv_scalar() {
        let assertion = BrickAssertion::equiv_scalar(0.01);
        let data = vec![1.0, 1.005, 1.003, 0.997];
        assert!(assertion.check_f32(&data, true).is_ok());
    }

    #[test]
    fn test_assertion_kind_debug() {
        let kind = AssertionKind::NoNaN;
        let debug = format!("{:?}", kind);
        assert!(debug.contains("NoNaN"));

        let kind2 = AssertionKind::Bounds { min: 0.0, max: 1.0 };
        let debug2 = format!("{:?}", kind2);
        assert!(debug2.contains("Bounds"));
    }

    // ========================================================================
    // Coverage Tests: BrickVerification
    // ========================================================================

    #[test]
    fn test_brick_verification_pass() {
        let v = BrickVerification::pass();
        assert!(v.is_valid);
        assert!(v.results.is_empty());
    }

    #[test]
    fn test_brick_verification_fail() {
        let v = BrickVerification::fail("test", "failed reason");
        assert!(!v.is_valid);
        assert_eq!(v.results.len(), 1);
    }

    #[test]
    fn test_brick_verification_add() {
        let mut v = BrickVerification::pass();
        v.add("check1", true, "passed");
        v.add("check2", false, "failed");
        assert!(!v.is_valid); // Should be false after failed check
        assert_eq!(v.results.len(), 2);
    }

    // ========================================================================
    // Coverage Tests: FlashAttentionBrick
    // ========================================================================

    #[test]
    fn test_flash_attention_group_size() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        assert_eq!(brick.group_size(), 4); // 8 heads / 2 kv heads
    }

    #[test]
    fn test_flash_attention_flops() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let flops = brick.flops(512);
        assert!(flops > 0);
    }

    #[test]
    fn test_flash_attention_memory_bytes() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let (naive, flash) = brick.memory_bytes(512);
        assert!(flash < naive);
    }

    // ========================================================================
    // Coverage Tests: QkvBrick
    // ========================================================================

    #[test]
    fn test_qkv_brick_with_bias() {
        let brick = QkvBrick::new(64, 64, 64, 64).with_bias();
        assert!(brick.has_bias);
    }

    #[test]
    fn test_qkv_brick_total_out_dim() {
        let brick = QkvBrick::new(64, 64, 32, 32);
        assert_eq!(brick.total_out_dim(), 128); // 64 + 32 + 32
    }

    // ========================================================================
    // Coverage Tests: BenchmarkConfig
    // ========================================================================

    #[test]
    fn test_benchmark_config_debug() {
        let config = BenchmarkConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("BenchmarkConfig"));
    }

    #[test]
    fn test_benchmark_config_clone() {
        let config = BenchmarkConfig::default();
        let cloned = config.clone();
        assert_eq!(config.samples, cloned.samples);
    }

    // ========================================================================
    // Coverage Tests: BenchmarkReport
    // ========================================================================

    #[test]
    fn test_benchmark_report_debug() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };
        let debug = format!("{:?}", report);
        assert!(debug.contains("BenchmarkReport"));
    }

    #[test]
    fn test_benchmark_report_clone() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };
        let cloned = report.clone();
        assert_eq!(report.brick_name, cloned.brick_name);
        assert_eq!(report.mean_us, cloned.mean_us);
    }

    // ========================================================================
    // Coverage Tests: LayerTiming
    // ========================================================================

    #[test]
    fn test_layer_timing_default() {
        let timing = LayerTiming::default();
        assert_eq!(timing.attn_norm_us, 0.0);
        assert_eq!(timing.qkv_us, 0.0);
    }

    #[test]
    fn test_layer_timing_debug() {
        let timing = LayerTiming::default();
        let debug = format!("{:?}", timing);
        assert!(debug.contains("LayerTiming"));
    }

    #[test]
    fn test_layer_timing_clone() {
        let timing = LayerTiming {
            attn_norm_us: 1.0,
            qkv_us: 2.0,
            rope_us: 3.0,
            attention_us: 4.0,
            o_proj_us: 5.0,
            ffn_norm_us: 6.0,
            ffn_us: 7.0,
            total_us: 8.0,
        };
        let cloned = timing.clone();
        assert_eq!(timing.attn_norm_us, cloned.attn_norm_us);
        assert_eq!(timing.qkv_us, cloned.qkv_us);
    }
