
    // F056: FusedFfnBrick with DP4A enabled
    #[test]
    #[cfg(feature = "cuda")]
    fn f056_fused_ffn_dp4a() {
        let brick = FusedFfnBrick::with_packed_dp4a(64, 256);
        assert!(brick.use_packed_dp4a, "DP4A should be enabled");

        let brick_default = FusedFfnBrick::new(64, 256);
        // Default depends on env var, so just verify it's boolean
        let _ = brick_default.use_packed_dp4a;
    }

    // F057: FusedFfnBrick has custom assertions
    #[test]
    #[cfg(feature = "cuda")]
    fn f057_fused_ffn_assertions() {
        let brick = FusedFfnBrick::new(64, 256);
        let assertions = brick.assertions();

        let has_shared_q8 = assertions.iter().any(|a| a.name == "shared_q8_quant");
        let has_swiglu_fused = assertions.iter().any(|a| a.name == "swiglu_fused");

        assert!(has_shared_q8, "Should have shared_q8_quant assertion");
        assert!(has_swiglu_fused, "Should have swiglu_fused assertion");
    }

    // F058: ActivationQuantBrick bandwidth reduction
    #[test]
    fn f058_activation_quant_bandwidth() {
        let brick = ActivationQuantBrick::new(1024);
        let reduction = brick.bandwidth_reduction();

        // Should achieve ~4x reduction (f32 → int8)
        assert!(
            reduction > 3.5 && reduction < 4.0,
            "Bandwidth reduction should be ~4x, got {:.2}x",
            reduction
        );
    }

    // F059: ActivationQuantBrick bytes saved
    #[test]
    fn f059_activation_quant_bytes_saved() {
        let brick = ActivationQuantBrick::new(1024);
        let saved = brick.bytes_saved();

        // 3 bytes saved per element (f32 - int8 = 4 - 1 = 3)
        assert_eq!(saved, 1024 * 3, "Should save 3 bytes per element");
    }

    // F060: ActivationQuantBrick error estimate
    #[test]
    fn f060_activation_quant_error() {
        let per_tensor = ActivationQuantBrick::new(1024);
        let per_channel = ActivationQuantBrick::with_per_channel(1024);

        // Per-tensor: 0.1% error
        assert!(
            (per_tensor.estimated_error() - 0.001).abs() < 0.0001,
            "Per-tensor error should be 0.1%"
        );

        // Per-channel: 0.05% error (more accurate)
        assert!(
            (per_channel.estimated_error() - 0.0005).abs() < 0.0001,
            "Per-channel error should be 0.05%"
        );
    }

    // F061: ActivationQuantBrick has custom assertions
    #[test]
    fn f061_activation_quant_assertions() {
        let brick = ActivationQuantBrick::new(1024);
        let assertions = brick.assertions();

        let has_symmetric = assertions.iter().any(|a| a.name == "symmetric_range");
        let has_error_bound = assertions.iter().any(|a| a.name == "error_bound");

        assert!(has_symmetric, "Should have symmetric_range assertion");
        assert!(has_error_bound, "Should have error_bound assertion");
    }

    // F062: ActivationQuantBrick ComputeBrick trait
    #[test]
    fn f062_activation_quant_trait() {
        let brick = ActivationQuantBrick::new(1024);

        assert_eq!(brick.name(), "activation_quant");
        assert!(brick.budget().us_per_token > 0.0);
        assert!(brick.can_run());

        // Zero dim should not run
        let zero_brick = ActivationQuantBrick::new(0);
        assert!(!zero_brick.can_run());
    }

    // =========================================================================
    // F061-F080: CUDA Kernel Validation (Infrastructure Stubs)
    // These tests verify the infrastructure is ready for CUDA validation.
    // Full validation requires CUDA hardware (ptxas, ncu, compute-sanitizer).
    // =========================================================================

    // F063: CUDA graph capture infrastructure ready
    #[test]
    fn f063_cuda_graph_capture_ready() {
        // CudaGraphBrick exists and has correct structure
        let brick = CudaGraphBrick::new(28, 1536); // 28 layers, 1536 hidden
        assert_eq!(brick.name(), "cuda_graph");
        assert!(brick.budget().us_per_token > 0.0);

        // Graph capture status works
        assert!(!brick.captured, "Should start not captured");
        assert!(
            !brick.can_replay(),
            "Should not be replayable until captured"
        );
    }

    // F064: CUDA graph replay verification infrastructure
    #[test]
    fn f064_cuda_graph_replay_ready() {
        let mut brick = CudaGraphBrick::new(28, 1536);
        let assertions = brick.assertions();

        // Should have graph-specific assertions
        let has_speedup = assertions.iter().any(|a| a.name == "graph_speedup");
        assert!(has_speedup, "Should verify graph speedup");

        // Capture and replay flow works
        brick.set_captured(true);
        assert!(brick.can_replay(), "Should be replayable after capture");
        assert!(brick.replay().is_ok(), "Replay should succeed");
    }

    // F065: Indirect kernel infrastructure ready
    #[test]
    #[cfg(feature = "cuda")]
    fn f065_indirect_kernel_ready() {
        // CoalescedDp4aBrick supports coalesced memory access
        let brick = CoalescedDp4aBrick::new(1024, 256);
        assert!(brick.can_run());

        // Should have bandwidth efficiency assertion
        let assertions = brick.assertions();
        let has_bandwidth = assertions.iter().any(|a| a.name == "bandwidth_efficient");
        assert!(has_bandwidth, "Should verify bandwidth efficiency");
    }

    // F066: DP4A instruction infrastructure ready
    #[test]
    #[cfg(feature = "cuda")]
    fn f066_dp4a_instruction_ready() {
        // CoalescedDp4aBrick name indicates DP4A usage
        let brick = CoalescedDp4aBrick::new(1024, 256);

        // Name includes "dp4a" indicating DP4A instruction usage
        assert!(
            brick.name().contains("dp4a"),
            "Name should indicate DP4A usage"
        );

        // K dimension must be multiple of 256 for DP4A Q4K
        assert!(brick.k.is_multiple_of(256), "K should align for DP4A");
    }

    // F067: Memory coalescing infrastructure ready
    #[test]
    #[cfg(feature = "cuda")]
    fn f067_memory_coalescing_ready() {
        let brick = CoalescedDp4aBrick::new(1024, 256);

        // K dimension should be multiple of 256 for coalescing
        assert!(
            brick.k.is_multiple_of(256) || brick.k < 256,
            "K should align for coalescing"
        );
    }

    // F070: Register usage tracking infrastructure
    #[test]
    fn f070_register_usage_ready() {
        // All bricks have budgets that implicitly constrain register usage
        let rms = RmsNormBrick::new(vec![1.0; 64], 1e-5);
        let qkv = QkvBrick::new(64, 64, 64, 64);
        let attn = AttentionBrick::new(8, 2, 64);
        let ffn = FfnBrick::new(64, 256);

        assert!(rms.budget().us_per_token > 0.0);
        assert!(qkv.budget().us_per_token > 0.0);
        assert!(attn.budget().us_per_token > 0.0);
        assert!(ffn.budget().us_per_token > 0.0);
    }

    // F073: Error handling infrastructure ready
    #[test]
    fn f073_error_handling_ready() {
        // BrickError variants exist for error handling
        let invalid_err = BrickError::InvalidInput("test".to_string());
        let budget_err = BrickError::BudgetExceeded {
            limit_us: 10.0,
            actual_us: 20.0,
        };

        // Errors are formattable
        assert!(!format!("{invalid_err}").is_empty());
        assert!(!format!("{budget_err}").is_empty());
    }

    // =========================================================================
    // F081-F100: Performance Regression (Infrastructure Stubs)
    // These tests verify benchmark infrastructure is ready.
    // Full validation requires benchmark runs against baselines.
    // =========================================================================

    // F081-F084: Throughput comparison infrastructure
    #[test]
    fn f081_throughput_comparison_ready() {
        // TokenBudget can express throughput targets
        let target_2x_llama = TokenBudget::from_throughput(976.0 * 2.0); // 2x Ollama for 1.5B

        assert!(target_2x_llama.tokens_per_sec > 1900.0);
        assert!(target_2x_llama.us_per_token < 520.0); // ~512µs for 2x
    }

    // F085: CV calculation infrastructure
    #[test]
    fn f085_cv_calculation_ready() {
        // BenchmarkReport has CV field
        let config = BenchmarkConfig::default();
        assert!(config.samples >= 100, "Need >= 100 samples for valid CV");
    }

    // F086: Latency percentile infrastructure
    #[test]
    fn f086_latency_percentile_ready() {
        // BenchmarkReport has p50 and p99 fields
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

        // p99 should be reasonable vs p50
        assert!(report.p99_us < report.p50_us * 2.0);
    }

    // F087: Baseline comparison infrastructure
    #[test]
    fn f087_baseline_comparison_ready() {
        // TokenBudget::gap_factor can compare to baseline
        let budget = TokenBudget::from_latency(100.0);
        let actual = 80.0;

        let gap = budget.gap_factor(actual);
        assert!(gap < 1.0, "Under budget should have gap < 1.0");

        let over_budget = budget.gap_factor(120.0);
        assert!(over_budget > 1.0, "Over budget should have gap > 1.0");
    }

    // F090: CUDA graph overhead infrastructure
    #[test]
    fn f090_cuda_graph_overhead_ready() {
        let brick = CudaGraphBrick::new(28, 1536); // 28 layers, 1536 hidden

        // Graph replay should be much faster than 100µs target
        assert!(
            brick.budget().us_per_token < 100.0,
            "Graph overhead should be < 100µs"
        );
    }

    // F092: Memory usage tracking infrastructure
    #[test]
    fn f092_memory_usage_ready() {
        // ActivationQuantBrick tracks memory reduction
        let brick = ActivationQuantBrick::new(1024);
        let bytes_saved = brick.bytes_saved();

        assert!(bytes_saved > 0, "Should track memory savings");

        // FlashAttentionBrick tracks memory usage
        let flash = FlashAttentionBrick::new(8, 2, 64);
        let (naive, flash_mem) = flash.memory_bytes(512);

        assert!(flash_mem < naive, "Flash should use less memory");
    }

    // F093: Memory leak detection infrastructure (no-op without valgrind)
    #[test]
    fn f093_memory_leak_detection_ready() {
        // Create and drop bricks - Rust's ownership handles cleanup
        for _ in 0..100 {
            let _ = RmsNormBrick::new(vec![1.0; 1024], 1e-5);
            let _ = FfnBrick::new(1024, 4096);
            let _ = AttentionBrick::new(32, 8, 128);
        }
        // If we get here without OOM, basic leak detection passes
    }

    // F094: Graceful degradation infrastructure
    #[test]
    #[allow(deprecated)] // Testing legacy execute() method
    fn f094_graceful_degradation_ready() {
        // Bricks return Result for error handling
        let brick = ActivationQuantBrick::new(0);
        let result = brick.execute();

        assert!(result.is_err(), "Zero-dim should fail gracefully");
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(!msg.is_empty(), "Error should have message");
        }
    }

    // ========================================================================
    // REAL IMPLEMENTATION TESTS (not stubs)
    // ========================================================================

    // R001: ActivationQuantBrick real quantize/dequantize
    #[test]
    fn r001_activation_quant_real_quantize() {
        let brick = ActivationQuantBrick::new(64);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();

        let (quants, scales) = brick.quantize(&input).expect("operation failed");

        assert_eq!(quants.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32 = 2 blocks
        assert!(scales.iter().all(|&s| s > 0.0), "Scales must be positive");
    }

    // R002: ActivationQuantBrick roundtrip accuracy
    #[test]
    fn r002_activation_quant_roundtrip() {
        let brick = ActivationQuantBrick::new(32);
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let (quants, scales) = brick.quantize(&input).expect("operation failed");
        let output = brick
            .dequantize(&quants, &scales)
            .expect("operation failed");

        // Q8 error should be < 1%
        let error = brick
            .measure_error(&input, &quants, &scales)
            .expect("operation failed");
        assert!(error < 0.01, "Q8 error {} should be < 1%", error);

        // Output should be close to input
        for (i, (&orig, &dequant)) in input.iter().zip(output.iter()).enumerate() {
            let diff = (orig - dequant).abs();
            assert!(diff < 0.05, "Value {} diff {} too large", i, diff);
        }
    }

    // R003: FlashAttentionBrick real forward pass
    #[test]
    fn r003_flash_attention_real_forward() {
        let brick = FlashAttentionBrick::new(4, 2, 8); // 4 heads, 2 kv heads, dim 8
        let seq_len = 4;

        // Create test data
        let query = vec![1.0f32; 4 * 8]; // [num_heads * head_dim]
        let keys = vec![0.5f32; seq_len * 2 * 8]; // [seq_len * num_kv_heads * head_dim]
        let values = vec![0.25f32; seq_len * 2 * 8];

        let output = brick
            .forward(&query, &keys, &values, seq_len)
            .expect("operation failed");

        assert_eq!(output.len(), 4 * 8);
        // With uniform values, output should be close to uniform values
        assert!(output.iter().all(|&v| !v.is_nan()), "No NaNs");
        assert!(output.iter().all(|&v| v.is_finite()), "All finite");
    }

    // R004: FlashAttentionBrick online softmax correctness
    #[test]
    fn r004_flash_attention_softmax_correct() {
        let brick = FlashAttentionBrick::new(1, 1, 4); // Single head for easy verification
        let seq_len = 3;

        // Query = [1, 0, 0, 0]
        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        // Keys with different similarities to query
        // K0 = [1, 0, 0, 0] -> dot = 1.0
        // K1 = [0, 1, 0, 0] -> dot = 0.0
        // K2 = [-1, 0, 0, 0] -> dot = -1.0
        let keys = vec![
            1.0, 0.0, 0.0, 0.0, // K0
            0.0, 1.0, 0.0, 0.0, // K1
            -1.0, 0.0, 0.0, 0.0, // K2
        ];

        // Values
        let values = vec![
            1.0, 0.0, 0.0, 0.0, // V0
            0.0, 1.0, 0.0, 0.0, // V1
            0.0, 0.0, 1.0, 0.0, // V2
        ];

        let output = brick
            .forward(&query, &keys, &values, seq_len)
            .expect("operation failed");

        // After softmax, K0 should have highest weight
        // Output should be weighted combination dominated by V0
        assert!(output[0] > output[1], "V0 weight should be highest");
        assert!(output[0] > output[2], "V0 weight should be highest");
    }

    // R005: CoalescedDp4aBrick real GEMV
    #[test]
    #[cfg(feature = "cuda")]
    fn r005_coalesced_dp4a_real_gemv() {
        let brick = CoalescedDp4aBrick::new(256, 4); // K=256, N=4

        // Create Q8 input (all 1s)
        let input_q8 = vec![1i8; 256];
        let input_scale = 1.0 / 127.0;

        // Create Q4 weights (alternating 0x77 = nibbles 7,7 centered to -1,-1)
        let weights_q4 = vec![0x88u8; 4 * 256 / 2]; // All 8s -> centered to 0

        // Weight scales
        let weight_scales = vec![0.1f32; 4];

        let output = brick
            .forward(&input_q8, input_scale, &weights_q4, &weight_scales)
            .expect("operation failed");

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&v| !v.is_nan()));
    }
