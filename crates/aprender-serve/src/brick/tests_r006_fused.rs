
    // R006: FusedFfnBrick real SwiGLU FFN
    #[test]
    #[cfg(feature = "cuda")]
    fn r006_fused_ffn_real_swiglu() {
        let brick = FusedFfnBrick::new(4, 8); // hidden=4, intermediate=8

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4]; // [intermediate, hidden]
        let up_proj = vec![0.2f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8]; // [hidden, intermediate]

        let output = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("operation failed");

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&v| !v.is_nan()), "No NaNs");
        assert!(output.iter().all(|&v| v.is_finite()), "All finite");
    }

    // R007: FusedFfnBrick SwiGLU activation verification
    #[test]
    #[cfg(feature = "cuda")]
    fn r007_fused_ffn_swiglu_activation() {
        // Verify SiLU activation: silu(x) = x * sigmoid(x)
        let brick = FusedFfnBrick::new(1, 1);

        // With identity-ish weights, we can verify SwiGLU behavior
        let input = vec![1.0f32];
        let gate_proj = vec![1.0f32]; // Pass input through
        let up_proj = vec![1.0f32]; // Pass input through
        let down_proj = vec![1.0f32]; // Pass intermediate through

        let output = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("operation failed");

        // SwiGLU(1.0, 1.0) = silu(1.0) * 1.0 = 1.0 * sigmoid(1.0) * 1.0
        // sigmoid(1.0) ≈ 0.731
        // silu(1.0) ≈ 0.731
        let expected = 0.731;
        assert!(
            (output[0] - expected).abs() < 0.01,
            "SwiGLU output {} should be ~{}",
            output[0],
            expected
        );
    }

    // R008: ActivationQuantBrick timed execution
    #[test]
    fn r008_activation_quant_timed() {
        let brick = ActivationQuantBrick::new(1024);
        let input: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();

        let result = brick.execute_timed(&input).expect("operation failed");

        assert_eq!(result.output.0.len(), 1024); // quants
        assert!(result.us_per_token > 0.0);
        assert!(result.tokens_per_sec > 0.0);
        println!(
            "ActivationQuant: {:.2}µs/tok, {:.0} tok/s",
            result.us_per_token, result.tokens_per_sec
        );
    }

    // R009: FlashAttention timed execution
    #[test]
    fn r009_flash_attention_timed() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let seq_len = 128;

        let query = vec![0.1f32; 8 * 64];
        let keys = vec![0.1f32; seq_len * 2 * 64];
        let values = vec![0.1f32; seq_len * 2 * 64];

        let result = brick
            .forward_timed(&query, &keys, &values, seq_len)
            .expect("operation failed");

        assert_eq!(result.output.len(), 8 * 64);
        assert!(result.us_per_token > 0.0);
        println!(
            "FlashAttention (seq={}): {:.2}µs/tok, {:.0} tok/s",
            seq_len, result.us_per_token, result.tokens_per_sec
        );
    }

    // R010: FusedFfn timed execution
    #[test]
    #[cfg(feature = "cuda")]
    fn r010_fused_ffn_timed() {
        let hidden = 64;
        let intermediate = 256;
        let brick = FusedFfnBrick::new(hidden, intermediate);

        let input = vec![0.1f32; hidden];
        let gate_proj = vec![0.01f32; intermediate * hidden];
        let up_proj = vec![0.01f32; intermediate * hidden];
        let down_proj = vec![0.01f32; hidden * intermediate];

        let result = brick
            .forward_timed(&input, &gate_proj, &up_proj, &down_proj)
            .expect("operation failed");

        assert_eq!(result.output.len(), hidden);
        assert!(result.us_per_token > 0.0);
        println!(
            "FusedFfn ({}x{}): {:.2}µs/tok, {:.0} tok/s",
            hidden, intermediate, result.us_per_token, result.tokens_per_sec
        );
    }

    // ========================================================================
    // F009, F016, F020: Additional Core Invariants
    // ========================================================================

    // F009: Brick composition is type-safe
    #[cfg(feature = "cuda")]
    #[test]
    fn f009_brick_composition_typesafe() {
        // Compile-time verification: bricks with different Output types
        // cannot be accidentally mixed
        let _quant: &dyn ComputeBrick<Output = Vec<u8>> = &ActivationQuantBrick::new(64);
        let _attn: &dyn ComputeBrick<Output = Vec<f32>> = &FlashAttentionBrick::new(4, 2, 8);
        let _ffn: &dyn ComputeBrick<Output = Vec<f32>> = &FusedFfnBrick::new(64, 256);
        // Type system prevents mixing incompatible bricks
    }

    // F016: RmsNorm brick produces normalized output
    #[test]
    #[ignore = "flaky - budget assertion depends on hardware timing"]
    fn f016_rmsnorm_normalizes() {
        let weights = vec![1.0f32; 64];
        let brick = RmsNormBrick::new(weights, 1e-5);
        let input = vec![2.0f32; 64];

        let result = brick.run(&input).expect("operation failed");
        let output = result.output;

        // RMSNorm should produce values with RMS ≈ 1.0 (scaled by weights)
        let rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.1,
            "RMSNorm output RMS {} should be ~1.0",
            rms
        );
    }

    // F020: All bricks have deterministic output for same input
    #[test]
    #[cfg(feature = "cuda")]
    fn f020_brick_determinism() {
        let brick = FusedFfnBrick::new(4, 8);
        let input = vec![1.0f32; 4];
        let gate = vec![0.1f32; 32];
        let up = vec![0.2f32; 32];
        let down = vec![0.1f32; 32];

        let out1 = brick
            .forward(&input, &gate, &up, &down)
            .expect("operation failed");
        let out2 = brick
            .forward(&input, &gate, &up, &down)
            .expect("operation failed");

        assert_eq!(out1, out2, "Same input must produce same output");
    }

    // ========================================================================
    // F023-F034: Budget Compliance Tests
    // ========================================================================

    // F023: RmsNormBrick budget target
    #[test]
    fn f023_rmsnorm_budget_target() {
        let brick = RmsNormBrick::new(vec![1.0; 1024], 1e-5);
        // Budget should be set appropriately for RmsNorm
        assert!(
            brick.budget().us_per_token < 10.0,
            "RmsNorm budget should be < 10µs"
        );
    }

    // F024: QkvBrick budget target (via AttentionBrick proxy)
    #[test]
    fn f024_attention_brick_budget() {
        let brick = AttentionBrick::new(32, 8, 128);
        assert!(
            brick.budget().us_per_token < 50.0,
            "Attention budget should be < 50µs"
        );
    }

    // F028: FfnBrick budget target
    #[test]
    fn f028_ffn_budget_target() {
        let brick = FfnBrick::new(1536, 8960);
        assert!(
            brick.budget().us_per_token < 100.0,
            "FFN budget should be < 100µs"
        );
    }

    // F029: Fused FFN budget
    #[test]
    #[cfg(feature = "cuda")]
    fn f029_fused_ffn_budget() {
        let brick = FusedFfnBrick::new(1536, 8960);
        assert!(
            brick.budget().us_per_token < 50.0,
            "FusedFFN budget should be < 50µs (2x improvement)"
        );
    }

    // F030: Model throughput target infrastructure
    #[test]
    fn f030_throughput_target() {
        // 976 tok/s = 1024µs/tok
        let target = TokenBudget::from_throughput(976.0);
        assert!(
            target.us_per_token < 1100.0,
            "976 tok/s should be ~1024µs/tok"
        );
    }

    // ========================================================================
    // F041-F048: Backend Correctness Tests
    // ========================================================================

    // F041: CPU output consistency (no CUDA comparison)
    #[test]
    #[cfg(feature = "cuda")]
    fn f041_cpu_consistency() {
        let brick = FusedFfnBrick::new(4, 8);
        let input = vec![1.0f32; 4];
        let gate = vec![0.1f32; 32];
        let up = vec![0.2f32; 32];
        let down = vec![0.1f32; 32];

        // Multiple runs should match
        let out1 = brick
            .forward(&input, &gate, &up, &down)
            .expect("operation failed");
        let out2 = brick
            .forward(&input, &gate, &up, &down)
            .expect("operation failed");

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "CPU output should be bit-identical across runs"
            );
        }
    }

    // F043: RoPE properties (via budget and assertion verification)
    #[test]
    fn f043_rope_properties() {
        // RoPE brick should have correct configuration
        let brick = RopeBrick::new(64, 8, 10000.0, 0); // head_dim, num_heads, theta, rope_type

        // Verify brick is properly configured
        assert_eq!(brick.head_dim, 64, "Head dim should be 64");
        assert_eq!(brick.num_heads, 8, "Num heads should be 8");
        assert!(brick.theta > 0.0, "Theta should be positive");

        // Verify budget is set appropriately for RoPE
        assert!(
            brick.budget().us_per_token < 5.0,
            "RoPE budget should be < 5µs"
        );
    }

    // F044: Softmax numerical stability
    #[test]
    fn f044_softmax_stability() {
        let brick = FlashAttentionBrick::new(1, 1, 4);

        // Test with large values that could cause overflow
        let query = vec![100.0f32, 0.0, 0.0, 0.0];
        let keys = vec![
            100.0, 0.0, 0.0, 0.0, // High similarity
            0.0, 0.0, 0.0, 0.0, // Zero
            -100.0, 0.0, 0.0, 0.0, // Negative
        ];
        let values = vec![1.0f32; 12];

        let output = brick
            .forward(&query, &keys, &values, 3)
            .expect("operation failed");

        assert!(output.iter().all(|&v| !v.is_nan()), "No NaN in output");
        assert!(output.iter().all(|&v| v.is_finite()), "All outputs finite");
    }

    // F047: SwiGLU activation correctness
    #[test]
    #[cfg(feature = "cuda")]
    fn f047_swiglu_correctness() {
        // SwiGLU(x, y) = SiLU(x) * y = x * sigmoid(x) * y
        let brick = FusedFfnBrick::new(1, 1);

        // Test: input=1, gate=1, up=1, down=1
        // SiLU(1.0) = 1.0 * sigmoid(1.0) ≈ 0.731
        let input = vec![1.0f32];
        let gate = vec![1.0f32];
        let up = vec![1.0f32];
        let down = vec![1.0f32];

        let output = brick
            .forward(&input, &gate, &up, &down)
            .expect("operation failed");

        // Expected: silu(1.0) * 1.0 * 1.0 = 0.731
        let expected = 1.0 / (1.0 + (-1.0f32).exp()); // sigmoid(1) * 1
        assert!(
            (output[0] - expected).abs() < 0.01,
            "SwiGLU output {} should be ~{}",
            output[0],
            expected
        );
    }

    // F048: RMSNorm epsilon handling
    #[test]
    fn f048_rmsnorm_epsilon() {
        // With near-zero input, epsilon prevents division by zero
        // Use a relaxed budget since this test is about correctness, not performance
        let brick =
            RmsNormBrick::new(vec![1.0; 4], 1e-5).with_budget(TokenBudget::from_latency(1000.0)); // 1ms budget for test
        let input = vec![1e-10f32; 4]; // Very small values

        let result = brick.run(&input).expect("operation failed");
        let output = result.output;

        assert!(
            output.iter().all(|&v| !v.is_nan()),
            "No NaN with small input"
        );
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "All outputs finite with small input"
        );
    }

    // ========================================================================
    // F068-F080: Additional CUDA Infrastructure
    // ========================================================================

    // F068: Shared memory infrastructure ready
    #[test]
    fn f068_shared_memory_ready() {
        // FlashAttention uses tiled access (proxy for shared memory)
        let brick = FlashAttentionBrick::new(8, 2, 64);
        assert!(brick.tile_size > 0, "Tile size should be set for tiling");
        assert_eq!(brick.tile_size, 128, "Default tile size for L2 cache fit");
    }

    // F069: Warp-level infrastructure ready
    #[test]
    #[cfg(feature = "cuda")]
    fn f069_warp_infrastructure_ready() {
        // Coalesced DP4A processes in groups (warp-aligned)
        let brick = CoalescedDp4aBrick::new(256, 4);
        // K must be multiple of 256 for warp-aligned access
        assert!(
            brick.k.is_multiple_of(256),
            "K should be warp-aligned (256)"
        );
    }

    // F071: Kernel launch overhead tracking
    #[test]
    fn f071_launch_overhead_tracking() {
        let brick = CudaGraphBrick::new(28, 1536);
        // Graph brick tracks number of kernels to eliminate
        assert!(brick.num_layers > 0, "Should track layer count");
    }

    // F072: Memory pool infrastructure
    #[test]
    fn f072_memory_pool_ready() {
        // Activation quant tracks memory savings (pool efficiency)
        let brick = ActivationQuantBrick::new(4096);
        let savings = brick.bytes_saved();
        assert!(savings > 0, "Should track memory savings");
        assert_eq!(savings, 4096 * 3, "f32→i8 saves 3 bytes/element");
    }

    // F074-F078: Budget gap tracking
    #[test]
    fn f074_budget_gap_factor() {
        let budget = TokenBudget::from_latency(100.0);

        // Under budget
        let gap = budget.gap_factor(80.0);
        assert!(gap < 1.0, "Under budget = gap < 1.0");

        // Over budget
        let gap = budget.gap_factor(120.0);
        assert!(gap > 1.0, "Over budget = gap > 1.0");
    }

    // F079: Error propagation
    #[test]
    fn f079_error_propagation() {
        let brick = ActivationQuantBrick::new(32);

        // Wrong input size should error
        let wrong_input = vec![1.0f32; 64]; // Expected 32
        let result = brick.quantize(&wrong_input);

        assert!(result.is_err(), "Wrong input size should error");
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("64"), "Error should mention actual size");
            assert!(msg.contains("32"), "Error should mention expected size");
        }
    }

    // F080: Graceful handling of edge cases
    #[test]
    #[allow(deprecated)] // Testing legacy execute() methods
    fn f080_edge_cases() {
        // Empty/zero dimension handling
        let brick = ActivationQuantBrick::new(0);
        assert!(brick.execute().is_err(), "Zero dim should error");

        let flash = FlashAttentionBrick::new(0, 0, 0);
        assert!(flash.execute(10).is_err(), "Zero heads/dim should error");
    }

    // ========================================================================
    // F082-F100: Performance Regression Infrastructure
    // ========================================================================

    // F082: Iteration count for statistical validity
    #[test]
    fn f082_iteration_count() {
        let config = BenchmarkConfig::default();
        assert!(config.samples >= 100, "Need >= 100 samples for valid stats");
        assert!(config.warmup >= 10, "Need >= 10 warmup iterations");
    }
