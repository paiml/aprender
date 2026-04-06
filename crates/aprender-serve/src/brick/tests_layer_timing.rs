
    #[test]
    fn test_layer_timing_bottleneck() {
        let timing = LayerTiming {
            attn_norm_us: 1.0,
            qkv_us: 10.0, // This should be the bottleneck
            rope_us: 2.0,
            attention_us: 5.0,
            o_proj_us: 3.0,
            ffn_norm_us: 1.0,
            ffn_us: 8.0,
            total_us: 30.0,
        };
        let (name, time) = timing.bottleneck();
        assert_eq!(name, "qkv");
        assert_eq!(time, 10.0);
    }

    // ========================================================================
    // Additional Coverage Tests: Fused Operations
    // ========================================================================

    // T001: CoalescedDp4aBrick dimension validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t001_coalesced_dp4a_invalid_input_length() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong input length
        let result = brick.forward(
            &[1i8; 128], // Wrong: 128 instead of 256
            1.0,
            &[0x88u8; 512],
            &[0.1f32; 4],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("128"), "Error should mention actual size");
            assert!(msg.contains("256"), "Error should mention expected size");
        }
    }

    // T002: CoalescedDp4aBrick weight length validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t002_coalesced_dp4a_invalid_weights() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong weights length (should be n * k / 2 = 4 * 256 / 2 = 512)
        let result = brick.forward(
            &[1i8; 256],
            1.0,
            &[0x88u8; 256], // Wrong: 256 instead of 512
            &[0.1f32; 4],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Weights"));
        }
    }

    // T003: CoalescedDp4aBrick scale length validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t003_coalesced_dp4a_invalid_scales() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong scale length (should be n = 4)
        let result = brick.forward(
            &[1i8; 256],
            1.0,
            &[0x88u8; 512],
            &[0.1f32; 2], // Wrong: 2 instead of 4
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("scale"));
        }
    }

    // T004: CoalescedDp4aBrick can_run for invalid dimensions
    #[test]
    #[cfg(feature = "cuda")]
    fn t004_coalesced_dp4a_can_run_invalid() {
        // K not multiple of 256
        let brick = CoalescedDp4aBrick::new(128, 4);
        assert!(!brick.can_run(), "K=128 not multiple of 256 should not run");

        // Zero dimensions
        let brick = CoalescedDp4aBrick::new(0, 4);
        assert!(!brick.can_run(), "K=0 should not run");

        let brick = CoalescedDp4aBrick::new(256, 0);
        assert!(!brick.can_run(), "N=0 should not run");
    }

    // T005: CoalescedDp4aBrick arithmetic intensity
    #[test]
    #[cfg(feature = "cuda")]
    fn t005_coalesced_dp4a_arithmetic_intensity() {
        let brick = CoalescedDp4aBrick::new(1024, 256);
        let ai = brick.arithmetic_intensity();

        // Should be positive and reasonable (AI can be high for compute-bound kernels)
        assert!(ai > 0.0, "Arithmetic intensity should be positive");
        assert!(ai < 1000.0, "Arithmetic intensity should be reasonable");
    }

    // T006: FusedFfnBrick invalid input length
    #[test]
    #[cfg(feature = "cuda")]
    fn t006_fused_ffn_invalid_input() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong input length
        let result = brick.forward(
            &[1.0f32; 8], // Wrong: 8 instead of 4
            &[0.1f32; 32],
            &[0.2f32; 32],
            &[0.1f32; 32],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Input"));
        }
    }

    // T007: FusedFfnBrick invalid gate/up proj length
    #[test]
    #[cfg(feature = "cuda")]
    fn t007_fused_ffn_invalid_gate_up() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong gate_proj length (should be 8 * 4 = 32)
        let result = brick.forward(
            &[1.0f32; 4],
            &[0.1f32; 16], // Wrong: 16 instead of 32
            &[0.2f32; 32],
            &[0.1f32; 32],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Gate/Up"));
        }
    }

    // T008: FusedFfnBrick invalid down proj length
    #[test]
    #[cfg(feature = "cuda")]
    fn t008_fused_ffn_invalid_down() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong down_proj length (should be 4 * 8 = 32)
        let result = brick.forward(
            &[1.0f32; 4],
            &[0.1f32; 32],
            &[0.2f32; 32],
            &[0.1f32; 16], // Wrong: 16 instead of 32
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Down"));
        }
    }

    // T009: FusedFfnBrick can_run for invalid dimensions
    #[test]
    #[cfg(feature = "cuda")]
    fn t009_fused_ffn_can_run_invalid() {
        let brick = FusedFfnBrick::new(0, 8);
        assert!(!brick.can_run(), "Zero hidden_dim should not run");

        let brick = FusedFfnBrick::new(4, 0);
        assert!(!brick.can_run(), "Zero intermediate_dim should not run");
    }

    // T010: FlashAttentionBrick invalid query length
    #[test]
    fn t010_flash_attention_invalid_query() {
        let brick = FlashAttentionBrick::new(4, 2, 8);

        // Wrong query length (should be 4 * 8 = 32)
        let result = brick.forward(
            &[1.0f32; 16], // Wrong: 16 instead of 32
            &[0.5f32; 64],
            &[0.25f32; 64],
            4,
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Query"));
        }
    }

    // T011: FlashAttentionBrick invalid keys/values length
    #[test]
    fn t011_flash_attention_invalid_kv() {
        let brick = FlashAttentionBrick::new(4, 2, 8);
        let seq_len = 4;

        // Wrong keys length (should be seq_len * num_kv_heads * head_dim = 4 * 2 * 8 = 64)
        let result = brick.forward(
            &[1.0f32; 32],
            &[0.5f32; 32], // Wrong: 32 instead of 64
            &[0.25f32; 64],
            seq_len,
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("KV"));
        }
    }

    // T012: FlashAttentionBrick zero dimension
    #[test]
    fn t012_flash_attention_zero_dim() {
        let brick = FlashAttentionBrick::new(0, 0, 0);

        let result = brick.forward(&[], &[], &[], 0);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Zero"));
        }
    }

    // T013: FlashAttentionBrick arithmetic intensity
    #[test]
    fn t013_flash_attention_arithmetic_intensity() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let ai = brick.arithmetic_intensity(512);

        assert!(ai > 0.0, "Arithmetic intensity should be positive");
    }

    // T014: ActivationQuantBrick zero dimension quantize
    #[test]
    fn t014_activation_quant_zero_dim_quantize() {
        let brick = ActivationQuantBrick::new(0);
        let result = brick.quantize(&[]);
        assert!(result.is_err());
    }

    // T015: ActivationQuantBrick dequantize length mismatch
    #[test]
    fn t015_activation_quant_dequantize_mismatch() {
        let brick = ActivationQuantBrick::new(32);

        // Wrong quants length
        let result = brick.dequantize(&[0i8; 16], &[0.1f32; 1]);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("16"));
        }
    }

    // T016: ActivationQuantBrick measure_error with zero max
    #[test]
    fn t016_activation_quant_measure_error_zero() {
        let brick = ActivationQuantBrick::new(32);
        let original = vec![0.0f32; 32];

        let (quants, scales) = brick.quantize(&original).expect("quantize failed");
        let error = brick
            .measure_error(&original, &quants, &scales)
            .expect("measure failed");

        // Error should be 0 for zero input
        assert_eq!(error, 0.0, "Error should be 0 for zero input");
    }

    // T017: RmsNormBrick input/weight length mismatch
    #[test]
    fn t017_rmsnorm_length_mismatch() {
        let brick =
            RmsNormBrick::new(vec![1.0; 32], 1e-5).with_budget(TokenBudget::from_latency(1000.0));

        // Wrong input length
        let result = brick.run(&[1.0f32; 16]);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("16"));
            assert!(msg.contains("32"));
        }
    }

    // T018: CudaGraphBrick replay without capture
    #[test]
    fn t018_cuda_graph_replay_not_captured() {
        let brick = CudaGraphBrick::new(28, 1536);

        let result = brick.replay();
        assert!(result.is_err());
        if let Err(BrickError::ComputeError(msg)) = result {
            assert!(msg.contains("not captured"));
        }
    }

    // T019: CudaGraphBrick can_run with invalid config
    #[test]
    fn t019_cuda_graph_can_run_invalid() {
        let brick = CudaGraphBrick::new(0, 1536);
        assert!(!brick.can_run(), "Zero layers should not run");

        let brick = CudaGraphBrick::new(28, 0);
        assert!(!brick.can_run(), "Zero hidden_dim should not run");
    }

    // T020: TransformerLayerBrick verify with invalid component
    #[test]
    fn t020_transformer_layer_verify() {
        let layer = TransformerLayerBrick::from_config(0, 64, 4, 2, 256, 1e-5, 10000.0, 0);

        // Should pass verification with valid config
        let verification = layer.verify();
        assert!(
            verification.is_valid,
            "Valid layer should pass verification"
        );
    }

    // T021: BottleneckReport display
    #[test]
    fn t021_bottleneck_report_display() {
        let report = BottleneckReport {
            layer_idx: 5,
            brick_name: "attention",
            actual_us: 15.0,
            budget_us: 10.0,
            gap_factor: 1.5,
        };

        let display = format!("{}", report);
        assert!(display.contains("attention"));
        assert!(display.contains("layer 5"));
        assert!(display.contains("15.0"));
        assert!(display.contains("10.0"));
        assert!(display.contains("1.50"));
    }

    // T022: BrickError std::error::Error impl
    #[test]
    fn t022_brick_error_std_error() {
        let err = BrickError::InvalidInput("test".to_string());

        // Verify Error trait is implemented (compiles)
        let _: &dyn std::error::Error = &err;
    }

    // T023: TokenResult with zero us_per_token
    #[test]
    fn t023_token_result_zero_latency() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 1, 0.0, &budget);

        // With 0 elapsed time, tokens_per_sec should be 0 (not infinity)
        assert_eq!(result.tokens_per_sec, 0.0);
    }

    // T024: FlashAttentionBrick with custom tile size
    #[test]
    fn t024_flash_attention_custom_tile() {
        let brick = FlashAttentionBrick::with_tile_size(8, 2, 64, 64);
        assert_eq!(brick.tile_size, 64);
        assert_eq!(brick.num_tiles(512), 8); // 512 / 64 = 8
    }

    // T025: FlashAttentionBrick can_run with zero tile
    #[test]
    fn t025_flash_attention_zero_tile() {
        let mut brick = FlashAttentionBrick::new(8, 2, 64);
        brick.tile_size = 0;
        assert!(!brick.can_run(), "Zero tile_size should not run");
    }

    // T026: AttentionBrick group_size edge case
    #[test]
    fn t026_attention_group_size_edge() {
        // Zero kv_heads should use max(1) to avoid division by zero
        let brick = AttentionBrick::new(8, 0, 64);
        assert_eq!(brick.group_size(), 8); // 8 / max(0, 1) = 8
    }

    // T027: benchmark_brick with empty samples
    #[test]
    fn t027_benchmark_empty_samples() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        let config = BenchmarkConfig {
            warmup: 0,
            samples: 1, // Minimal samples
            max_cv: 1.0,
        };

        let report = benchmark_brick(
            &brick,
            || 10.0, // Fixed timing
            &config,
        );

        assert_eq!(report.mean_us, 10.0);
        assert_eq!(report.std_us, 0.0); // Single sample = no variance
    }

    // T028: BrickAssertion budget_met check
    #[test]
    fn t028_assertion_budget_not_met() {
        let assertion = BrickAssertion::budget_met();
        let data = &[1.0f32, 2.0, 3.0];

        // Should pass when budget is met
        assert!(assertion.check_f32(data, true).is_ok());

        // Should fail when budget is not met
        let result = assertion.check_f32(data, false);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed {
            name,
            expected,
            actual,
        }) = result
        {
            assert_eq!(name, "budget_met");
            assert!(expected.contains("met"));
            assert!(actual.contains("exceeded"));
        }
    }

    // T029: BrickAssertion no_inf check
    #[test]
    fn t029_assertion_inf_check() {
        let assertion = BrickAssertion::no_inf();

        // Should pass without infinity
        assert!(assertion.check_f32(&[1.0, 2.0, 3.0], true).is_ok());

        // Should fail with positive infinity
        let result = assertion.check_f32(&[1.0, f32::INFINITY, 3.0], true);
        assert!(result.is_err());

        // Should fail with negative infinity
        let result = assertion.check_f32(&[1.0, f32::NEG_INFINITY, 3.0], true);
        assert!(result.is_err());
    }
