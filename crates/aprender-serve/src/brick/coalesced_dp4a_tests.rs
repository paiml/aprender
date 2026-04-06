
    // ============================================================================
    // CoalescedDp4aBrick Tests
    // ============================================================================

    #[test]
    fn coalesced_dp4a_new_basic() {
        let brick = CoalescedDp4aBrick::new(256, 64);
        assert_eq!(brick.k, 256);
        assert_eq!(brick.n, 64);
    }

    #[test]
    fn coalesced_dp4a_with_budget() {
        let custom_budget = TokenBudget::from_latency(50.0);
        let brick = CoalescedDp4aBrick::new(256, 64).with_budget(custom_budget);
        assert!((brick.budget().us_per_token - 50.0).abs() < 0.001);
    }

    #[test]
    fn coalesced_dp4a_flops_calculation() {
        let brick = CoalescedDp4aBrick::new(512, 128);
        // GEMV: 2 * K * N = 2 * 512 * 128 = 131072
        assert_eq!(brick.flops(), 2 * 512 * 128);
    }

    #[test]
    fn coalesced_dp4a_arithmetic_intensity() {
        let brick = CoalescedDp4aBrick::new(1024, 256);
        let ai = brick.arithmetic_intensity();
        // AI should be positive and reasonable (AI can be high for compute-bound kernels)
        assert!(ai > 0.0, "Arithmetic intensity should be positive");
        assert!(ai < 1000.0, "Arithmetic intensity should be reasonable");
    }

    #[test]
    fn coalesced_dp4a_forward_basic() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8: Vec<i8> = (0..256).map(|i| (i % 127) as i8).collect();
        let input_scale = 0.1f32;
        let weights_q4: Vec<u8> = vec![0x88; 256 * 4 / 2]; // 4-bit packed, centered at 8
        let weight_scales: Vec<f32> = vec![0.01; 4];

        let result = brick
            .forward(&input_q8, input_scale, &weights_q4, &weight_scales)
            .expect("forward should succeed");

        assert_eq!(result.len(), 4);
        // Result should not contain NaN
        assert!(!result.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn coalesced_dp4a_forward_timed() {
        let brick =
            CoalescedDp4aBrick::new(256, 4).with_budget(TokenBudget::from_latency(100_000.0));

        let input_q8: Vec<i8> = vec![1; 256];
        let input_scale = 1.0f32;
        let weights_q4: Vec<u8> = vec![0x88; 256 * 4 / 2];
        let weight_scales: Vec<f32> = vec![1.0; 4];

        let result = brick
            .forward_timed(&input_q8, input_scale, &weights_q4, &weight_scales)
            .expect("forward_timed should succeed");

        assert_eq!(result.output.len(), 4);
        assert_eq!(result.tokens_processed, 1);
        assert!(result.us_per_token > 0.0);
        assert!(result.tokens_per_sec > 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn coalesced_dp4a_execute_valid_dimensions() {
        // K must be multiple of 256
        let brick = CoalescedDp4aBrick::new(256, 64);
        let result = brick.execute();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 64);
    }

    #[test]
    #[allow(deprecated)]
    fn coalesced_dp4a_execute_invalid_k_not_multiple_256() {
        // K=128 is NOT multiple of 256
        let brick = CoalescedDp4aBrick::new(128, 64);
        let result = brick.execute();
        assert!(result.is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn coalesced_dp4a_execute_zero_k() {
        let brick = CoalescedDp4aBrick::new(0, 64);
        let result = brick.execute();
        assert!(result.is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn coalesced_dp4a_execute_zero_n() {
        let brick = CoalescedDp4aBrick::new(256, 0);
        let result = brick.execute();
        assert!(result.is_err());
    }

    #[test]
    fn coalesced_dp4a_trait_name() {
        let brick = CoalescedDp4aBrick::new(256, 64);
        assert_eq!(brick.name(), "coalesced_dp4a");
    }

    #[test]
    fn coalesced_dp4a_trait_assertions() {
        let brick = CoalescedDp4aBrick::new(256, 64);
        let assertions = brick.assertions();
        assert!(!assertions.is_empty());

        let names: Vec<&str> = assertions.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"no_nan"));
        assert!(names.contains(&"no_inf"));
        assert!(names.contains(&"budget_met"));
        assert!(names.contains(&"bandwidth_efficient"));
    }

    #[test]
    fn coalesced_dp4a_can_run_valid() {
        let brick = CoalescedDp4aBrick::new(256, 64);
        assert!(brick.can_run());
    }

    #[test]
    fn coalesced_dp4a_can_run_k_not_multiple_256() {
        let brick = CoalescedDp4aBrick::new(128, 64);
        assert!(!brick.can_run());
    }

    #[test]
    fn coalesced_dp4a_can_run_zero_dimensions() {
        assert!(!CoalescedDp4aBrick::new(0, 64).can_run());
        assert!(!CoalescedDp4aBrick::new(256, 0).can_run());
    }

    #[test]
    fn coalesced_dp4a_forward_input_length_error() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8: Vec<i8> = vec![1; 128]; // Wrong length
        let input_scale = 1.0f32;
        let weights_q4: Vec<u8> = vec![0x88; 256 * 4 / 2];
        let weight_scales: Vec<f32> = vec![1.0; 4];

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
    }

    #[test]
    fn coalesced_dp4a_forward_weights_length_error() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8: Vec<i8> = vec![1; 256];
        let input_scale = 1.0f32;
        let weights_q4: Vec<u8> = vec![0x88; 100]; // Wrong length
        let weight_scales: Vec<f32> = vec![1.0; 4];

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
    }

    #[test]
    fn coalesced_dp4a_forward_scale_length_error() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8: Vec<i8> = vec![1; 256];
        let input_scale = 1.0f32;
        let weights_q4: Vec<u8> = vec![0x88; 256 * 4 / 2];
        let weight_scales: Vec<f32> = vec![1.0; 2]; // Wrong length

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
    }

    #[test]
    fn coalesced_dp4a_debug_and_clone() {
        let brick = CoalescedDp4aBrick::new(256, 64);
        // Test Debug
        let debug_str = format!("{:?}", brick);
        assert!(debug_str.contains("CoalescedDp4aBrick"));
        // Test Clone
        let cloned = brick.clone();
        assert_eq!(cloned.k, brick.k);
        assert_eq!(cloned.n, brick.n);
    }

    // ============================================================================
    // FusedFfnBrick Tests
    // ============================================================================

    #[test]
    fn fused_ffn_new_basic() {
        let brick = FusedFfnBrick::new(64, 256);
        assert_eq!(brick.hidden_dim, 64);
        assert_eq!(brick.intermediate_dim, 256);
        assert!(!brick.use_packed_dp4a); // default is false
    }

    #[test]
    fn fused_ffn_with_packed_dp4a() {
        let brick = FusedFfnBrick::with_packed_dp4a(64, 256);
        assert!(brick.use_packed_dp4a);
        assert_eq!(brick.hidden_dim, 64);
        assert_eq!(brick.intermediate_dim, 256);
    }

    #[test]
    fn fused_ffn_with_budget() {
        let custom_budget = TokenBudget::from_latency(25.0);
        let brick = FusedFfnBrick::new(64, 256).with_budget(custom_budget);
        assert!((brick.budget().us_per_token - 25.0).abs() < 0.001);
    }

    #[test]
    fn fused_ffn_flops_calculation() {
        let brick = FusedFfnBrick::new(64, 256);
        // Total: 6 * hidden * intermediate = 6 * 64 * 256 = 98304
        assert_eq!(brick.flops(), 6 * 64 * 256);
    }

    #[test]
    fn fused_ffn_arithmetic_intensity() {
        let brick = FusedFfnBrick::new(1536, 8960);
        let ai = brick.arithmetic_intensity();
        assert!(ai > 0.0, "Arithmetic intensity should be positive");
        assert!(ai < 100.0, "Arithmetic intensity should be reasonable");
    }

    #[test]
    fn fused_ffn_forward_basic() {
        let brick = FusedFfnBrick::new(4, 8);

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward should succeed");

        assert_eq!(result.len(), 4);
        assert!(!result.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn fused_ffn_forward_timed() {
        let brick = FusedFfnBrick::new(4, 8).with_budget(TokenBudget::from_latency(100_000.0));

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick
            .forward_timed(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward_timed should succeed");

        assert_eq!(result.output.len(), 4);
        assert_eq!(result.tokens_processed, 1);
        assert!(result.us_per_token > 0.0);
        assert!(result.tokens_per_sec > 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn fused_ffn_execute_valid_dimensions() {
        let brick = FusedFfnBrick::new(64, 256);
        let result = brick.execute();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 64);
    }

    #[test]
    #[allow(deprecated)]
    fn fused_ffn_execute_zero_hidden() {
        let brick = FusedFfnBrick::new(0, 256);
        let result = brick.execute();
        assert!(result.is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn fused_ffn_execute_zero_intermediate() {
        let brick = FusedFfnBrick::new(64, 0);
        let result = brick.execute();
        assert!(result.is_err());
    }

    #[test]
    fn fused_ffn_trait_name() {
        let brick = FusedFfnBrick::new(64, 256);
        assert_eq!(brick.name(), "fused_ffn");
    }

    #[test]
    fn fused_ffn_trait_assertions() {
        let brick = FusedFfnBrick::new(64, 256);
        let assertions = brick.assertions();
        assert!(!assertions.is_empty());

        let names: Vec<&str> = assertions.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"no_nan"));
        assert!(names.contains(&"no_inf"));
        assert!(names.contains(&"budget_met"));
        assert!(names.contains(&"shared_q8_quant"));
        assert!(names.contains(&"swiglu_fused"));
    }

    #[test]
    fn fused_ffn_can_run_valid() {
        let brick = FusedFfnBrick::new(64, 256);
        assert!(brick.can_run());
    }

    #[test]
    fn fused_ffn_can_run_zero_dimensions() {
        assert!(!FusedFfnBrick::new(0, 256).can_run());
        assert!(!FusedFfnBrick::new(64, 0).can_run());
    }

    #[test]
    fn fused_ffn_forward_input_length_error() {
        let brick = FusedFfnBrick::new(4, 8);

        let input = vec![1.0f32; 2]; // Wrong length
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
    }

    #[test]
    fn fused_ffn_forward_gate_proj_length_error() {
        let brick = FusedFfnBrick::new(4, 8);

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 10]; // Wrong length
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
    }

    #[test]
    fn fused_ffn_forward_up_proj_length_error() {
        let brick = FusedFfnBrick::new(4, 8);

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 10]; // Wrong length
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
    }

    #[test]
    fn fused_ffn_forward_down_proj_length_error() {
        let brick = FusedFfnBrick::new(4, 8);

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 10]; // Wrong length

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
    }

    #[test]
    fn fused_ffn_debug_and_clone() {
        let brick = FusedFfnBrick::new(64, 256);
        // Test Debug
        let debug_str = format!("{:?}", brick);
        assert!(debug_str.contains("FusedFfnBrick"));
        // Test Clone
        let cloned = brick.clone();
        assert_eq!(cloned.hidden_dim, brick.hidden_dim);
        assert_eq!(cloned.intermediate_dim, brick.intermediate_dim);
    }

    #[test]
    fn fused_ffn_swiglu_activation_correctness() {
        // Test that SwiGLU activation is computed correctly
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let brick = FusedFfnBrick::new(1, 1);

        // Use identity projections to isolate SwiGLU behavior
        let input = vec![1.0f32];
        let gate_proj = vec![1.0f32]; // gate = input * 1.0 = 1.0
        let up_proj = vec![1.0f32]; // up = input * 1.0 = 1.0
        let down_proj = vec![1.0f32]; // output = hidden * 1.0

        let result = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .unwrap();

        // silu(1.0) = 1.0 / (1 + exp(-1.0)) = 1.0 * 0.731... = 0.731...
        // hidden = silu(gate) * up = 0.731... * 1.0 = 0.731...
        // output = hidden * down = 0.731... * 1.0 = 0.731...
        let expected_silu = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!(
            (result[0] - expected_silu).abs() < 0.001,
            "Expected SwiGLU output ~{}, got {}",
            expected_silu,
            result[0]
        );
    }

    #[test]
    fn fused_ffn_negative_input_values() {
        let brick = FusedFfnBrick::new(2, 4);

        let input = vec![-1.0f32, -2.0f32];
        let gate_proj = vec![0.1f32; 4 * 2];
        let up_proj = vec![0.1f32; 4 * 2];
        let down_proj = vec![0.1f32; 2 * 4];

        let result = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward with negative input should succeed");

        assert_eq!(result.len(), 2);
        assert!(!result.iter().any(|x| x.is_nan()));
        assert!(!result.iter().any(|x| x.is_infinite()));
    }
