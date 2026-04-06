
impl ComputeBrick for FusedFfnBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "fused_ffn"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
            BrickAssertion {
                name: "shared_q8_quant".to_string(),
                description: "Input quantized once, shared by gate & up projections".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "shared_q8_quant".to_string(),
                },
            },
            BrickAssertion {
                name: "swiglu_fused".to_string(),
                description: "SwiGLU activation fused (silu(gate) * up in single kernel)"
                    .to_string(),
                kind: AssertionKind::Custom {
                    check_name: "swiglu_fused".to_string(),
                },
            },
        ]
    }

    fn can_run(&self) -> bool {
        self.hidden_dim > 0 && self.intermediate_dim > 0
    }
}

// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CoalescedDp4aBrick Tests
    // =========================================================================

    #[test]
    fn test_coalesced_dp4a_new() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        assert_eq!(brick.k, 256);
        assert_eq!(brick.n, 128);
    }

    #[test]
    fn test_coalesced_dp4a_flops() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        // FLOPS = 2 * K * N = 2 * 256 * 128 = 65536
        assert_eq!(brick.flops(), 65536);
    }

    #[test]
    fn test_coalesced_dp4a_arithmetic_intensity() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        let intensity = brick.arithmetic_intensity();
        // Should be positive and reasonable
        assert!(intensity > 0.0);
        assert!(intensity < 100.0);
    }

    #[test]
    fn test_coalesced_dp4a_forward_simple() {
        let brick = CoalescedDp4aBrick::new(4, 2);
        let input_q8 = vec![1i8, 2, 3, 4];
        let input_scale = 1.0;
        let weights_q4 = vec![0x88u8, 0x88, 0x88, 0x88]; // 4 bytes = 2 * 4 / 2
        let weight_scales = vec![1.0, 1.0];

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), 2);
    }

    #[test]
    fn test_coalesced_dp4a_forward_invalid_input_length() {
        let brick = CoalescedDp4aBrick::new(4, 2);
        let input_q8 = vec![1i8, 2, 3]; // Wrong length: 3 instead of 4
        let input_scale = 1.0;
        let weights_q4 = vec![0x88u8, 0x88, 0x88, 0x88];
        let weight_scales = vec![1.0, 1.0];

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Input length"));
    }

    #[test]
    fn test_coalesced_dp4a_forward_invalid_weights_length() {
        let brick = CoalescedDp4aBrick::new(4, 2);
        let input_q8 = vec![1i8, 2, 3, 4];
        let input_scale = 1.0;
        let weights_q4 = vec![0x88u8, 0x88]; // Wrong length
        let weight_scales = vec![1.0, 1.0];

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Weights length"));
    }

    #[test]
    fn test_coalesced_dp4a_forward_invalid_scales_length() {
        let brick = CoalescedDp4aBrick::new(4, 2);
        let input_q8 = vec![1i8, 2, 3, 4];
        let input_scale = 1.0;
        let weights_q4 = vec![0x88u8, 0x88, 0x88, 0x88];
        let weight_scales = vec![1.0]; // Wrong length: 1 instead of 2

        let result = brick.forward(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scales length"));
    }

    #[test]
    fn test_coalesced_dp4a_forward_timed() {
        let brick = CoalescedDp4aBrick::new(4, 2);
        let input_q8 = vec![1i8, 2, 3, 4];
        let input_scale = 1.0;
        let weights_q4 = vec![0x88u8, 0x88, 0x88, 0x88];
        let weight_scales = vec![1.0, 1.0];

        let result = brick.forward_timed(&input_q8, input_scale, &weights_q4, &weight_scales);
        assert!(result.is_ok());
        let token_result = result.expect("token_result");
        assert_eq!(token_result.tokens_processed, 1);
        assert!(token_result.us_per_token > 0.0);
        assert!(token_result.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_coalesced_dp4a_can_run() {
        // Valid: k is multiple of 256
        let brick = CoalescedDp4aBrick::new(256, 128);
        assert!(brick.can_run());

        // Invalid: k is not multiple of 256
        let brick_invalid = CoalescedDp4aBrick::new(100, 128);
        assert!(!brick_invalid.can_run());

        // Invalid: k is 0
        let brick_zero_k = CoalescedDp4aBrick::new(0, 128);
        assert!(!brick_zero_k.can_run());

        // Invalid: n is 0
        let brick_zero_n = CoalescedDp4aBrick::new(256, 0);
        assert!(!brick_zero_n.can_run());
    }

    #[test]
    fn test_coalesced_dp4a_compute_brick_name() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        assert_eq!(brick.name(), "coalesced_dp4a");
    }

    #[test]
    fn test_coalesced_dp4a_compute_brick_budget() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        let budget = brick.budget();
        assert!(budget.us_per_token > 0.0);
    }

    #[test]
    fn test_coalesced_dp4a_compute_brick_assertions() {
        let brick = CoalescedDp4aBrick::new(256, 128);
        let assertions = brick.assertions();
        assert!(!assertions.is_empty());
        // Should have no_nan, no_inf, budget_met, bandwidth_efficient
        assert!(assertions.iter().any(|a| a.name == "no_nan"));
        assert!(assertions.iter().any(|a| a.name == "no_inf"));
        assert!(assertions.iter().any(|a| a.name == "budget_met"));
        assert!(assertions.iter().any(|a| a.name == "bandwidth_efficient"));
    }

    #[test]
    fn test_coalesced_dp4a_with_budget() {
        let brick = CoalescedDp4aBrick::new(256, 128).with_budget(TokenBudget::from_latency(100.0));
        assert!((brick.budget().us_per_token - 100.0).abs() < 0.01);
    }

    #[test]
    #[allow(deprecated)]
    fn test_coalesced_dp4a_execute_legacy() {
        // Valid dimensions
        let brick = CoalescedDp4aBrick::new(256, 128);
        let result = brick.execute();
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), 128);

        // Invalid: k not multiple of 256
        let brick_invalid = CoalescedDp4aBrick::new(100, 128);
        let result_invalid = brick_invalid.execute();
        assert!(result_invalid.is_err());
    }

    // =========================================================================
    // FusedFfnBrick Tests
    // =========================================================================

    #[test]
    fn test_fused_ffn_new() {
        let brick = FusedFfnBrick::new(128, 512);
        assert_eq!(brick.hidden_dim, 128);
        assert_eq!(brick.intermediate_dim, 512);
    }

    #[test]
    fn test_fused_ffn_with_packed_dp4a() {
        let brick = FusedFfnBrick::with_packed_dp4a(128, 512);
        assert!(brick.use_packed_dp4a);
    }

    #[test]
    fn test_fused_ffn_flops() {
        let brick = FusedFfnBrick::new(128, 512);
        // FLOPS = 6 * hidden * intermediate = 6 * 128 * 512 = 393216
        assert_eq!(brick.flops(), 393216);
    }

    #[test]
    fn test_fused_ffn_arithmetic_intensity() {
        let brick = FusedFfnBrick::new(128, 512);
        let intensity = brick.arithmetic_intensity();
        assert!(intensity > 0.0);
        assert!(intensity < 100.0);
    }

    #[test]
    fn test_fused_ffn_forward_simple() {
        let brick = FusedFfnBrick::new(2, 4);
        let input = vec![1.0, 2.0];
        let gate_proj = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 4 * 2 = 8
        let up_proj = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let down_proj = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]; // 2 * 4 = 8

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 2);
        // Output should be finite
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_fused_ffn_forward_invalid_input() {
        let brick = FusedFfnBrick::new(4, 8);
        let input = vec![1.0, 2.0]; // Wrong: 2 instead of 4
        let gate_proj = vec![0.1; 32];
        let up_proj = vec![0.1; 32];
        let down_proj = vec![0.1; 32];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Input length"));
    }

    #[test]
    fn test_fused_ffn_forward_invalid_gate_proj() {
        let brick = FusedFfnBrick::new(4, 8);
        let input = vec![1.0; 4];
        let gate_proj = vec![0.1; 16]; // Wrong: 16 instead of 32
        let up_proj = vec![0.1; 32];
        let down_proj = vec![0.1; 32];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Gate/Up length"));
    }

    #[test]
    fn test_fused_ffn_forward_invalid_down_proj() {
        let brick = FusedFfnBrick::new(4, 8);
        let input = vec![1.0; 4];
        let gate_proj = vec![0.1; 32];
        let up_proj = vec![0.1; 32];
        let down_proj = vec![0.1; 16]; // Wrong: 16 instead of 32

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Down length"));
    }

    #[test]
    fn test_fused_ffn_forward_timed() {
        let brick = FusedFfnBrick::new(2, 4);
        let input = vec![1.0, 2.0];
        let gate_proj = vec![0.1; 8];
        let up_proj = vec![0.1; 8];
        let down_proj = vec![0.1; 8];

        let result = brick.forward_timed(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_ok());
        let token_result = result.expect("token_result");
        assert_eq!(token_result.tokens_processed, 1);
        assert!(token_result.us_per_token > 0.0);
        assert!(token_result.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_fused_ffn_can_run() {
        // Valid
        let brick = FusedFfnBrick::new(128, 512);
        assert!(brick.can_run());

        // Invalid: hidden_dim is 0
        let brick_zero_h = FusedFfnBrick::new(0, 512);
        assert!(!brick_zero_h.can_run());

        // Invalid: intermediate_dim is 0
        let brick_zero_i = FusedFfnBrick::new(128, 0);
        assert!(!brick_zero_i.can_run());
    }

    #[test]
    fn test_fused_ffn_compute_brick_name() {
        let brick = FusedFfnBrick::new(128, 512);
        assert_eq!(brick.name(), "fused_ffn");
    }

    #[test]
    fn test_fused_ffn_compute_brick_budget() {
        let brick = FusedFfnBrick::new(128, 512);
        let budget = brick.budget();
        // Default budget is 12.2µs
        assert!((budget.us_per_token - 12.2).abs() < 0.1);
    }

    #[test]
    fn test_fused_ffn_compute_brick_assertions() {
        let brick = FusedFfnBrick::new(128, 512);
        let assertions = brick.assertions();
        assert!(!assertions.is_empty());
        // Should have no_nan, no_inf, budget_met, shared_q8_quant, swiglu_fused
        assert!(assertions.iter().any(|a| a.name == "no_nan"));
        assert!(assertions.iter().any(|a| a.name == "no_inf"));
        assert!(assertions.iter().any(|a| a.name == "budget_met"));
        assert!(assertions.iter().any(|a| a.name == "shared_q8_quant"));
        assert!(assertions.iter().any(|a| a.name == "swiglu_fused"));
    }

    #[test]
    fn test_fused_ffn_with_budget() {
        let brick = FusedFfnBrick::new(128, 512).with_budget(TokenBudget::from_latency(50.0));
        assert!((brick.budget().us_per_token - 50.0).abs() < 0.01);
    }

    #[test]
    #[allow(deprecated)]
    fn test_fused_ffn_execute_legacy() {
        // Valid dimensions
        let brick = FusedFfnBrick::new(128, 512);
        let result = brick.execute();
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), 128);

        // Invalid: zero dimension
        let brick_invalid = FusedFfnBrick::new(0, 512);
        let result_invalid = brick_invalid.execute();
        assert!(result_invalid.is_err());
    }

    #[test]
    fn test_fused_ffn_swiglu_activation() {
        // Test that SwiGLU activation works correctly
        // silu(x) = x / (1 + exp(-x))
        // For x=0: silu(0) = 0 / (1 + 1) = 0
        let brick = FusedFfnBrick::new(2, 2);
        let input = vec![1.0, 1.0];
        // Gate proj gives 0 (so silu(0) = 0)
        let gate_proj = vec![0.0, 0.0, 0.0, 0.0];
        let up_proj = vec![1.0, 1.0, 1.0, 1.0];
        let down_proj = vec![1.0, 1.0, 1.0, 1.0];

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_ok());
        let output = result.expect("output");
        // With gate=0, silu(0)=0, so output should be 0
        for &val in &output {
            assert!(val.abs() < 0.001, "Expected ~0, got {}", val);
        }
    }

    #[test]
    fn test_fused_ffn_identity_down_proj() {
        // Test with identity-like down projection
        let brick = FusedFfnBrick::new(2, 2);
        let input = vec![1.0, 0.0];
        let gate_proj = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let up_proj = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let down_proj = vec![1.0, 0.0, 0.0, 1.0]; // Identity

        let result = brick.forward(&input, &gate_proj, &up_proj, &down_proj);
        assert!(result.is_ok());
        let output = result.expect("output");
        // Output should be finite
        for &val in &output {
            assert!(val.is_finite());
        }
    }
}
