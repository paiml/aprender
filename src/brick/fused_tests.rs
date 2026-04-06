//! Comprehensive tests for fused.rs - CoalescedDp4aBrick and FusedFfnBrick
//!
//! These tests cover uncovered functions and edge cases for the fused CUDA bricks.

#[cfg(test)]
mod tests {
    use crate::brick::{
        fused::{CoalescedDp4aBrick, FusedFfnBrick},
        ComputeBrick, TokenBudget,
    };

    #[test]
    fn fused_ffn_zero_input_values() {
        let brick = FusedFfnBrick::new(2, 4);

        let input = vec![0.0f32, 0.0f32];
        let gate_proj = vec![0.1f32; 4 * 2];
        let up_proj = vec![0.1f32; 4 * 2];
        let down_proj = vec![0.1f32; 2 * 4];

        let result = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward with zero input should succeed");

        assert_eq!(result.len(), 2);
        // With zero input, output should be zero (all projections zero)
        for v in &result {
            assert!(v.abs() < 1e-6, "Expected near-zero output, got {}", v);
        }
    }

    // ============================================================================
    // Boundary and Edge Case Tests
    // ============================================================================

    #[test]
    fn coalesced_dp4a_large_dimensions() {
        let brick = CoalescedDp4aBrick::new(4096, 1024);
        assert_eq!(brick.flops(), 2 * 4096 * 1024);
        assert!(brick.can_run());
    }

    #[test]
    fn fused_ffn_large_dimensions() {
        let brick = FusedFfnBrick::new(4096, 22528);
        assert_eq!(brick.flops(), 6 * 4096 * 22528);
        assert!(brick.can_run());
    }

    #[test]
    fn coalesced_dp4a_budget_throughput_consistency() {
        let budget = TokenBudget::from_latency(100.0);
        let brick = CoalescedDp4aBrick::new(256, 64).with_budget(budget);

        let retrieved_budget = brick.budget();
        // Verify throughput = 1e6 / latency
        let expected_throughput = 1_000_000.0 / 100.0;
        assert!(
            (retrieved_budget.tokens_per_sec - expected_throughput).abs() < 1.0,
            "Throughput should be ~{}, got {}",
            expected_throughput,
            retrieved_budget.tokens_per_sec
        );
    }

    #[test]
    fn fused_ffn_budget_throughput_consistency() {
        let budget = TokenBudget::from_latency(50.0);
        let brick = FusedFfnBrick::new(64, 256).with_budget(budget);

        let retrieved_budget = brick.budget();
        let expected_throughput = 1_000_000.0 / 50.0;
        assert!(
            (retrieved_budget.tokens_per_sec - expected_throughput).abs() < 1.0,
            "Throughput should be ~{}, got {}",
            expected_throughput,
            retrieved_budget.tokens_per_sec
        );
    }

    #[test]
    fn coalesced_dp4a_forward_timed_budget_check() {
        // Use a very lenient budget that should pass
        let brick =
            CoalescedDp4aBrick::new(256, 4).with_budget(TokenBudget::from_latency(1_000_000.0));

        let input_q8: Vec<i8> = vec![1; 256];
        let input_scale = 1.0f32;
        let weights_q4: Vec<u8> = vec![0x88; 256 * 4 / 2];
        let weight_scales: Vec<f32> = vec![1.0; 4];

        let result = brick
            .forward_timed(&input_q8, input_scale, &weights_q4, &weight_scales)
            .unwrap();

        assert!(
            result.budget_met,
            "Budget should be met with lenient budget"
        );
    }

    #[test]
    fn fused_ffn_forward_timed_budget_check() {
        // Use a very lenient budget that should pass
        let brick = FusedFfnBrick::new(4, 8).with_budget(TokenBudget::from_latency(1_000_000.0));

        let input = vec![1.0f32; 4];
        let gate_proj = vec![0.1f32; 8 * 4];
        let up_proj = vec![0.1f32; 8 * 4];
        let down_proj = vec![0.1f32; 4 * 8];

        let result = brick
            .forward_timed(&input, &gate_proj, &up_proj, &down_proj)
            .unwrap();

        assert!(
            result.budget_met,
            "Budget should be met with lenient budget"
        );
    }
include!("coalesced_dp4a_tests.rs");
}
