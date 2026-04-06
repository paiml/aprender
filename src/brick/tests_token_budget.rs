//! Additional brick tests - Part 02
//!
//! Focuses on TokenBudget, TokenResult, BrickError, and BrickAssertion tests.
//! This file kept under 400 lines per PMAT file health rules.

#[cfg(test)]
mod tests {
    use crate::brick::*;

    // =========================================================================
    // TokenBudget Edge Cases
    // =========================================================================

    #[test]
    fn token_budget_default_values() {
        let budget = TokenBudget::default();
        // Default is 100us = 10k tok/s
        assert!((budget.us_per_token - 100.0).abs() < 0.001);
        assert!((budget.tokens_per_sec - 10_000.0).abs() < 1.0);
        assert_eq!(budget.batch_size, 1);
    }

    #[test]
    fn token_budget_with_batch_size() {
        let budget = TokenBudget::from_latency(50.0).with_batch_size(4);
        assert_eq!(budget.batch_size, 4);
        assert!((budget.us_per_token - 50.0).abs() < 0.001);
    }

    #[test]
    fn token_budget_extreme_values() {
        // Very fast (1us = 1M tok/s)
        let fast = TokenBudget::from_latency(1.0);
        assert!((fast.tokens_per_sec - 1_000_000.0).abs() < 1.0);

        // Very slow (10000us = 100 tok/s)
        let slow = TokenBudget::from_throughput(100.0);
        assert!((slow.us_per_token - 10_000.0).abs() < 0.1);
    }

    #[test]
    fn token_budget_boundary_is_met() {
        let budget = TokenBudget::from_latency(100.0);
        // Exactly at boundary
        assert!(budget.is_met(100.0));
        // Just under
        assert!(budget.is_met(99.999));
        // Just over
        assert!(!budget.is_met(100.001));
    }

    // =========================================================================
    // TokenResult Edge Cases
    // =========================================================================

    #[test]
    fn token_result_default() {
        let result: TokenResult<Vec<f32>> = TokenResult::default();
        assert!(result.output.is_empty());
        assert_eq!(result.tokens_processed, 0);
        assert_eq!(result.us_per_token, 0.0);
        assert_eq!(result.tokens_per_sec, 0.0);
        assert!(result.budget_met);
    }

    #[test]
    fn token_result_zero_tokens_handled() {
        let budget = TokenBudget::from_latency(100.0);
        // Zero tokens should not cause division by zero
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 0, 100.0, &budget);
        // With 0 tokens, us_per_token = 100.0 / max(0, 1) = 100.0
        assert!((result.us_per_token - 100.0).abs() < 0.001);
    }

    #[test]
    fn token_result_zero_elapsed_time() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 10, 0.0, &budget);
        // Zero elapsed time should not cause issues
        assert!(result.us_per_token == 0.0);
        assert!(result.tokens_per_sec == 0.0); // 1M / 0 special case
        assert!(result.budget_met); // 0 <= 100
    }

    // =========================================================================
    // BrickError Display and Error Trait
    // =========================================================================

    #[test]
    fn brick_error_display_assertion_failed() {
        let err = BrickError::AssertionFailed {
            name: "test_assertion".to_string(),
            expected: "42".to_string(),
            actual: "0".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test_assertion"));
        assert!(msg.contains("42"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn brick_error_display_budget_exceeded() {
        let err = BrickError::BudgetExceeded {
            limit_us: 10.0,
            actual_us: 20.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("20"));
    }

    #[test]
    fn brick_error_display_compute_error() {
        let err = BrickError::ComputeError("test error message".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test error message"));
    }

    #[test]
    fn brick_error_display_invalid_input() {
        let err = BrickError::InvalidInput("bad input".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("bad input"));
    }

    #[test]
    fn brick_error_is_std_error() {
        let err = BrickError::InvalidInput("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    // =========================================================================
    // BrickAssertion Tests
    // =========================================================================

    #[test]
    fn assertion_equiv_scalar_description() {
        let assertion = BrickAssertion::equiv_scalar(0.001);
        assert_eq!(assertion.name, "equiv_scalar");
        assert!(assertion.description.contains("0.001"));
    }

    #[test]
    fn assertion_bounds_check_pass() {
        let assertion = BrickAssertion::bounds(-10.0, 10.0);
        let data = [-5.0f32, 0.0, 5.0];
        assert!(assertion.check_f32(&data, true).is_ok());
    }

    #[test]
    fn assertion_bounds_check_fail_low() {
        let assertion = BrickAssertion::bounds(-10.0, 10.0);
        let data = [-15.0f32, 0.0, 5.0];
        let result = assertion.check_f32(&data, true);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed { actual, .. }) = result {
            assert!(actual.contains("-15"));
            assert!(actual.contains("index 0"));
        }
    }

    #[test]
    fn assertion_bounds_check_fail_high() {
        let assertion = BrickAssertion::bounds(-10.0, 10.0);
        let data = [0.0f32, 0.0, 15.0];
        let result = assertion.check_f32(&data, true);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed { actual, .. }) = result {
            assert!(actual.contains("15"));
            assert!(actual.contains("index 2"));
        }
    }

    #[test]
    fn assertion_no_inf_check_pass() {
        let assertion = BrickAssertion::no_inf();
        let data = [1e30f32, -1e30, 0.0];
        assert!(assertion.check_f32(&data, true).is_ok());
    }

    #[test]
    fn assertion_no_inf_check_fail_positive() {
        let assertion = BrickAssertion::no_inf();
        let data = [1.0f32, f32::INFINITY, 3.0];
        let result = assertion.check_f32(&data, true);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed { actual, .. }) = result {
            assert!(actual.contains("index 1"));
        }
    }

    #[test]
    fn assertion_no_inf_check_fail_negative() {
        let assertion = BrickAssertion::no_inf();
        let data = [f32::NEG_INFINITY, 2.0, 3.0];
        let result = assertion.check_f32(&data, true);
        assert!(result.is_err());
    }

    #[test]
    fn assertion_budget_met_pass() {
        let assertion = BrickAssertion::budget_met();
        assert!(assertion.check_f32(&[1.0, 2.0, 3.0], true).is_ok());
    }

    #[test]
    fn assertion_budget_met_fail() {
        let assertion = BrickAssertion::budget_met();
        let result = assertion.check_f32(&[1.0, 2.0, 3.0], false);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed {
            expected, actual, ..
        }) = result
        {
            assert!(expected.contains("budget met"));
            assert!(actual.contains("budget exceeded"));
        }
    }

    // =========================================================================
    // BrickVerification Tests
    // =========================================================================

    #[test]
    fn verification_pass() {
        let v = BrickVerification::pass();
        assert!(v.is_valid);
        assert!(v.results.is_empty());
    }

    #[test]
    fn verification_fail() {
        let v = BrickVerification::fail("test_brick", "test failure reason");
        assert!(!v.is_valid);
        assert_eq!(v.results.len(), 1);
        assert_eq!(v.results[0].0, "test_brick");
        assert!(!v.results[0].1); // passed = false
        assert_eq!(v.results[0].2, "test failure reason");
    }

    #[test]
    fn verification_add_passed() {
        let mut v = BrickVerification::pass();
        v.add("check1", true, "ok");
        assert!(v.is_valid); // Still valid
        assert_eq!(v.results.len(), 1);
    }

    #[test]
    fn verification_add_failed() {
        let mut v = BrickVerification::pass();
        v.add("check1", false, "failed");
        assert!(!v.is_valid); // Now invalid
        assert_eq!(v.results.len(), 1);
    }

    #[test]
    fn verification_multiple_adds() {
        let mut v = BrickVerification::pass();
        v.add("check1", true, "ok");
        v.add("check2", true, "ok");
        v.add("check3", false, "failed");
        v.add("check4", true, "ok");
        assert!(!v.is_valid); // One failure makes it invalid
        assert_eq!(v.results.len(), 4);
    }

    // =========================================================================
    // RmsNormBrick Tests
    // =========================================================================

    #[test]
    fn rmsnorm_input_length_mismatch() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        let input = vec![1.0; 8]; // Wrong size
        let result = brick.run(&input);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("8"));
            assert!(msg.contains("4"));
        }
    }

    #[test]
    fn rmsnorm_verify_passes() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        let v = brick.verify();
        assert!(v.is_valid);
    }

    #[test]
    fn rmsnorm_can_run() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        assert!(brick.can_run());
    }

    #[test]
    fn rmsnorm_with_custom_budget() {
        let brick =
            RmsNormBrick::new(vec![1.0; 4], 1e-5).with_budget(TokenBudget::from_latency(50.0));
        assert!((brick.budget().us_per_token - 50.0).abs() < 0.001);
    }

    // =========================================================================
    // QkvBrick Tests
    // =========================================================================

    #[test]
    fn qkv_total_out_dim() {
        let brick = QkvBrick::new(128, 64, 32, 32);
        assert_eq!(brick.total_out_dim(), 128); // 64 + 32 + 32
    }

    #[test]
    fn qkv_with_bias() {
        let brick = QkvBrick::new(128, 64, 32, 32).with_bias();
        assert!(brick.has_bias);
    }

    #[test]
    fn qkv_without_bias() {
        let brick = QkvBrick::new(128, 64, 32, 32);
        assert!(!brick.has_bias);
    }

    // =========================================================================
    // AttentionBrick Tests
    // =========================================================================

    #[test]
    fn attention_gqa_group_size() {
        // 8 heads, 2 KV heads = 4 groups
        let brick = AttentionBrick::new(8, 2, 64);
        assert_eq!(brick.group_size(), 4);
    }

    #[test]
    fn attention_gqa_group_size_mha() {
        // MHA: 8 heads, 8 KV heads = 1 group
        let brick = AttentionBrick::new(8, 8, 64);
        assert_eq!(brick.group_size(), 1);
    }

    #[test]
    fn attention_gqa_group_size_mqa() {
        // MQA: 8 heads, 1 KV head = 8 groups
        let brick = AttentionBrick::new(8, 1, 64);
        assert_eq!(brick.group_size(), 8);
    }

    #[test]
    fn attention_gqa_group_size_zero_kv() {
        // Edge case: 0 KV heads (should not divide by zero)
        let brick = AttentionBrick::new(8, 0, 64);
        assert_eq!(brick.group_size(), 8); // max(1, 0) = 1, 8/1 = 8
    }
}
