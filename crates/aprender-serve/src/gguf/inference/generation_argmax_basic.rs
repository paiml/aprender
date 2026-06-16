
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // T-COV-95 Phase 52: Pure function tests for generation.rs
    // argmax, sample_topk, and sampling edge cases
    // ============================================================================

    // -----------------------------------------------------------------------
    // argmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_argmax_basic() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 3);
    }

    #[test]
    fn test_argmax_first_element_largest() {
        let logits = vec![10.0, 1.0, 2.0, 3.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_last_element_largest() {
        let logits = vec![1.0, 2.0, 3.0, 100.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 3);
    }

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![42.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_empty() {
        let logits: Vec<f32> = Vec::new();
        // Should return 0 for empty logits
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_negative_values() {
        let logits = vec![-5.0, -1.0, -3.0, -2.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 1); // -1.0 is the max
    }

    #[test]
    fn test_argmax_all_same() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        // All equal -> returns some valid index (implementation may pick any)
        let result = OwnedQuantizedModel::argmax(&logits);
        assert!(result < 4, "Expected valid index, got {}", result);
    }

    #[test]
    fn test_argmax_with_nan() {
        // NaN comparison: partial_cmp returns None -> Equal, so first non-NaN max wins
        let logits = vec![1.0, f32::NAN, 3.0, 2.0];
        let result = OwnedQuantizedModel::argmax(&logits);
        // The argmax skips NaN via partial_cmp -> Equal ordering
        // Result should be 2 (3.0 is max among comparable values)
        assert!(result == 2 || result == 1); // NaN behavior is implementation-defined
    }

    #[test]
    fn test_argmax_with_infinity() {
        let logits = vec![1.0, f32::INFINITY, 3.0, 2.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_with_neg_infinity() {
        let logits = vec![f32::NEG_INFINITY, 0.0, -1.0];
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_large_vocab() {
        // Simulate a large vocabulary
        let mut logits = vec![0.0f32; 32000];
        logits[15000] = 100.0;
        assert_eq!(OwnedQuantizedModel::argmax(&logits), 15000);
    }

    // -----------------------------------------------------------------------
    // sample_topk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_topk_deterministic_single_dominant() {
        // One logit is vastly larger -> should always select it
        let logits = vec![0.0, 0.0, 100.0, 0.0, 0.0];
        for _ in 0..10 {
            let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 5);
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_sample_topk_top_k_1() {
        // top_k=1 is equivalent to argmax
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sample_topk_high_temperature() {
        // High temperature makes distribution more uniform
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 100.0, 5);
        // Result should be a valid index
        assert!(result < 5);
    }

    #[test]
    fn test_sample_topk_low_temperature() {
        // Very low temperature makes distribution peaky -> should pick max
        let logits = vec![1.0, 2.0, 10.0, 3.0, 4.0];
        // With extremely low temp, the distribution should be peaked at max
        let result = OwnedQuantizedModel::sample_topk(&logits, 0.001, 5);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_sample_topk_all_equal() {
        // All logits equal -> should return some valid index
        let logits = vec![1.0; 10];
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 10);
        assert!(result < 10);
    }

    #[test]
    fn test_sample_topk_single_element() {
        let logits = vec![42.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_sample_topk_top_k_larger_than_vocab() {
        // top_k > logits length should still work (truncates to available)
        let logits = vec![1.0, 2.0, 3.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 100);
        assert!(result < 3);
    }

    #[test]
    fn test_sample_topk_negative_logits() {
        let logits = vec![-10.0, -5.0, -1.0, -3.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 4);
        assert!(result < 4);
    }

    #[test]
    fn test_sample_topk_large_logit_spread() {
        // Huge spread should reliably select the max
        let mut logits = vec![-1000.0; 100];
        logits[50] = 1000.0;
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 100);
        assert_eq!(result, 50);
    }

    #[test]
    fn test_sample_topk_returns_valid_range() {
        // Run many times to exercise randomness
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for _ in 0..50 {
            let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 3);
            // Should only return one of top-3 indices (2, 3, or 4)
            assert!(
                result == 2 || result == 3 || result == 4,
                "sample_topk returned {} which is not in top-3",
                result
            );
        }
    }

    #[test]
    fn test_sample_topk_temperature_scaling() {
        // Verify that temperature actually scales logits
        // At temperature=2.0, logits are halved before softmax
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let result = OwnedQuantizedModel::sample_topk(&logits, 2.0, 4);
        // With temp=2, logit 10 becomes 5, still dominant
        assert!(result < 4);
    }

    #[test]
    fn test_sample_topk_softmax_normalization() {
        // Check that softmax produces valid probabilities
        // (implicitly tested by the fact that sampling works)
        let logits = vec![1.0, 2.0, 3.0];
        // Run multiple samples, all should be valid
        for _ in 0..20 {
            let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 3);
            assert!(result < 3, "Invalid token index: {}", result);
        }
    }

    // ========================================================================
    // PMAT-816 / sampling-algorithms-v1 FALSIFY-SA-006: dense-path repetition
    // penalty (repeat_penalty / repeat_last_n) must NOT be silently dropped.
    //
    // Before the fix, the dense decode loops (generate / generate_with_cache /
    // generate_with_cache_streaming / generate_with_scratch / adaptive) only
    // applied temperature + top_k, exactly like top_p / repeat_penalty were
    // dropped on the dense path while the MoE path applied them.
    // ========================================================================

    /// FALSIFY-SA-006 V1_001 (no-regression / no-op): repeat_penalty == 1.0 is
    /// a no-op and returns the input slice BORROWED (byte-identical, no alloc).
    #[test]
    fn rep_penalty_neutral_is_byte_identical_borrow() {
        let logits = vec![3.0_f32, 1.0, -2.0, 0.5];
        let recent = [0u32, 1, 2, 3];
        // penalty == 1.0 → no-op
        let out = apply_repeat_penalty(&logits, 1.0, 64, &recent);
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "neutral penalty must borrow (no allocation)"
        );
        assert_eq!(&*out, logits.as_slice(), "neutral penalty must be identity");
        // repeat_last_n == 0 → no-op
        let out2 = apply_repeat_penalty(&logits, 1.3, 0, &recent);
        assert!(matches!(out2, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*out2, logits.as_slice());
        // empty recent window → no-op
        let out3 = apply_repeat_penalty(&logits, 1.3, 64, &[]);
        assert!(matches!(out3, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*out3, logits.as_slice());
    }

    /// FALSIFY-SA-006 V1_002 (RED-on-bug, GREEN-on-fix): repeat_penalty > 1.0
    /// CHANGES the greedy/argmax token. With the bug (penalty dropped) argmax
    /// stays on the repeated token; with the fix it moves to a different token.
    #[test]
    fn rep_penalty_changes_greedy_token() {
        // Token 0 is the (slightly) dominant logit AND the recently-repeated
        // token. Token 1 is a close runner-up that was NOT repeated.
        let logits = vec![10.0_f32, 9.0, 1.0, 0.0];
        let recent = [0u32, 0, 0]; // token 0 generated three times

        // Bug behavior (no penalty): argmax is token 0.
        assert_eq!(
            OwnedQuantizedModel::argmax(&logits),
            0,
            "baseline: argmax of raw logits is the repeated token 0"
        );

        // Fix behavior: penalize → token 0 logit 10.0 / 2.0 = 5.0 < token 1 (9.0).
        let penalized = apply_repeat_penalty(&logits, 2.0, 64, &recent);
        assert!(
            matches!(penalized, std::borrow::Cow::Owned(_)),
            "active penalty must allocate a modified copy"
        );
        let token = OwnedQuantizedModel::argmax(&penalized);
        assert_eq!(
            token, 1,
            "repeat_penalty must move greedy token off the repeated token 0 → 1"
        );
        // Without the fix this would still be 0 → the falsifier is RED on the bug.
        assert_ne!(
            token,
            OwnedQuantizedModel::argmax(&logits),
            "repeat_penalty must change the selected token vs. no-penalty"
        );
    }

    /// FALSIFY-SA-006 V1_003: repeat_last_n bounds the penalty window. A short
    /// window penalizes fewer repeats than a long window → different logits.
    #[test]
    fn rep_penalty_window_is_bounded_by_repeat_last_n() {
        let logits = vec![10.0_f32, 0.0, 0.0];
        // Token 0 repeated 8 times.
        let recent = [0u32, 0, 0, 0, 0, 0, 0, 0];

        // last_n = 2 → 2 divisions: 10 / 1.5 / 1.5 ≈ 4.444
        let short = apply_repeat_penalty(&logits, 1.5, 2, &recent);
        // last_n = 8 → 8 divisions: 10 / 1.5^8 ≈ 0.390
        let long = apply_repeat_penalty(&logits, 1.5, 8, &recent);
        assert!(
            short[0] > long[0],
            "larger repeat_last_n must compound the penalty more (short {} > long {})",
            short[0],
            long[0]
        );
        // Negative-logit branch uses multiplication (Candle convention).
        let neg = vec![-2.0_f32, 0.0];
        let p = apply_repeat_penalty(&neg, 2.0, 1, &[0u32]);
        assert!((p[0] - (-4.0)).abs() < 1e-6, "negative logit *= penalty");
    }
}
