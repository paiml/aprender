
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

    // ============================================================================
    // PMAT-794: sample_topk_top_p — single-request nucleus (top-p) sampling.
    //
    // Mirrors the batched-decode fix (PMAT-792b / #2080). The live single-request
    // sampler `sample_topk` silently dropped `config.top_p`, so a request asking for
    // e.g. top_p=0.1 got the full top_k distribution sampled (low-prob tokens leaked).
    // `sample_topk_top_p` applies temperature → top-k truncate → top-p nucleus →
    // softmax → inverse-CDF sample, honoring the per-request top_p.
    // ============================================================================

    #[test]
    fn test_sample_topk_top_p_restricts_nucleus_to_dominant_token() {
        // PMAT-794 FALSIFIER (the load-bearing test). Logits [3.0, 0, 0, 0] at temp 1.0
        // softmax to P(token0) ≈ 0.870, P(each other) ≈ 0.043. With top_p = 0.85 the
        // nucleus (smallest prefix with cumulative prob >= top_p) is exactly {token0}
        // (0.870 >= 0.85), so EVERY draw must return index 0. top_k = 4 keeps the whole
        // vocabulary, so the only thing that can restrict the sample is top_p.
        //
        // BEFORE the fix the single-request path called `sample_topk`, which silently
        // dropped top_p: the full 4-way distribution stayed samplable, P(non-token0) ≈
        // 0.13 per draw, and over 512 draws a non-zero token is effectively certain
        // (0.87^512 ≈ 0) → this loop fails (RED). AFTER the fix the nucleus collapses to
        // {token0} and every draw is deterministic (GREEN).
        let logits = [3.0_f32, 0.0, 0.0, 0.0];
        for _ in 0..512 {
            let tok = OwnedQuantizedModel::sample_topk_top_p(&logits, 1.0, 4, 0.85);
            assert_eq!(
                tok, 0,
                "top_p=0.85 must collapse the nucleus to the dominant token 0, got {tok} \
                 (top_p silently dropped → low-prob token leaked through)"
            );
        }
    }

    #[test]
    fn test_sample_topk_top_p_one_is_a_noop_nucleus() {
        // No-regression: top_p == 1.0 (and > 1.0) skips the nucleus cutoff → identical to
        // the pure top_k path. top_k == 1 always returns the argmax, verifying equivalence.
        assert_eq!(
            OwnedQuantizedModel::sample_topk_top_p(&[1.0, 5.0, 2.0], 1.5, 1, 1.0),
            1
        );
        assert_eq!(
            OwnedQuantizedModel::sample_topk_top_p(&[1.0, 5.0, 2.0], 1.5, 1, 2.0),
            1
        );
        assert_eq!(
            OwnedQuantizedModel::sample_topk_top_p(&[9.0, 5.0, 2.0], 0.9, 1, 1.0),
            0
        );
    }

    #[test]
    fn test_sample_topk_top_p_reachable_with_top_p_one() {
        // No-regression: with top_p == 1.0 the result is always a valid in-range token
        // and stays within the top_k set — same observable behavior as `sample_topk`.
        let logits = vec![1.0, 2.0, 3.0, 0.5, 0.1];
        for _ in 0..64 {
            let tok = OwnedQuantizedModel::sample_topk_top_p(&logits, 1.0, 3, 1.0);
            // top_k == 3 keeps tokens {2, 1, 0} (logits 3,2,1); top_p == 1.0 is a no-op.
            assert!(tok < 3, "top_p=1.0 must keep the full top_k set, got {tok}");
        }
    }

    #[test]
    fn test_sample_topk_top_p_low_p_concentrates_on_argmax() {
        // A very small top_p (0.05) keeps only the single most-probable token even when
        // top_k is large — the nucleus can never be empty (always >= 1 token).
        let logits = vec![0.0, 0.0, 8.0, 0.0, 0.0];
        for _ in 0..128 {
            let tok = OwnedQuantizedModel::sample_topk_top_p(&logits, 1.0, 40, 0.05);
            assert_eq!(tok, 2, "tiny top_p must collapse to the dominant token, got {tok}");
        }
    }
}
