//! Property-based tests for text generation and sampling strategies
//!
//! These tests use proptest to verify mathematical properties of sampling algorithms.

use proptest::prelude::*;
use realizar::{
    generate::{
        apply_temperature, sample_greedy, sample_token, sample_top_k, sample_top_p,
        GenerationConfig,
    },
    tensor::Tensor,
};

/// Strategy for generating valid logits tensors
fn logits_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Tensor<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, min_len..=max_len).prop_map(|data| {
        let len = data.len();
        Tensor::from_vec(vec![len], data).expect("test")
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Temperature scaling preserves tensor shape
    #[test]
    fn test_temperature_preserves_shape(
        logits in logits_strategy(1, 100),
        temperature in 0.1f32..10.0f32
    ) {
        let scaled = apply_temperature(&logits, temperature).expect("test");
        prop_assert_eq!(scaled.shape(), logits.shape());
    }

    /// Temperature scaling with t=1 preserves values
    #[test]
    fn test_temperature_one_identity(logits in logits_strategy(1, 100)) {
        let scaled = apply_temperature(&logits, 1.0).expect("test");
        for (a, b) in logits.data().iter().zip(scaled.data().iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    /// Temperature scaling is monotonic (higher temp = more uniform)
    #[test]
    fn test_temperature_scaling_effect(
        logits in logits_strategy(2, 50),
        temp in 0.1f32..5.0f32
    ) {
        let scaled = apply_temperature(&logits, temp).expect("test");
        // Verify scaling: scaled[i] = logits[i] / temp
        for (orig, scaled_val) in logits.data().iter().zip(scaled.data().iter()) {
            let expected = orig / temp;
            prop_assert!((scaled_val - expected).abs() < 1e-5);
        }
    }

    /// Greedy sampling always returns valid index
    #[test]
    fn test_greedy_returns_valid_index(logits in logits_strategy(1, 100)) {
        let idx = sample_greedy(&logits).expect("test");
        prop_assert!(idx < logits.size());
    }

    /// Greedy sampling returns argmax
    #[test]
    fn test_greedy_returns_argmax(logits in logits_strategy(1, 100)) {
        let idx = sample_greedy(&logits).expect("test");
        let data = logits.data();
        let max_val = data[idx];

        // Verify this is the maximum
        for &val in data {
            prop_assert!(val <= max_val + 1e-6);
        }
    }

    /// Top-k sampling returns valid index
    #[test]
    fn test_top_k_returns_valid_index(
        logits in logits_strategy(1, 100),
        k in 1usize..50,
        rng in 0.0f32..1.0f32
    ) {
        let idx = sample_top_k(&logits, k.min(logits.size()), rng).expect("test");
        prop_assert!(idx < logits.size());
    }

    /// Top-k with k=1 behaves like greedy
    #[test]
    fn test_top_k_one_is_greedy(
        logits in logits_strategy(1, 100),
        rng in 0.0f32..1.0f32
    ) {
        let greedy_idx = sample_greedy(&logits).expect("test");
        let top_k_idx = sample_top_k(&logits, 1, rng).expect("test");
        prop_assert_eq!(greedy_idx, top_k_idx);
    }

    /// Top-p sampling returns valid index
    #[test]
    fn test_top_p_returns_valid_index(
        logits in logits_strategy(1, 100),
        p in 0.01f32..1.0f32,
        rng in 0.0f32..1.0f32
    ) {
        let idx = sample_top_p(&logits, p, rng).expect("test");
        prop_assert!(idx < logits.size());
    }

    /// Top-p with p=1.0 includes all tokens
    #[test]
    fn test_top_p_full_includes_all(
        logits in logits_strategy(1, 50),
        rng in 0.0f32..1.0f32
    ) {
        let idx = sample_top_p(&logits, 1.0, rng).expect("test");
        prop_assert!(idx < logits.size());
    }

    /// Sample token with greedy config returns argmax
    #[test]
    fn test_sample_token_greedy(logits in logits_strategy(1, 100)) {
        let config = GenerationConfig::greedy();
        let idx = sample_token(&logits, &config, 0.5).expect("test");
        let greedy_idx = sample_greedy(&logits).expect("test");
        prop_assert_eq!(idx, greedy_idx);
    }

    /// Different temperatures produce different distributions
    #[test]
    fn test_different_temperatures(
        logits in logits_strategy(2, 50),
        t1 in 0.1f32..1.0f32,
        t2 in 1.5f32..5.0f32
    ) {
        let scaled1 = apply_temperature(&logits, t1).expect("test");
        let scaled2 = apply_temperature(&logits, t2).expect("test");

        // Lower temperature should have larger differences between values
        if logits.size() >= 2 {
            let range1 = scaled1.data().iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - scaled1.data().iter().copied().fold(f32::INFINITY, f32::min);
            let range2 = scaled2.data().iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - scaled2.data().iter().copied().fold(f32::INFINITY, f32::min);

            // Higher temperature (t2) should have smaller range
            prop_assert!(range1 >= range2 - 1e-5);
        }
    }
}
