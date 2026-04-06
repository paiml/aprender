//! Pipeline Fuzz Tests
//!
//! Property-based and fuzz tests for the full inference pipeline.
//! Validates correctness under various edge cases and random inputs.

use proptest::prelude::*;
use realizar::{
    generate::{GenerationConfig, SamplingStrategy},
    layers::{Model, ModelConfig},
    Tensor,
};
use trueno::{Matrix, Vector};

// ============================================================================
// Fuzz Test Helpers
// ============================================================================

/// Create a small test model configuration
fn test_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
    }
}

// ============================================================================
// Property-Based Fuzz Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Fuzz test: Model forward should always produce finite outputs
    #[test]
    fn fuzz_forward_output_finite(seq_len in 1usize..10) {
        let config = test_config();
        let model = Model::new(config).expect("model creation");

        let tokens: Vec<usize> = (0..seq_len).map(|i| i % 100).collect();
        let result = model.forward(&tokens);

        prop_assert!(result.is_ok(), "Forward failed");
        let logits = result.expect("test");
        prop_assert!(
            logits.data().iter().all(|x| x.is_finite()),
            "Forward output contains non-finite values"
        );
    }

    /// Fuzz test: Output shape should match [seq_len, vocab_size]
    #[test]
    fn fuzz_forward_output_shape(seq_len in 1usize..10) {
        let config = test_config();
        let model = Model::new(config.clone()).expect("model creation");

        let tokens: Vec<usize> = (0..seq_len).map(|i| i % 100).collect();
        let logits = model.forward(&tokens).expect("forward");

        let shape = logits.shape();
        prop_assert_eq!(shape.len(), 2, "Expected 2D output");
        prop_assert_eq!(shape[0], seq_len, "Sequence dimension mismatch");
        prop_assert_eq!(shape[1], config.vocab_size, "Vocab dimension mismatch");
    }

    /// Fuzz test: Generation should produce valid token IDs
    #[test]
    fn fuzz_generate_valid_tokens(
        prompt_len in 1usize..5,
        max_tokens in 1usize..10,
    ) {
        let config = test_config();
        let model = Model::new(config.clone()).expect("model creation");

        let prompt: Vec<usize> = (0..prompt_len).map(|i| i % 100).collect();
        let gen_config = GenerationConfig {
            max_tokens,
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            eos_token_id: None,
            seed: Some(42),
        };

        let result = model.generate(&prompt, &gen_config);
        prop_assert!(result.is_ok(), "Generate failed");

        let generated = result.expect("test");
        prop_assert!(
            generated.iter().all(|&t| t < config.vocab_size),
            "Generated invalid token IDs"
        );
        prop_assert!(
            generated.len() >= prompt_len,
            "Output shorter than prompt"
        );
    }

    /// Fuzz test: Softmax should always sum to 1
    #[test]
    fn fuzz_softmax_sums_to_one(len in 10usize..500) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.1).sin() * 10.0)
            .collect();

        let v = Vector::from_slice(&input);
        let output = v.softmax().expect("test");

        let sum: f32 = output.as_slice().iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 0.001,
            "Softmax sum {} is not close to 1.0", sum
        );

        prop_assert!(
            output.as_slice().iter().all(|&x| x >= 0.0),
            "Softmax contains negative values"
        );
    }

    /// Fuzz test: Softmax should be stable for extreme values
    #[test]
    fn fuzz_softmax_numerical_stability(
        offset in -500.0f32..500.0f32,
        scale in 0.01f32..50.0f32,
    ) {
        let input: Vec<f32> = (0..100)
            .map(|i| offset + (i as f32) * scale)
            .collect();

        let v = Vector::from_slice(&input);
        let output = v.softmax().expect("test");

        // Should be finite
        prop_assert!(
            output.as_slice().iter().all(|x| x.is_finite()),
            "Softmax output contains non-finite values for offset={}, scale={}",
            offset, scale
        );

        // Should sum to 1
        let sum: f32 = output.as_slice().iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax sum {} not close to 1.0 for offset={}, scale={}",
            sum, offset, scale
        );
    }

    /// Fuzz test: Matrix multiply dimensions should be preserved
    #[test]
    fn fuzz_matmul_dimensions(
        rows in 1usize..32,
        inner in 1usize..64,
        cols in 1usize..64,
    ) {
        let a: Vec<f32> = (0..rows * inner)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..inner * cols)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        let ma = Matrix::from_vec(rows, inner, a).expect("test");
        let mb = Matrix::from_vec(inner, cols, b).expect("test");

        let mc = ma.matmul(&mb).expect("test");

        prop_assert_eq!(
            mc.rows(),
            rows,
            "Output rows wrong: expected {}, got {}",
            rows, mc.rows()
        );
        prop_assert_eq!(
            mc.cols(),
            cols,
            "Output cols wrong: expected {}, got {}",
            cols, mc.cols()
        );

        prop_assert!(
            mc.as_slice().iter().all(|x| x.is_finite()),
            "Matmul output contains non-finite values"
        );
    }

    /// Fuzz test: GELU should be bounded
    #[test]
    fn fuzz_gelu_bounded(len in 10usize..500) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.05) - 12.5) // Range [-12.5, 12.5]
            .collect();

        let v = Vector::from_slice(&input);
        let output = v.gelu().expect("test");

        prop_assert!(
            output.as_slice().iter().all(|x| x.is_finite()),
            "GELU output contains non-finite values"
        );

        // GELU is approximately linear for large positive values
        // and approximately 0 for large negative values
        for (&inp, &out) in input.iter().zip(output.as_slice().iter()) {
            if inp > 3.0 {
                prop_assert!(out > 0.0, "GELU({}) = {} should be positive", inp, out);
            }
            if inp < -5.0 {
                prop_assert!(out.abs() < 0.1, "GELU({}) = {} should be near 0", inp, out);
            }
        }
    }

    /// Fuzz test: Dot product should be commutative
    #[test]
    fn fuzz_dot_commutative(len in 4usize..256) {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let dot_ab = va.dot(&vb).expect("test");
        let dot_ba = vb.dot(&va).expect("test");

        prop_assert!(
            (dot_ab - dot_ba).abs() < 1e-4,
            "Dot product not commutative: {} vs {}", dot_ab, dot_ba
        );
    }

    /// Fuzz test: Tensor creation and data consistency
    #[test]
    fn fuzz_tensor_consistency(
        dim1 in 1usize..20,
        dim2 in 1usize..20,
    ) {
        let data: Vec<f32> = (0..dim1 * dim2)
            .map(|i| i as f32 * 0.1)
            .collect();

        let tensor = Tensor::from_vec(vec![dim1, dim2], data.clone()).expect("test");

        prop_assert_eq!(tensor.shape(), &[dim1, dim2]);
        prop_assert_eq!(tensor.size(), dim1 * dim2);
        prop_assert_eq!(tensor.data(), &data[..]);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_token_forward() {
    let config = test_config();
    let model = Model::new(config.clone()).expect("model");

    let result = model.forward(&[0]);
    assert!(result.is_ok());

    let logits = result.expect("test");
    assert_eq!(logits.shape()[0], 1);
    assert_eq!(logits.shape()[1], config.vocab_size);
}

#[test]
fn test_empty_softmax() {
    let empty: Vec<f32> = vec![];
    let v = Vector::from_slice(&empty);
    let result = v.softmax();
    // Empty vector softmax should either error or return empty
    if let Ok(out) = result {
        assert!(out.as_slice().is_empty());
    }
}

#[test]
fn test_single_element_softmax() {
    let single = vec![5.0_f32];
    let v = Vector::from_slice(&single);
    let result = v.softmax().expect("test");
    assert_eq!(result.as_slice().len(), 1);
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_uniform_softmax() {
    let uniform = vec![1.0_f32; 10];
    let v = Vector::from_slice(&uniform);
    let result = v.softmax().expect("test");
    for x in result.as_slice() {
        assert!((*x - 0.1).abs() < 0.01);
    }
}

#[test]
fn test_extreme_softmax() {
    // Very large values
    let large = vec![500.0_f32, 501.0, 499.0];
    let v = Vector::from_slice(&large);
    let result = v.softmax().expect("test");
    assert!(result.as_slice().iter().all(|x| x.is_finite()));
    assert!((result.as_slice().iter().sum::<f32>() - 1.0).abs() < 0.01);

    // Very negative values
    let negative = vec![-500.0_f32, -501.0, -499.0];
    let v = Vector::from_slice(&negative);
    let result = v.softmax().expect("test");
    assert!(result.as_slice().iter().all(|x| x.is_finite()));
    assert!((result.as_slice().iter().sum::<f32>() - 1.0).abs() < 0.01);
}

#[test]
fn test_greedy_sampling_deterministic() {
    let config = test_config();
    let model = Model::new(config).expect("model");

    let prompt: Vec<usize> = vec![1, 2, 3];
    let gen_config = GenerationConfig {
        max_tokens: 10,
        strategy: SamplingStrategy::Greedy,
        temperature: 1.0,
        eos_token_id: None,
        seed: None,
    };

    let result1 = model.generate(&prompt, &gen_config).expect("gen1");
    let result2 = model.generate(&prompt, &gen_config).expect("gen2");

    // Greedy should be deterministic
    assert_eq!(result1, result2);
}

#[test]
fn test_different_sampling_strategies() {
    let config = test_config();
    let model = Model::new(config).expect("model");

    let prompt: Vec<usize> = vec![1, 2, 3];

    let strategies = [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK { k: 5 },
        SamplingStrategy::TopP { p: 0.9 },
    ];

    for strategy in strategies {
        let gen_config = GenerationConfig {
            max_tokens: 5,
            strategy,
            temperature: 1.0,
            eos_token_id: None,
            seed: Some(42),
        };

        let result = model.generate(&prompt, &gen_config);
        assert!(result.is_ok(), "Strategy {:?} failed", strategy);

        let generated = result.expect("test");
        assert!(generated.len() >= prompt.len());
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn stress_test_repeated_forward() {
    let config = test_config();
    let model = Model::new(config).expect("model");

    let tokens: Vec<usize> = vec![1, 2, 3, 4, 5];

    // Run forward 50 times
    for i in 0..50 {
        let result = model.forward(&tokens);
        assert!(result.is_ok(), "Forward failed at iteration {}", i);

        let logits = result.expect("test");
        assert!(
            logits.data().iter().all(|x| x.is_finite()),
            "Non-finite output at iteration {}",
            i
        );
    }
}

#[test]
fn stress_test_vocab_projection() {
    // Test large vocab projection (1 x 256 x 32000 pattern)
    let hidden_dim = 256;
    let vocab_size = 32000;

    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let embedding: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| (i as f32 * 0.0001).sin())
        .collect();

    let h = Matrix::from_vec(1, hidden_dim, hidden).expect("test");
    let e = Matrix::from_vec(hidden_dim, vocab_size, embedding).expect("test");

    let logits = h.matmul(&e).expect("test");
    assert_eq!(logits.rows(), 1);
    assert_eq!(logits.cols(), vocab_size);
    assert!(logits.as_slice().iter().all(|x| x.is_finite()));
}

#[test]
fn stress_test_batch_matmul() {
    // Test batch matrix multiply
    let batch_size = 32;
    let hidden_dim = 128;

    let a: Vec<f32> = (0..batch_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let b: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();

    let ma = Matrix::from_vec(batch_size, hidden_dim, a).expect("test");
    let mb = Matrix::from_vec(hidden_dim, hidden_dim, b).expect("test");

    let mc = ma.matmul(&mb).expect("test");
    assert_eq!(mc.rows(), batch_size);
    assert_eq!(mc.cols(), hidden_dim);
    assert!(mc.as_slice().iter().all(|x| x.is_finite()));
}
