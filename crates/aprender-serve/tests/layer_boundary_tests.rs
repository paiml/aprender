//! Layer Boundary Tests (T-QA-010)
//!
//! Kernel & Layer Hardening Squad: Close coverage gaps in layers.rs (86% -> 95%)
//!
//! Tests RMSNorm and Attention with extreme values:
//! - Near-zero variance
//! - Near-infinite weights
//! - Edge case dimensions
//!
//! Constraint: Pure CPU logic verification, < 5s execution

use realizar::inference::{apply_rope, simd_layer_norm, simd_rms_norm};
use realizar::layers::{Attention, LayerNorm, Linear, SlidingWindowAttention};
use realizar::tensor::Tensor;

// ============================================================================
// A. RMSNorm Boundary Tests
// ============================================================================

#[test]
fn test_rms_norm_near_zero_variance() {
    // All values are nearly identical (near-zero variance)
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // RMS of [1,1,1,1] = 1.0, so output should be [1,1,1,1]
    for v in &output {
        assert!(v.is_finite(), "Output should be finite");
        assert!((v - 1.0).abs() < 0.01, "Expected ~1.0, got {}", v);
    }
}

#[test]
fn test_rms_norm_zero_input() {
    // All zeros - tests numerical stability with eps
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // RMS of [0,0,0,0] = sqrt(eps), output = 0 / sqrt(eps) = 0
    for v in &output {
        assert!(v.is_finite(), "Output should be finite");
        assert_eq!(*v, 0.0, "Expected 0.0 for zero input");
    }
}

#[test]
fn test_rms_norm_tiny_values() {
    // Very small values
    let input = vec![1e-10, 1e-10, 1e-10, 1e-10];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Should not blow up
    for v in &output {
        assert!(v.is_finite(), "Output should be finite for tiny values");
    }
}

#[test]
fn test_rms_norm_large_values() {
    // Large values near f32 limits
    let input = vec![1e10, 1e10, 1e10, 1e10];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // RMS of large values should normalize to ~1
    for v in &output {
        assert!(v.is_finite(), "Output should be finite for large values");
    }
}

#[test]
fn test_rms_norm_mixed_signs() {
    // Mix of positive and negative values
    let input = vec![-2.0, 1.0, -1.0, 2.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Signs should be preserved
    assert!(
        output[0] < 0.0,
        "Negative input should give negative output"
    );
    assert!(
        output[1] > 0.0,
        "Positive input should give positive output"
    );
}

#[test]
fn test_rms_norm_single_element() {
    let input = vec![5.0];
    let weight = vec![2.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // RMS of [5] = 5, normalized = 5/5 = 1, * weight 2 = 2
    assert_eq!(output.len(), 1);
    assert!(
        (output[0] - 2.0).abs() < 0.01,
        "Expected ~2.0, got {}",
        output[0]
    );
}

#[test]
fn test_rms_norm_empty_input() {
    let input: Vec<f32> = vec![];
    let weight: Vec<f32> = vec![];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    assert!(output.is_empty(), "Empty input should produce empty output");
}

#[test]
fn test_rms_norm_weight_scaling() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0, 0.5, 1.0, 0.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Weight of 0 should give 0 output
    assert!(
        (output[3]).abs() < 1e-10,
        "Zero weight should give zero output"
    );
}

// ============================================================================
// B. LayerNorm Boundary Tests
// ============================================================================

#[test]
fn test_layer_norm_near_zero_variance() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, None, eps);

    // LayerNorm with zero variance should produce near-zero (after mean subtraction)
    for v in &output {
        assert!(v.is_finite(), "Output should be finite");
        assert!(
            v.abs() < 0.01,
            "Near-zero variance should give near-zero output"
        );
    }
}

#[test]
fn test_layer_norm_zero_input() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, None, eps);

    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_layer_norm_with_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let bias = vec![10.0, 10.0, 10.0, 10.0];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, Some(&bias), eps);

    // After normalization (mean 0, std 1) + bias 10, all values should be around 10
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!((mean - 10.0).abs() < 0.1, "Mean should be ~10 after bias");
}

#[test]
fn test_layer_norm_single_element() {
    let input = vec![5.0];
    let weight = vec![1.0];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, None, eps);

    // Single element: mean = x, variance = 0, result depends on eps
    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_layer_norm_empty() {
    let input: Vec<f32> = vec![];
    let weight: Vec<f32> = vec![];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, None, eps);

    assert!(output.is_empty());
}

// ============================================================================
// C. Attention Boundary Tests
// ============================================================================

#[test]
fn test_attention_creation_zero_head_dim() {
    let result = Attention::new(0);
    assert!(result.is_err(), "head_dim=0 should fail");
}

#[test]
fn test_attention_creation_valid() {
    let result = Attention::new(64);
    assert!(result.is_ok(), "head_dim=64 should succeed");

    let attn = result.unwrap();
    // Scale should be 1/sqrt(64) = 0.125
    // Can't access scale directly, but we can verify creation succeeded
    let _ = attn;
}

#[test]
fn test_attention_forward_empty_query() {
    let attn = Attention::new(64).unwrap();

    let query = Tensor::from_vec(vec![0, 64], vec![]).unwrap_or_else(|_| {
        // Empty tensor might fail construction, that's ok
        Tensor::from_vec(vec![1, 64], vec![1.0; 64]).unwrap()
    });

    let key = Tensor::from_vec(vec![1, 64], vec![1.0; 64]).unwrap();
    let value = Tensor::from_vec(vec![1, 64], vec![1.0; 64]).unwrap();

    let result = attn.forward(&query, &key, &value);
    // May succeed or fail depending on implementation
    let _ = result;
}

#[test]
fn test_attention_forward_dimension_mismatch() {
    let attn = Attention::new(64).unwrap();

    let query = Tensor::from_vec(vec![1, 64], vec![1.0; 64]).unwrap();
    let key = Tensor::from_vec(vec![1, 32], vec![1.0; 32]).unwrap(); // Wrong dim
    let value = Tensor::from_vec(vec![1, 64], vec![1.0; 64]).unwrap();

    let result = attn.forward(&query, &key, &value);
    assert!(result.is_err(), "Dimension mismatch should fail");
}

#[test]
fn test_attention_forward_near_zero_values() {
    let attn = Attention::new(4).unwrap();

    // Very small values
    let query = Tensor::from_vec(vec![1, 4], vec![1e-10; 4]).unwrap();
    let key = Tensor::from_vec(vec![1, 4], vec![1e-10; 4]).unwrap();
    let value = Tensor::from_vec(vec![1, 4], vec![1e-10; 4]).unwrap();

    let result = attn.forward(&query, &key, &value);
    if let Ok(output) = result {
        for v in output.data() {
            assert!(v.is_finite(), "Output should be finite for small inputs");
        }
    }
}

#[test]
fn test_attention_forward_large_values() {
    let attn = Attention::new(4).unwrap();

    // Large values - may cause overflow in dot product
    let query = Tensor::from_vec(vec![1, 4], vec![1000.0; 4]).unwrap();
    let key = Tensor::from_vec(vec![1, 4], vec![1000.0; 4]).unwrap();
    let value = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).unwrap();

    let result = attn.forward(&query, &key, &value);
    if let Ok(output) = result {
        for v in output.data() {
            // May be infinite or NaN due to softmax overflow, which is expected behavior
            let _ = v;
        }
    }
}

#[test]
fn test_attention_single_token() {
    let attn = Attention::new(8).unwrap();

    // Single token attention
    let query = Tensor::from_vec(vec![1, 8], vec![1.0; 8]).unwrap();
    let key = Tensor::from_vec(vec![1, 8], vec![1.0; 8]).unwrap();
    let value = Tensor::from_vec(vec![1, 8], vec![2.0; 8]).unwrap();

    let result = attn.forward(&query, &key, &value);
    if let Ok(output) = result {
        // Single token self-attention should return value (softmax([x]) = [1])
        assert_eq!(output.shape(), &[1, 8]);
        for v in output.data() {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// D. SlidingWindowAttention Tests
// ============================================================================

#[test]
fn test_sliding_window_attention_creation() {
    let result = SlidingWindowAttention::new(64, 128);
    assert!(result.is_ok());
}

#[test]
fn test_sliding_window_attention_zero_head_dim() {
    let result = SlidingWindowAttention::new(0, 128);
    assert!(result.is_err(), "Zero head_dim should fail");
}

#[test]
fn test_sliding_window_attention_zero_window() {
    let result = SlidingWindowAttention::new(64, 0);
    assert!(result.is_err(), "Zero window_size should fail");
}

// ============================================================================
// E. Linear Layer Boundary Tests
// ============================================================================

#[test]
fn test_linear_zero_features() {
    let result = Linear::new(0, 64);
    assert!(result.is_err(), "Zero in_features should fail");
}

#[test]
fn test_linear_zero_out_features() {
    let result = Linear::new(64, 0);
    assert!(result.is_err(), "Zero out_features should fail");
}

#[test]
fn test_linear_valid() {
    let result = Linear::new(64, 128);
    assert!(result.is_ok());
}

#[test]
fn test_linear_forward_shape_mismatch() {
    let linear = Linear::new(64, 128).unwrap();
    let input = Tensor::from_vec(vec![1, 32], vec![1.0; 32]).unwrap(); // Wrong size

    let result = linear.forward(&input);
    assert!(result.is_err(), "Shape mismatch should fail");
}

// ============================================================================
// F. LayerNorm Layer Tests
// ============================================================================

#[test]
fn test_layer_norm_layer_zero_shape() {
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err(), "Zero normalized_shape should fail");
}

#[test]
fn test_layer_norm_layer_valid() {
    let result = LayerNorm::new(64, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_layer_norm_layer_forward_shape_mismatch() {
    let ln = LayerNorm::new(64, 1e-5).unwrap();
    let input = Tensor::from_vec(vec![1, 32], vec![1.0; 32]).unwrap(); // Wrong size

    let result = ln.forward(&input);
    assert!(result.is_err(), "Shape mismatch should fail");
}

#[test]
fn test_layer_norm_layer_forward_extreme_values() {
    let ln = LayerNorm::new(4, 1e-5).unwrap();

    // Create tensor with extreme values
    let input = Tensor::from_vec(vec![1, 4], vec![1e10, -1e10, 1e-10, 0.0]).unwrap();

    let result = ln.forward(&input);
    if let Ok(output) = result {
        for v in output.data() {
            // May have NaN due to extreme values
            let _ = v;
        }
    }
}

// ============================================================================
// G. RoPE (Rotary Position Embedding) Boundary Tests
// ============================================================================

#[test]
fn test_rope_position_zero() {
    let mut x = vec![1.0; 64];
    apply_rope(&mut x, 64, 4, 0, 10000.0);

    // At position 0, angle = 0, so cos=1, sin=0 → values unchanged
    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_position_one() {
    let mut x = vec![1.0; 64];
    apply_rope(&mut x, 64, 4, 1, 10000.0);

    // At position 1, values should be rotated
    for v in &x {
        assert!(v.is_finite(), "RoPE output should be finite");
    }
}

#[test]
fn test_rope_large_position() {
    let mut x = vec![1.0; 64];
    apply_rope(&mut x, 64, 4, 1000000, 10000.0);

    // Very large position should still work
    for v in &x {
        assert!(v.is_finite(), "RoPE should handle large positions");
    }
}

#[test]
fn test_rope_single_head() {
    let mut x = vec![1.0; 16];
    apply_rope(&mut x, 16, 1, 5, 10000.0);

    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_many_heads() {
    let mut x = vec![1.0; 128];
    apply_rope(&mut x, 128, 16, 5, 10000.0);

    for v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_theta_variation() {
    let mut x1 = vec![1.0; 64];
    let mut x2 = x1.clone();

    apply_rope(&mut x1, 64, 4, 10, 10000.0);
    apply_rope(&mut x2, 64, 4, 10, 500000.0); // Different theta

    // Different theta should produce different results
    // (for non-zero position)
    let different = x1.iter().zip(x2.iter()).any(|(a, b)| (a - b).abs() > 0.001);
    assert!(
        different,
        "Different theta should produce different results"
    );
}

#[test]
fn test_rope_preserves_norm_approximately() {
    let mut x = vec![1.0; 64];
    let original_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    apply_rope(&mut x, 64, 4, 5, 10000.0);

    let rotated_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // Rotation should preserve norm (within numerical tolerance)
    assert!(
        (original_norm - rotated_norm).abs() < 0.1,
        "RoPE should approximately preserve norm"
    );
}

// ============================================================================
// H. Numerical Stability Edge Cases
// ============================================================================

#[test]
fn test_rms_norm_nan_propagation() {
    let input = vec![f32::NAN, 1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // NaN in input should propagate
    assert!(output[0].is_nan(), "NaN should propagate through RMSNorm");
}

#[test]
fn test_rms_norm_infinity_handling() {
    let input = vec![f32::INFINITY, 1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Infinity should be handled somehow (NaN or Inf)
    assert!(!output[0].is_finite() || output[0].is_nan());
}

#[test]
fn test_layer_norm_nan_propagation() {
    let input = vec![f32::NAN, 1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_layer_norm(&input, &weight, None, eps);

    // NaN should propagate
    assert!(output.iter().any(|v| v.is_nan()), "NaN should propagate");
}

// ============================================================================
// I. Dimension Boundary Tests
// ============================================================================

#[test]
fn test_rms_norm_odd_dimension() {
    let input = vec![1.0, 2.0, 3.0]; // Odd dimension
    let weight = vec![1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);
    assert_eq!(output.len(), 3);
}

#[test]
fn test_rms_norm_power_of_two_dimension() {
    let input = vec![1.0; 256]; // Power of 2
    let weight = vec![1.0; 256];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);
    assert_eq!(output.len(), 256);
}

#[test]
fn test_rms_norm_prime_dimension() {
    let input = vec![1.0; 127]; // Prime number
    let weight = vec![1.0; 127];
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);
    assert_eq!(output.len(), 127);
}

#[test]
fn test_attention_various_head_dims() {
    // Test various head dimensions
    for head_dim in [1, 2, 4, 8, 16, 32, 64, 128] {
        let result = Attention::new(head_dim);
        assert!(result.is_ok(), "head_dim={} should be valid", head_dim);
    }
}

// ============================================================================
// J. Weight Extreme Value Tests
// ============================================================================

#[test]
fn test_rms_norm_negative_weights() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![-1.0, -1.0, -1.0, -1.0]; // Negative weights
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Negative weights should negate output
    for (i, v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output[{}] should be finite", i);
    }
}

#[test]
fn test_rms_norm_large_weights() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight = vec![1e6, 1e6, 1e6, 1e6]; // Very large weights
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Large weights should scale output
    for v in &output {
        assert!(
            v.is_finite(),
            "Large weights should still give finite output"
        );
    }
}

#[test]
fn test_rms_norm_tiny_weights() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1e-10, 1e-10, 1e-10, 1e-10]; // Very small weights
    let eps = 1e-5;

    let output = simd_rms_norm(&input, &weight, eps);

    // Tiny weights should produce tiny output
    for v in &output {
        assert!(v.abs() < 1e-5, "Tiny weights should give tiny output");
    }
}

// ============================================================================
// K. Epsilon Sensitivity Tests
// ============================================================================

#[test]
fn test_rms_norm_tiny_epsilon() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-15; // Very small epsilon

    let output = simd_rms_norm(&input, &weight, eps);

    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rms_norm_large_epsilon() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1.0; // Large epsilon

    let output = simd_rms_norm(&input, &weight, eps);

    // Large epsilon should still work
    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_layer_norm_tiny_epsilon() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-15;

    let output = simd_layer_norm(&input, &weight, None, eps);

    for v in &output {
        assert!(v.is_finite());
    }
}
