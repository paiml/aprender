//! Part 05: Batch Generation Tests (PMAT-803)
//!
//! Tests for `gpu/scheduler/batch.rs` functions:
//! - `argmax` - Vectorized argmax for both small and large vocabularies
//! - `optimized_lm_head_argmax_transposed` - Parallel argmax with transposed weights
//! - `simplified_attention` - Fallback attention implementation

use crate::gpu::scheduler::batch::{
    argmax, optimized_lm_head_argmax_transposed, simplified_attention,
};
use crate::gpu::scheduler::GpuModelConfig;

// ============================================================================
// argmax Tests
// ============================================================================

#[test]
fn test_argmax_empty_input() {
    let logits: Vec<f32> = vec![];
    let result = argmax(&logits);
    assert_eq!(result, 0, "Empty input should return 0");
}

#[test]
fn test_argmax_single_element() {
    let logits = vec![42.0];
    let result = argmax(&logits);
    assert_eq!(result, 0, "Single element should return index 0");
}

#[test]
fn test_argmax_small_vocab_first_max() {
    let logits = vec![10.0, 5.0, 3.0, 1.0];
    let result = argmax(&logits);
    assert_eq!(result, 0, "First element is max");
}

#[test]
fn test_argmax_small_vocab_last_max() {
    let logits = vec![1.0, 2.0, 3.0, 100.0];
    let result = argmax(&logits);
    assert_eq!(result, 3, "Last element is max");
}

#[test]
fn test_argmax_small_vocab_middle_max() {
    let logits = vec![1.0, 2.0, 100.0, 3.0, 4.0];
    let result = argmax(&logits);
    assert_eq!(result, 2, "Middle element is max");
}

#[test]
fn test_argmax_small_vocab_negative_values() {
    let logits = vec![-10.0, -5.0, -1.0, -100.0];
    let result = argmax(&logits);
    assert_eq!(result, 2, "Largest negative is at index 2");
}

#[test]
fn test_argmax_small_vocab_all_equal() {
    let logits = vec![5.0, 5.0, 5.0, 5.0];
    let result = argmax(&logits);
    // When all values are equal, any index is valid - just verify it points to max value
    assert!(result < logits.len(), "Index should be in bounds");
    assert!(
        (logits[result] - 5.0).abs() < 1e-6,
        "Should point to a max value"
    );
}

#[test]
fn test_argmax_small_vocab_infinity() {
    let logits = vec![1.0, f32::INFINITY, 3.0, 4.0];
    let result = argmax(&logits);
    assert_eq!(result, 1, "Infinity should be max");
}

#[test]
fn test_argmax_small_vocab_neg_infinity() {
    let logits = vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0];
    let result = argmax(&logits);
    assert_eq!(result, 3, "Neg infinity should not be max");
}

#[test]
fn test_argmax_small_vocab_nan_handling() {
    // NaN comparisons return Ordering::Equal by default
    let logits = vec![1.0, f32::NAN, 3.0, 4.0];
    let result = argmax(&logits);
    // The result depends on implementation - just verify it doesn't panic
    assert!(result < logits.len());
}

#[test]
fn test_argmax_exactly_1024_elements() {
    // Boundary test: <= 1024 uses simple iterator path
    let mut logits = vec![0.0f32; 1024];
    logits[500] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 500);
}

#[test]
fn test_argmax_large_vocab_1025_elements() {
    // Boundary test: > 1024 uses chunked parallel path
    let mut logits = vec![0.0f32; 1025];
    logits[1000] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 1000);
}

#[test]
fn test_argmax_large_vocab_32k() {
    // Typical LLM vocab size
    let mut logits = vec![0.0f32; 32000];
    logits[12345] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 12345);
}

#[test]
fn test_argmax_large_vocab_65k() {
    // Large vocab size
    let mut logits = vec![0.0f32; 65536];
    logits[65000] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 65000);
}

#[test]
fn test_argmax_large_vocab_max_at_chunk_boundary() {
    // Test max at exactly 4096 (chunk boundary)
    let mut logits = vec![0.0f32; 8192];
    logits[4096] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 4096);
}

#[test]
fn test_argmax_large_vocab_max_in_last_chunk() {
    // Test max in last (partial) chunk
    let mut logits = vec![0.0f32; 5000];
    logits[4999] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 4999);
}

#[test]
fn test_argmax_large_vocab_max_in_first_chunk() {
    // Test max in first chunk
    let mut logits = vec![0.0f32; 10000];
    logits[100] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 100);
}

#[test]
fn test_argmax_large_vocab_multiple_local_maxes() {
    // Each chunk has a local max, but one is global
    let mut logits = vec![0.0f32; 16384];
    logits[1000] = 50.0; // First chunk max
    logits[5000] = 75.0; // Second chunk max
    logits[9000] = 100.0; // Third chunk max (global)
    logits[13000] = 60.0; // Fourth chunk max
    let result = argmax(&logits);
    assert_eq!(result, 9000);
}

// ============================================================================
// optimized_lm_head_argmax_transposed Tests
// ============================================================================

#[test]
fn test_optimized_lm_head_argmax_simple() {
    // hidden = [1, 0, 0, 0], weight_t[0] = [1, 0, 0, 0] -> dot = 1
    // weight_t[1] = [0, 1, 0, 0] -> dot = 0
    let hidden = vec![1.0, 0.0, 0.0, 0.0];
    let weight_t = vec![
        1.0, 0.0, 0.0, 0.0, // Row 0: dot product = 1
        0.0, 1.0, 0.0, 0.0, // Row 1: dot product = 0
        0.0, 0.0, 1.0, 0.0, // Row 2: dot product = 0
        0.0, 0.0, 0.0, 1.0, // Row 3: dot product = 0
    ];
    let bias = vec![0.0, 0.0, 0.0, 0.0];

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 4, 4);
    assert_eq!(result, 0);
}

#[test]
fn test_optimized_lm_head_argmax_with_bias() {
    // hidden = [1, 0, 0, 0]
    // All weight rows give dot = 1, but bias makes index 2 largest
    let hidden = vec![1.0, 0.0, 0.0, 0.0];
    let weight_t = vec![
        1.0, 0.0, 0.0, 0.0, // Row 0: dot = 1
        1.0, 0.0, 0.0, 0.0, // Row 1: dot = 1
        1.0, 0.0, 0.0, 0.0, // Row 2: dot = 1
        1.0, 0.0, 0.0, 0.0, // Row 3: dot = 1
    ];
    let bias = vec![0.0, 0.0, 10.0, 0.0]; // bias[2] = 10

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 4, 4);
    assert_eq!(result, 2);
}

#[test]
fn test_optimized_lm_head_argmax_larger_vocab() {
    // Test with vocab size > chunk size to trigger parallel path
    let hidden_dim = 64;
    let vocab_size = 5000;

    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let mut weight_t = vec![0.0f32; vocab_size * hidden_dim];
    let bias = vec![0.0f32; vocab_size];

    // Make row 2500 produce the highest logit
    for i in 0..hidden_dim {
        weight_t[2500 * hidden_dim + i] = hidden[i]; // Dot product = sum of squares
    }

    let result =
        optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
    assert_eq!(result, 2500);
}

#[test]
fn test_optimized_lm_head_argmax_single_vocab() {
    let hidden = vec![1.0, 2.0, 3.0, 4.0];
    let weight_t = vec![1.0, 1.0, 1.0, 1.0]; // Vocab size 1
    let bias = vec![0.0];

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 4, 1);
    assert_eq!(result, 0);
}

#[test]
fn test_optimized_lm_head_argmax_negative_logits() {
    let hidden = vec![1.0, 1.0];
    let weight_t = vec![
        -1.0, -1.0, // Row 0: dot = -2
        -0.5, -0.5, // Row 1: dot = -1 (largest)
    ];
    let bias = vec![0.0, 0.0];

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 2);
    assert_eq!(result, 1);
}

// ============================================================================
// simplified_attention Tests
// ============================================================================

#[test]
fn test_simplified_attention_single_token() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let seq_len = 1;
    let hidden_dim = config.hidden_dim;

    // QKV for single position
    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![2.0f32; hidden_dim];

    let mut qkv = Vec::with_capacity(seq_len * 3 * hidden_dim);
    qkv.extend(&q);
    qkv.extend(&k);
    qkv.extend(&v);

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), hidden_dim);

    // For single token: softmax of single element = 1.0, so output = v
    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - v[i]).abs() < 1e-5,
            "Output should equal V for single token"
        );
    }
}

#[test]
fn test_simplified_attention_two_tokens() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 4, // Small for easy verification
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 16,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let seq_len = 2;
    let hidden_dim = config.hidden_dim;

    // Create QKV data for 2 positions
    // Position 0: Q0, K0, V0
    // Position 1: Q1, K1, V1
    let q0 = vec![1.0f32; hidden_dim];
    let q1 = vec![1.0f32; hidden_dim];
    let k0 = vec![1.0f32; hidden_dim];
    let k1 = vec![1.0f32; hidden_dim];
    let v0 = vec![1.0f32; hidden_dim];
    let v1 = vec![2.0f32; hidden_dim];

    let mut qkv = Vec::with_capacity(seq_len * 3 * hidden_dim);
    // Q: all positions
    qkv.extend(&q0);
    qkv.extend(&q1);
    // K: all positions
    qkv.extend(&k0);
    qkv.extend(&k1);
    // V: all positions
    qkv.extend(&v0);
    qkv.extend(&v1);

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), seq_len * hidden_dim);

    // Position 0: can only attend to itself (causal), output = v0
    for i in 0..hidden_dim {
        assert!(
            (output[i] - v0[i]).abs() < 1e-5,
            "Position 0 should only attend to itself"
        );
    }

    // Position 1: attends to both positions with some weighting
    // Since all Qs and Ks are identical, scores are equal, softmax gives 0.5, 0.5
    // Expected: 0.5 * v0 + 0.5 * v1 = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    for i in 0..hidden_dim {
        let expected = 0.5 * v0[i] + 0.5 * v1[i];
        assert!(
            (output[hidden_dim + i] - expected).abs() < 1e-4,
            "Position 1 should be weighted average: got {} expected {}",
            output[hidden_dim + i],
            expected
        );
    }
}

#[test]
fn test_simplified_attention_causal_mask() {
    // Verify causal masking works correctly
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 8,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 32,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let seq_len = 3;
    let hidden_dim = config.hidden_dim;

    // Create distinct V vectors so we can verify causal masking
    let v0 = vec![1.0f32; hidden_dim];
    let v1 = vec![2.0f32; hidden_dim];
    let v2 = vec![3.0f32; hidden_dim];

    // All Q/K are identical for equal attention weights
    let qk = vec![1.0f32; hidden_dim];

    let mut qkv = Vec::with_capacity(seq_len * 3 * hidden_dim);
    // Q for all positions
    for _ in 0..seq_len {
        qkv.extend(&qk);
    }
    // K for all positions
    for _ in 0..seq_len {
        qkv.extend(&qk);
    }
    // V for all positions
    qkv.extend(&v0);
    qkv.extend(&v1);
    qkv.extend(&v2);

    let result = simplified_attention(&config, &qkv, seq_len);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), seq_len * hidden_dim);

    // Position 0: only attends to itself -> v0
    for i in 0..hidden_dim {
        assert!((output[i] - 1.0).abs() < 1e-4);
    }

    // Position 1: attends to pos 0 and 1 -> avg(v0, v1) = 1.5
    for i in 0..hidden_dim {
        assert!((output[hidden_dim + i] - 1.5).abs() < 1e-4);
    }

    // Position 2: attends to pos 0, 1, 2 -> avg(v0, v1, v2) = 2.0
    for i in 0..hidden_dim {
        assert!((output[2 * hidden_dim + i] - 2.0).abs() < 1e-4);
    }
}

// ============================================================================
// Edge Case and Error Path Tests
// ============================================================================

#[test]
fn test_argmax_with_subnormal_numbers() {
    let logits = vec![f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE * 2.0];
    let result = argmax(&logits);
    assert_eq!(result, 2);
}

#[test]
fn test_argmax_very_close_values() {
    let logits = vec![
        1.0000001, 1.0000002, // Slightly larger
        1.0000000,
    ];
    let result = argmax(&logits);
    assert_eq!(result, 1);
}

#[test]
fn test_optimized_lm_head_argmax_zero_hidden() {
    let hidden = vec![0.0, 0.0, 0.0, 0.0];
    let weight_t = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    // Bias determines the result
    let bias = vec![5.0, 10.0];

    let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 4, 2);
    assert_eq!(result, 1); // Higher bias
}

// ============================================================================
// Performance Boundary Tests
// ============================================================================

#[test]
fn test_argmax_chunk_boundary_exact() {
    // Exactly 4096 elements (one full chunk)
    let mut logits = vec![0.0f32; 4096];
    logits[4095] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 4095);
}

#[test]
fn test_argmax_chunk_boundary_plus_one() {
    // 4097 elements (one full chunk + 1)
    let mut logits = vec![0.0f32; 4097];
    logits[4096] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 4096);
}

#[test]
fn test_argmax_two_full_chunks() {
    // 8192 elements (two full chunks)
    let mut logits = vec![0.0f32; 8192];
    logits[8191] = 100.0;
    let result = argmax(&logits);
    assert_eq!(result, 8191);
}
