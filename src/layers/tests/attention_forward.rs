//! Additional coverage tests for attention.rs
//!
//! Targets uncovered paths in:
#![allow(clippy::many_single_char_names)]
//! - Attention: empty tensor errors, head_dim validation
//! - Flash Attention: block_size=0, shape mismatches
//! - Flash Attention v2: shape validation, SIMD paths
//! - Flash Attention parallel: shape errors, single position
//! - SlidingWindowAttention: bidirectional mode, edge cases
//! - FusedQKVAttention: weight accessors, input validation
//! - MultiHeadAttention: GQA configurations, error paths

use crate::layers::*;

// =========================================================================
// Attention: Empty and Invalid Tensor Tests
// =========================================================================

#[test]
fn test_attention_forward_empty_query_shape_error() {
    let attn = Attention::new(4).expect("test");

    // Create tensors with valid last dimension but test shape validation
    // Empty shape is caught at Tensor creation, so test with wrong head_dim
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on Q head_dim mismatch");
}

#[test]
fn test_attention_forward_empty_key_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on K head_dim mismatch");
}

#[test]
fn test_attention_forward_empty_value_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err(), "Should error on V head_dim mismatch");
}

#[test]
fn test_attention_forward_single_dim_tensors() {
    let attn = Attention::new(4).expect("test");

    // 1D tensors: [4] treated as seq_len=1, head_dim=4
    let q = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Single position self-attention should return the value
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_attention_scale_computation() {
    // Test scale is computed correctly: 1 / sqrt(head_dim)
    for head_dim in [1, 4, 16, 64, 128] {
        let attn = Attention::new(head_dim).expect("test");
        let expected_scale = 1.0 / (head_dim as f32).sqrt();
        assert!(
            (attn.scale() - expected_scale).abs() < 1e-6,
            "Scale for head_dim={} should be {}",
            head_dim,
            expected_scale
        );
    }
}

// =========================================================================
// Flash Attention: Shape Validation and Edge Cases
// =========================================================================

#[test]
fn test_flash_forward_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_single_position() {
    let attn = Attention::new(4).expect("test");

    // Single position: seq_len=1
    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).expect("test");

    let output = attn.flash_forward(&q, &k, &v, 1).expect("test");

    // Self-attention on single position returns the value
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_flash_forward_block_size_larger_than_seq() {
    let attn = Attention::new(4).expect("test");

    // seq_len=2, block_size=10 (larger than sequence)
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.flash_forward(&q, &k, &v, 10).expect("test");
    assert_eq!(output.shape(), &[2, 4]);
}

// =========================================================================
// Flash Attention v2: Shape Validation and SIMD Testing
// =========================================================================

#[test]
fn test_flash_forward_v2_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_v2_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward_v2(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_v2 should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_v2_single_position() {
    let attn = Attention::new(8).expect("test");

    // Single position with head_dim=8 (exercises SIMD path with remainder)
    let q = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("test");

    let output = attn.flash_forward_v2(&q, &k, &v, 1).expect("test");

    assert_eq!(output.shape(), &[1, 8]);
    // Self-attention on single position returns the value
    for i in 0..8 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-5);
    }
}

#[test]
fn test_flash_forward_v2_simd_aligned_dimensions() {
    // Test with dimensions that align with SIMD (8 floats for AVX2)
    let attn = Attention::new(16).expect("test");

    let q = Tensor::from_vec(vec![4, 16], vec![0.1; 64]).expect("test");
    let k = Tensor::from_vec(vec![4, 16], vec![0.2; 64]).expect("test");
    let v = Tensor::from_vec(vec![4, 16], vec![0.3; 64]).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "SIMD aligned mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

#[test]
fn test_flash_forward_v2_simd_unaligned_dimensions() {
    // Test with dimensions that don't align with SIMD (7 floats, has remainder)
    let attn = Attention::new(7).expect("test");

    let q = Tensor::from_vec(vec![3, 7], vec![0.1; 21]).expect("test");
    let k = Tensor::from_vec(vec![3, 7], vec![0.2; 21]).expect("test");
    let v = Tensor::from_vec(vec![3, 7], vec![0.3; 21]).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 2).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "SIMD unaligned mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

// =========================================================================
// Flash Attention Parallel: Shape Validation and Edge Cases
// =========================================================================

#[test]
fn test_flash_forward_parallel_empty_q_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on Q head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_empty_k_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on K head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_empty_v_shape_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test"); // Wrong head_dim

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on V head_dim mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_kv_seq_len_mismatch() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test"); // seq_len=3
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // seq_len=2

    let result = attn.flash_forward_parallel(&q, &k, &v, 2);
    assert!(
        result.is_err(),
        "flash_forward_parallel should error on K/V seq_len mismatch"
    );
}

#[test]
fn test_flash_forward_parallel_single_position() {
    let attn = Attention::new(4).expect("test");

    // Single position
    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]).expect("test");

    let output = attn.flash_forward_parallel(&q, &k, &v, 1).expect("test");

    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

// =========================================================================
// SlidingWindowAttention: Bidirectional Mode and Edge Cases
// =========================================================================

#[test]
fn test_sliding_window_bidirectional_shape_errors() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // Q head_dim mismatch
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on Q head_dim mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_k_shape_error() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // K head_dim mismatch
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on K head_dim mismatch"
    );
}

#[test]
fn test_sliding_window_bidirectional_v_shape_error() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    // V head_dim mismatch
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");

    let result = swa.forward_with_mask(&q, &k, &v, false);
    assert!(
        result.is_err(),
        "Bidirectional should error on V head_dim mismatch"
    );
}

include!("sliding_window_02.rs");
