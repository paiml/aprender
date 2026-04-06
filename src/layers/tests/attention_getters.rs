//! Additional coverage tests for attention.rs to reach 100% coverage
//!
//! Targets specific uncovered paths:
//! - SlidingWindowAttention::forward - minimal window edge cases
//! - SlidingWindowAttention::forward_with_mask - bidirectional mode
//! - SIMD remainder paths in simd_dot_avx2/simd_dot_product
//! - 1D tensor inputs for various flash_forward methods
//! - Attention getters (head_dim, scale)
//! - Debug/Clone trait implementations

use crate::layers::*;
use crate::tensor::Tensor;

// =========================================================================
// Attention: Basic Getters and Trait Implementations
// =========================================================================

#[test]
fn test_attention_getters_and_traits() {
    let attn = Attention::new(64).expect("test");

    // Test head_dim() getter
    assert_eq!(attn.head_dim(), 64);

    // Test scale() getter
    let expected = 1.0 / (64.0f32).sqrt();
    assert!((attn.scale() - expected).abs() < 1e-7);

    // Test Debug
    let debug = format!("{:?}", attn);
    assert!(debug.contains("Attention"));

    // Test Clone
    let cloned = attn.clone();
    assert_eq!(attn.head_dim(), cloned.head_dim());
}

// =========================================================================
// SlidingWindowAttention: Window Edge Cases
// =========================================================================

#[test]
fn test_sliding_window_minimal_window() {
    // Test with minimum window size of 1
    let swa = SlidingWindowAttention::new(4, 1).expect("test");

    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![2.0, 3.0, 4.0, 5.0]).expect("test");

    let output = swa.forward(&q, &k, &v).expect("test");
    assert_eq!(output.shape(), &[1, 4]);

    // Self-attention returns the value
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-5);
    }
}

#[test]
fn test_sliding_window_bidirectional_mode() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");

    let q = Tensor::from_vec(vec![4, 4], vec![0.1; 16]).expect("test");
    let k = Tensor::from_vec(vec![4, 4], vec![0.2; 16]).expect("test");
    let v = Tensor::from_vec(vec![4, 4], (1..17).map(|x| x as f32 * 0.1).collect()).expect("test");

    // Test causal=false (bidirectional)
    let bidirectional = swa.forward_with_mask(&q, &k, &v, false).expect("test");
    assert_eq!(bidirectional.shape(), &[4, 4]);

    // Test causal=true delegates to forward()
    let forward_result = swa.forward(&q, &k, &v).expect("test");
    let causal = swa.forward_with_mask(&q, &k, &v, true).expect("test");

    for i in 0..forward_result.data().len() {
        assert!((forward_result.data()[i] - causal.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_sliding_window_getters_and_traits() {
    let swa = SlidingWindowAttention::new(64, 4096).expect("test");

    assert_eq!(swa.head_dim(), 64);
    assert_eq!(swa.window_size(), 4096);
    assert!((swa.scale() - 1.0 / 8.0).abs() < 1e-7); // 1/sqrt(64)

    // effective_context: at position p, we can see min(p+1, window_size) positions
    assert_eq!(swa.effective_context(0, 10), 1); // position 0: can see [0] = 1
    assert_eq!(swa.effective_context(5, 10), 6); // position 5: can see [0,1,2,3,4,5] = 6

    // Debug/Clone
    let debug = format!("{:?}", swa);
    assert!(debug.contains("SlidingWindowAttention"));

    let cloned = swa.clone();
    assert_eq!(swa.window_size(), cloned.window_size());
}

// =========================================================================
// Flash Attention: 1D Tensor Inputs (seq_len=1 path)
// =========================================================================

#[test]
fn test_flash_forward_1d_inputs() {
    let attn = Attention::new(8).expect("test");

    let q = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("test");
    let v = Tensor::from_vec(vec![8], (1..9).map(|x| x as f32).collect()).expect("test");

    // All flash variants with 1D input
    let flash = attn.flash_forward(&q, &k, &v, 1).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 1).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 1).expect("test");

    assert_eq!(flash.shape(), &[1, 8]);
    assert_eq!(v2.shape(), &[1, 8]);
    assert_eq!(parallel.shape(), &[1, 8]);

    // Self-attention returns the value
    for i in 0..8 {
        assert!((flash.data()[i] - v.data()[i]).abs() < 1e-5);
    }
}

// =========================================================================
// SIMD Dot Product: Remainder Paths
// =========================================================================

#[test]
fn test_simd_remainder_paths() {
    // head_dim = 9 (8 SIMD + 1 remainder)
    let attn9 = Attention::new(9).expect("test");
    let q9 = Tensor::from_vec(
        vec![2, 9],
        (0..18).map(|i| (i as f32 * 0.1).sin()).collect(),
    )
    .expect("test");
    let k9 = Tensor::from_vec(
        vec![2, 9],
        (0..18).map(|i| (i as f32 * 0.1).cos()).collect(),
    )
    .expect("test");
    let v9 =
        Tensor::from_vec(vec![2, 9], (0..18).map(|i| i as f32 * 0.05).collect()).expect("test");

    let std9 = attn9.forward(&q9, &k9, &v9).expect("test");
    let v2_9 = attn9.flash_forward_v2(&q9, &k9, &v9, 1).expect("test");

    for i in 0..std9.data().len() {
        assert!((std9.data()[i] - v2_9.data()[i]).abs() < 1e-4);
    }

    // head_dim = 15 (8 SIMD + 7 remainder)
    let attn15 = Attention::new(15).expect("test");
    let q15 = Tensor::from_vec(vec![2, 15], vec![0.1; 30]).expect("test");
    let k15 = Tensor::from_vec(vec![2, 15], vec![0.2; 30]).expect("test");
    let v15 =
        Tensor::from_vec(vec![2, 15], (0..30).map(|i| i as f32 * 0.02).collect()).expect("test");

    let std15 = attn15.forward(&q15, &k15, &v15).expect("test");
    let par15 = attn15
        .flash_forward_parallel(&q15, &k15, &v15, 2)
        .expect("test");

    for i in 0..std15.data().len() {
        assert!((std15.data()[i] - par15.data()[i]).abs() < 1e-4);
    }
}

// =========================================================================
// FusedQKVAttention: Traits and Single/Many Heads
// =========================================================================

#[test]
fn test_fused_qkv_traits_and_heads() {
    let fused = FusedQKVAttention::new(8, 32).expect("test");

    // Debug/Clone
    let debug = format!("{:?}", fused);
    assert!(debug.contains("FusedQKVAttention"));

    let cloned = fused.clone();
    assert_eq!(fused.head_dim(), cloned.head_dim());
    assert_eq!(fused.hidden_dim(), cloned.hidden_dim());

    // Single head
    let single = FusedQKVAttention::new(8, 8).expect("test");
    assert_eq!(single.num_heads(), 1);

    let input = Tensor::from_vec(vec![2, 8], vec![0.1; 16]).expect("test");
    let output = single.forward(&input).expect("test");
    assert_eq!(output.shape(), &[2, 8]);

    // Many heads
    let many = FusedQKVAttention::new(4, 64).expect("test");
    assert_eq!(many.num_heads(), 16);
}

// =========================================================================
// MultiHeadAttention: Modes and Traits
// =========================================================================

#[test]
fn test_mha_modes_and_traits() {
    let mha = MultiHeadAttention::mha(64, 8).expect("test");

    // Debug/Clone
    let debug = format!("{:?}", mha);
    assert!(debug.contains("MultiHeadAttention"));

    let cloned = mha.clone();
    assert_eq!(mha.num_heads(), cloned.num_heads());

    // Test is_mha/is_mqa/is_gqa
    assert!(mha.is_mha());
    assert!(!mha.is_mqa());
    assert!(!mha.is_gqa());

    let mqa = MultiHeadAttention::mqa(64, 8).expect("test");
    assert!(mqa.is_mqa());
    assert!(!mqa.is_mha());

    let gqa = MultiHeadAttention::gqa(128, 16, 4).expect("test");
    assert!(gqa.is_gqa());
    assert!(!gqa.is_mha());
    assert!(!gqa.is_mqa());
    assert_eq!(gqa.head_dim(), 8);
}

#[test]
fn test_mha_single_head_edge_case() {
    // 1 head, 1 KV head: both MHA and MQA conditions met
    let mha = MultiHeadAttention::new(16, 1, 1).expect("test");

    assert!(mha.is_mha()); // num_kv_heads == num_heads
    assert!(mha.is_mqa()); // num_kv_heads == 1
    assert!(!mha.is_gqa());

    let input = Tensor::from_vec(vec![2, 16], vec![0.1; 32]).expect("test");
    let output = mha.forward(&input).expect("test");
    assert_eq!(output.shape(), &[2, 16]);
}

// =========================================================================
// Attention Forward: Cross-Attention and Long Sequence
// =========================================================================

#[test]
fn test_attention_cross_attention_shape() {
    // Q and K can have different sequence lengths
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test");
    let v = Tensor::from_vec(vec![3, 4], (1..13).map(|x| x as f32 * 0.1).collect()).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");
    assert_eq!(output.shape(), &[2, 4]); // Q's seq_len
}

#[test]
fn test_attention_long_sequence() {
    let attn = Attention::new(8).expect("test");

    let q = Tensor::from_vec(vec![64, 8], vec![0.1; 512]).expect("test");
    let k = Tensor::from_vec(vec![64, 8], vec![0.1; 512]).expect("test");
    let v =
        Tensor::from_vec(vec![64, 8], (0..512).map(|i| i as f32 * 0.001).collect()).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");
    assert_eq!(output.shape(), &[64, 8]);

    for &val in output.data() {
        assert!(val.is_finite());
    }
}

// =========================================================================
// Flash Attention: Block Size Larger Than Sequence
// =========================================================================

#[test]
fn test_flash_block_larger_than_seq() {
    let attn = Attention::new(8).expect("test");

    let q = Tensor::from_vec(vec![4, 8], vec![0.1; 32]).expect("test");
    let k = Tensor::from_vec(vec![4, 8], vec![0.2; 32]).expect("test");
    let v = Tensor::from_vec(vec![4, 8], (0..32).map(|i| i as f32 * 0.05).collect()).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let flash = attn.flash_forward(&q, &k, &v, 16).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 16).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 16).expect("test");

    for i in 0..standard.data().len() {
        assert!((standard.data()[i] - flash.data()[i]).abs() < 1e-4);
        assert!((standard.data()[i] - v2.data()[i]).abs() < 1e-4);
        assert!((standard.data()[i] - parallel.data()[i]).abs() < 1e-4);
    }
}

// =========================================================================
// SlidingWindowAttention: Bidirectional vs Causal Difference
// =========================================================================

#[test]
fn test_sliding_window_bidirectional_vs_causal() {
    let swa = SlidingWindowAttention::new(4, 5).expect("test");

    let q = Tensor::from_vec(
        vec![6, 4],
        (0..24).map(|i| (i as f32 * 0.15).sin()).collect(),
    )
    .expect("test");
    let k = Tensor::from_vec(
        vec![6, 4],
        (0..24).map(|i| (i as f32 * 0.15).cos()).collect(),
    )
    .expect("test");
    let v = Tensor::from_vec(vec![6, 4], (0..24).map(|i| i as f32 * 0.1).collect()).expect("test");

    let causal = swa.forward(&q, &k, &v).expect("test");
    let bidirectional = swa.forward_with_mask(&q, &k, &v, false).expect("test");

    assert_eq!(causal.shape(), bidirectional.shape());

    for &val in causal.data() {
        assert!(val.is_finite());
    }
    for &val in bidirectional.data() {
        assert!(val.is_finite());
    }
}
