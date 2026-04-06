//! Scheduler logic tests for Phase 46 - The Soft Target Sweep
//!
//! Goal: Push gpu/scheduler/batch.rs from 46% to >80% coverage.
//! Focus on pure CPU functions that don't need GPU.

use realizar::gpu::scheduler::batch::{argmax, simplified_attention};
use realizar::gpu::{GpuModelConfig, ResourceTracker};

// ============================================================================
// argmax Tests
// ============================================================================

#[test]
fn test_argmax_basic() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(argmax(&logits), 4);
}

#[test]
fn test_argmax_first() {
    let logits = vec![10.0, 1.0, 2.0, 3.0];
    assert_eq!(argmax(&logits), 0);
}

#[test]
fn test_argmax_middle() {
    let logits = vec![1.0, 2.0, 100.0, 3.0, 4.0];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn test_argmax_single() {
    let logits = vec![42.0];
    assert_eq!(argmax(&logits), 0);
}

#[test]
fn test_argmax_negatives() {
    let logits = vec![-5.0, -2.0, -1.0, -10.0];
    assert_eq!(argmax(&logits), 2); // -1.0 is the largest
}

#[test]
fn test_argmax_mixed() {
    let logits = vec![-1.0, 0.0, 1.0, -2.0];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn test_argmax_ties() {
    // When there are ties, max_by returns the last max element
    let logits = vec![5.0, 5.0, 5.0];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn test_argmax_large() {
    // Large vocabulary
    let mut logits = vec![0.0; 32000];
    logits[12345] = 100.0;
    assert_eq!(argmax(&logits), 12345);
}

#[test]
fn test_argmax_zeros() {
    // All zeros are equal, max_by returns the last one
    let logits = vec![0.0, 0.0, 0.0];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn test_argmax_very_small_diff() {
    // 1e-10 is below f32 precision (~1.19e-7), so these are effectively equal
    // max_by returns the last equal element
    let logits = vec![1.0, 1.0 + 1e-10, 1.0];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn test_argmax_distinguishable_diff() {
    // 1e-5 is above f32 precision, so should be distinguishable
    let logits = vec![1.0, 1.0 + 1e-5, 1.0];
    assert_eq!(argmax(&logits), 1);
}

// ============================================================================
// simplified_attention Tests
// ============================================================================

fn create_test_config(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> GpuModelConfig {
    GpuModelConfig {
        vocab_size: 1000,
        hidden_dim,
        num_heads,
        num_kv_heads,
        num_layers: 1,
        intermediate_dim: hidden_dim * 4,
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
    }
}

/// simplified_attention expects QKV in MHA layout: 3 * hidden_dim * seq_len
/// Q: seq_len * hidden_dim
/// K: seq_len * hidden_dim
/// V: seq_len * hidden_dim
fn qkv_mha_size(hidden_dim: usize, seq_len: usize) -> usize {
    3 * hidden_dim * seq_len
}

#[test]
fn test_simplified_attention_basic() {
    let config = create_test_config(64, 2, 2);

    // simplified_attention expects MHA layout: 3 * hidden_dim * seq_len
    let qkv_size = qkv_mha_size(config.hidden_dim, 1);
    let qkv = vec![0.1f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_simplified_attention_seq_len_2() {
    let config = create_test_config(64, 2, 2);

    // QKV for seq_len=2
    let qkv_size = qkv_mha_size(config.hidden_dim, 2);
    let qkv = vec![0.1f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 2);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim * 2);
}

#[test]
fn test_simplified_attention_4_heads() {
    // Test with 4 attention heads
    let config = create_test_config(64, 4, 4);

    let qkv_size = qkv_mha_size(config.hidden_dim, 1);
    let qkv = vec![0.1f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_simplified_attention_mha() {
    // Test MHA: num_heads=4, num_kv_heads=4
    let config = create_test_config(64, 4, 4);

    // MHA layout: 3 * hidden_dim
    let qkv_size = qkv_mha_size(config.hidden_dim, 1);
    let qkv = vec![0.1f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), config.hidden_dim);
}

#[test]
fn test_simplified_attention_zeros() {
    let config = create_test_config(32, 2, 2);

    let qkv_size = qkv_mha_size(config.hidden_dim, 1);
    let qkv = vec![0.0f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());
}

#[test]
fn test_simplified_attention_ones() {
    let config = create_test_config(32, 2, 2);

    let qkv_size = qkv_mha_size(config.hidden_dim, 1);
    let qkv = vec![1.0f32; qkv_size];

    let result = simplified_attention(&config, &qkv, 1);
    assert!(result.is_ok());
}

// ============================================================================
// ResourceTracker Tests
// ============================================================================

#[test]
fn test_resource_tracker_new() {
    let tracker = ResourceTracker::new(1024 * 1024 * 1024, 100); // 1GB, 100%
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_allocate() {
    let mut tracker = ResourceTracker::new(1024 * 1024, 100); // 1MB

    // Allocate 512KB
    let id = tracker
        .allocate(512 * 1024, 50)
        .expect("allocation should succeed");
    assert_eq!(tracker.memory_usage(), 512 * 1024);
    assert_eq!(tracker.compute_usage(), 50);

    // Release
    tracker.release(id);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_multiple_allocations() {
    let mut tracker = ResourceTracker::new(1024 * 1024, 100);

    let id1 = tracker
        .allocate(256 * 1024, 25)
        .expect("allocation 1 should succeed");
    let id2 = tracker
        .allocate(256 * 1024, 25)
        .expect("allocation 2 should succeed");

    assert_eq!(tracker.memory_usage(), 512 * 1024);
    assert_eq!(tracker.compute_usage(), 50);

    tracker.release(id1);
    assert_eq!(tracker.memory_usage(), 256 * 1024);
    assert_eq!(tracker.compute_usage(), 25);

    tracker.release(id2);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_allocation_fail_memory() {
    let mut tracker = ResourceTracker::new(1024, 100); // 1KB

    // Try to allocate more than capacity
    let result = tracker.allocate(2048, 50);
    assert!(result.is_none());
}

#[test]
fn test_resource_tracker_allocation_fail_compute() {
    let mut tracker = ResourceTracker::new(1024 * 1024, 100);

    // Allocate 90% compute
    tracker
        .allocate(1024, 90)
        .expect("first allocation should succeed");

    // Try to allocate another 20% (would exceed 100%)
    let result = tracker.allocate(1024, 20);
    assert!(result.is_none());
}

#[test]
fn test_resource_tracker_can_allocate() {
    let mut tracker = ResourceTracker::new(1024, 100);

    assert!(tracker.can_allocate(512, 50));
    assert!(tracker.can_allocate(1024, 100));
    assert!(!tracker.can_allocate(2048, 50)); // Memory overflow
    assert!(!tracker.can_allocate(512, 150)); // Compute overflow

    // After allocation, less capacity
    tracker
        .allocate(512, 50)
        .expect("allocation should succeed");
    assert!(tracker.can_allocate(512, 50));
    assert!(!tracker.can_allocate(1024, 50)); // Memory would overflow
}

#[test]
fn test_resource_tracker_release_invalid() {
    let mut tracker = ResourceTracker::new(1024, 100);

    // Releasing non-existent ID should not panic
    tracker.release(999);
    assert_eq!(tracker.memory_usage(), 0);
}

#[test]
fn test_resource_tracker_release_twice() {
    let mut tracker = ResourceTracker::new(1024, 100);

    let id = tracker
        .allocate(512, 50)
        .expect("allocation should succeed");
    tracker.release(id);

    // Second release should not panic or change state
    tracker.release(id);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_usage_percentage() {
    let mut tracker = ResourceTracker::new(1000, 100);

    let (mem_pct, compute_pct) = tracker.usage_percentage();
    assert!((mem_pct - 0.0).abs() < 0.01);
    assert!((compute_pct - 0.0).abs() < 0.01);

    tracker
        .allocate(500, 50)
        .expect("allocation should succeed");
    let (mem_pct, compute_pct) = tracker.usage_percentage();
    assert!((mem_pct - 50.0).abs() < 0.01);
    assert!((compute_pct - 50.0).abs() < 0.01);
}

#[test]
fn test_resource_tracker_zero_capacity() {
    let mut tracker = ResourceTracker::new(0, 0);

    // Any allocation should fail
    assert!(tracker.allocate(1, 0).is_none());
    assert!(tracker.allocate(0, 1).is_none());

    // Zero allocation behavior - depends on implementation
    // Just verify no panic
    let _ = tracker.allocate(0, 0);
}

#[test]
fn test_resource_tracker_edge_exact_fit() {
    let mut tracker = ResourceTracker::new(1000, 100);

    // Exact fit allocation
    let id = tracker
        .allocate(1000, 100)
        .expect("exact fit should succeed");
    assert_eq!(tracker.memory_usage(), 1000);
    assert_eq!(tracker.compute_usage(), 100);

    // No more room
    assert!(!tracker.can_allocate(1, 0));
    assert!(!tracker.can_allocate(0, 1));

    tracker.release(id);
    assert!(tracker.can_allocate(1000, 100));
}

// ============================================================================
// GpuModelConfig Tests
// ============================================================================

#[test]
fn test_config_head_dim() {
    let config = create_test_config(64, 4, 4);
    assert_eq!(config.head_dim(), 16); // 64/4
}

#[test]
fn test_config_kv_dim() {
    let config = create_test_config(64, 4, 2);
    assert_eq!(config.kv_dim(), 32); // 2 * (64/4) = 32
}

#[test]
fn test_config_qkv_dim() {
    // MHA: hidden_dim + 2 * hidden_dim = 3 * hidden_dim
    let config = create_test_config(64, 4, 4);
    assert_eq!(config.qkv_dim(), 192); // 64 + 2*64

    // GQA: hidden_dim + 2 * kv_dim
    let config2 = create_test_config(64, 4, 2);
    assert_eq!(config2.qkv_dim(), 128); // 64 + 2*32
}

#[test]
fn test_config_is_gqa() {
    let mha_config = create_test_config(64, 4, 4);
    assert!(!mha_config.is_gqa());

    let gqa_config = create_test_config(64, 4, 2);
    assert!(gqa_config.is_gqa());
}
