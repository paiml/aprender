//! Tests for demo batch creation

use crate::config::train::create_demo_batches;

#[test]
fn test_create_demo_batches_default() {
    let batches = create_demo_batches(4);
    assert!(!batches.is_empty());
    // Each batch should have 4 * 4 = 16 elements (batch_size * feature_dim)
    assert_eq!(batches[0].inputs.len(), 16);
    assert_eq!(batches[0].targets.len(), 16);
}

#[test]
fn test_create_demo_batches_small_batch() {
    let batches = create_demo_batches(1);
    assert!(!batches.is_empty());
    // With batch_size 1, should create multiple batches
    assert!(batches.len() >= 2);
}

#[test]
fn test_create_demo_batches_zero_batch_size() {
    // Should handle zero gracefully (uses max(1))
    let batches = create_demo_batches(0);
    assert!(!batches.is_empty());
}

#[test]
fn test_create_demo_batches_large_batch() {
    let batches = create_demo_batches(16);
    assert!(!batches.is_empty());
    // With large batch size, should have at least 2 batches (from the max)
    assert!(batches.len() >= 2);
}
