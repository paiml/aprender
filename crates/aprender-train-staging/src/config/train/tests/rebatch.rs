//! Tests for rebatching functionality

use crate::config::train::rebatch;
use crate::train::Batch;
use crate::Tensor;

#[test]
fn test_rebatch_empty() {
    let batches: Vec<Batch> = Vec::new();
    let result = rebatch(batches, 4);
    assert!(result.is_empty());
}

#[test]
fn test_rebatch_single_batch() {
    // Create batch with 4 examples, each with 2 features (8 elements total)
    // rebatch determines input_dim from first batch's length = 8
    // So this represents 1 example (8/8=1)
    let batch = Batch::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false),
        Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], false),
    );
    // With input_dim=4 and 4 elements, we have 1 example
    // Rebatching 1 example into batch_size 2 gives 1 batch
    let result = rebatch(vec![batch], 2);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_multiple_batches() {
    // Two batches with same dimensions
    let batch1 = Batch::new(
        Tensor::from_vec(vec![1.0, 2.0], false),
        Tensor::from_vec(vec![3.0, 4.0], false),
    );
    let batch2 = Batch::new(
        Tensor::from_vec(vec![5.0, 6.0], false),
        Tensor::from_vec(vec![7.0, 8.0], false),
    );
    // input_dim = 2 (from first batch), total 4 elements = 2 examples
    // Rebatching 2 examples with batch_size 2 = 1 batch
    let result = rebatch(vec![batch1, batch2], 2);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_creates_multiple_batches() {
    // Create 4 batches each with 2 elements (input_dim=2)
    let batches: Vec<Batch> = (0..4)
        .map(|i| {
            Batch::new(
                Tensor::from_vec(vec![(i * 2) as f32, (i * 2 + 1) as f32], false),
                Tensor::from_vec(vec![(i * 2 + 10) as f32, (i * 2 + 11) as f32], false),
            )
        })
        .collect();
    // input_dim = 2, total 8 elements = 4 examples
    // Rebatching 4 examples with batch_size 2 = 2 batches
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_rebatch_uneven_split() {
    // Create 5 batches each with 2 elements
    let batches: Vec<Batch> = (0..5)
        .map(|i| {
            Batch::new(
                Tensor::from_vec(vec![(i * 2) as f32, (i * 2 + 1) as f32], false),
                Tensor::from_vec(vec![(i * 2 + 10) as f32, (i * 2 + 11) as f32], false),
            )
        })
        .collect();
    // input_dim = 2, total 10 elements = 5 examples
    // Rebatching 5 examples with batch_size 2 = 3 batches (2, 2, 1)
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 3);
}
