use super::*;

// ==========================================================================
// FALSIFICATION: WidthPruner configuration
// ==========================================================================
#[test]
fn test_width_pruner_new() {
    let pruner = WidthPruner::new(512, 2048, 8);

    assert_eq!(pruner.target_hidden_dim(), 512);
    assert_eq!(pruner.target_intermediate_dim(), 2048);
    assert_eq!(pruner.num_attention_heads(), 8);
}

#[test]
fn test_width_pruner_default() {
    let pruner = WidthPruner::default();
    assert_eq!(pruner.target_hidden_dim(), 0);
    assert_eq!(pruner.num_attention_heads(), 1);
}

// ==========================================================================
// FALSIFICATION: Validation
// ==========================================================================
#[test]
fn test_validate_success() {
    let pruner = WidthPruner::new(512, 2048, 8); // 512/8 = 64
    assert!(pruner.validate(1024, 4096).is_ok());
}

#[test]
fn test_validate_head_divisibility() {
    let pruner = WidthPruner::new(500, 2048, 8); // 500/8 = 62.5 - not divisible
    let result = pruner.validate(1024, 4096);

    assert!(
        result.is_err(),
        "WID-01 FALSIFIED: Should error on non-divisible"
    );
    match result.unwrap_err() {
        PruningError::InvalidPattern { message } => {
            assert!(message.contains("divisible"));
        }
        _ => panic!("WID-01 FALSIFIED: Expected InvalidPattern error"),
    }
}

#[test]
fn test_validate_target_exceeds_original_hidden() {
    let pruner = WidthPruner::new(2048, 2048, 8);
    let result = pruner.validate(1024, 4096); // 2048 > 1024

    assert!(
        result.is_err(),
        "WID-02 FALSIFIED: Should error when target > original"
    );
}

#[test]
fn test_validate_target_exceeds_original_intermediate() {
    let pruner = WidthPruner::new(512, 8192, 8);
    let result = pruner.validate(1024, 4096); // 8192 > 4096

    assert!(
        result.is_err(),
        "WID-03 FALSIFIED: Should error when target > original"
    );
}

// ==========================================================================
// FALSIFICATION: Channel importance computation
// ==========================================================================
#[test]
fn test_compute_channel_importance_basic() {
    let pruner = WidthPruner::new(2, 2, 1);

    // 2 samples, 4 hidden channels
    let hidden = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // sample 0
            1.0, 2.0, 3.0, 4.0, // sample 1
        ],
        &[2, 4],
    );

    // 2 samples, 3 intermediate channels
    let intermediate = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);

    let importance = pruner
        .compute_channel_importance(&hidden, &intermediate)
        .unwrap();

    assert_eq!(importance.hidden_dim(), 4);
    assert_eq!(importance.intermediate_dim(), 3);
    assert_eq!(importance.num_samples, 2);

    // importance[d] = mean(x^2) = x^2 for constant values
    let h_data = importance.hidden.data();
    assert!(
        (h_data[0] - 1.0).abs() < 1e-5,
        "WID-04 FALSIFIED: Channel 0 importance should be 1.0"
    );
    assert!(
        (h_data[1] - 4.0).abs() < 1e-5,
        "WID-04 FALSIFIED: Channel 1 importance should be 4.0"
    );
    assert!(
        (h_data[2] - 9.0).abs() < 1e-5,
        "WID-04 FALSIFIED: Channel 2 importance should be 9.0"
    );
    assert!(
        (h_data[3] - 16.0).abs() < 1e-5,
        "WID-04 FALSIFIED: Channel 3 importance should be 16.0"
    );
}

#[test]
fn test_compute_channel_importance_varying() {
    let pruner = WidthPruner::new(2, 2, 1);

    // importance = mean(x^2) across samples
    let hidden = Tensor::new(
        &[
            1.0, 3.0, // sample 0: [1, 9]
            3.0, 1.0, // sample 1: [9, 1]
        ],
        &[2, 2],
    );

    // Intermediate must have same number of samples
    let intermediate = Tensor::new(
        &[
            0.0, 0.0, // sample 0
            0.0, 0.0, // sample 1
        ],
        &[2, 2],
    );

    let importance = pruner
        .compute_channel_importance(&hidden, &intermediate)
        .unwrap();

    let h_data = importance.hidden.data();
    // Channel 0: mean(1, 9) = 5
    // Channel 1: mean(9, 1) = 5
    assert!(
        (h_data[0] - 5.0).abs() < 1e-5,
        "WID-05 FALSIFIED: Channel 0 importance should be 5.0, got {}",
        h_data[0]
    );
    assert!(
        (h_data[1] - 5.0).abs() < 1e-5,
        "WID-05 FALSIFIED: Channel 1 importance should be 5.0, got {}",
        h_data[1]
    );
}

#[test]
fn test_compute_channel_importance_invalid_shape() {
    let pruner = WidthPruner::new(2, 2, 1);

    // 1D tensor (invalid)
    let hidden = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let intermediate = Tensor::new(&[1.0, 2.0], &[1, 2]);

    let result = pruner.compute_channel_importance(&hidden, &intermediate);
    assert!(
        result.is_err(),
        "WID-06 FALSIFIED: Should error on 1D tensor"
    );
}

#[test]
fn test_compute_channel_importance_mismatched_samples() {
    let pruner = WidthPruner::new(2, 2, 1);

    // Different number of samples
    let hidden = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let intermediate = Tensor::new(&[1.0, 2.0], &[1, 2]); // Only 1 sample

    let result = pruner.compute_channel_importance(&hidden, &intermediate);
    assert!(
        result.is_err(),
        "WID-06b FALSIFIED: Should error on mismatched sample counts"
    );
}

// ==========================================================================
// FALSIFICATION: Top-k channel selection
// ==========================================================================
#[test]
fn test_top_hidden_channels() {
    let importance = ChannelImportance::new(
        Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]),
        Tensor::new(&[], &[0]),
        1,
    );

    let top3 = importance.top_hidden_channels(3);

    assert_eq!(top3.len(), 3, "WID-07 FALSIFIED: Should return 3 channels");
    // Top 3 by importance: index 4 (5.0), index 2 (4.0), index 0 (3.0)
    // Sorted: [0, 2, 4]
    assert!(
        top3.contains(&4),
        "WID-07 FALSIFIED: Should contain index 4 (highest)"
    );
    assert!(
        top3.contains(&2),
        "WID-07 FALSIFIED: Should contain index 2"
    );
    assert!(
        top3.contains(&0),
        "WID-07 FALSIFIED: Should contain index 0"
    );
}

#[test]
fn test_top_intermediate_channels() {
    let importance = ChannelImportance::new(
        Tensor::new(&[], &[0]),
        Tensor::new(&[10.0, 20.0, 5.0, 15.0], &[4]),
        1,
    );

    let top2 = importance.top_intermediate_channels(2);

    assert_eq!(top2.len(), 2);
    // Top 2: index 1 (20.0), index 3 (15.0)
    assert!(top2.contains(&1));
    assert!(top2.contains(&3));
}

// ==========================================================================
// FALSIFICATION: Select channels to keep
// ==========================================================================
#[test]
fn test_select_channels_to_keep() {
    let pruner = WidthPruner::new(2, 2, 1);
    let importance = ChannelImportance::new(
        Tensor::new(&[1.0, 5.0, 3.0, 4.0], &[4]), // top-2: indices 1, 3
        Tensor::new(&[10.0, 5.0, 15.0], &[3]),    // top-2: indices 0, 2
        1,
    );

    let (hidden_keep, intermediate_keep) = pruner.select_channels_to_keep(&importance).unwrap();

    assert_eq!(
        hidden_keep.len(),
        2,
        "WID-08 FALSIFIED: Should keep 2 hidden"
    );
    assert_eq!(
        intermediate_keep.len(),
        2,
        "WID-08 FALSIFIED: Should keep 2 intermediate"
    );

    // Verify correct channels selected
    assert!(hidden_keep.contains(&1)); // 5.0
    assert!(hidden_keep.contains(&3)); // 4.0
    assert!(intermediate_keep.contains(&0)); // 10.0
    assert!(intermediate_keep.contains(&2)); // 15.0
}

#[test]
fn test_select_channels_validates() {
    let pruner = WidthPruner::new(100, 100, 8); // 100/8 = 12.5 - invalid
    let importance = ChannelImportance::new(
        Tensor::new(&[1.0; 50], &[50]),
        Tensor::new(&[1.0; 200], &[200]),
        1,
    );

    let result = pruner.select_channels_to_keep(&importance);
    assert!(
        result.is_err(),
        "WID-09 FALSIFIED: Should validate before selecting"
    );
}

// ==========================================================================
// FALSIFICATION: Generate hidden mask
// ==========================================================================
#[test]
fn test_generate_hidden_mask() {
    let pruner = WidthPruner::new(3, 3, 1);
    let mask = pruner.generate_hidden_mask(5, &[0, 2, 4]);

    let data = mask.data();
    assert_eq!(data.len(), 5);
    assert_eq!(data[0], 1.0, "WID-10 FALSIFIED: Channel 0 should be kept");
    assert_eq!(data[1], 0.0, "WID-10 FALSIFIED: Channel 1 should be pruned");
    assert_eq!(data[2], 1.0, "WID-10 FALSIFIED: Channel 2 should be kept");
    assert_eq!(data[3], 0.0, "WID-10 FALSIFIED: Channel 3 should be pruned");
    assert_eq!(data[4], 1.0, "WID-10 FALSIFIED: Channel 4 should be kept");
}

#[test]
fn test_generate_hidden_mask_out_of_bounds() {
    let pruner = WidthPruner::new(3, 3, 1);
    let mask = pruner.generate_hidden_mask(3, &[0, 1, 100]); // 100 is out of bounds

    let data = mask.data();
    assert_eq!(data.len(), 3);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 1.0);
    assert_eq!(data[2], 0.0); // index 100 ignored, index 2 not set
}

// ==========================================================================
// FALSIFICATION: Head dimension calculation
// ==========================================================================
#[test]
fn test_head_dim_after_pruning() {
    let pruner = WidthPruner::new(512, 2048, 8);
    assert_eq!(pruner.head_dim_after_pruning(), 64); // 512/8
}

#[test]
fn test_head_dim_zero_heads() {
    let pruner = WidthPruner::new(512, 2048, 0);
    assert_eq!(pruner.head_dim_after_pruning(), 0);
}

// ==========================================================================
// FALSIFICATION: WidthPruningResult
// ==========================================================================
#[test]
fn test_width_pruning_result_compression() {
    let result = WidthPruningResult::new(
        1024,
        512, // hidden: 1024 -> 512
        4096,
        2048, // intermediate: 4096 -> 2048
        vec![0, 1, 2, 3],
        vec![0, 1],
    );

    assert!(
        (result.hidden_compression_ratio() - 2.0).abs() < 1e-5,
        "WID-11 FALSIFIED: Hidden compression should be 2x"
    );
    assert!(
        (result.intermediate_compression_ratio() - 2.0).abs() < 1e-5,
        "WID-11 FALSIFIED: Intermediate compression should be 2x"
    );
}

#[test]
fn test_width_pruning_result_removal_percentage() {
    let result = WidthPruningResult::new(1000, 750, 2000, 1500, vec![], vec![]);

    assert!(
        (result.hidden_removal_percentage() - 25.0).abs() < 1e-5,
        "WID-12 FALSIFIED: Hidden removal should be 25%"
    );
    assert!(
        (result.intermediate_removal_percentage() - 25.0).abs() < 1e-5,
        "WID-12 FALSIFIED: Intermediate removal should be 25%"
    );
}

#[test]
fn test_width_pruning_result_edge_cases() {
    let result = WidthPruningResult::new(0, 0, 0, 0, vec![], vec![]);

    assert_eq!(result.hidden_compression_ratio(), f32::INFINITY);
    assert_eq!(result.hidden_removal_percentage(), 0.0);
}

// ==========================================================================
// FALSIFICATION: top_k_indices helper
// ==========================================================================
#[test]
fn test_top_k_indices_basic() {
    let data = &[5.0, 1.0, 3.0, 2.0, 4.0];
    let top3 = top_k_indices(data, 3);

    // Top 3 values: 5.0 (idx 0), 4.0 (idx 4), 3.0 (idx 2)
    // Sorted: [0, 2, 4]
    assert_eq!(top3, vec![0, 2, 4]);
}

#[test]
fn test_top_k_indices_all() {
    let data = &[1.0, 2.0, 3.0];
    let all = top_k_indices(data, 5); // k > len

    assert_eq!(all, vec![0, 1, 2]);
}

#[test]
fn test_top_k_indices_empty() {
    let data: &[f32] = &[];
    let result = top_k_indices(data, 3);
    assert!(result.is_empty());
}

#[test]
fn test_top_k_indices_ties() {
    let data = &[1.0, 2.0, 2.0, 1.0];
    let top2 = top_k_indices(data, 2);

    // Two 2.0 values at indices 1 and 2
    assert_eq!(top2.len(), 2);
    assert!(top2.contains(&1) || top2.contains(&2)); // At least one of them
}

// ==========================================================================
// FALSIFICATION: Clone and Debug
// ==========================================================================
#[test]
fn test_width_pruner_clone() {
    let orig = WidthPruner::new(512, 2048, 8);
    let cloned = orig.clone();

    assert_eq!(orig.target_hidden_dim(), cloned.target_hidden_dim());
    assert_eq!(
        orig.target_intermediate_dim(),
        cloned.target_intermediate_dim()
    );
    assert_eq!(orig.num_attention_heads(), cloned.num_attention_heads());
}

#[test]
fn test_width_pruner_debug() {
    let pruner = WidthPruner::new(512, 2048, 8);
    let debug = format!("{:?}", pruner);
    assert!(debug.contains("WidthPruner"));
}

#[test]
fn test_channel_importance_debug() {
    let imp = ChannelImportance::new(Tensor::new(&[1.0], &[1]), Tensor::new(&[1.0], &[1]), 1);
    let debug = format!("{:?}", imp);
    assert!(debug.contains("ChannelImportance"));
}

#[test]
fn test_width_pruning_result_debug() {
    let result = WidthPruningResult::new(100, 50, 200, 100, vec![], vec![]);
    let debug = format!("{:?}", result);
    assert!(debug.contains("WidthPruningResult"));
}
