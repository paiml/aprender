//! Tests for LoRA layer

use super::*;
use crate::autograd::matmul;
use crate::Tensor;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

// ========================================================================
// PROPERTY TESTS
// ========================================================================

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    #[test]
    fn prop_zero_b_gives_base_output(
        d_out in 2usize..10,
        d_in in 2usize..10,
        rank in 1usize..5,
    ) {
        // When B is zeros, LoRA output should equal base output
        let size = d_out * d_in;
        let base_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
        let base_weight = Tensor::from_vec(base_data, false);
        let lora = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, 1.0);

        // B is initialized to zeros by default
        let x_data: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.5).collect();
        let x = Tensor::from_vec(x_data.clone(), true);

        let lora_output = lora.forward(&x);

        // Compute expected base output: W @ x
        let base_output = matmul(&base_weight, &Tensor::from_vec(x_data, false), d_out, d_in, 1);

        for i in 0..d_out {
            prop_assert!(
                (lora_output.data()[i] - base_output.data()[i]).abs() < 1e-4,
                "Zero B should give base output at index {}", i
            );
        }
    }

    #[test]
    fn prop_merge_preserves_forward_output(
        d_out in 2usize..8,
        d_in in 2usize..8,
        rank in 1usize..4,
    ) {
        let size = d_out * d_in;
        let base_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();
        let base_weight = Tensor::from_vec(base_data, false);
        let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, 2.0);

        // Set non-zero LoRA weights
        let a_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32 * 0.2).sin() * 0.1).collect();
        let b_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32 * 0.3).cos() * 0.1).collect();
        *lora.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
        *lora.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

        let x_data: Vec<f32> = (0..d_in).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::from_vec(x_data.clone(), true);

        // Forward before merge
        let output_before = lora.forward(&x);

        // Merge
        lora.merge();
        prop_assert!(lora.is_merged());

        // Forward after merge
        let x2 = Tensor::from_vec(x_data, true);
        let output_after = lora.forward(&x2);

        // Outputs should match
        for i in 0..d_out {
            prop_assert!(
                (output_before.data()[i] - output_after.data()[i]).abs() < 1e-3,
                "Merge should preserve output at index {}: before={} after={}",
                i, output_before.data()[i], output_after.data()[i]
            );
        }
    }

    #[test]
    fn prop_unmerge_restores_weights(
        d_out in 2usize..8,
        d_in in 2usize..8,
        rank in 1usize..4,
    ) {
        let size = d_out * d_in;
        let base_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let base_weight = Tensor::from_vec(base_data.clone(), false);
        let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, 1.0);

        // Set non-zero LoRA weights
        let a_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * 0.02).collect();
        *lora.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
        *lora.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

        // Merge then unmerge
        lora.merge();
        lora.unmerge();

        // Base weights should be restored
        for i in 0..size {
            prop_assert!(
                (lora.base_weight().data()[i] - base_data[i]).abs() < 1e-4,
                "Unmerge should restore weight at index {}", i
            );
        }
    }

    #[test]
    fn prop_scale_factor_correct(
        rank in 1usize..32,
        alpha in 1.0f32..64.0,
    ) {
        let base_weight = Tensor::from_vec(vec![1.0], false);
        let lora = LoRALayer::new(base_weight, 1, 1, rank, alpha);

        let expected_scale = alpha / rank as f32;
        prop_assert!(
            (lora.scale() - expected_scale).abs() < 1e-6,
            "Scale should be alpha/rank: expected {} got {}", expected_scale, lora.scale()
        );
    }

    #[test]
    fn prop_lora_dimensions_correct(
        d_out in 2usize..20,
        d_in in 2usize..20,
        rank in 1usize..10,
    ) {
        let size = d_out * d_in;
        let base_data: Vec<f32> = vec![0.0; size];
        let base_weight = Tensor::from_vec(base_data, false);
        let lora = LoRALayer::new(base_weight, d_out, d_in, rank, 1.0);

        // Verify all dimensions
        prop_assert_eq!(lora.d_out(), d_out);
        prop_assert_eq!(lora.d_in(), d_in);
        prop_assert_eq!(lora.rank(), rank);
        prop_assert_eq!(lora.lora_a().len(), rank * d_in);
        prop_assert_eq!(lora.lora_b().len(), d_out * rank);
        prop_assert_eq!(lora.base_weight().len(), d_out * d_in);
    }
}

// ========================================================================
// DETERMINISTIC UNIT TESTS
// ========================================================================

#[test]
fn test_lora_layer_creation() {
    // 3x2 weight matrix
    let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    let lora = LoRALayer::new(base_weight, 3, 2, 2, 2.0);

    assert_eq!(lora.rank(), 2);
    assert_eq!(lora.d_out(), 3);
    assert_eq!(lora.d_in(), 2);
    assert_abs_diff_eq!(lora.scale(), 1.0, epsilon = 1e-6); // alpha/rank = 2/2 = 1
    assert!(!lora.is_merged());

    // Check dimensions
    assert_eq!(lora.lora_a().len(), 2 * 2); // [r * d_in]
    assert_eq!(lora.lora_b().len(), 3 * 2); // [d_out * r]
}

#[test]
fn test_lora_forward_unmerged() {
    // Simple 2x2 identity weight matrix
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // Set LoRA weights to known values for testing
    // A: [1, 2] (1x2 matrix)
    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
    // B: [3, 4] (2x1 matrix) - stored as column-major [3, 4]
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[3.0, 4.0]);

    // Input vector [1, 2]
    let x = Tensor::from_vec(vec![1.0, 2.0], true);

    // Forward pass
    let output = lora.forward(&x);

    // Expected:
    // Base: [[1, 0], [0, 1]] @ [1, 2] = [1, 2]
    // LoRA:
    //   A @ x: [1, 2] @ [1, 2] = 1*1 + 2*2 = 5 (scalar)
    //   B @ (A@x): [[3], [4]] @ [5] = [15, 20]
    //   scale = 1.0, so LoRA output = [15, 20]
    // Total: [1, 2] + [15, 20] = [16, 22]
    assert_eq!(output.len(), 2);
    assert_abs_diff_eq!(output.data()[0], 16.0, epsilon = 1e-4);
    assert_abs_diff_eq!(output.data()[1], 22.0, epsilon = 1e-4);
}

#[test]
fn test_lora_merge_unmerge() {
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // Set LoRA weights: A = [1, 2], B = [0.5, 0.5]
    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

    // Original base weight
    let original_weight = lora.base_weight().data().to_owned();

    // Merge
    lora.merge();
    assert!(lora.is_merged());

    // After merge, base weight should be W + scale * (B @ A)
    // B @ A = [[0.5], [0.5]] @ [[1, 2]] = [[0.5, 1.0], [0.5, 1.0]]
    //       = [0.5, 1.0, 0.5, 1.0] in row-major
    // scale = 1.0, so delta = [0.5, 1.0, 0.5, 1.0]
    // W' = [1, 0, 0, 1] + [0.5, 1.0, 0.5, 1.0] = [1.5, 1.0, 0.5, 2.0]
    let merged_weight = lora.base_weight().data();
    assert_abs_diff_eq!(merged_weight[0], 1.5, epsilon = 1e-4);
    assert_abs_diff_eq!(merged_weight[1], 1.0, epsilon = 1e-4);
    assert_abs_diff_eq!(merged_weight[2], 0.5, epsilon = 1e-4);
    assert_abs_diff_eq!(merged_weight[3], 2.0, epsilon = 1e-4);

    // Unmerge
    lora.unmerge();
    assert!(!lora.is_merged());

    // Should restore original weight
    let restored_weight = lora.base_weight().data();
    for i in 0..4 {
        assert_abs_diff_eq!(restored_weight[i], original_weight[i], epsilon = 1e-4);
    }
}

#[test]
fn test_lora_forward_merged() {
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);

    let x = Tensor::from_vec(vec![1.0, 1.0], true);

    // Forward before merge
    let output_unmerged = lora.forward(&x);

    // Merge
    lora.merge();

    // Forward after merge - should give same result
    let output_merged = lora.forward(&x);

    assert_eq!(output_unmerged.len(), output_merged.len());
    for i in 0..output_unmerged.len() {
        assert_abs_diff_eq!(output_unmerged.data()[i], output_merged.data()[i], epsilon = 1e-4);
    }
}

#[test]
fn test_lora_trainable_params() {
    let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 2, 4.0);

    let params = lora.trainable_params();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].len(), 2 * 2); // A: [r * d_in]
    assert_eq!(params[1].len(), 2 * 2); // B: [d_out * r]

    // All should require gradients
    assert!(params[0].requires_grad());
    assert!(params[1].requires_grad());
}

#[test]
fn test_lora_zero_initialization() {
    // With B initialized to zeros, initial LoRA contribution should be zero
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let lora = LoRALayer::new(base_weight, 2, 2, 2, 2.0);

    let x = Tensor::from_vec(vec![2.0, 3.0], true);

    // Forward pass
    let output = lora.forward(&x);

    // Should match base forward since B is zeros
    // Base: [[1, 0], [0, 1]] @ [2, 3] = [2, 3]
    assert_abs_diff_eq!(output.data()[0], 2.0, epsilon = 1e-4);
    assert_abs_diff_eq!(output.data()[1], 3.0, epsilon = 1e-4);
}

#[test]
fn test_lora_rank_scaling() {
    let base_weight = Tensor::from_vec(vec![1.0], false);

    // Different ranks with same alpha should give different scales
    let lora_r4 = LoRALayer::new(base_weight.clone(), 1, 1, 4, 8.0);
    let lora_r8 = LoRALayer::new(base_weight, 1, 1, 8, 8.0);

    assert_abs_diff_eq!(lora_r4.scale(), 2.0, epsilon = 1e-6); // 8/4 = 2
    assert_abs_diff_eq!(lora_r8.scale(), 1.0, epsilon = 1e-6); // 8/8 = 1
}

// ========================================================================
// ENT-LoRA-004: rsLoRA scaling tests
// ========================================================================

#[test]
fn test_rslora_scaling_compute() {
    // Standard: alpha / rank
    assert_abs_diff_eq!(LoRAScaling::Standard.compute(32.0, 16), 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(LoRAScaling::Standard.compute(32.0, 64), 0.5, epsilon = 1e-6);

    // rsLoRA: alpha / sqrt(rank)
    assert_abs_diff_eq!(LoRAScaling::RsLoRA.compute(32.0, 16), 8.0, epsilon = 1e-6); // 32/4
    assert_abs_diff_eq!(LoRAScaling::RsLoRA.compute(32.0, 64), 4.0, epsilon = 1e-6); // 32/8
    assert_abs_diff_eq!(LoRAScaling::RsLoRA.compute(8.0, 4), 4.0, epsilon = 1e-6);
    // 8/2
}

#[test]
fn test_rslora_scaling_all_ranks() {
    // FALSIFY-LoRA-MATH-002: rsLoRA scale for r=4,8,16,32,64,128
    let alpha = 32.0;
    for &rank in &[4usize, 8, 16, 32, 64, 128] {
        let standard = LoRAScaling::Standard.compute(alpha, rank);
        let rslora = LoRAScaling::RsLoRA.compute(alpha, rank);
        assert_abs_diff_eq!(standard, alpha / rank as f32, epsilon = 1e-6);
        assert_abs_diff_eq!(rslora, alpha / (rank as f32).sqrt(), epsilon = 1e-6);
        // rsLoRA should always be >= standard for rank > 1
        assert!(rslora >= standard, "rsLoRA should be >= standard for rank={rank}");
    }
}

#[test]
fn test_lora_layer_with_rslora() {
    let base_weight = Tensor::from_vec(vec![1.0; 4], false);
    let layer = LoRALayer::new_with_scaling(base_weight, 2, 2, 4, 8.0, LoRAScaling::RsLoRA);
    // rsLoRA: 8.0 / sqrt(4) = 8.0 / 2.0 = 4.0
    assert_abs_diff_eq!(layer.scale(), 4.0, epsilon = 1e-6);
}

#[test]
fn test_lora_layer_standard_scaling_matches_new() {
    let base_weight = Tensor::from_vec(vec![1.0; 4], false);
    let standard = LoRALayer::new(base_weight.clone(), 2, 2, 4, 8.0);
    let explicit = LoRALayer::new_with_scaling(base_weight, 2, 2, 4, 8.0, LoRAScaling::Standard);
    assert_abs_diff_eq!(standard.scale(), explicit.scale(), epsilon = 1e-10);
}

// ========================================================================
// COVERAGE GAP TESTS — early returns, Clone, accessors
// ========================================================================

#[test]
fn test_merge_already_merged_is_noop() {
    // Covers the "already merged" early return in merge() (lines 149-151)
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // Set non-zero LoRA weights
    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

    // First merge
    lora.merge();
    assert!(lora.is_merged());
    let weight_after_first_merge = lora.base_weight().data().to_owned();

    // Second merge — should be a no-op (early return path)
    lora.merge();
    assert!(lora.is_merged());
    let weight_after_second_merge = lora.base_weight().data().to_owned();

    // Weights must be identical — no double-merge
    for i in 0..4 {
        assert_abs_diff_eq!(
            weight_after_first_merge[i],
            weight_after_second_merge[i],
            epsilon = 1e-10
        );
    }
}

#[test]
fn test_unmerge_not_merged_is_noop() {
    // Covers the "not merged" early return in unmerge() (lines 169-171)
    let base_data = vec![1.0, 2.0, 3.0, 4.0];
    let base_weight = Tensor::from_vec(base_data.clone(), false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // Set non-zero LoRA weights
    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

    // Never merged — unmerge should be a no-op (early return path)
    assert!(!lora.is_merged());
    lora.unmerge();
    assert!(!lora.is_merged());

    // Weights must be unchanged
    for i in 0..4 {
        assert_abs_diff_eq!(lora.base_weight().data()[i], base_data[i], epsilon = 1e-10);
    }
}

#[test]
fn test_lora_layer_clone() {
    // Covers the Clone derive on LoRALayer
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut original = LoRALayer::new(base_weight, 2, 2, 1, 2.0);

    *original.lora_a_mut().data_mut() = ndarray::arr1(&[0.3, 0.7]);
    *original.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.9]);

    let cloned = original.clone();

    // Verify all fields match
    assert_eq!(cloned.d_out(), original.d_out());
    assert_eq!(cloned.d_in(), original.d_in());
    assert_eq!(cloned.rank(), original.rank());
    assert_abs_diff_eq!(cloned.scale(), original.scale(), epsilon = 1e-10);
    assert_eq!(cloned.is_merged(), original.is_merged());

    // Verify tensor data matches
    for (a, b) in cloned.lora_a().data().iter().zip(original.lora_a().data().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
    for (a, b) in cloned.lora_b().data().iter().zip(original.lora_b().data().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }
    for (a, b) in cloned.base_weight().data().iter().zip(original.base_weight().data().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }

    // Verify independence: modifying clone doesn't affect original
    let x = Tensor::from_vec(vec![1.0, 1.0], true);
    let output_original = original.forward(&x);
    let output_cloned = cloned.forward(&x);
    for i in 0..2 {
        assert_abs_diff_eq!(output_original.data()[i], output_cloned.data()[i], epsilon = 1e-6);
    }
}

#[test]
fn test_lora_a_mut_modifies_weights() {
    // Explicitly covers lora_a_mut() accessor
    let base_weight = Tensor::from_vec(vec![1.0; 4], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // Verify initial A has small values (from sin init)
    let initial_a: Vec<f32> = lora.lora_a().data().to_vec();

    // Mutate through lora_a_mut
    *lora.lora_a_mut().data_mut() = ndarray::arr1(&[99.0, 99.0]);

    // Verify mutation took effect
    assert_abs_diff_eq!(lora.lora_a().data()[0], 99.0, epsilon = 1e-6);
    assert_abs_diff_eq!(lora.lora_a().data()[1], 99.0, epsilon = 1e-6);
    assert!(
        (lora.lora_a().data()[0] - initial_a[0]).abs() > 1.0,
        "lora_a_mut should have changed the values"
    );
}

#[test]
fn test_lora_b_mut_modifies_weights() {
    // Explicitly covers lora_b_mut() accessor
    let base_weight = Tensor::from_vec(vec![1.0; 4], false);
    let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

    // B is initialized to zeros
    assert_abs_diff_eq!(lora.lora_b().data()[0], 0.0, epsilon = 1e-10);

    // Mutate through lora_b_mut
    *lora.lora_b_mut().data_mut() = ndarray::arr1(&[42.0, 42.0]);

    // Verify mutation took effect
    assert_abs_diff_eq!(lora.lora_b().data()[0], 42.0, epsilon = 1e-6);
    assert_abs_diff_eq!(lora.lora_b().data()[1], 42.0, epsilon = 1e-6);
}

#[test]
fn test_lora_scaling_clone_and_eq() {
    // Covers Clone and PartialEq derives on LoRAScaling
    let standard = LoRAScaling::Standard;
    let rslora = LoRAScaling::RsLoRA;
    let standard_clone = standard;

    assert_eq!(standard, standard_clone);
    assert_ne!(standard, rslora);
    assert_eq!(rslora, LoRAScaling::RsLoRA);
}

#[test]
fn test_lora_scaling_debug() {
    // Covers Debug derive on LoRAScaling
    let s = format!("{:?}", LoRAScaling::Standard);
    assert!(s.contains("Standard"));
    let r = format!("{:?}", LoRAScaling::RsLoRA);
    assert!(r.contains("RsLoRA"));
}
