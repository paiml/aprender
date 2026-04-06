use super::super::super::*;

// Tests for zscore() - Z-score normalization (standardization)
// ========================================================================

#[test]
fn test_zscore_basic() {
    // [1, 2, 3, 4, 5] has mean=3, stddev=sqrt(2)
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let z = v.zscore().unwrap();

    // Verify mean ≈ 0
    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5, "mean = {}, expected ≈ 0", mean);

    // Verify stddev ≈ 1
    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5, "stddev = {}, expected ≈ 1", std);
}

#[test]
fn test_zscore_negative_values() {
    // [-2, -1, 0, 1, 2] has mean=0, stddev=sqrt(2)
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let z = v.zscore().unwrap();

    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5);

    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5);
}

#[test]
fn test_zscore_single_element() {
    // Single element has zero stddev → DivisionByZero
    let v = Vector::from_slice(&[5.0]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_zscore_constant_vector() {
    // All identical elements have zero stddev → DivisionByZero
    let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_zscore_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_zscore_already_normalized() {
    // Vector already with mean≈0, std≈1 should stay similar
    let v = Vector::from_slice(&[-1.0, 0.0, 1.0]);
    let z = v.zscore().unwrap();

    // Should be close to the original (scaling might differ slightly)
    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5);

    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5);
}

// ========================================================================
// Tests for minmax_normalize() - Min-max normalization to [0, 1]
// ========================================================================

#[test]
fn test_minmax_normalize_basic() {
    // [1, 2, 3, 4, 5] → [0, 0.25, 0.5, 0.75, 1.0]
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let normalized = v.minmax_normalize().unwrap();

    // Verify min = 0
    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5, "min = {}, expected 0", min);

    // Verify max = 1
    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5, "max = {}, expected 1", max);

    // Verify specific values
    assert!((normalized.data[0] - 0.0).abs() < 1e-5);
    assert!((normalized.data[2] - 0.5).abs() < 1e-5);
    assert!((normalized.data[4] - 1.0).abs() < 1e-5);
}

#[test]
fn test_minmax_normalize_negative_values() {
    // [-10, -5, 0, 5, 10] → [0, 0.25, 0.5, 0.75, 1.0]
    let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
    let normalized = v.minmax_normalize().unwrap();

    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5);

    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5);

    // Middle value should be 0.5
    assert!((normalized.data[2] - 0.5).abs() < 1e-5);
}

#[test]
fn test_minmax_normalize_single_element() {
    // Single element has zero range → DivisionByZero
    let v = Vector::from_slice(&[5.0]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_minmax_normalize_constant_vector() {
    // All identical elements have zero range → DivisionByZero
    let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_minmax_normalize_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_minmax_normalize_already_normalized() {
    // Vector already in [0, 1] should stay in [0, 1]
    let v = Vector::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    let normalized = v.minmax_normalize().unwrap();

    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5);

    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5);
}

// ========================================================================
// Tests for layer_norm() - Layer normalization (Issue #61)
// ========================================================================

#[test]
fn test_layer_norm_basic() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Output should have mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

    // Output should have variance ≈ 1
    let var: f32 = y.as_slice().iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / y.len() as f32;
    assert!((var - 1.0).abs() < 1e-3, "Variance should be ~1, got {}", var);
}

#[test]
fn test_layer_norm_with_scale_shift() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let gamma = Vector::from_slice(&[2.0, 2.0, 2.0, 2.0]); // Scale by 2
    let beta = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]); // Shift by 1

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Output should have mean ≈ 1 (beta)
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!((mean - 1.0).abs() < 1e-3, "Mean should be ~1, got {}", mean);

    // Output should have std ≈ 2 (gamma)
    let var: f32 = y.as_slice().iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / y.len() as f32;
    let std = var.sqrt();
    assert!((std - 2.0).abs() < 1e-3, "Std should be ~2, got {}", std);
}

#[test]
fn test_layer_norm_empty_vector() {
    let x: Vector<f32> = Vector::from_slice(&[]);
    let gamma = Vector::from_slice(&[]);
    let beta = Vector::from_slice(&[]);

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_layer_norm_size_mismatch_gamma() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0]); // Wrong size
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0]);

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_layer_norm_size_mismatch_beta() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0]); // Wrong size

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_layer_norm_simple() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = x.layer_norm_simple(1e-5).unwrap();

    // Output should have mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
}

#[test]
fn test_layer_norm_constant_input() {
    // Constant input [5, 5, 5, 5] should produce zeros (with gamma=1, beta=0)
    let x = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // All outputs should be 0 (or very close due to eps)
    for &v in y.as_slice() {
        assert!(v.abs() < 1e-3, "Expected ~0, got {}", v);
    }
}

#[test]
fn test_layer_norm_single_element() {
    // Single element should normalize to 0 (x - mean = 0)
    let x = Vector::from_slice(&[42.0]);
    let gamma = Vector::from_slice(&[1.0]);
    let beta = Vector::from_slice(&[0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    assert!(y.as_slice()[0].abs() < 1e-3);
}

#[test]
fn test_layer_norm_negative_values() {
    let x = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let gamma = Vector::from_slice(&[1.0; 5]);
    let beta = Vector::from_slice(&[0.0; 5]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Should still produce mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5);
}

// ========================================================================
