use super::*;
use crate::backends::scalar::ScalarBackend;

mod binary_ops;
mod large_tests;
mod reductions;
mod unary_transforms;

fn avx512_test<F>(test_fn: F)
where
    F: FnOnce(),
{
    if is_x86_feature_detected!("avx512f") {
        test_fn();
    } else {
        println!("Skipping AVX-512 test (CPU does not support avx512f)");
    }
}

/// Helper: test a binary element-wise op (`a OP b => expected` for all elements).
fn assert_binary_op(
    a_val: f32,
    b_val: f32,
    expected: f32,
    op: unsafe fn(&[f32], &[f32], &mut [f32]),
) {
    let a = vec![a_val; 32];
    let b = vec![b_val; 32];
    let mut result = vec![0.0; 32];
    // SAFETY: test-only; vectors are identically sized, backend selected by caller
    unsafe { op(&a, &b, &mut result) };
    assert!(
        result.iter().all(|&x| (x - expected).abs() < 1e-6),
        "expected all {expected}, got {:?}",
        &result[..4]
    );
}

/// Helper: test a unary transform against per-element expected values.
fn assert_unary_transform(
    input: &[f32],
    expected: &[f32],
    tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(input, &mut result) };
    for (i, (&val, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((val - exp).abs() < tol, "mismatch at {i}: got {val}, expected {exp}");
    }
}

/// Helper: test a unary transform on a large array against a scalar reference function.
fn assert_unary_large(
    input: Vec<f32>,
    tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
    reference_fn: fn(f32) -> f32,
    label: &str,
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(&input, &mut result) };
    for (i, &val) in result.iter().enumerate() {
        let expected = reference_fn(input[i]);
        assert!((val - expected).abs() < tol, "{label} large mismatch at {i}: {val} vs {expected}");
    }
}

/// Helper: test a unary transform on a large array using relative error tolerance.
fn assert_unary_large_relative(
    input: Vec<f32>,
    rel_tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
    reference_fn: fn(f32) -> f32,
    label: &str,
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(&input, &mut result) };
    for (i, &val) in result.iter().enumerate() {
        let expected = reference_fn(input[i]);
        assert!(
            (val - expected).abs() / expected.max(1e-6) < rel_tol,
            "{label} mismatch at {i}: {val} vs {expected}"
        );
    }
}

/// Helper: test a scalar reduction on sequential 1..=32 input.
// SAFETY: caller verifies SSE2 support, input slices meet alignment/length requirements
fn assert_reduction_f32(expected: f32, tol: f32, op: unsafe fn(&[f32]) -> f32) {
    let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { op(&a) };
    assert!((result - expected).abs() < tol, "expected {expected}, got {result}");
}

/// Helper: test an index-returning reduction on sequential 1..=32 input.
// SAFETY: caller verifies SSE2 support, input slices meet alignment/length requirements
fn assert_reduction_usize(expected: usize, op: unsafe fn(&[f32]) -> usize) {
    let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
    let result = unsafe { op(&a) };
    assert_eq!(result, expected);
}
