//! Shared helpers for Phase 5 falsification tests.

#![allow(dead_code)]

/// Asserts every element in a slice is finite (not NaN or infinite).
pub fn assert_all_finite(slice: &[f32]) {
    for (i, &val) in slice.iter().enumerate() {
        assert!(val.is_finite(), "Element [{i}] is not finite: {val}");
    }
}

/// Asserts slice forms a valid probability distribution: all >= 0 and sum ≈ 1.
pub fn assert_probability_distribution(slice: &[f32], tol: f32) {
    let sum: f32 = slice.iter().sum();
    assert!(
        (sum - 1.0).abs() < tol,
        "Distribution sums to {sum}, expected ≈1.0 (tol={tol})"
    );
    for (i, &val) in slice.iter().enumerate() {
        assert!(val >= 0.0, "Element [{i}] is negative: {val}");
    }
}

/// L2 norm of a vector.
pub fn l2_norm(slice: &[f32]) -> f32 {
    slice.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L2 distance between two vectors.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Dot product of two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Generate a random stochastic row (sums to 1, all >= 0) of size `n`.
#[allow(clippy::cast_precision_loss)]
pub fn random_stochastic_row(n: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 13) as f32 + 1.0).collect();
    let sum: f32 = raw.iter().sum();
    raw.iter().map(|x| x / sum).collect()
}

/// Generate a column-major stochastic transition matrix of size n x n.
#[allow(clippy::cast_precision_loss)]
pub fn stochastic_transition_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; n * n];
    for col in 0..n {
        let mut sum = 0.0f32;
        for row in 0..n {
            let val = ((row * 7 + col * 13 + 3) % 11) as f32 + 1.0;
            mat[col * n + row] = val;
            sum += val;
        }
        for row in 0..n {
            mat[col * n + row] /= sum;
        }
    }
    mat
}

/// Generate an identity matrix (row-major) of size n x n.
pub fn identity_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Arithmetic mean of a slice.
#[allow(clippy::cast_precision_loss)]
pub fn mean(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().sum::<f32>() / slice.len() as f32
}

/// Population variance of a slice.
#[allow(clippy::cast_precision_loss)]
pub fn variance(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let m = mean(slice);
    slice.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / slice.len() as f32
}

/// Maximum absolute ULP difference between two slices.
#[allow(clippy::cast_possible_wrap)]
pub fn max_ulp_distance(a: &[f32], b: &[f32]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            if x.is_nan() || y.is_nan() {
                return u32::MAX;
            }
            let xi = x.to_bits() as i32;
            let yi = y.to_bits() as i32;
            (xi.wrapping_sub(yi)).unsigned_abs()
        })
        .max()
        .unwrap_or(0)
}
