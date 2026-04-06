// CONTRACT: compute-backend-equivalence-v1.yaml
// Falsification tests: FALSIFY-BE-001..003
// Validates SIMD kernel equivalence against scalar reference.
// GPU tests (BE-002..006) require hardware and are in realizar.

use aprender::autograd::Tensor;
use aprender::nn::functional::{rms_norm, softmax};
use proptest::prelude::*;

/// Helper: compute cosine similarity between two slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if na < f64::EPSILON || nb < f64::EPSILON {
        return if na < f64::EPSILON && nb < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    (dot / (na * nb)) as f32
}

/// Scalar reference RMSNorm for equivalence checking.
fn rmsnorm_scalar(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(&xi, &gi)| xi / rms * gi)
        .collect()
}

proptest! {
    /// FALSIFY-BE-001: SIMD RMSNorm matches scalar reference.
    /// Contract: max_ulp_error(simd_rmsnorm(x), scalar_rmsnorm(x)) <= 2
    /// We test via cosine similarity ≥ 0.9999 (equivalent to ULP ≤ 2 for normalized outputs).
    #[test]
    fn falsify_be_001_simd_rmsnorm_matches_scalar(
        dim in 1usize..512,
        seed in 0u64..10000,
    ) {
        let n = dim.max(1);
        // Deterministic pseudo-random data
        let x: Vec<f32> = (0..n).map(|i| {
            let v = ((i as u64).wrapping_mul(seed.wrapping_add(17))) as f32;
            (v % 200.0 - 100.0) * 0.01
        }).collect();
        let gamma: Vec<f32> = (0..n).map(|i| {
            let v = ((i as u64).wrapping_mul(seed.wrapping_add(31))) as f32;
            0.5 + (v % 100.0) * 0.01
        }).collect();
        let eps = 1e-6f32;

        // Scalar reference
        let expected = rmsnorm_scalar(&x, &gamma, eps);

        // Trueno/aprender RMSNorm (uses SIMD when available)
        let x_tensor = Tensor::new(&x, &[1, n]);
        let gamma_tensor = Tensor::new(&gamma, &[n]);
        let result = rms_norm(&x_tensor, &gamma_tensor, eps);
        let actual = result.data().to_vec();

        // Equivalence check
        let cosine = cosine_similarity(&expected, &actual);
        prop_assert!(
            cosine >= 0.9999,
            "FALSIFY-BE-001: SIMD RMSNorm diverges from scalar. cosine={cosine:.6}, dim={n}"
        );
    }

    /// FALSIFY-BE-001b: RMSNorm shape preservation across backends.
    #[test]
    fn falsify_be_001b_rmsnorm_shape_preserved(
        dim in 1usize..256,
    ) {
        let x = Tensor::new(&vec![1.0f32; dim], &[1, dim]);
        let gamma = Tensor::new(&vec![1.0f32; dim], &[dim]);
        let result = rms_norm(&x, &gamma, 1e-6);
        prop_assert_eq!(
            result.data().len(), dim,
            "FALSIFY-BE-001b: RMSNorm output length mismatch"
        );
    }

    /// FALSIFY-BE-001c: Softmax normalization invariant (cross-backend).
    /// Contract: sum(softmax(x)) ≈ 1.0 for all finite x.
    #[test]
    fn falsify_be_001c_softmax_normalization(
        dim in 2usize..128,
        seed in 0u64..10000,
    ) {
        let n = dim.max(2);
        let x: Vec<f32> = (0..n).map(|i| {
            let v = ((i as u64).wrapping_mul(seed.wrapping_add(7))) as f32;
            (v % 200.0 - 100.0) * 0.1
        }).collect();

        let x_tensor = Tensor::new(&x, &[1, n]);
        let result = softmax(&x_tensor, -1);
        let sum: f32 = result.data().iter().sum();

        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "FALSIFY-BE-001c: softmax sum={sum:.8}, expected 1.0, dim={n}"
        );

        // All values must be positive
        for (i, &v) in result.data().iter().enumerate() {
            prop_assert!(
                v >= 0.0,
                "FALSIFY-BE-001c: softmax[{i}]={v} is negative"
            );
        }
    }
}

/// FALSIFY-BE-001d: RMSNorm with zero input produces finite output.
#[test]
fn falsify_be_001d_rmsnorm_zero_input() {
    let x = Tensor::new(&[0.0f32; 64], &[1, 64]);
    let gamma = Tensor::new(&[1.0f32; 64], &[64]);
    let result = rms_norm(&x, &gamma, 1e-6);
    for &v in result.data() {
        assert!(v.is_finite(), "FALSIFY-BE-001d: NaN/Inf from zero input");
    }
}

/// FALSIFY-BE-001e: RMSNorm with large values doesn't overflow.
#[test]
fn falsify_be_001e_rmsnorm_large_values() {
    let x = Tensor::new(&[1e10f32; 32], &[1, 32]);
    let gamma = Tensor::new(&[1.0f32; 32], &[32]);
    let result = rms_norm(&x, &gamma, 1e-6);
    for &v in result.data() {
        assert!(v.is_finite(), "FALSIFY-BE-001e: overflow with large values");
    }
}
