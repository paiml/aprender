//! Property-based tests for matmul operations

use super::test_utils::finite_difference;
use crate::autograd::{backward, matmul, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_matmul_backward_gradient_check(
        // Generate random matrix dimensions (smaller to reduce accumulated float errors)
        m in 2usize..5,
        k in 2usize..5,
        n in 2usize..5,
        // Generate seed for random data
        seed in 0u64..1000,
    ) {
        // Generate random matrices A (m x k) and B (k x n) deterministically
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let base = hasher.finish();

        let a_data: Vec<f32> = (0..m*k).map(|i| {
            ((base.wrapping_add(i as u64) % 1000) as f32 / 100.0) - 5.0
        }).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| {
            ((base.wrapping_add((m*k + i) as u64) % 1000) as f32 / 100.0) - 5.0
        }).collect();

        let a = Tensor::from_vec(a_data.clone(), true);
        let b = Tensor::from_vec(b_data.clone(), true);
        let mut c = matmul(&a, &b, m, k, n);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().expect("gradient should be available");

        // Numerical gradient for A
        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(b_data.clone(), false);
                let t_c = matmul(&t_a, &t_b, m, k, n);
                t_c.data().sum()
            },
            &a_data,
            1e-3,
        );

        // Check gradient with tolerance (slightly higher for accumulated float errors in larger matrices)
        for i in 0..a_data.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.2,
                "Gradient mismatch at index {}: m={}, k={}, n={}, analytical={}, numerical={}, diff={}",
                i, m, k, n, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_matmul_dimensions(
        m in 1usize..10,
        k in 1usize..10,
        n in 1usize..10,
    ) {
        let a = Tensor::from_vec(vec![1.0; m * k], false);
        let b = Tensor::from_vec(vec![1.0; k * n], false);
        let c = matmul(&a, &b, m, k, n);

        // Output should be m x n
        prop_assert_eq!(c.len(), m * n);
    }
}
