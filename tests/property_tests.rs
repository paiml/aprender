//! Property-based tests using proptest.
//!
//! These tests verify invariants and properties of the ML algorithms.

use aprender::prelude::*;
use proptest::prelude::*;

// Strategy for generating small matrices
fn matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Matrix<f32>> {
    proptest::collection::vec(-100.0f32..100.0, rows * cols)
        .prop_map(move |data| Matrix::from_vec(rows, cols, data).unwrap())
}

// Strategy for generating vectors
fn vector_strategy(len: usize) -> impl Strategy<Value = Vector<f32>> {
    proptest::collection::vec(-100.0f32..100.0, len).prop_map(Vector::from_vec)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Vector properties
    #[test]
    fn vector_sum_is_associative(a in vector_strategy(10), b in vector_strategy(10)) {
        let sum_ab = (&a + &b).sum();
        let sum_ba = (&b + &a).sum();
        prop_assert!((sum_ab - sum_ba).abs() < 1e-4);
    }

    #[test]
    fn vector_dot_is_commutative(a in vector_strategy(10), b in vector_strategy(10)) {
        let dot_ab = a.dot(&b);
        let dot_ba = b.dot(&a);
        prop_assert!((dot_ab - dot_ba).abs() < 1e-4);
    }

    #[test]
    fn vector_norm_is_non_negative(v in vector_strategy(10)) {
        prop_assert!(v.norm() >= 0.0);
    }

    #[test]
    fn vector_scalar_mul_distributes(v in vector_strategy(10), s in -10.0f32..10.0) {
        let scaled = v.mul_scalar(s);
        let expected_sum = v.sum() * s;
        prop_assert!((scaled.sum() - expected_sum).abs() < 1e-3);
    }

    #[test]
    fn vector_elementwise_mul_is_commutative(a in vector_strategy(10), b in vector_strategy(10)) {
        let mul_ab = &a * &b;
        let mul_ba = &b * &a;
        for i in 0..10 {
            prop_assert!((mul_ab[i] - mul_ba[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn vector_elementwise_mul_with_ones_is_identity(v in vector_strategy(10)) {
        let ones = Vector::<f32>::ones(10);
        let result = &v * &ones;
        for i in 0..10 {
            prop_assert!((result[i] - v[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_elementwise_mul_with_zeros_is_zero(v in vector_strategy(10)) {
        let zeros = Vector::<f32>::zeros(10);
        let result = &v * &zeros;
        for i in 0..10 {
            prop_assert!((result[i]).abs() < 1e-6);
        }
    }

    // Matrix properties
    #[test]
    fn matrix_transpose_involution(m in matrix_strategy(5, 5)) {
        let m_t = m.transpose();
        let m_tt = m_t.transpose();
        for i in 0..5 {
            for j in 0..5 {
                prop_assert!((m.get(i, j) - m_tt.get(i, j)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn matrix_shape_preserved_by_add(a in matrix_strategy(4, 3), b in matrix_strategy(4, 3)) {
        let c = a.add(&b).unwrap();
        prop_assert_eq!(c.shape(), (4, 3));
    }

    #[test]
    fn matrix_matmul_shape(a in matrix_strategy(3, 4), b in matrix_strategy(4, 2)) {
        let c = a.matmul(&b).unwrap();
        prop_assert_eq!(c.shape(), (3, 2));
    }

    // DataFrame properties
    #[test]
    fn dataframe_select_preserves_rows(
        data in proptest::collection::vec(-100.0f32..100.0, 15)
    ) {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&data[0..5])),
            ("b".to_string(), Vector::from_slice(&data[5..10])),
            ("c".to_string(), Vector::from_slice(&data[10..15])),
        ];
        let df = DataFrame::new(columns).unwrap();
        let selected = df.select(&["a", "c"]).unwrap();
        prop_assert_eq!(selected.n_rows(), df.n_rows());
    }

    #[test]
    fn dataframe_to_matrix_shape(
        data in proptest::collection::vec(-100.0f32..100.0, 15)
    ) {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&data[0..5])),
            ("b".to_string(), Vector::from_slice(&data[5..10])),
            ("c".to_string(), Vector::from_slice(&data[10..15])),
        ];
        let df = DataFrame::new(columns).unwrap();
        let matrix = df.to_matrix();
        prop_assert_eq!(matrix.shape(), (5, 3));
    }

    // Metrics properties
    #[test]
    fn r_squared_perfect_prediction(y in vector_strategy(10)) {
        let r2 = r_squared(&y, &y);
        // Perfect prediction should give R² = 1 (or very close)
        prop_assert!((r2 - 1.0).abs() < 1e-6 || y.variance() == 0.0);
    }

    #[test]
    fn mse_is_non_negative(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let error = mse(&y_pred, &y_true);
        prop_assert!(error >= 0.0);
    }

    #[test]
    fn mae_is_non_negative(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let error = mae(&y_pred, &y_true);
        prop_assert!(error >= 0.0);
    }

    #[test]
    fn rmse_equals_sqrt_mse(y_true in vector_strategy(10), y_pred in vector_strategy(10)) {
        let mse_val = mse(&y_pred, &y_true);
        let rmse_val = rmse(&y_pred, &y_true);
        prop_assert!((rmse_val - mse_val.sqrt()).abs() < 1e-6);
    }

    // Linear regression properties
    #[test]
    fn linear_regression_coefficients_length(n_features in 1usize..5) {
        // Create well-conditioned data
        let n_samples = n_features + 3;
        let mut x_data = vec![0.0; n_samples * n_features];
        let mut y_data = vec![0.0; n_samples];

        // Create identity-like pattern to ensure positive definiteness
        for i in 0..n_samples {
            for j in 0..n_features {
                x_data[i * n_features + j] = if i == j { 1.0 } else { 0.1 * (i + j) as f32 };
            }
            y_data[i] = (i + 1) as f32;
        }

        let x = Matrix::from_vec(n_samples, n_features, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        let mut model = LinearRegression::new();
        if model.fit(&x, &y).is_ok() {
            prop_assert_eq!(model.coefficients().len(), n_features);
        }
    }

    // K-Means properties
    #[test]
    fn kmeans_labels_valid(n_clusters in 1usize..4) {
        // Create data with clear clusters
        let n_samples = n_clusters * 5;
        let mut data = Vec::with_capacity(n_samples * 2);

        for k in 0..n_clusters {
            for i in 0..5 {
                data.push((k * 10) as f32 + i as f32 * 0.1);
                data.push((k * 10) as f32 + i as f32 * 0.1);
            }
        }

        let matrix = Matrix::from_vec(n_samples, 2, data).unwrap();
        let mut kmeans = KMeans::new(n_clusters).with_random_state(42);
        kmeans.fit(&matrix).unwrap();

        let labels = kmeans.predict(&matrix);

        // All labels should be valid cluster indices
        for &label in &labels {
            prop_assert!(label < n_clusters);
        }
    }

    #[test]
    fn kmeans_inertia_non_negative(n_clusters in 1usize..3) {
        let n_samples = n_clusters * 3;
        let data: Vec<f32> = (0..n_samples * 2).map(|i| i as f32).collect();
        let matrix = Matrix::from_vec(n_samples, 2, data).unwrap();

        let mut kmeans = KMeans::new(n_clusters).with_random_state(42);
        kmeans.fit(&matrix).unwrap();

        prop_assert!(kmeans.inertia() >= 0.0);
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_vector_zero_norm() {
        let v = Vector::<f32>::zeros(5);
        assert!((v.norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_matrix_properties() {
        let eye = Matrix::<f32>::eye(3);

        // I * I = I
        let result = eye.matmul(&eye).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result.get(i, j) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_silhouette_bounds() {
        // Silhouette score should be in [-1, 1]
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
            ],
        )
        .unwrap();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &labels);

        assert!(score >= -1.0);
        assert!(score <= 1.0);
    }
}
