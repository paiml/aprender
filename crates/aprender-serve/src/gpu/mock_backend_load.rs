
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_is_always_available() {
        assert!(MockBackend::is_available());
    }

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockBackend::new(0).expect("Mock backend should always succeed");
        assert_eq!(backend.device_name(), "MockGPU (CPU fallback)");
        assert_eq!(backend.cached_weight_count(), 0);
    }

    #[test]
    fn test_load_weights() {
        let mut backend = MockBackend::new_mock();
        let weights = vec![1.0, 2.0, 3.0, 4.0];

        let handle = backend.load_weights("test", &weights).expect("handle");

        assert_eq!(handle, 4);
        assert!(backend.has_weights("test"));
        assert!(!backend.has_weights("nonexistent"));
        assert_eq!(backend.cached_weight_count(), 1);
        assert_eq!(backend.get_weights("test"), Some(&weights));
    }

    #[test]
    fn test_load_quantized_weights() {
        let mut backend = MockBackend::new_mock();
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];

        let handle = backend.load_quantized_weights("q4_test", &data, 2).expect("handle");

        assert_eq!(handle, 8);
        assert!(backend.has_weights("q4_test"));
        assert_eq!(backend.get_quantized_weights("q4_test"), Some(&data));
        assert_eq!(backend.get_quant_type("q4_test"), Some(2));
    }

    #[test]
    fn test_clear_weights() {
        let mut backend = MockBackend::new_mock();
        backend.load_weights("test", &[1.0, 2.0]).expect("load_weights");
        backend
            .load_quantized_weights("q_test", &[0u8; 18], 2)
            .expect("expected value");

        assert_eq!(backend.cached_weight_count(), 2);

        backend.clear_weights();

        assert_eq!(backend.cached_weight_count(), 0);
        assert!(!backend.has_weights("test"));
        assert!(!backend.has_weights("q_test"));
    }

    #[test]
    fn test_matmul_identity() {
        let mut backend = MockBackend::new_mock();

        // 2x2 identity @ 2x2 matrix = same matrix
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let matrix = vec![1.0, 2.0, 3.0, 4.0];

        let result = backend.matmul(&identity, &matrix, 2, 2, 2).expect("result");

        assert_eq!(result, matrix);
    }

    #[test]
    fn test_matmul_simple() {
        let mut backend = MockBackend::new_mock();

        // [1, 2] @ [[1], [2]] = [5]
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];

        let result = backend.matmul(&a, &b, 1, 2, 1).expect("result");

        assert_eq!(result.len(), 1);
        assert!((result[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let mut backend = MockBackend::new_mock();

        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[7, 8], [9, 10], [11, 12]] (3x2)
        // C = A @ B = [[58, 64], [139, 154]] (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = backend.matmul(&a, &b, 2, 3, 2).expect("result");

        assert_eq!(result.len(), 4);
        assert!((result[0] - 58.0).abs() < 1e-5);
        assert!((result[1] - 64.0).abs() < 1e-5);
        assert!((result[2] - 139.0).abs() < 1e-5);
        assert!((result[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_3x3_identity() {
        let mut backend = MockBackend::new_mock();

        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // B = identity
        // A @ B = A
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = backend.matmul(&a, &identity, 3, 3, 3).expect("result");

        for (i, (&expected, &actual)) in a.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_matmul_dimension_validation_a() {
        let mut backend = MockBackend::new_mock();

        let a = vec![1.0, 2.0]; // 2 elements
        let b = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements

        // This should fail: m=2, k=2 means a should have 4 elements
        let result = backend.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_dimension_validation_b() {
        let mut backend = MockBackend::new_mock();

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements (2x2)
        let b = vec![1.0, 2.0]; // 2 elements

        // This should fail: k=2, n=2 means b should have 4 elements
        let result = backend.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_cached() {
        let mut backend = MockBackend::new_mock();

        // Weight: 3x2 matrix stored as [k=3, n=2]
        // [[1, 2], [3, 4], [5, 6]]
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        backend.load_weights("w", &weight).expect("load_weights");

        // Input: 1x3 vector (m=1, k=3)
        let x = vec![1.0, 1.0, 1.0];

        // y = x @ W = [1*1+1*3+1*5, 1*2+1*4+1*6] = [9, 12]
        let result = backend.matmul_cached("w", &x, 1, 3, 2).expect("result");

        assert_eq!(result.len(), 2);
        assert!((result[0] - 9.0).abs() < 1e-6);
        assert!((result[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_cached_weight_not_found() {
        let mut backend = MockBackend::new_mock();

        let x = vec![1.0, 2.0, 3.0];
        let result = backend.matmul_cached("nonexistent", &x, 1, 3, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_q4k_gemv_cached() {
        let mut backend = MockBackend::new_mock();

        // Load quantized weight
        let q_data = vec![0u8; 256]; // Dummy quantized data
        backend
            .load_quantized_weights("q_weight", &q_data, 3)
            .expect("expected value"); // 3 = Q4_K

        let input = vec![1.0f32; 768];
        let result = backend
            .q4k_gemv_cached("q_weight", &input, 3072, 768)
            .expect("expected value");

        assert_eq!(result.len(), 3072);
        // Result should be finite (no NaN/Inf)
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_q4k_gemv_cached_not_found() {
        let mut backend = MockBackend::new_mock();

        let input = vec![1.0f32; 768];
        let result = backend.q4k_gemv_cached("nonexistent", &input, 3072, 768);

        assert!(result.is_err());
    }

    #[test]
    fn test_synchronize() {
        let backend = MockBackend::new_mock();

        // Should always succeed for mock backend
        backend.synchronize().expect("synchronize");
    }

    #[test]
    fn test_debug_impl() {
        let mut backend = MockBackend::new_mock();
        backend.load_weights("test", &[1.0, 2.0]).expect("load_weights");

        let debug_str = format!("{:?}", backend);

        assert!(debug_str.contains("MockBackend"));
        assert!(debug_str.contains("MockGPU"));
        assert!(debug_str.contains("num_weights"));
    }

    #[test]
    fn test_multiple_weights() {
        let mut backend = MockBackend::new_mock();

        backend.load_weights("w1", &[1.0, 2.0]).expect("load_weights");
        backend.load_weights("w2", &[3.0, 4.0]).expect("load_weights");
        backend.load_quantized_weights("q1", &[0u8; 18], 2).expect("load_quantized_weights");

        assert_eq!(backend.cached_weight_count(), 3);
        assert!(backend.has_weights("w1"));
        assert!(backend.has_weights("w2"));
        assert!(backend.has_weights("q1"));
    }

    #[test]
    fn test_overwrite_weight() {
        let mut backend = MockBackend::new_mock();

        backend.load_weights("test", &[1.0, 2.0]).expect("load_weights");
        assert_eq!(backend.get_weights("test"), Some(&vec![1.0, 2.0]));

        // Overwrite with different values
        backend.load_weights("test", &[3.0, 4.0, 5.0]).expect("load_weights");
        assert_eq!(backend.get_weights("test"), Some(&vec![3.0, 4.0, 5.0]));

        // Count should still be 1 (not 2)
        assert_eq!(backend.weights.len(), 1);
    }
}
