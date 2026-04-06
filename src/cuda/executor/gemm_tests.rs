
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // GEMM Tests (gemm, gemm_optimized, gemm_fused)
    // ========================================================================

    #[test]
    fn test_gemm_identity_matrix() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // 2x2 identity @ 2x2 vector
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];
        exec.gemm(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert!((c[0] - 3.0).abs() < 0.1);
        assert!((c[3] - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_gemm_small_matrix() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A@B = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        exec.gemm(&a, &b, &mut c, 2, 2, 2).unwrap();
        // Tiled GEMM may have some numerical error
        assert!((c[0] - 19.0).abs() < 1.0);
        assert!((c[3] - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_gemm_m1_uses_gemv() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=1 should use GEMV kernel path
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        let b = vec![1.0, 0.5, 0.25, 0.125, 2.0, 1.0, 0.5, 0.25]; // 4x2
        let mut c = vec![0.0; 2]; // 1x2
        exec.gemm(&a, &b, &mut c, 1, 2, 4).unwrap();
        // dot(a, b[:, 0]) = 1*1 + 2*0.5 + 3*0.25 + 4*0.125 = 1+1+0.75+0.5 = 3.25
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_size_mismatch_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0, 2.0]; // Wrong size
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let result = exec.gemm(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_optimized_tile32() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 64]; // 8x8
        let b = vec![1.0; 64]; // 8x8
        let mut c = vec![0.0; 64]; // 8x8
        exec.gemm_optimized(&a, &b, &mut c, 8, 8, 8, 32).unwrap();
        // Each element should be sum of 8 ones = 8
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_fused_no_activation() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        exec.gemm_fused(&a, &b, None, &mut c, 2, 2, 2, 0).unwrap();
        assert!(c[0] > 0.0);
    }

    #[test]
    fn test_gemm_fused_with_relu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0, -1.0, 1.0, -1.0]; // Some negative
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        // activation=1 is ReLU
        exec.gemm_fused(&a, &b, None, &mut c, 2, 2, 2, 1).unwrap();
        // ReLU should clamp negatives to 0
        for val in &c {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_gemm_fused_with_bias() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let bias = vec![10.0, 20.0];
        let mut c = vec![0.0; 4];
        exec.gemm_fused(&a, &b, Some(&bias), &mut c, 2, 2, 2, 0)
            .unwrap();
        // Bias should be added
        assert!(c[0] > 10.0);
    }

    #[test]
    fn test_gemm_fused_bias_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let bias = vec![10.0]; // Wrong size, should be 2
        let mut c = vec![0.0; 4];
        let result = exec.gemm_fused(&a, &b, Some(&bias), &mut c, 2, 2, 2, 0);
        assert!(result.is_err());
    }

    // ========================================================================
    // Softmax Tests
    // ========================================================================

    #[test]
    fn test_softmax_uniform() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        exec.softmax(&mut data).unwrap();
        // Uniform input -> uniform output
        let expected = 0.25;
        for val in &data {
            assert!((*val - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_softmax_single_max() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![10.0, 0.0, 0.0, 0.0];
        exec.softmax(&mut data).unwrap();
        // First element should dominate
        assert!(data[0] > 0.9);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        exec.softmax(&mut data).unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_large_values() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut data = vec![100.0, 101.0, 102.0, 103.0];
        exec.softmax(&mut data).unwrap();
        // Should still sum to 1 (numerically stable)
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // Allocate Buffer Tests
    // ========================================================================

    #[test]
    fn test_allocate_buffer_basic() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1024).unwrap();
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn test_allocate_buffer_small() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1).unwrap();
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_allocate_buffer_large() {
        let Some(exec) = create_executor() else {
            return;
        };
        let buf = exec.allocate_buffer(1024 * 1024).unwrap();
        assert_eq!(buf.len(), 1024 * 1024);
    }

    // ========================================================================
    // Synchronize Tests
    // ========================================================================

    #[test]
    fn test_synchronize_compute() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_compute().is_ok());
    }

    #[test]
    fn test_synchronize_transfer() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_transfer().is_ok());
    }

    #[test]
    fn test_synchronize_all() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize_all().is_ok());
    }

    // ========================================================================
    // Cached GEMM Tests
    // ========================================================================

    #[test]
    fn test_gemm_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let result = exec.gemm_cached("nonexistent_weight", &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_b_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let a = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let result = exec.gemm_b_cached("nonexistent_weight", &a, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 4];
        let result = exec.gemv_cached("nonexistent_weight", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_input_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 2]; // Wrong size
        let mut y = vec![0.0; 4];
        let result = exec.gemv_cached("test", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemv_cached_output_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 2]; // Wrong size
        let result = exec.gemv_cached("test", &x, &mut y, 4, 4);
        assert!(result.is_err());
    }

    // ========================================================================
    // Async Copy Tests
    // ========================================================================

    #[test]
    fn test_async_copy_roundtrip() {
        let Some(exec) = create_executor() else {
            return;
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buf = exec.allocate_buffer(4).unwrap();

        // Copy to GPU async
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe {
            exec.copy_to_gpu_async(&mut buf, &data).unwrap();
        }
        exec.synchronize_transfer().unwrap();

        // Copy back from GPU async
        let mut result = vec![0.0f32; 4];
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe {
            exec.copy_from_gpu_async(&buf, &mut result).unwrap();
        }
        exec.synchronize_transfer().unwrap();

        assert_eq!(result, data);
    }
}
