
// ============================================================================
// EXTREME TDD: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_bf16_to_f32_various_values() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 2.0, -0.5, 100.0, -100.0];
        let mut bf16_bytes = Vec::new();
        for &v in &values {
            bf16_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }

        let result = simd_bf16_to_f32(&bf16_bytes);
        assert_eq!(result.len(), values.len());

        for (i, (&expected, &actual)) in values.iter().zip(result.iter()).enumerate() {
            // BF16 has limited precision, allow some tolerance
            let tol = expected.abs().max(1.0) * 0.01;
            assert!(
                (actual - expected).abs() < tol,
                "Value {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_simd_bf16_to_f32_large_batch() {
        // Test with more than 8 values to verify SIMD remainder handling
        let count = 17; // 2 SIMD chunks (8+8) + 1 remainder
        let mut bf16_bytes = Vec::with_capacity(count * 2);
        for i in 0..count {
            let v = i as f32 * 0.1;
            bf16_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }

        let result = simd_bf16_to_f32(&bf16_bytes);
        assert_eq!(result.len(), count);

        for i in 0..count {
            let expected = i as f32 * 0.1;
            let tol = 0.01;
            assert!(
                (result[i] - expected).abs() < tol,
                "Index {} mismatch: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_simd_f16_to_f32_single() {
        let f16_bytes = half::f16::from_f32(1.0).to_le_bytes();
        let result = simd_f16_to_f32(&f16_bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_simd_f16_to_f32_various_values() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 2.0, -0.5];
        let mut f16_bytes = Vec::new();
        for &v in &values {
            f16_bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }

        let result = simd_f16_to_f32(&f16_bytes);
        assert_eq!(result.len(), values.len());

        for (expected, actual) in values.iter().zip(result.iter()) {
            // F16 has limited precision
            assert!(
                (actual - expected).abs() < 0.01,
                "Mismatch: expected {}, got {}",
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_simd_bf16_dot_basic() {
        // Create two simple BF16 vectors
        let a_vals = [1.0f32, 2.0, 3.0, 4.0];
        let b_vals = [1.0f32, 1.0, 1.0, 1.0];

        let mut a_bytes = Vec::new();
        let mut b_bytes = Vec::new();
        for (&a, &b) in a_vals.iter().zip(b_vals.iter()) {
            a_bytes.extend_from_slice(&half::bf16::from_f32(a).to_le_bytes());
            b_bytes.extend_from_slice(&half::bf16::from_f32(b).to_le_bytes());
        }

        let result = simd_bf16_dot(&a_bytes, &b_bytes);
        // 1+2+3+4 = 10
        assert!((result - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_simd_bf16_dot_large() {
        // Test with many chunks to exercise chunked processing
        let n = 256; // 4 chunks of 64
        let mut a_bytes = Vec::with_capacity(n * 2);
        let mut b_bytes = Vec::with_capacity(n * 2);

        for i in 0..n {
            let v = ((i % 10) as f32) * 0.1;
            a_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            b_bytes.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
        }

        let result = simd_bf16_dot(&a_bytes, &b_bytes);
        // Sum of 0.0, 0.1, 0.2, ..., 0.9, 0.0, 0.1, ... (26 complete cycles)
        // Each cycle sums to 0+0.1+0.2+...+0.9 = 4.5
        // 256/10 = 25 full cycles + 6 remainder (0.0+0.1+0.2+0.3+0.4+0.5 = 1.5)
        let expected = 25.0 * 4.5 + 1.5;
        assert!(
            (result - expected).abs() < 1.0,
            "Large dot: expected ~{}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_simd_bf16_matmul_identity() {
        // 3x3 identity in BF16
        let input = vec![1.0f32, 2.0, 3.0];
        let mut weight_bytes = Vec::new();
        let identity = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for row in &identity {
            for &v in row {
                weight_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }

        let output = simd_bf16_matmul(&input, &weight_bytes, 3, 3);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[1] - 2.0).abs() < 0.01);
        assert!((output[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_simd_bf16_matmul_projection() {
        // 2x4 projection matrix
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = [
            [1.0f32, 1.0, 1.0, 1.0], // row 0: sum = 10
            [1.0, 0.0, 0.0, -1.0],   // row 1: 1-4 = -3
        ];
        let mut weight_bytes = Vec::new();
        for row in &weight {
            for &v in row {
                weight_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }

        let output = simd_bf16_matmul(&input, &weight_bytes, 4, 2);
        assert_eq!(output.len(), 2);
        assert!(
            (output[0] - 10.0).abs() < 0.1,
            "Sum: expected 10, got {}",
            output[0]
        );
        assert!(
            (output[1] - (-3.0)).abs() < 0.1,
            "Diff: expected -3, got {}",
            output[1]
        );
    }
include!("simd_matmul.rs");
}
