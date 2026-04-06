
// ============================================================================
// Tests for SIMD Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // simd_dot_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_f32_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = simd_dot_f32(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_zeros() {
        let a = vec![0.0f32; 16];
        let b = vec![1.0f32; 16];
        let result = simd_dot_f32(&a, &b);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_ones() {
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let result = simd_dot_f32(&a, &b);
        assert!((result - 16.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_large_vector() {
        // Test with vector size > 8 to exercise AVX2 path
        let n = 64;
        let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let b = vec![1.0f32; n];
        let result = simd_dot_f32(&a, &b);
        // Sum of 1 to 64 = 64*65/2 = 2080
        assert!((result - 2080.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_remainder() {
        // Test with size not divisible by 8 to exercise remainder handling
        let n = 13;
        let a = vec![2.0f32; n];
        let b = vec![3.0f32; n];
        let result = simd_dot_f32(&a, &b);
        // 2*3*13 = 78
        assert!((result - 78.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_small_vector() {
        // Test with size < 8 to exercise scalar fallback
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = simd_dot_f32(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_f32_negative() {
        let a = vec![-1.0f32, 2.0, -3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let result = simd_dot_f32(&a, &b);
        // -1 + 2 - 3 + 4 = 2
        assert!((result - 2.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // simd_add_weighted Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_add_weighted_basic() {
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let val = vec![10.0f32, 20.0, 30.0, 40.0];
        simd_add_weighted(&mut out, &val, 0.5);
        // out[i] = out[i] + 0.5 * val[i]
        assert!((out[0] - 6.0).abs() < 0.001); // 1 + 0.5*10 = 6
        assert!((out[1] - 12.0).abs() < 0.001); // 2 + 0.5*20 = 12
        assert!((out[2] - 18.0).abs() < 0.001); // 3 + 0.5*30 = 18
        assert!((out[3] - 24.0).abs() < 0.001); // 4 + 0.5*40 = 24
    }

    #[test]
    fn test_simd_add_weighted_zero_weight() {
        let mut out = vec![1.0f32, 2.0, 3.0, 4.0];
        let val = vec![100.0f32; 4];
        simd_add_weighted(&mut out, &val, 0.0);
        // out should remain unchanged
        assert!((out[0] - 1.0).abs() < 0.001);
        assert!((out[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_add_weighted_large_vector() {
        // Test with vector size > 8 to exercise AVX2 path
        let n = 64;
        let mut out = vec![1.0f32; n];
        let val = vec![2.0f32; n];
        simd_add_weighted(&mut out, &val, 3.0);
        // out[i] = 1 + 3*2 = 7
        for &v in &out {
            assert!((v - 7.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_add_weighted_remainder() {
        // Test with size not divisible by 8
        let n = 11;
        let mut out = vec![0.0f32; n];
        let val: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        simd_add_weighted(&mut out, &val, 2.0);
        // out[i] = 0 + 2*val[i]
        for (i, &v) in out.iter().enumerate() {
            let expected = 2.0 * (i + 1) as f32;
            assert!((v - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_add_weighted_small_vector() {
        // Test with size < 8 to exercise scalar fallback
        let mut out = vec![5.0f32, 10.0];
        let val = vec![1.0f32, 1.0];
        simd_add_weighted(&mut out, &val, -2.0);
        // out[0] = 5 + (-2)*1 = 3
        // out[1] = 10 + (-2)*1 = 8
        assert!((out[0] - 3.0).abs() < 0.001);
        assert!((out[1] - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_add_weighted_accumulate() {
        // Test multiple accumulations
        let mut out = vec![0.0f32; 16];
        let val1 = vec![1.0f32; 16];
        let val2 = vec![2.0f32; 16];

        simd_add_weighted(&mut out, &val1, 1.0);
        simd_add_weighted(&mut out, &val2, 0.5);
        // out = 0 + 1*1 + 0.5*2 = 2
        for &v in &out {
            assert!((v - 2.0).abs() < 0.001);
        }
    }
}
