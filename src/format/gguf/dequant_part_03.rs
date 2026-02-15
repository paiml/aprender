
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_positive_zero() {
        let result = f16_to_f32(0x0000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_negative_zero() {
        let result = f16_to_f32(0x8000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_positive_infinity() {
        let result = f16_to_f32(0x7C00);
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_negative_infinity() {
        let result = f16_to_f32(0xFC00);
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_nan() {
        // NaN has exp=31 and non-zero mantissa
        let result = f16_to_f32(0x7C01);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_subnormal() {
        // Smallest subnormal: 0x0001 = 2^-24 ~= 5.96e-8
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 1e-4);
    }

    #[test]
    fn test_f16_normal_one() {
        // f16 representation of 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_normal_negative() {
        // f16 representation of -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    // =========================================================================
    // Dequantize Q4_0
    // =========================================================================

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Build a minimal Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes per block of 32
        let mut data = vec![0u8; 18];
        // Scale = 1.0 in f16 = 0x3C00
        data[0] = 0x00;
        data[1] = 0x3C;
        // Fill 16 quant bytes with 0x88 (both nibbles = 8, so value = 8-8 = 0)
        for i in 2..18 {
            data[i] = 0x88;
        }

        let result = dequantize_q4_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
        // All values should be 0 (quant value 8 - 8 = 0, scaled by 1.0)
        for v in &result {
            assert!(v.abs() < 1e-6, "Expected ~0.0 but got {v}");
        }
    }

    #[test]
    fn test_dequantize_q4_0_exceeds_file_size() {
        let data = vec![0u8; 10]; // Too small for 1 block (needs 18 bytes)
        let result = dequantize_q4_0(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_partial_block() {
        // Request fewer elements than a full block
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C;
        let result = dequantize_q4_0(&data, 0, 16).expect("should succeed");
        assert_eq!(result.len(), 16);
    }

    // =========================================================================
    // Dequantize Q8_0
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Build Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes
        let mut data = vec![0u8; 34];
        // Scale = 1.0 in f16
        data[0] = 0x00;
        data[1] = 0x3C;
        // All quant bytes = 0 (int8 value 0)
        let result = dequantize_q8_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequantize_q8_0_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q8_0(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_0
    // =========================================================================

    #[test]
    fn test_dequantize_q5_0_basic() {
        // Q5_0 block: 2 + 4 + 16 = 22 bytes
        let data = vec![0u8; 22];
        let result = dequantize_q5_0(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q5_0_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q5_0(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_1
    // =========================================================================

    #[test]
    fn test_dequantize_q5_1_basic() {
        // Q5_1 block: 2 + 2 + 4 + 16 = 24 bytes
        let data = vec![0u8; 24];
        let result = dequantize_q5_1(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q5_1_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q5_1(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q4_K
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Q4_K block: 2 + 2 + 12 + 128 = 144 bytes
        let data = vec![0u8; 144];
        let result = dequantize_q4_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q4_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q5_K
    // =========================================================================

    #[test]
    fn test_dequantize_q5_k_basic() {
        // Q5_K block: 2 + 2 + 12 + 32 + 128 = 176 bytes
        let data = vec![0u8; 176];
        let result = dequantize_q5_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q5_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q5_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q6_K
    // =========================================================================

    #[test]
    fn test_dequantize_q6_k_basic() {
        // Q6_K block: 128 + 64 + 16 + 2 = 210 bytes
        let data = vec![0u8; 210];
        let result = dequantize_q6_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_exceeds_file_size() {
        let data = vec![0u8; 100];
        let result = dequantize_q6_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q4_1
    // =========================================================================

    #[test]
    fn test_dequantize_q4_1_basic() {
        // Q4_1 block: 2 + 2 + 16 = 20 bytes
        let data = vec![0u8; 20];
        let result = dequantize_q4_1(&data, 0, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q4_1_exceeds_file_size() {
        let data = vec![0u8; 10];
        let result = dequantize_q4_1(&data, 0, 32);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q2_K
    // =========================================================================

    #[test]
    fn test_dequantize_q2_k_basic() {
        // Q2_K block: 2 + 2 + 16 + 64 = 84 bytes
        let data = vec![0u8; 84];
        let result = dequantize_q2_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q2_k_exceeds_file_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q2_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize Q3_K
    // =========================================================================

    #[test]
    fn test_dequantize_q3_k_basic() {
        // Q3_K block: 32 + 64 + 12 + 2 = 110 bytes
        let data = vec![0u8; 110];
        let result = dequantize_q3_k(&data, 0, 256).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q3_k_exceeds_file_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q3_k(&data, 0, 256);
        assert!(result.is_err());
    }

    // =========================================================================
    // Dequantize IQ approximate
    // =========================================================================

    #[test]
    fn test_dequantize_iq_approximate_iq2() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 13); // IQ2_XXS
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_iq3() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 16); // IQ3_XXS
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_iq1() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 18); // IQ1_S
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_default_dtype() {
        let data = vec![128u8; 1024];
        let result = dequantize_iq_approximate(&data, 0, 64, 99); // Unknown dtype
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_iq_approximate_byte_out_of_range() {
        // Small data so byte_idx exceeds data.len()
        let data = vec![128u8; 4];
        let result = dequantize_iq_approximate(&data, 0, 256, 13);
        // Should still produce 256 elements (some will be 0.0 for out-of-range bytes)
        assert_eq!(result.len(), 256);
        // Verify some elements are 0.0 (from the byte_idx >= data.len() path)
        assert!(result.iter().any(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_q4_0_with_nonzero_start() {
        let mut data = vec![0u8; 36]; // 18 bytes padding + 18 bytes block
                                      // Put scale at offset 18
        data[18] = 0x00;
        data[19] = 0x3C;
        let result = dequantize_q4_0(&data, 18, 32).expect("should succeed");
        assert_eq!(result.len(), 32);
    }

    // =========================================================================
    // GH-186: safe_f16_scale NaN/Inf/subnormal clamping
    // =========================================================================

    #[test]
    fn test_safe_f16_scale_normal() {
        // 1.0 in f16 = 0x3C00
        assert!((safe_f16_scale(0x3C00) - 1.0).abs() < 1e-3);
        // 2.0 in f16 = 0x4000
        assert!((safe_f16_scale(0x4000) - 2.0).abs() < 1e-3);
        // -1.0 in f16 = 0xBC00
        assert!((safe_f16_scale(0xBC00) - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_safe_f16_scale_nan_clamped() {
        // NaN in f16: exp=31, mantissa!=0 → 0x7E00
        assert_eq!(safe_f16_scale(0x7E00), 0.0);
        // Another NaN pattern
        assert_eq!(safe_f16_scale(0x7C01), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_inf_clamped() {
        // +Inf in f16: 0x7C00
        assert_eq!(safe_f16_scale(0x7C00), 0.0);
        // -Inf in f16: 0xFC00
        assert_eq!(safe_f16_scale(0xFC00), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_subnormal_preserved() {
        // PMAT-238: Subnormals are now PRESERVED (valid Q6_K scale factors)
        // Smallest subnormal: 0x0001 → ~5.96e-8
        assert!(safe_f16_scale(0x0001) > 0.0);
        // Largest subnormal: 0x03FF → ~6.09e-5
        assert!(safe_f16_scale(0x03FF) > 0.0);
    }

    #[test]
    fn test_safe_f16_scale_zero_preserved() {
        // Positive zero
        assert_eq!(safe_f16_scale(0x0000), 0.0);
        // Negative zero - abs(0.0) < F16_MIN_NORMAL so clamped to 0.0
        assert_eq!(safe_f16_scale(0x8000), 0.0);
    }

    #[test]
    fn test_gh186_nan_does_not_propagate_q4_0() {
        // Build a Q4_0 block with NaN scale (0x7E00)
        let mut data = vec![0u8; 18]; // 2-byte scale + 16-byte quants
        data[0] = 0x00; // NaN f16 = 0x7E00 (little-endian: 0x00, 0x7E)
        data[1] = 0x7E;
        // Fill quants with non-zero data
        for i in 2..18 {
            data[i] = 0x55; // non-zero nibbles
        }
        let result = dequantize_q4_0(&data, 0, 32).expect("should succeed");
        // With NaN clamping, all values should be finite (0.0 * anything = 0.0)
        assert!(
            result.iter().all(|v| v.is_finite()),
            "GH-186: NaN scale should not propagate to output"
        );
    }

    #[test]
    fn test_gh186_nan_does_not_propagate_q4_k() {
        // Build a Q4_K block with NaN scale
        // Q4_K block: 4 bytes (d+dmin) + 12 bytes (scales) + 128 bytes (quants) = 144 bytes
        let mut data = vec![0u8; 144];
        data[0] = 0x00; // d = NaN f16 = 0x7E00 (LE)
        data[1] = 0x7E;
        data[2] = 0x00; // dmin = NaN
        data[3] = 0x7E;
        // Fill scales and quants with non-zero
        for i in 4..144 {
            data[i] = 0x33;
        }
        let result = dequantize_q4_k(&data, 0, 256).expect("should succeed");
        assert!(
            result.iter().all(|v| v.is_finite()),
            "GH-186: NaN scale should not propagate to Q4_K output"
        );
    }
}
