
// ============================================================================
// Tests for Dequantization Functions (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // f16_to_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_zero() {
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let result = f16_to_f32(0x8000);
        assert!(result == 0.0 || result == -0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 0.001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_0_empty() {
        let result = dequantize_q4_0(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_0_invalid_length() {
        let result = dequantize_q4_0(&[0; 17]); // Not multiple of 18
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_zeros() {
        // One block: 2 (scale) + 16 (quants) = 18 bytes
        let data = vec![0u8; 18];
        let result = dequantize_q4_0(&data).expect("result");
        assert_eq!(result.len(), 32);
        // With scale=0, all values should be 0 (or close due to -8 offset)
    }

    #[test]
    fn test_dequantize_q4_0_one_block() {
        // scale = 1.0 (f16: 0x3C00)
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 for 1.0
        let result = dequantize_q4_0(&data).expect("result");
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q8_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q8_0_empty() {
        let result = dequantize_q8_0(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_invalid_length() {
        let result = dequantize_q8_0(&[0; 33]); // Not multiple of 34
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_zeros() {
        // One block: 2 (scale) + 32 (quants) = 34 bytes
        let data = vec![0u8; 34];
        let result = dequantize_q8_0(&data).expect("result");
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q8_0_one_block() {
        // scale = 1.0 (f16: 0x3C00)
        let mut data = vec![0u8; 34];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 for 1.0
                        // quants[0] = 1 (as i8)
        data[2] = 1;
        let result = dequantize_q8_0(&data).expect("result");
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // dequantize_f16 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_f16_empty() {
        let result = dequantize_f16(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_f16_invalid_length() {
        let result = dequantize_f16(&[0]); // Not multiple of 2
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_f16_zeros() {
        let data = vec![0u8; 4]; // 2 values
        let result = dequantize_f16(&data).expect("result");
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_f16_one() {
        // f16 for 1.0 = 0x3C00
        let data = vec![0x00, 0x3C];
        let result = dequantize_f16(&data).expect("result");
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_1_empty() {
        let result = dequantize_q4_1(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_1_invalid_length() {
        let result = dequantize_q4_1(&[0; 19]); // Not multiple of 20
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_1_zeros() {
        let data = vec![0u8; 20];
        let result = dequantize_q4_1(&data).expect("result");
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_0 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_0_empty() {
        let result = dequantize_q5_0(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_0_invalid_length() {
        let result = dequantize_q5_0(&[0; 21]); // Not multiple of 22
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_0_zeros() {
        let data = vec![0u8; 22];
        let result = dequantize_q5_0(&data).expect("result");
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_1_empty() {
        let result = dequantize_q5_1(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_1_invalid_length() {
        let result = dequantize_q5_1(&[0; 23]); // Not multiple of 24
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_1_zeros() {
        let data = vec![0u8; 24];
        let result = dequantize_q5_1(&data).expect("result");
        assert_eq!(result.len(), 32);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = dequantize_q4_k(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_invalid_length() {
        let result = dequantize_q4_k(&[0; 100]); // Not multiple of 144
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_zeros() {
        let data = vec![0u8; 144]; // One super-block
        let result = dequantize_q4_k(&data).expect("result");
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q5_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q5_k_empty() {
        let result = dequantize_q5_k(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q5_k_invalid_length() {
        let result = dequantize_q5_k(&[0; 100]); // Not multiple of 176
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_k_zeros() {
        let data = vec![0u8; 176]; // One super-block
        let result = dequantize_q5_k(&data).expect("result");
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q6_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = dequantize_q6_k(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_invalid_length() {
        let result = dequantize_q6_k(&[0; 100]); // Not multiple of 210
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q6_k_zeros() {
        let data = vec![0u8; 210]; // One super-block
        let result = dequantize_q6_k(&data).expect("result");
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // dequantize_q2_k Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q2_k_empty() {
        let result = dequantize_q2_k(&[]).expect("result");
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q2_k_invalid_length() {
        let result = dequantize_q2_k(&[0; 50]); // Not multiple of 84
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q2_k_zeros() {
        let data = vec![0u8; 84]; // One super-block
        let result = dequantize_q2_k(&data).expect("result");
        assert_eq!(result.len(), QK_K);
    }

    // -------------------------------------------------------------------------
    // read_f16 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_read_f16_zero() {
        let result = read_f16(&[0x00, 0x00]);
        assert!((result - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_read_f16_one() {
        let result = read_f16(&[0x00, 0x3C]); // f16 for 1.0
        assert!((result - 1.0).abs() < 0.0001);
    }
}
