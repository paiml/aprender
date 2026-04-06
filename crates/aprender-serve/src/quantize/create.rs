
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Q6_K FUSED DOT PRODUCT TESTS
    // ============================================================================

    #[test]
    fn test_fused_q6k_dot_invalid_data_length() {
        // Q6_K super-block is 210 bytes, test with non-multiple
        let data = vec![0u8; 100]; // Not a multiple of 210
        let activations = vec![0.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q6k_dot_activation_length_mismatch() {
        // One super-block = 210 bytes = 256 values
        let data = vec![0u8; 210];
        let activations = vec![0.0f32; 128]; // Should be 256

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q6k_dot_zero_data() {
        // All zeros should produce zero dot product
        let mut data = vec![0u8; 210];
        // Set d to 0 (f16 zero)
        data[208..210].copy_from_slice(&[0x00, 0x00]);
        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations).expect("result");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_fused_q6k_dot_single_super_block() {
        // Create valid Q6_K data: ql (128) + qh (64) + scales (16) + d (2)
        let mut data = vec![0u8; 210];

        // Set d = 1.0 (f16: 0x3C00)
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set some scales (signed i8)
        for i in 0..16 {
            data[192 + i] = 1; // Small positive scale
        }

        // Set ql values to small patterns
        for i in 0..128 {
            data[i] = ((i % 16) as u8) | (((i % 16) as u8) << 4);
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q6k_dot_simd_matches_scalar() {
        // Create valid Q6_K data for comparison
        let mut data = vec![0u8; 210];

        // Set d = 1.0 (f16: 0x3C00)
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set scales to 1
        for i in 0..16 {
            data[192 + i] = 1;
        }

        // Set ql/qh to deterministic pattern
        for i in 0..128 {
            data[i] = (i % 256) as u8;
        }
        for i in 0..64 {
            data[128 + i] = (i % 256) as u8;
        }

        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q6k_dot(&data, &activations).expect("scalar_result");
        let simd_result = fused_q6k_dot_simd(&data, &activations).expect("simd_result");

        // Allow 1% tolerance for SIMD vs scalar differences
        let rel_err = if scalar_result.abs() > 1e-6 {
            (simd_result - scalar_result).abs() / scalar_result.abs()
        } else {
            (simd_result - scalar_result).abs()
        };
        assert!(
            rel_err < 0.01,
            "scalar={} simd={} rel_err={}",
            scalar_result,
            simd_result,
            rel_err
        );
    }

    #[test]
    fn test_fused_q6k_dot_simd_invalid_input() {
        let data = vec![0u8; 100]; // Invalid length
        let activations = vec![0.0f32; 256];

        let result = fused_q6k_dot_simd(&data, &activations);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q6k_dot_multiple_super_blocks() {
        // Two super-blocks = 420 bytes = 512 values
        let mut data = vec![0u8; 420];

        // Set d = 0.5 for both blocks (f16: 0x3800)
        data[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        data[418..420].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set scales
        for i in 0..16 {
            data[192 + i] = 2;
            data[402 + i] = 2;
        }

        let activations = vec![0.5f32; 512];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    // ============================================================================
    // Q5_K FUSED DOT PRODUCT TESTS
    // ============================================================================

    #[test]
    fn test_fused_q5k_dot_invalid_data_length() {
        // Q5_K super-block is 176 bytes
        let data = vec![0u8; 100]; // Not a multiple of 176
        let activations = vec![0.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q5k_dot_activation_length_mismatch() {
        // One super-block = 176 bytes = 256 values
        let data = vec![0u8; 176];
        let activations = vec![0.0f32; 128]; // Should be 256

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q5k_dot_zero_data() {
        // All zeros should produce zero dot product
        let data = vec![0u8; 176];
        // d = 0, dmin = 0
        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations).expect("result");
        // With zero scales, result should be close to zero
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_fused_q5k_dot_single_super_block() {
        // Q5_K layout: d (2) + dmin (2) + scales (12) + qh (32) + qs (128)
        let mut data = vec![0u8; 176];

        // Set d = 1.0 (f16: 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set dmin = 0.5 (f16: 0x3800)
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set qs values to pattern
        for i in 0..128 {
            data[48 + i] = ((i % 16) as u8) | (((i + 1) % 16) << 4) as u8;
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q5k_dot_simd_matches_scalar() {
        // Create valid Q5_K data
        let mut data = vec![0u8; 176];

        // Set d = 1.0, dmin = 0.5
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set scales (12 bytes packed)
        for i in 0..12 {
            data[4 + i] = 0x11; // Small positive scales
        }

        // Set qh and qs
        for i in 0..32 {
            data[16 + i] = (i % 256) as u8;
        }
        for i in 0..128 {
            data[48 + i] = ((i * 3) % 256) as u8;
        }

        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q5k_dot(&data, &activations).expect("scalar_result");
        let simd_result = fused_q5k_dot_simd(&data, &activations).expect("simd_result");

        // SIMD currently uses scalar, so should be exact match
        assert!(
            (scalar_result - simd_result).abs() < 1e-6,
            "scalar={} simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_fused_q5k_dot_simd_invalid_input() {
        let data = vec![0u8; 100]; // Invalid length
        let activations = vec![0.0f32; 256];

        let result = fused_q5k_dot_simd(&data, &activations);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q5k_dot_multiple_super_blocks() {
        // Two super-blocks = 352 bytes = 512 values
        let mut data = vec![0u8; 352];

        // Set d and dmin for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        data[176..178].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[178..180].copy_from_slice(&0x3800u16.to_le_bytes());

        let activations = vec![0.5f32; 512];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    // ============================================================================
    // Q4_K × Q8 DOT PRODUCT TESTS
    // ============================================================================

    /// Helper to create a Q8_0Block with zero values
    fn zero_q8_block() -> Q8_0Block {
        Q8_0Block {
            scale: 0.0,
            quants: [0i8; 32],
        }
    }

    /// Helper to create a Q8_0Block with specified scale and quant value
    fn make_q8_block(scale: f32, quant_val: i8) -> Q8_0Block {
        Q8_0Block {
            scale,
            quants: [quant_val; 32],
        }
    }

    #[test]
    fn test_fused_q4k_q8_dot_invalid_data_length() {
        // Q4_K super-block is 144 bytes
        let data = vec![0u8; 100]; // Not a multiple of 144
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_fused_q4k_q8_dot_block_count_mismatch() {
        // One super-block = 144 bytes = 256 values = 8 Q8 blocks
        let data = vec![0u8; 144];
        let q8_blocks: Vec<Q8_0Block> = (0..4).map(|_| zero_q8_block()).collect(); // Should be 8

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q4k_q8_dot_zero_data() {
        // All zeros should produce zero dot product
        let data = vec![0u8; 144];
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks).expect("result");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_fused_q4k_q8_dot_single_super_block() {
        // Create valid Q4_K data: d (2) + dmin (2) + scales (12) + qs (128)
        let mut data = vec![0u8; 144];

        // Set d = 1.0, dmin = 0.5
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        // Set qs values
        for i in 0..128 {
            data[16 + i] = 0x55; // Pattern: low=5, high=5
        }

        // Create Q8 blocks with some values
        let q8_blocks: Vec<Q8_0Block> = (0..8).map(|_| make_q8_block(0.1, 10)).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q4k_q8_dot_multiple_super_blocks() {
        // Two super-blocks = 288 bytes = 512 values = 16 Q8 blocks
        let mut data = vec![0u8; 288];

        // Set headers for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

        let q8_blocks: Vec<Q8_0Block> = (0..16).map(|_| zero_q8_block()).collect();

        let result = fused_q4k_q8_dot(&data, &q8_blocks);
        assert!(result.is_ok());
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_fused_q6k_dot_negative_scales() {
        let mut data = vec![0u8; 210];

        // Set d = 1.0
        data[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());

        // Set negative scales (i8)
        #[allow(clippy::cast_sign_loss)]
        for i in 0..16 {
            data[192 + i] = (-5i8) as u8;
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q5k_dot_with_high_bits() {
        let mut data = vec![0u8; 176];

        // Set d = 1.0, dmin = 0.1
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[2..4].copy_from_slice(&0x2E66u16.to_le_bytes()); // ~0.1

        // Set qh (high bits) to all 1s
        for i in 0..32 {
            data[16 + i] = 0xFF;
        }

        // Set qs
        for i in 0..128 {
            data[48 + i] = 0xF0; // high nibbles
        }

        let activations = vec![1.0f32; 256];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q6k_dot_empty_data() {
        let data = vec![];
        let activations = vec![];

        // Empty is valid (0 super-blocks)
        let result = fused_q6k_dot(&data, &activations);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), 0.0);
    }

    #[test]
    fn test_fused_q5k_dot_empty_data() {
        let data = vec![];
        let activations = vec![];

        let result = fused_q5k_dot(&data, &activations);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), 0.0);
    }
}
