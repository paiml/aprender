
#[cfg(test)]
mod tests {
    use super::*;

    // ============= Q4_K parallel tests =============

    #[test]
    fn test_dequantize_q4_k_parallel_empty() {
        let result = dequantize_q4_k_parallel(&[]);
        assert!(result.is_ok());
        assert!(result.expect("result").is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_parallel_invalid_size() {
        // Q4_K super-block is 144 bytes; 100 bytes is invalid
        let data = vec![0u8; 100];
        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RealizarError::InvalidShape { .. }));
    }

    #[test]
    fn test_dequantize_q4_k_parallel_single_block() {
        // Create a valid super-block (144 bytes)
        let mut data = vec![0u8; 144];
        // Set d = 1.0 (f16 encoding)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0 in f16
        // Set dmin = 0.0
        data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());

        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_ok());
        let dequant = result.expect("dequant");
        assert_eq!(dequant.len(), 256); // QK_K = 256
    }

    #[test]
    fn test_dequantize_q4_k_parallel_multiple_blocks() {
        // Create 2 valid super-blocks (288 bytes)
        let mut data = vec![0u8; 288];
        // Set d = 1.0 for both blocks
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());

        let result = dequantize_q4_k_parallel(&data);
        assert!(result.is_ok());
        let dequant = result.expect("dequant");
        assert_eq!(dequant.len(), 512); // 2 * QK_K = 512
    }

    #[test]
    fn test_dequantize_q4_k_superblock_zero_data() {
        // All zeros should dequantize correctly
        let sb_data = vec![0u8; 144];
        let result = dequantize_q4_k_superblock(&sb_data);
        assert_eq!(result.len(), 256);
        // With d=0 and dmin=0, all values should be 0
        for val in &result {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dequantize_q4_k_superblock_scale_factor() {
        let mut sb_data = vec![0u8; 144];
        // d = 2.0 in f16 = 0x4000
        sb_data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        // dmin = 0
        sb_data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        // Set qs to known values (nibbles = 1)
        for i in 16..144 {
            sb_data[i] = 0x11; // Low nibble = 1, high nibble = 1
        }

        let result = dequantize_q4_k_superblock(&sb_data);
        assert_eq!(result.len(), 256);
        // Values should reflect the scale factor
    }

    // ============= Q4_K SIMD tests =============

    #[test]
    fn test_dequantize_q4_k_simd_empty() {
        let result = dequantize_q4_k_simd(&[]);
        assert!(result.is_ok());
        assert!(result.expect("result").is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_simd_invalid_size() {
        let data = vec![0u8; 50];
        let result = dequantize_q4_k_simd(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_simd_matches_parallel() {
        let mut data = vec![0u8; 144];
        // Set meaningful values
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
        for i in 16..144 {
            data[i] = (i % 256) as u8;
        }

        let simd_result = dequantize_q4_k_simd(&data).expect("simd_result");
        let parallel_result = dequantize_q4_k_parallel(&data).expect("parallel_result");

        assert_eq!(simd_result.len(), parallel_result.len());
        for (s, p) in simd_result.iter().zip(parallel_result.iter()) {
            assert!((s - p).abs() < 1e-5, "simd={} parallel={}", s, p);
        }
    }

    // ============= Q8_0 parallel tests =============

    #[test]
    fn test_dequantize_q8_0_parallel_empty() {
        let result = dequantize_q8_0_parallel(&[]);
        assert!(result.is_ok());
        assert!(result.expect("result").is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_parallel_invalid_size() {
        // Q8_0 block is 34 bytes; 20 bytes is invalid
        let data = vec![0u8; 20];
        let result = dequantize_q8_0_parallel(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_parallel_single_block() {
        // Create valid Q8_0 block (34 bytes)
        let mut data = vec![0u8; 34];
        // Scale = 1.0 (f16 = 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Set quants to known values
        for i in 2..34 {
            data[i] = 10; // Each i8 = 10
        }

        let result = dequantize_q8_0_parallel(&data).expect("result");
        assert_eq!(result.len(), 32);
        for val in &result {
            assert!((val - 10.0).abs() < 0.01, "expected 10.0, got {}", val);
        }
    }

    #[test]
    fn test_dequantize_q8_0_parallel_negative_values() {
        let mut data = vec![0u8; 34];
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
        // Set quants to -5 (i8 represented as u8)
        for i in 2..34 {
            data[i] = (-5i8) as u8;
        }

        let result = dequantize_q8_0_parallel(&data).expect("result");
        for val in &result {
            assert!((val - (-5.0)).abs() < 0.01, "expected -5.0, got {}", val);
        }
    }

    #[test]
    fn test_dequantize_q8_0_block_identity() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
        for i in 0..32 {
            block[2 + i] = i as u8;
        }

        let result = dequantize_q8_0_block(&block);
        assert_eq!(result.len(), 32);
        for (i, val) in result.iter().enumerate() {
            assert!((val - i as f32).abs() < 0.01);
        }
    }

    // ============= Q8_0 SIMD tests =============

    #[test]
    fn test_dequantize_q8_0_simd_empty() {
        let result = dequantize_q8_0_simd(&[]);
        assert!(result.is_ok());
        assert!(result.expect("result").is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_simd_invalid_size() {
        let data = vec![0u8; 30];
        let result = dequantize_q8_0_simd(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_simd_matches_parallel() {
        let mut data = vec![0u8; 34];
        data[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // scale = 2.0
        for i in 2..34 {
            data[i] = ((i - 2) as i8 * 3) as u8;
        }

        let simd_result = dequantize_q8_0_simd(&data).expect("simd_result");
        let parallel_result = dequantize_q8_0_parallel(&data).expect("parallel_result");

        assert_eq!(simd_result.len(), parallel_result.len());
        for (s, p) in simd_result.iter().zip(parallel_result.iter()) {
            assert!((s - p).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequantize_q8_0_simd_multiple_blocks() {
        // 3 blocks = 102 bytes
        let mut data = vec![0u8; 102];
        for block in 0..3 {
            let offset = block * 34;
            data[offset..offset + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
            for i in 0..32 {
                data[offset + 2 + i] = (block * 10 + i) as u8;
            }
        }

        let result = dequantize_q8_0_simd(&data).expect("result");
        assert_eq!(result.len(), 96); // 3 * 32
    }

    // ============= RoPE rotation tests =============

    #[test]
    fn test_apply_rope_rotation_scalar_identity() {
        let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x2 = vec![0.0, 0.0, 0.0, 0.0];
        let cos_vals = vec![1.0, 1.0, 1.0, 1.0]; // cos(0) = 1
        let sin_vals = vec![0.0, 0.0, 0.0, 0.0]; // sin(0) = 0

        apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

        // With cos=1, sin=0: x1' = x1*1 - x2*0 = x1, x2' = x1*0 + x2*1 = 0
        assert_eq!(x1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x2, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_rope_rotation_scalar_90_degrees() {
        let mut x1 = vec![1.0, 2.0];
        let mut x2 = vec![0.0, 0.0];
        let cos_vals = vec![0.0, 0.0]; // cos(90°) ≈ 0
        let sin_vals = vec![1.0, 1.0]; // sin(90°) = 1

        apply_rope_rotation_scalar(&mut x1, &mut x2, &cos_vals, &sin_vals);

        // x1' = x1*0 - x2*1 = 0
        // x2' = x1*1 + x2*0 = 1, 2
        assert!((x1[0] - 0.0).abs() < 1e-5);
        assert!((x2[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_rotation_simd_matches_scalar() {
        let mut x1_simd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut x2_simd = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let cos_vals = vec![0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let sin_vals = vec![0.6, 0.4, 0.7, 0.8, 0.9, 0.9, 0.95, 0.98, 0.995, 1.0];

        let mut x1_scalar = x1_simd.clone();
        let mut x2_scalar = x2_simd.clone();

        apply_rope_rotation_scalar(&mut x1_scalar, &mut x2_scalar, &cos_vals, &sin_vals);
        apply_rope_rotation_simd(&mut x1_simd, &mut x2_simd, &cos_vals, &sin_vals);

        for i in 0..x1_simd.len() {
            assert!(
                (x1_simd[i] - x1_scalar[i]).abs() < 1e-5,
                "x1 mismatch at {}: simd={} scalar={}",
                i,
                x1_simd[i],
                x1_scalar[i]
            );
            assert!(
                (x2_simd[i] - x2_scalar[i]).abs() < 1e-5,
                "x2 mismatch at {}: simd={} scalar={}",
                i,
                x2_simd[i],
                x2_scalar[i]
            );
        }
    }

    #[test]
    fn test_apply_rope_rotation_simd_large() {
        // Test with length > 16 to exercise AVX-512 path
        let n = 64;
        let mut x1 = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let mut x2 = (0..n).map(|i| (i + 100) as f32).collect::<Vec<_>>();
        let cos_vals = (0..n)
            .map(|i| ((i as f32) * 0.01).cos())
            .collect::<Vec<_>>();
        let sin_vals = (0..n)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect::<Vec<_>>();

        let mut x1_ref = x1.clone();
        let mut x2_ref = x2.clone();
        apply_rope_rotation_scalar(&mut x1_ref, &mut x2_ref, &cos_vals, &sin_vals);

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for i in 0..n {
            assert!((x1[i] - x1_ref[i]).abs() < 1e-4);
            assert!((x2[i] - x2_ref[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_apply_rope_rotation_preserves_magnitude() {
        // Rotation should preserve magnitude: |x|^2 = x1^2 + x2^2
        let mut x1: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0];
        let mut x2: Vec<f32> = vec![4.0, 3.0, 12.0, 8.0];
        let angle = 0.5f32;
        let cos_vals = vec![angle.cos(); 4];
        let sin_vals = vec![angle.sin(); 4];

        let mag_before: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a * a + b * b).sqrt())
            .collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        let mag_after: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a * a + b * b).sqrt())
            .collect();

        for (before, after) in mag_before.iter().zip(mag_after.iter()) {
            assert!((before - after).abs() < 1e-5);
        }
    }
}
