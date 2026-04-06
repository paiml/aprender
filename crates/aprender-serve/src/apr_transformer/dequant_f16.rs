
    // -------------------------------------------------------------------------
    // f16_to_f32 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_zero() {
        // +0.0 in f16 = 0x0000
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.0001);
        // -0.0 in f16 = 0x8000
        assert!((f16_to_f32(0x8000) - (-0.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.0001);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_two() {
        // 2.0 in f16 = 0x4000
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // +Inf in f16 = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
        // -Inf in f16 = 0xFC00
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // NaN in f16: exp=31, mantissa!=0
        assert!(f16_to_f32(0x7C01).is_nan());
        assert!(f16_to_f32(0x7FFF).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest positive subnormal: 0x0001 = 2^-24
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0);
        assert!(result < 0.001); // Very small
    }

    // -------------------------------------------------------------------------
    // extract_scale_min_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_scale_min_apr_first_four_blocks() {
        // 12 bytes of scales
        let scales = [10, 20, 30, 40, 5, 15, 25, 35, 0, 0, 0, 0];

        // Block 0: scale = scales[0] & 63 = 10, min = scales[4] & 63 = 5
        let (s, m) = extract_scale_min_apr(&scales, 0);
        assert!((s - 10.0).abs() < 0.001);
        assert!((m - 5.0).abs() < 0.001);

        // Block 1: scale = scales[1] & 63 = 20, min = scales[5] & 63 = 15
        let (s, m) = extract_scale_min_apr(&scales, 1);
        assert!((s - 20.0).abs() < 0.001);
        assert!((m - 15.0).abs() < 0.001);

        // Block 3: scale = scales[3] & 63 = 40, min = scales[7] & 63 = 35
        let (s, m) = extract_scale_min_apr(&scales, 3);
        assert!((s - 40.0).abs() < 0.001);
        assert!((m - 35.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_scale_min_apr_last_four_blocks() {
        // 12 bytes of scales with specific values for testing packed layout
        // For block 4+: d = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //               m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        let scales = [0, 0, 0, 0, 0, 0, 0, 0, 0x12, 0x34, 0x56, 0x78];

        // Block 4: j=4, uses scales[8] and scales[0]
        let (s, m) = extract_scale_min_apr(&scales, 4);
        // scale = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        //       = (0x12 & 0x0F) | ((0 >> 6) << 4) = 0x02 = 2
        // min = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
        //     = (0x12 >> 4) | ((0 >> 6) << 4) = 0x01 = 1
        assert!((s - 2.0).abs() < 0.001);
        assert!((m - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // dequantize_q4_k_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q4_k_apr_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q4_k_apr(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4_k_apr_insufficient_data() {
        let data: Vec<u8> = vec![0; 10]; // Less than 144 bytes
        let result = dequantize_q4_k_apr(&data, 256);
        // Should return zeros when data is insufficient
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q4_k_apr_zeros() {
        // 144 bytes of zeros (one super-block)
        let data = vec![0u8; 144];
        let result = dequantize_q4_k_apr(&data, 256);
        assert_eq!(result.len(), 256);
        // With d=0.0, all values should be 0.0
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q4_k_apr_truncation() {
        // Request fewer elements than super-block size
        let data = vec![0u8; 144];
        let result = dequantize_q4_k_apr(&data, 100);
        assert_eq!(result.len(), 100);
    }

    // -------------------------------------------------------------------------
    // dequantize_q6_k_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q6_k_apr_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q6_k_apr(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_apr_insufficient_data() {
        let data: Vec<u8> = vec![0; 100]; // Less than 210 bytes
        let result = dequantize_q6_k_apr(&data, 256);
        // Should return zeros when data is insufficient
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q6_k_apr_zeros() {
        // 210 bytes of zeros (one super-block)
        let data = vec![0u8; 210];
        let result = dequantize_q6_k_apr(&data, 256);
        assert_eq!(result.len(), 256);
        // With d=0.0, all values should be 0.0
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q6_k_apr_truncation() {
        // Request fewer elements than super-block size
        let data = vec![0u8; 210];
        let result = dequantize_q6_k_apr(&data, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q6_k_apr_multiple_blocks() {
        // Two super-blocks (420 bytes)
        let data = vec![0u8; 420];
        let result = dequantize_q6_k_apr(&data, 512);
        assert_eq!(result.len(), 512);
    }

    // -------------------------------------------------------------------------
    // dequantize_q8_0_apr Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_q8_0_apr_zeros() {
        // Q8_0: 34 bytes per block (2 f16 scale + 32 i8 quants)
        // All zeros => scale=0, so all outputs should be 0
        let data = vec![0u8; 34];
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_q8_0_apr_single_block() {
        // scale = 1.0 in f16 = 0x3C00 little-endian
        let mut data = vec![0u8; 34];
        data[0] = 0x00; // f16 1.0 low byte
        data[1] = 0x3C; // f16 1.0 high byte
                        // Set quants: all i8 = 1 (unsigned byte 1)
        for i in 2..34 {
            data[i] = 1;
        }
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        // Each value should be scale * q = 1.0 * 1 = 1.0
        for &v in &result {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q8_0_apr_negative_quants() {
        let mut data = vec![0u8; 34];
        data[0] = 0x00;
        data[1] = 0x3C; // scale = 1.0
                        // Set quants: all i8 = -1 (0xFF as u8)
        for i in 2..34 {
            data[i] = 0xFF;
        }
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - (-1.0)).abs() < 0.01, "expected ~-1.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_q8_0_apr_truncation() {
        // Request fewer elements than block size
        let data = vec![0u8; 34];
        let result = dequantize_q8_0_apr(&data, 16);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_dequantize_q8_0_apr_multiple_blocks() {
        // Two blocks (68 bytes)
        let data = vec![0u8; 68];
        let result = dequantize_q8_0_apr(&data, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q8_0_apr_insufficient_data() {
        // Not enough data for even one block
        let data = vec![0u8; 10];
        let result = dequantize_q8_0_apr(&data, 32);
        assert_eq!(result.len(), 32);
        // Should return zeros when data is insufficient
        assert!(result.iter().all(|&x| x == 0.0));
    }

    // -------------------------------------------------------------------------
    // dequantize_apr_q8_native Tests (GH-239)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_apr_q8_native_zeros() {
        // 4 bytes scale (0.0) + 32 bytes quants
        let data = vec![0u8; 36];
        let result = dequantize_apr_q8_native(&data, 32);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_apr_q8_native_unit_scale() {
        // scale = 1.0 as f32 LE + 32 quants all = 1 (i8)
        let mut data = Vec::with_capacity(36);
        data.extend_from_slice(&1.0f32.to_le_bytes()); // scale = 1.0
        for _ in 0..32 {
            data.push(1); // i8 = 1
        }
        let result = dequantize_apr_q8_native(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 1.0).abs() < 0.001, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_apr_q8_native_negative_quants() {
        // scale = 2.0, quants = -1 (0xFF as u8 → -1 as i8)
        let mut data = Vec::with_capacity(36);
        data.extend_from_slice(&2.0f32.to_le_bytes());
        for _ in 0..32 {
            data.push(0xFF); // i8 = -1
        }
        let result = dequantize_apr_q8_native(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - (-2.0)).abs() < 0.001, "expected ~-2.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_apr_q8_native_insufficient_data() {
        // Less than 4 bytes
        let data = vec![0u8; 2];
        let result = dequantize_apr_q8_native(&data, 32);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_apr_q8_native_partial_data() {
        // scale + only 10 quants, but request 32 elements
        let mut data = Vec::with_capacity(14);
        data.extend_from_slice(&1.0f32.to_le_bytes());
        for i in 0..10 {
            data.push(i + 1); // i8 values 1..10
        }
        let result = dequantize_apr_q8_native(&data, 32);
        assert_eq!(result.len(), 32);
        // First 10 should be 1.0..10.0
        for i in 0..10 {
            let expected = (i + 1) as f32;
            assert!(
                (result[i] - expected).abs() < 0.001,
                "result[{i}] = {}, expected {expected}",
                result[i]
            );
        }
        // Remaining 22 should be 0.0 (padding)
        for i in 10..32 {
            assert!(
                result[i] == 0.0,
                "result[{i}] = {}, expected 0.0",
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_apr_q8_native_roundtrip() {
        // Simulate what add_q8_tensor produces: scale = max_abs / 127.0
        let original: Vec<f32> = vec![0.5, -0.3, 1.0, -1.0, 0.0];
        let max_abs = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;

        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_le_bytes());
        for &v in &original {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            data.push(q as u8);
        }

        let result = dequantize_apr_q8_native(&data, 5);
        assert_eq!(result.len(), 5);
        for (i, (&orig, &deq)) in original.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.02,
                "element {i}: orig={orig}, deq={deq}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // dequantize_apr_q4_native Tests (GH-239)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dequantize_apr_q4_native_zeros() {
        // 18 bytes of zeros (1 block): f16 scale=0.0 + 16 nibble bytes
        let data = vec![0u8; 18];
        let result = dequantize_apr_q4_native(&data, 32);
        assert_eq!(result.len(), 32);
        // scale=0.0, so all values = 0.0 * (nibble - 8) = 0.0
        // Note: even nibble=0 → (0-8)*0.0 = 0.0
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_apr_q4_native_unit_scale_nibble_8() {
        // scale = 1.0 in f16, all nibble bytes = 0x88 (nibble 8 → q=(8-8)=0)
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 1.0
        for i in 2..18 {
            data[i] = 0x88; // low=8, high=8 → both q=0
        }
        let result = dequantize_apr_q4_native(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 0.0).abs() < 0.001, "expected ~0.0, got {v}");
        }
    }

    #[test]
    fn test_dequantize_apr_q4_native_known_values() {
        // scale = 1.0 in f16 (0x3C00), nibble byte 0x0F → low=15, high=0
        // low: (15-8)=7 → 1.0*7=7.0
        // high: (0-8)=-8 → 1.0*(-8)=-8.0
        let mut data = vec![0u8; 18];
        data[0] = 0x00;
        data[1] = 0x3C; // f16 1.0
        data[2] = 0x0F; // first nibble byte
        // rest are zeros
        let result = dequantize_apr_q4_native(&data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 7.0).abs() < 0.01, "expected 7.0, got {}", result[0]);
        assert!((result[1] - (-8.0)).abs() < 0.01, "expected -8.0, got {}", result[1]);
    }

    #[test]
    fn test_dequantize_apr_q4_native_insufficient_data() {
        // Empty data
        let data: Vec<u8> = vec![];
        let result = dequantize_apr_q4_native(&data, 32);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dequantize_apr_q4_native_multiple_blocks() {
        // 2 blocks = 36 bytes, 64 elements
        let data = vec![0u8; 36];
        let result = dequantize_apr_q4_native(&data, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_apr_q4_native_truncation() {
        // Request fewer elements than block size
        let data = vec![0u8; 18];
        let result = dequantize_apr_q4_native(&data, 10);
        assert_eq!(result.len(), 10);
    }
