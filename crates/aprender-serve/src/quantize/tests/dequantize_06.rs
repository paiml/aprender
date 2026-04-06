
#[test]
fn test_dequantize_q8_0_direct() {
    // Create a Q8_0 block (34 bytes): 2 bytes scale + 32 bytes quants
    let mut block_data = vec![0u8; 34];

    // Set scale = 1.0 (f16 encoding)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    block_data[0] = scale_bits as u8;
    block_data[1] = (scale_bits >> 8) as u8;

    // Set quants to sequential values 0-31 (as i8)
    for i in 0..32 {
        block_data[2 + i] = i as u8;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    // Should produce 32 values
    assert_eq!(result.len(), 32, "expected 32 values, got {}", result.len());

    // Values should be 0.0, 1.0, 2.0, ... 31.0 (scale * quant = 1.0 * i)
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32;
        assert!(
            (val - expected).abs() < 0.01,
            "result[{}] = {}, expected {}",
            i,
            val,
            expected
        );
    }
}

#[test]
fn test_dequantize_q8_0_negative_quants() {
    // Test with negative quantized values
    let mut block_data = vec![0u8; 34];

    // Set scale = 2.0
    let scale_bits = half::f16::from_f32(2.0).to_bits();
    block_data[0] = scale_bits as u8;
    block_data[1] = (scale_bits >> 8) as u8;

    // Set quants to -128, -64, 0, 64, 127 pattern repeated
    let pattern: [i8; 5] = [-128, -64, 0, 64, 127];
    for i in 0..32 {
        block_data[2 + i] = pattern[i % 5] as u8;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    assert_eq!(result.len(), 32);

    // Check first few values: scale * quant = 2.0 * i8
    assert!(
        (result[0] - (-256.0)).abs() < 0.1,
        "result[0] = {}",
        result[0]
    ); // 2.0 * -128
    assert!(
        (result[1] - (-128.0)).abs() < 0.1,
        "result[1] = {}",
        result[1]
    ); // 2.0 * -64
    assert!((result[2] - 0.0).abs() < 0.1, "result[2] = {}", result[2]); // 2.0 * 0
    assert!((result[3] - 128.0).abs() < 0.1, "result[3] = {}", result[3]); // 2.0 * 64
    assert!((result[4] - 254.0).abs() < 0.1, "result[4] = {}", result[4]); // 2.0 * 127
}

#[test]
fn test_dequantize_q8_0_zero_scale() {
    // Test with zero scale (edge case)
    let mut block_data = vec![0u8; 34];

    // Set scale = 0.0
    block_data[0] = 0;
    block_data[1] = 0;

    // Set quants to 127 (max)
    for i in 0..32 {
        block_data[2 + i] = 127;
    }

    let result = dequantize_q8_0(&block_data).expect("test");

    // All values should be 0 (0.0 * 127 = 0)
    for (i, &val) in result.iter().enumerate() {
        assert!(val.abs() < 0.001, "result[{}] should be 0, got {}", i, val);
    }
}

// ============================================================================
// DIRECT TESTS FOR EXPOSED HELPER FUNCTIONS (Force-Path Coverage)
// ============================================================================

#[test]
fn test_f16_to_f32_lut_zero() {
    // Test zero conversion
    let result = f16_to_f32_lut(0);
    assert!(
        result.abs() < 1e-10,
        "Zero should convert to 0.0, got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_lut_one() {
    // Test 1.0 conversion (f16 bits for 1.0 = 0x3C00)
    let result = f16_to_f32_lut(0x3C00);
    assert!((result - 1.0).abs() < 1e-6, "Should be 1.0, got {}", result);
}

#[test]
fn test_f16_to_f32_lut_negative_one() {
    // Test -1.0 conversion (f16 bits for -1.0 = 0xBC00)
    let result = f16_to_f32_lut(0xBC00);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "Should be -1.0, got {}",
        result
    );
}

#[test]
fn test_f16_to_f32_lut_small_value() {
    // Test small value (0.5 = 0x3800 in f16)
    let result = f16_to_f32_lut(0x3800);
    assert!((result - 0.5).abs() < 1e-6, "Should be 0.5, got {}", result);
}

#[test]
fn test_read_f16_zero() {
    // Test reading f16 zero
    let bytes = [0x00, 0x00]; // f16 zero
    let result = read_f16(&bytes);
    assert!(result.abs() < 1e-10, "Should be 0.0, got {}", result);
}

#[test]
fn test_read_f16_one() {
    // Test reading f16 1.0
    let bytes = [0x00, 0x3C]; // f16 1.0 in little-endian
    let result = read_f16(&bytes);
    assert!((result - 1.0).abs() < 1e-6, "Should be 1.0, got {}", result);
}

#[test]
fn test_read_f16_negative() {
    // Test reading f16 -1.0
    let bytes = [0x00, 0xBC]; // f16 -1.0 in little-endian
    let result = read_f16(&bytes);
    assert!(
        (result - (-1.0)).abs() < 1e-6,
        "Should be -1.0, got {}",
        result
    );
}

#[test]
fn test_extract_scale_min_from_slice_first_block() {
    // Test extraction for index 0 (first block - simple layout)
    let scales: [u8; 8] = [0x3F, 0x1F, 0x0F, 0x07, 0x20, 0x10, 0x08, 0x04];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    // idx=0: scale = scales[0] & 0x3F = 0x3F = 63
    // idx=0: min = scales[4] & 0x3F = 0x20 = 32
    assert!(
        (scale - 63.0).abs() < 0.001,
        "scale should be 63, got {}",
        scale
    );
    assert!((min - 32.0).abs() < 0.001, "min should be 32, got {}", min);
}

#[test]
fn test_extract_scale_min_from_slice_second_block() {
    // Test extraction for index 2 (even, simple layout)
    let scales: [u8; 8] = [0x3F, 0x1F, 0x0F, 0x07, 0x20, 0x10, 0x08, 0x04];
    let (scale, min) = extract_scale_min_from_slice(&scales, 2);
    // idx=2: scale_idx = 1, scale = scales[1] & 0x3F = 0x1F = 31
    // idx=2: min_idx = 5, min = scales[5] & 0x3F = 0x10 = 16
    assert!(
        (scale - 31.0).abs() < 0.001,
        "scale should be 31, got {}",
        scale
    );
    assert!((min - 16.0).abs() < 0.001, "min should be 16, got {}", min);
}

#[test]
fn test_extract_scale_min_first_blocks() {
    // Test extract_scale_min for first 4 blocks (simple layout)
    let scales: [u8; 12] = [
        0x3F, 0x2F, 0x1F, 0x0F, // First 4 scale values (low 6 bits)
        0x20, 0x18, 0x10, 0x08, // First 4 min values (low 6 bits)
        0x00, 0x00, 0x00, 0x00, // High bits (unused for first 4)
    ];

    // Block 0: scale = 0x3F & 63 = 63, min = 0x20 & 63 = 32
    let (scale, min) = extract_scale_min(&scales, 0);
    assert!(
        (scale - 63.0).abs() < 0.001,
        "Block 0 scale should be 63, got {}",
        scale
    );
    assert!(
        (min - 32.0).abs() < 0.001,
        "Block 0 min should be 32, got {}",
        min
    );

    // Block 1: scale = 0x2F & 63 = 47, min = 0x18 & 63 = 24
    let (scale, min) = extract_scale_min(&scales, 1);
    assert!(
        (scale - 47.0).abs() < 0.001,
        "Block 1 scale should be 47, got {}",
        scale
    );
    assert!(
        (min - 24.0).abs() < 0.001,
        "Block 1 min should be 24, got {}",
        min
    );
}

#[test]
fn test_extract_scale_min_packed_blocks() {
    // Test extract_scale_min for blocks 4-7 (packed layout)
    // The packed layout uses high bits from bytes 0-3 combined with bytes 8-11
    let scales: [u8; 12] = [
        0x40, 0x80, 0xC0, 0x00, // Low 6 bits + high 2 bits for blocks 4-7
        0x00, 0x00, 0x00, 0x00, // Min values for blocks 0-3
        0x12, 0x34, 0x56, 0x78, // Packed values for blocks 4-7
    ];

    // Block 4: d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x12 & 0x0F) | (1 << 4) = 2 | 16 = 18
    // Block 4: m = (scales[8] >> 4) | ((scales[4] >> 6) << 4) = (0x12 >> 4) | (0 << 4) = 1 | 0 = 1
    let (scale, min) = extract_scale_min(&scales, 4);
    assert!(
        (scale - 18.0).abs() < 0.001,
        "Block 4 scale should be 18, got {}",
        scale
    );
    assert!(
        (min - 1.0).abs() < 0.001,
        "Block 4 min should be 1, got {}",
        min
    );
}

#[test]
fn test_extract_scale_min_all_zeros() {
    // Test with all zeros
    let scales: [u8; 12] = [0; 12];

    for block_idx in 0..8 {
        let (scale, min) = extract_scale_min(&scales, block_idx);
        assert!(
            scale.abs() < 0.001,
            "Block {} scale should be 0, got {}",
            block_idx,
            scale
        );
        assert!(
            min.abs() < 0.001,
            "Block {} min should be 0, got {}",
            block_idx,
            min
        );
    }
}

// =========================================================================
// Dequantization Edge Cases (Coverage: error paths)
// =========================================================================

#[test]
fn test_dequantize_f16_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_f16(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_odd_length_edge() {
    // Odd number of bytes (not valid for f16)
    let data = [0u8; 3];
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q8_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q6_k_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q6_k(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q4_1 Tests (Coverage: Q4_1 format)
// =========================================================================

#[test]
fn test_dequantize_q4_1_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q4_1(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q5_0 Tests (Coverage: Q5_0 format)
// =========================================================================

#[test]
fn test_dequantize_q5_0_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_0(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Dequantize Q5_1 Tests (Coverage: Q5_1 format)
// =========================================================================

#[test]
fn test_dequantize_q5_1_empty_edge() {
    let data: &[u8] = &[];
    let result = dequantize_q5_1(data);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Block Size Constants Tests (Coverage: constants)
// =========================================================================

#[test]
fn test_block_size_constant_check() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_check() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Q4_0Block Tests (Coverage: Q4_0Block struct)
// =========================================================================

#[test]
fn test_q4_0_block_struct_fields_check() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0x12; 16],
    };
    assert!((block.scale - 2.5).abs() < 1e-6);
    assert_eq!(block.quants[0], 0x12);
}

#[test]
fn test_q4_0_block_clone_check() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0xAB; 16],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q4_0_block_debug_check() {
    let block = Q4_0Block {
        scale: 0.5,
        quants: [0; 16],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_0Block"));
    assert!(debug.contains("scale"));
}

// =========================================================================
// Q8_0Block Tests (Coverage: Q8_0Block struct)
// =========================================================================

#[test]
fn test_q8_0_block_struct_fields_check() {
    let block = Q8_0Block {
        scale: 0.125,
        quants: [64i8; 32],
    };
    assert!((block.scale - 0.125).abs() < 1e-6);
    assert_eq!(block.quants[0], 64);
}

#[test]
fn test_q8_0_block_clone_check() {
    let block = Q8_0Block {
        scale: 0.25,
        quants: [-10i8; 32],
    };
    let cloned = block.clone();
    assert!((cloned.scale - block.scale).abs() < 1e-6);
    assert_eq!(cloned.quants, block.quants);
}
