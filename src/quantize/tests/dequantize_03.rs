use crate::quantize::*;
#[test]
fn test_dequantize_q4_0_two_blocks_ext_cov() {
    let mut data = vec![0u8; 36]; // 2 blocks at 18 bytes each
    // First block scale = 1.0
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Second block scale = 2.0
    data[18..20].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    let result = dequantize_q4_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 64);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q8_0 success path
// =========================================================================

#[test]
fn test_dequantize_q8_0_one_block_ext_cov() {
    // Q8_0 block = 2 byte f16 scale + 32 byte quants = 34 bytes
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // All zero quants
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
}

#[test]
fn test_dequantize_q8_0_with_values_ext_cov() {
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // Set some quants (as signed i8 but stored as u8)
    for i in 0..32 {
        data[2 + i] = 10; // int8 value 10
    }
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 32);
    // Each value should be approximately 10 * 0.1 = 1.0
    for v in &vals {
        assert!((v - 1.0).abs() < 0.05);
    }
}

// =========================================================================
// Extended Coverage Tests for dequantize_q4_k success path
// =========================================================================

#[test]
fn test_dequantize_q4_k_one_superblock_cov() {
    // Q4_K super-block = 144 bytes
    let data = vec![0u8; 144];
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q5_k success path
// =========================================================================

#[test]
fn test_dequantize_q5_k_one_superblock_cov() {
    // Q5_K super-block = 176 bytes
    let data = vec![0u8; 176];
    let result = dequantize_q5_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q6_k success path
// =========================================================================

#[test]
fn test_dequantize_q6_k_one_superblock_cov() {
    // Q6_K super-block = 210 bytes
    let data = vec![0u8; 210];
    let result = dequantize_q6_k(&data);
    assert!(result.is_ok());
    let vals = result.expect("quantization failed");
    assert_eq!(vals.len(), 256);
}

// =========================================================================
// Extended Coverage Tests for Q4_KBlock struct
// =========================================================================

#[test]
fn test_q4_k_block_full_init_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    assert!((block.d - 1.0).abs() < 1e-6);
    assert!((block.dmin - 0.5).abs() < 1e-6);
    assert_eq!(block.scales.len(), 12);
    assert_eq!(block.qs.len(), 128);
}

// =========================================================================
// Extended Coverage Tests for Q5_KBlock struct
// =========================================================================

#[test]
fn test_q5_k_block_full_init_cov() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 1.0,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    assert!((block.d - 2.0).abs() < 1e-6);
    assert!((block.dmin - 1.0).abs() < 1e-6);
    assert_eq!(block.qh.len(), 32);
}

// =========================================================================
// Extended Coverage Tests for Q6_KBlock struct
// =========================================================================

#[test]
fn test_q6_k_block_full_init_cov() {
    let block = Q6_KBlock {
        d: 3.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    assert!((block.d - 3.0).abs() < 1e-6);
    assert_eq!(block.scales.len(), 16);
    assert_eq!(block.qh.len(), 64);
    assert_eq!(block.qs.len(), 128);
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32
// =========================================================================

#[test]
fn test_f16_to_f32_zero_ext_cov() {
    let zero = f16_to_f32(0);
    assert!((zero - 0.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_negative_zero_ext_cov() {
    // Negative zero: sign bit set, everything else zero
    let neg_zero = f16_to_f32(0x8000);
    assert!(neg_zero == 0.0 || neg_zero == -0.0);
}

#[test]
fn test_f16_to_f32_one_ext_cov() {
    let one = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32(one);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_negative_one_ext_cov() {
    let neg_one = half::f16::from_f32(-1.0).to_bits();
    let result = f16_to_f32(neg_one);
    assert!((result - (-1.0)).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_infinity_ext_cov() {
    // Positive infinity: exp=31, mantissa=0
    let pos_inf = 0x7C00;
    assert!(f16_to_f32(pos_inf).is_infinite());
    assert!(f16_to_f32(pos_inf) > 0.0);
}

#[test]
fn test_f16_to_f32_neg_infinity_ext_cov() {
    // Negative infinity: sign=1, exp=31, mantissa=0
    let neg_inf = 0xFC00;
    assert!(f16_to_f32(neg_inf).is_infinite());
    assert!(f16_to_f32(neg_inf) < 0.0);
}

#[test]
fn test_f16_to_f32_nan_ext_cov() {
    // NaN: exp=31, mantissa!=0
    let nan = 0x7C01;
    assert!(f16_to_f32(nan).is_nan());
}

#[test]
fn test_f16_to_f32_subnormal_ext_cov() {
    // Subnormal: exp=0, mantissa!=0
    let subnormal = 0x0001; // Smallest positive subnormal
    let result = f16_to_f32(subnormal);
    assert!(result > 0.0);
    assert!(result < 1e-5);
}

#[test]
fn test_f16_to_f32_neg_subnormal_ext_cov() {
    // Negative subnormal
    let neg_subnormal = 0x8001;
    let result = f16_to_f32(neg_subnormal);
    assert!(result < 0.0);
}

#[test]
fn test_f16_to_f32_max_normal_ext_cov() {
    // Max normal: exp=30, mantissa=all 1s (just below infinity)
    let max_normal = 0x7BFF;
    let result = f16_to_f32(max_normal);
    assert!(result > 60000.0);
    assert!(result < 70000.0);
}

// =========================================================================
// Extended Coverage Tests for f16_to_f32_lut
// =========================================================================

#[test]
fn test_f16_to_f32_lut_zero_ext_cov() {
    let result = f16_to_f32_lut(0);
    assert!((result - 0.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_lut_one_ext_cov() {
    let one_bits = half::f16::from_f32(1.0).to_bits();
    let result = f16_to_f32_lut(one_bits);
    assert!((result - 1.0).abs() < 1e-3);
}

#[test]
fn test_f16_to_f32_lut_all_values_ext_cov() {
    // Test that LUT returns reasonable values for all inputs
    for i in 0..1000u16 {
        let val = f16_to_f32_lut(i);
        assert!(!val.is_nan() || i >= 0x7C01);
    }
}

// =========================================================================
// Extended Coverage Tests for Q8_0Block methods
// =========================================================================

#[test]
fn test_q8_0_block_quantize_zeros_ext_cov() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Scale should be minimal for zeros
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8_0_block_quantize_uniform_ext_cov() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    // All quants should be 127 (max) since all values are equal to max
    assert!(block.quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8_0_block_quantize_negative_ext_cov() {
    let values = [-1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // All quants should be -127 (min + 1) since all values are negative max
    assert!(block.quants.iter().all(|&q| q == -127));
}

#[test]
fn test_q8_0_block_quantize_mixed_ext_cov() {
    let mut values = [0.0f32; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 15.5) * 0.1;
    }
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8_0_block_dequantize_ext_cov() {
    let block = Q8_0Block {
        scale: 0.1,
        quants: [10i8; 32],
    };
    let result = block.dequantize();
    assert_eq!(result.len(), 32);
    for val in result {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_q8_0_block_quantization_error_cov2() {
    let original = [1.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);
    // Error should be small for uniform values
    assert!(error < 0.02);
}

#[test]
fn test_q8_0_block_relative_error_zeros_cov2() {
    let original = [0.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let error = block.relative_error(&original);
    assert!((error - 0.0).abs() < 1e-6);
}

#[test]
fn test_q8_0_block_relative_error_nonzero_cov() {
    let mut original = [0.0f32; 32];
    for i in 0..32 {
        original[i] = (i as f32) * 0.1;
    }
    let block = Q8_0Block::quantize(&original);
    let error = block.relative_error(&original);
    // Relative error should be small
    assert!(error < 0.1);
}

#[test]
fn test_q8_0_block_clone_cov2() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [1i8; 32],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);
}

#[test]
fn test_q8_0_block_debug_cov2() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [0i8; 32],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

// =========================================================================
// Extended Coverage Tests for Q4_0Block
// =========================================================================

#[test]
fn test_q4_0_block_clone_cov2() {
    let block = Q4_0Block {
        scale: 2.0,
        quants: [0x12u8; 16],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
    assert_eq!(block.quants, cloned.quants);
}

#[test]
fn test_q4_0_block_debug_cov2() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
    assert!(debug_str.contains("quants"));
}

// =========================================================================
// Extended Coverage Tests for Q8KSuperBlock
// =========================================================================

#[test]
fn test_q8k_superblock_quantize_zeros_ext_cov() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale < 0.01);
}

#[test]
fn test_q8k_superblock_quantize_uniform_ext_cov() {
    let values = [1.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8k_superblock_quantize_mixed_ext_cov() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 127.5) * 0.01;
    }
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
}

#[test]
fn test_q8k_superblock_dequantize_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 0.1,
        quants: [10i8; 256],
    };
    let result = block.dequantize();
    assert_eq!(result.len(), 256);
    for val in result {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_q8k_superblock_quantize_into_ext_cov() {
    let values = [1.0f32; 256];
    let mut scale = 0.0f32;
    let mut quants = [0i8; 256];
    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    assert!(quants.iter().all(|&q| q == 127));
}

#[test]
fn test_q8k_superblock_clone_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 0.5,
        quants: [1i8; 256],
    };
    let cloned = block.clone();
    assert!((block.scale - cloned.scale).abs() < 1e-6);
}

#[test]
fn test_q8k_superblock_debug_ext_cov() {
    let block = Q8KSuperBlock {
        scale: 1.0,
        quants: [0i8; 256],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("scale"));
}

include!("quantize_activations_02.rs");
include!("fused_q5k_q6k.rs");
