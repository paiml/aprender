
#[test]
fn test_q8_0_block_debug_check() {
    let block = Q8_0Block {
        scale: 1.0,
        quants: [0; 32],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8_0Block"));
    assert!(debug.contains("scale"));
}

// =========================================================================
// f16_to_f32_lut Additional Tests (Coverage: LUT edge cases)
// =========================================================================

#[test]
fn test_f16_to_f32_lut_max_positive_check() {
    // f16 max positive = 0x7BFF = ~65504
    let result = f16_to_f32_lut(0x7BFF);
    assert!(result > 60000.0);
}

#[test]
fn test_f16_to_f32_lut_max_negative_check() {
    // f16 max negative = 0xFBFF = ~-65504
    let result = f16_to_f32_lut(0xFBFF);
    assert!(result < -60000.0);
}

#[test]
fn test_f16_to_f32_lut_infinity_check() {
    // f16 positive infinity = 0x7C00
    let result = f16_to_f32_lut(0x7C00);
    assert!(result.is_infinite());
    assert!(result > 0.0);
}

#[test]
fn test_f16_to_f32_lut_neg_infinity_check() {
    // f16 negative infinity = 0xFC00
    let result = f16_to_f32_lut(0xFC00);
    assert!(result.is_infinite());
    assert!(result < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan_check() {
    // f16 NaN = 0x7C01 (exponent all 1s, nonzero mantissa)
    let result = f16_to_f32_lut(0x7C01);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_lut_subnormal_check() {
    // f16 smallest subnormal = 0x0001
    let result = f16_to_f32_lut(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-6);
}

// =========================================================================
// Softmax SIMD Tests (Coverage: softmax_simd function - extended)
// =========================================================================

#[test]
fn test_softmax_simd_single_element_extended() {
    let mut values = [1.0f32];
    softmax_simd(&mut values);
    assert!((values[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_two_equal_elements_extended() {
    let mut values = [1.0f32, 1.0];
    softmax_simd(&mut values);
    assert!((values[0] - 0.5).abs() < 1e-6);
    assert!((values[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_simd_dominant_element_extended() {
    let mut values = [100.0f32, 0.0, 0.0];
    softmax_simd(&mut values);
    assert!(values[0] > 0.99); // Dominant
    assert!(values[1] < 0.01);
    assert!(values[2] < 0.01);
}

#[test]
fn test_softmax_simd_negative_values_extended() {
    let mut values = [-1.0f32, -2.0, -3.0];
    softmax_simd(&mut values);
    // Sum should be 1
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // First element should be largest
    assert!(values[0] > values[1]);
    assert!(values[1] > values[2]);
}

#[test]
fn test_softmax_simd_large_values_extended() {
    let mut values = [1000.0f32, 1000.0, 1000.0];
    softmax_simd(&mut values);
    // Should still sum to 1 despite large inputs
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_empty_extended() {
    let mut values: [f32; 0] = [];
    softmax_simd(&mut values);
    // Should not panic
}

// =========================================================================
// Fused Q4_0 Q8_0 Parallel Matvec Tests (Coverage: fused operations)
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_empty_input_check() {
    let result = fused_q4_0_q8_0_parallel_matvec(&[], &[], 0, 0);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_into_empty_check() {
    let mut output: Vec<f32> = vec![];
    let result = fused_q4_0_q8_0_parallel_matvec_into(&[], &[], 0, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Quantize Activations Tests (Coverage: activation quantization)
// =========================================================================

#[test]
fn test_quantize_activations_q8_0_returns_tuple() {
    let (scales, quants) = quantize_activations_q8_0(&[1.0, 2.0, 3.0, 4.0]);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_empty_returns_tuple() {
    let (scales, quants) = quantize_activations_q8_0(&[]);
    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

#[test]
fn test_quantize_activations_q8_0_uniform_values() {
    let input = vec![2.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&input);
    assert!(!scales.is_empty());
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_zeros_values() {
    let input = vec![0.0f32; 32];
    let (scales, quants) = quantize_activations_q8_0(&input);
    // Should handle zeros gracefully
    let _ = (scales, quants);
}

// =========================================================================
// Extract Scale Min from Slice Tests (Coverage: scale extraction helpers)
// =========================================================================

#[test]
fn test_extract_scale_min_from_slice_all_max_check() {
    let scales: [u8; 8] = [0x3F; 8];
    let (scale, _min) = extract_scale_min_from_slice(&scales, 0);
    assert!((scale - 63.0).abs() < 0.001);
}

#[test]
fn test_extract_scale_min_from_slice_alternating_check() {
    let scales: [u8; 8] = [0x15, 0x2A, 0x15, 0x2A, 0x15, 0x2A, 0x15, 0x2A];
    let (scale0, _min0) = extract_scale_min_from_slice(&scales, 0);
    let (scale2, _min2) = extract_scale_min_from_slice(&scales, 2);
    // Should be consistent
    assert!((scale0 - 21.0).abs() < 0.001);
    assert!((scale2 - 42.0).abs() < 0.001);
}

// =========================================================================
// Coverage Tests: Q4_0Block struct
// =========================================================================

#[test]
fn test_q4_0_block_debug_cov() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_0Block"));
}

#[test]
fn test_q4_0_block_clone_cov() {
    let block = Q4_0Block {
        scale: 2.5,
        quants: [0x12; 16],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q4_0_block_zero_scale_cov() {
    let block = Q4_0Block {
        scale: 0.0,
        quants: [0xFF; 16],
    };
    // Q4_0Block stores raw bytes, verify fields
    assert_eq!(block.scale, 0.0);
    assert_eq!(block.quants[0], 0xFF);
}

// =========================================================================
// Coverage Tests: Q8_0Block struct
// =========================================================================

#[test]
fn test_q8_0_block_debug_cov() {
    let block = Q8_0Block {
        scale: 0.5,
        quants: [0i8; 32],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q8_0Block"));
}

#[test]
fn test_q8_0_block_clone_cov() {
    let block = Q8_0Block {
        scale: 1.0,
        quants: [127i8; 32],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0_block_negative_quants_cov() {
    let mut quants = [0i8; 32];
    for i in 0..32 {
        quants[i] = -((i % 128) as i8);
    }
    let block = Q8_0Block { scale: 0.1, quants };
    let deq = block.dequantize();
    assert_eq!(deq.len(), 32);
    // First non-zero should be negative
    assert!(deq[1] < 0.0);
}

// =========================================================================
// Coverage Tests: Q4_KBlock struct
// =========================================================================

#[test]
fn test_q4_k_block_debug_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_KBlock"));
}

#[test]
fn test_q4_k_block_clone_cov() {
    let block = Q4_KBlock {
        d: 2.0,
        dmin: 1.0,
        scales: [0x3F; 12],
        qs: [0xAA; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
    assert_eq!(cloned.scales, block.scales);
    assert_eq!(cloned.qs, block.qs);
}

// =========================================================================
// Coverage Tests: Q5_KBlock struct
// =========================================================================

#[test]
fn test_q5_k_block_debug_cov() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q5_KBlock"));
}

#[test]
fn test_q5_k_block_clone_cov() {
    let block = Q5_KBlock {
        d: 3.0,
        dmin: 1.5,
        scales: [0x55; 12],
        qh: [0xFF; 32],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.qh, block.qh);
}

// =========================================================================
// Coverage Tests: Q6_KBlock struct
// =========================================================================

#[test]
fn test_q6_k_block_debug_cov() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q6_KBlock"));
}

#[test]
fn test_q6_k_block_clone_cov() {
    let block = Q6_KBlock {
        d: 4.0,
        scales: [127i8; 16],
        qh: [0xAA; 64],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.scales, block.scales);
}

// =========================================================================
// Coverage Tests: Q8KSuperBlock struct
// =========================================================================

#[test]
fn test_q8k_superblock_debug_cov() {
    let sb = Q8KSuperBlock {
        scale: 1.0,
        quants: [0i8; 256],
    };
    let debug_str = format!("{:?}", sb);
    assert!(debug_str.contains("Q8KSuperBlock"));
}

#[test]
fn test_q8k_superblock_clone_cov() {
    let sb = Q8KSuperBlock {
        scale: 2.0,
        quants: [64i8; 256],
    };
    let cloned = sb.clone();
    assert_eq!(cloned.scale, sb.scale);
    assert_eq!(cloned.quants[0], sb.quants[0]);
}

#[test]
fn test_q8k_superblock_quantize_zeros_cov() {
    let values = [0.0f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    // All zeros should produce near-zero quants
    for q in &sb.quants {
        assert_eq!(*q, 0);
    }
}

#[test]
fn test_q8k_superblock_quantize_max_values_cov() {
    let values = [127.0f32; 256];
    let sb = Q8KSuperBlock::quantize(&values);
    // Scale should handle max values
    assert!(sb.scale > 0.0);
    // All quants should be at max
    for q in &sb.quants {
        assert_eq!(*q, 127);
    }
}

#[test]
fn test_q8k_superblock_dequantize_roundtrip_cov() {
    let mut values = [0.0f32; 256];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32 - 128.0) * 0.1;
    }
    let sb = Q8KSuperBlock::quantize(&values);
    let deq = sb.dequantize();
    // Check roundtrip is approximate
    for (orig, deq_val) in values.iter().zip(deq.iter()) {
        let diff = (orig - deq_val).abs();
        assert!(diff < 0.2); // Quantization error tolerance
    }
}

// =========================================================================
// Coverage Tests: InterleavedQ4K struct
// =========================================================================

#[test]
fn test_interleaved_q4k_debug_cov() {
    let iq4k = InterleavedQ4K {
        d: vec![1.0],
        dmin: vec![0.5],
        scales: vec![0u8; 12],
        qs: vec![0u8; 128],
        num_super_blocks: 1,
    };
    let debug_str = format!("{:?}", iq4k);
    assert!(debug_str.contains("InterleavedQ4K"));
}

#[test]
fn test_interleaved_q4k_clone_cov() {
    let iq4k = InterleavedQ4K {
        d: vec![2.0, 3.0],
        dmin: vec![1.0, 1.5],
        scales: vec![0x55; 24],
        qs: vec![0xAA; 256],
        num_super_blocks: 2,
    };
    let cloned = iq4k.clone();
    assert_eq!(cloned.num_super_blocks, iq4k.num_super_blocks);
    assert_eq!(cloned.d, iq4k.d);
}
