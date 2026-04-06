
#[test]
fn test_fused_swiglu_simd_larger_cov() {
    // Test with larger vectors for better coverage
    let mut gate = vec![1.0f32; 16];
    let up = vec![1.0f32; 16];
    fused_swiglu_simd(&mut gate, &up);
    // gate = 1 * sigmoid(1) * 1 ≈ 0.731
    for v in &gate {
        assert!((*v - 0.731).abs() < 0.05);
    }
}

// =========================================================================
// Coverage Tests: quantize_activations_q8_0 (cov suffix)
// =========================================================================

#[test]
fn test_quantize_activations_q8_0_basic_cov() {
    let activations = vec![1.0f32; 64];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // 64 / 32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_zeros_cov() {
    let activations = vec![0.0f32; 32];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    // Quants should all be zero
    for q in &quants {
        assert_eq!(*q, 0);
    }
}

// =========================================================================
// Coverage Tests: Struct Debug/Clone implementations
// =========================================================================

#[test]
fn test_q4_0block_debug_clone() {
    let block = Q4_0Block {
        scale: 1.5,
        quants: [0u8; 16],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_0Block"));
    assert!(debug.contains("1.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0block_debug_clone() {
    let block = Q8_0Block {
        scale: 2.5,
        quants: [1i8; 32],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8_0Block"));
    assert!(debug.contains("2.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8_0block_relative_error() {
    let original = [1.0f32; 32];
    let block = Q8_0Block::quantize(&original);
    let rel_err = block.relative_error(&original);
    assert!(
        rel_err < 0.01,
        "Relative error should be small for uniform values"
    );

    // Test with zeros
    let zeros = [0.0f32; 32];
    let block_zeros = Q8_0Block::quantize(&zeros);
    let rel_err_zeros = block_zeros.relative_error(&zeros);
    assert_eq!(rel_err_zeros, 0.0, "Relative error for zeros should be 0");
}

#[test]
fn test_q8_0block_quantization_error() {
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) / 10.0);
    let block = Q8_0Block::quantize(&original);
    let error = block.quantization_error(&original);
    // Max error should be bounded
    assert!(error < 0.05, "Quantization error should be bounded");
}

#[test]
fn test_q8ksuperblock_debug_clone() {
    let block = Q8KSuperBlock {
        scale: 3.5,
        quants: [1i8; 256],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q8KSuperBlock"));
    assert!(debug.contains("3.5"));

    let cloned = block.clone();
    assert_eq!(cloned.scale, block.scale);
    assert_eq!(cloned.quants, block.quants);
}

#[test]
fn test_q8ksuperblock_dequantize() {
    let mut block = Q8KSuperBlock {
        scale: 0.1,
        quants: [0i8; 256],
    };
    // Set some values
    block.quants[0] = 127;
    block.quants[1] = -128;
    block.quants[255] = 50;

    let dequant = block.dequantize();
    assert_eq!(dequant.len(), 256);
    assert!((dequant[0] - 12.7).abs() < 0.01); // 127 * 0.1
    assert!((dequant[1] - (-12.8)).abs() < 0.01); // -128 * 0.1
    assert!((dequant[255] - 5.0).abs() < 0.01); // 50 * 0.1
}

#[test]
fn test_q4_kblock_debug_clone() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
}

#[test]
fn test_q5_kblock_debug_clone() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.25,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q5_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
    assert_eq!(cloned.dmin, block.dmin);
}

#[test]
fn test_q6_kblock_debug_clone() {
    let block = Q6_KBlock {
        d: 1.5,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q6_KBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.d, block.d);
}

#[test]
fn test_interleaved_q4k_debug_clone() {
    // Create minimal valid Q4K data (144 bytes)
    let mut q4k_data = vec![0u8; 144];
    // Set d and dmin as f16
    q4k_data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    q4k_data[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    let interleaved = InterleavedQ4K::from_q4k(&q4k_data).expect("test");
    let debug = format!("{:?}", interleaved);
    assert!(debug.contains("InterleavedQ4K"));

    let cloned = interleaved.clone();
    assert_eq!(cloned.num_super_blocks, interleaved.num_super_blocks);
    assert_eq!(cloned.num_values(), 256);
}

#[test]
fn test_interleaved_q4k_invalid_length_cov() {
    let bad_data = vec![0u8; 143]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&bad_data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: f16_to_f32 edge cases
// =========================================================================

#[test]
fn test_f16_to_f32_positive_zero() {
    let result = f16_to_f32(0x0000);
    assert_eq!(result, 0.0);
    assert!(!result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_negative_zero() {
    let result = f16_to_f32(0x8000);
    assert_eq!(result, -0.0);
    assert!(result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_positive_infinity() {
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite());
    assert!(result.is_sign_positive());
}

#[test]
fn test_f16_to_f32_negative_infinity() {
    let result = f16_to_f32(0xFC00);
    assert!(result.is_infinite());
    assert!(result.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan_cov() {
    let result = f16_to_f32(0x7C01); // NaN with non-zero mantissa
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_subnormal() {
    // Smallest positive subnormal: 2^-24 ≈ 5.96e-8
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-6);

    // Negative subnormal
    let result_neg = f16_to_f32(0x8001);
    assert!(result_neg < 0.0);
    assert!(result_neg > -1e-6);
}

#[test]
fn test_f16_to_f32_normal_values() {
    // Test 1.0
    let one = f16_to_f32(0x3C00);
    assert!((one - 1.0).abs() < 1e-5);

    // Test -1.0
    let neg_one = f16_to_f32(0xBC00);
    assert!((neg_one - (-1.0)).abs() < 1e-5);

    // Test 2.0
    let two = f16_to_f32(0x4000);
    assert!((two - 2.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: dequantize_q4_1 valid case
// =========================================================================

#[test]
fn test_dequantize_q4_1_valid() {
    // Create valid Q4_1 data: 20 bytes per block
    let mut data = vec![0u8; 20];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 (f16)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // 16 bytes of quants (all zeros = 0s)
    // zeros should produce: d * 0 + min = 0 for low nibble, d * 0 + min = 0 for high nibble

    let result = dequantize_q4_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // With zero quants and zero min, all values should be 0
    for v in &result {
        assert!(v.abs() < 0.001);
    }
}

#[test]
fn test_dequantize_q4_1_invalid_length_cov() {
    let data = vec![0u8; 19]; // Not multiple of 20
    let result = dequantize_q4_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_q5_0 and dequantize_q5_1
// =========================================================================

#[test]
fn test_dequantize_q5_0_valid() {
    // Q5_0: 22 bytes per block
    let mut data = vec![0u8; 22];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // High bits (4 bytes)
    // Low bits (16 bytes)
    // All zeros = 0 - 16 = -16 (per dequant formula)

    let result = dequantize_q5_0(&data).expect("test");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_0_invalid_length_cov() {
    let data = vec![0u8; 21]; // Not multiple of 22
    let result = dequantize_q5_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_1_valid() {
    // Q5_1: 24 bytes per block
    let mut data = vec![0u8; 24];
    // Scale = 1.0 (f16)
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    // Min = 0.0 (f16)
    data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    // High bits (4 bytes), low bits (16 bytes)

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_dequantize_q5_1_invalid_length_cov() {
    let data = vec![0u8; 23]; // Not multiple of 24
    let result = dequantize_q5_1(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: dequantize_f16
// =========================================================================

#[test]
fn test_dequantize_f16_valid() {
    // Create f16 data for value 1.0
    let f16_one = half::f16::from_f32(1.0).to_le_bytes();
    let f16_two = half::f16::from_f32(2.0).to_le_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&f16_one);
    data.extend_from_slice(&f16_two);

    let result = dequantize_f16(&data).expect("test");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_dequantize_f16_invalid_length_cov() {
    let data = vec![0u8; 3]; // Not multiple of 2
    let result = dequantize_f16(&data);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_dequant_stats_debug_clone() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 800,
        simd_backend: SimdBackend::Avx2,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("DequantStats"));
    assert!(debug.contains("100"));

    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, stats.blocks_processed);
    assert_eq!(cloned.bytes_processed, stats.bytes_processed);
}

#[test]
fn test_simd_backend_debug_clone() {
    let backend = SimdBackend::Avx2;
    let debug = format!("{:?}", backend);
    assert!(debug.contains("Avx2"));

    let cloned = backend;
    assert_eq!(cloned, backend);

    // Test equality
    assert_eq!(SimdBackend::Sse2, SimdBackend::Sse2);
    assert_ne!(SimdBackend::Avx2, SimdBackend::Scalar);
}

#[test]
fn test_detect_simd_backend_cov() {
    let backend = detect_simd_backend();
    let debug = format!("{:?}", backend);
    // Just verify it returns something valid
    assert!(
        backend == SimdBackend::Avx2
            || backend == SimdBackend::Sse2
            || backend == SimdBackend::Neon
            || backend == SimdBackend::Scalar
    );
    println!("Detected SIMD backend: {}", debug);
}

// =========================================================================
// Coverage Tests: quantize_activations_q8k_into error paths
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_invalid_length_cov() {
    let activations = vec![0.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_small_scales_buffer() {
    let activations = vec![0.0f32; 512]; // 2 super-blocks
    let mut scales = vec![0.0f32; 1]; // Too small
    let mut quants = vec![0i8; 512];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_small_quants_buffer() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}
