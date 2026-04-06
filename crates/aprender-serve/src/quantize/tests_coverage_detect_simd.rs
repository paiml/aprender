
// ============================================================================
// detect_simd_backend
// ============================================================================

#[test]
fn test_detect_simd_backend_returns_valid() {
    let backend = detect_simd_backend();
    // On x86_64, should be at least SSE2 or AVX2
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            backend == SimdBackend::Avx2 || backend == SimdBackend::Sse2,
            "On x86_64, should detect AVX2 or SSE2, got {:?}",
            backend
        );
    }
    // On any platform, should return a valid variant
    let display = format!("{}", backend);
    assert!(!display.is_empty());
}

// ============================================================================
// Q8_0Block
// ============================================================================

#[test]
fn test_q8_0_block_quantize_ones() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    assert!(block.scale > 0.0);
    assert!(block.quants.iter().all(|&q| q == 127)); // All max
}

#[test]
fn test_q8_0_block_quantize_zeros() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    // Near-zero values -> minimal scale, all quants zero
    assert!(block.quants.iter().all(|&q| q == 0));
}

#[test]
fn test_q8_0_block_quantize_mixed() {
    let mut values = [0.0f32; 32];
    values[0] = 1.0;
    values[1] = -1.0;
    values[2] = 0.5;
    let block = Q8_0Block::quantize(&values);
    assert!(block.quants[0] > 0);
    assert!(block.quants[1] < 0);
    assert!(block.quants[2] > 0);
}

#[test]
fn test_q8_0_block_dequantize() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let dequantized = block.dequantize();
    for &v in dequantized.iter() {
        assert!(
            (v - 1.0).abs() < 0.02,
            "Dequantized should be near 1.0, got {}",
            v
        );
    }
}

#[test]
fn test_q8_0_block_quantization_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let error = block.quantization_error(&values);
    assert!(
        error < 0.02,
        "Quantization error should be small, got {}",
        error
    );
}

#[test]
fn test_q8_0_block_relative_error() {
    let values = [1.0f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert!(
        rel_error < 0.02,
        "Relative error should be small, got {}",
        rel_error
    );
}

#[test]
fn test_q8_0_block_relative_error_near_zero() {
    let values = [1e-12f32; 32];
    let block = Q8_0Block::quantize(&values);
    let rel_error = block.relative_error(&values);
    assert_eq!(
        rel_error, 0.0,
        "Near-zero values should return 0 relative error"
    );
}

// ============================================================================
// Q8KSuperBlock
// ============================================================================

#[test]
fn test_q8k_superblock_quantize() {
    let values = [0.5f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.scale > 0.0);
    // All same value -> all quants should be the same
    assert!(block.quants.iter().all(|&q| q == block.quants[0]));
}

#[test]
fn test_q8k_superblock_quantize_zeros() {
    let values = [0.0f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    assert!(block.quants.iter().all(|&q| q == 0));
}

#[test]
fn test_q8k_superblock_dequantize() {
    let values = [0.5f32; 256];
    let block = Q8KSuperBlock::quantize(&values);
    let dequantized = block.dequantize();
    for &v in dequantized.iter() {
        assert!(
            (v - 0.5).abs() < 0.01,
            "Dequantized should be near 0.5, got {}",
            v
        );
    }
}

#[test]
fn test_q8k_superblock_quantize_into() {
    let values = vec![0.3f32; 256];
    let mut scale = 0.0f32;
    let mut quants = vec![0i8; 256];

    Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);
    assert!(scale > 0.0);
    assert!(quants.iter().all(|&q| q == quants[0]));
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant() {
    assert_eq!(QK_K, 256);
}

// ============================================================================
// T-COV-95: Additional coverage for quantize/mod.rs pure functions
// ============================================================================

// --- fused_q4_0_q8_0_dot_scalar: known value computation ---

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_known_values() {
    // Q4_0 block: 2 bytes (f16 scale) + 16 bytes (quants) = 18 bytes
    // f16 for 1.0 = 0x3C00
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // scale = 1.0

    // Set quants: byte 0 = 0x98 -> low nibble=8, high nibble=9
    // Q4_0 dequant: (nibble - 8) * scale
    // low_quant = 8 - 8 = 0, high_quant = 9 - 8 = 1
    q4_data[2] = 0x98;
    // All other quant bytes are 0 => (0 - 8) = -8 for both nibbles

    // Q8 activations: scale = 1.0, quants all = 1
    let q8_scales = vec![1.0f32];
    let q8_quants = vec![1i8; 32];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);

    // Manually compute:
    // byte 0: low_quant = (8 - 8) = 0, act[0] = 1 => 0*1 = 0
    //         high_quant = (9 - 8) = 1, act[16] = 1 => 1*1 = 1
    // bytes 1-15: low nibble = 0-8 = -8, high = 0-8 = -8
    //   15 * (-8 * 1) + 15 * (-8 * 1) = -240
    // total integer sum = 0 + 1 + (-240) = -239
    // combined_scale = 1.0 * 1.0 = 1.0
    // result = 1.0 * (-239.0) = -239.0
    assert!(
        (result - (-239.0)).abs() < 0.01,
        "Expected -239.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_multi_block() {
    // 2 blocks = 36 bytes, 64 elements
    let mut q4_data = vec![0u8; 36];
    // Block 0: scale = 0.5 (f16 = 0x3800)
    q4_data[0] = 0x00;
    q4_data[1] = 0x38;
    // All zero quants => all nibbles = 0, offset by -8 => val = -8

    // Block 1: scale = 0
    // All zeros => no contribution

    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![1i8; 64];

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 64);
    // Block 0: 32 values, all = (0 - 8) * 0.5 = -4.0 as weight, act = 1
    // Sum of q4*q8 (integer) = 32 * (-8 * 1) = -256
    // scale = 0.5 * 1.0 = 0.5, block_sum = 0.5 * -256 = -128.0
    // Block 1: scale = 0, so 0 contribution
    assert!(
        (result - (-128.0)).abs() < 0.1,
        "Expected about -128.0, got {}",
        result
    );
}

#[test]
fn test_fused_q4_0_q8_0_dot_scalar_negative_quants() {
    let mut q4_data = vec![0u8; 18];
    q4_data[0] = 0x00;
    q4_data[1] = 0x3C; // scale = 1.0
                       // All quant bytes = 0xFF => low nibble = 15, high nibble = 15
                       // (15 - 8) = 7 for each
    for i in 2..18 {
        q4_data[i] = 0xFF;
    }

    let q8_scales = vec![1.0f32];
    let q8_quants = vec![-1i8; 32]; // All negative activations

    let result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 32);
    // All q4 values = 7, all q8 = -1
    // block_sum = 32 * (7 * -1) = -224
    // result = 1.0 * (-224) = -224.0
    assert!(
        (result - (-224.0)).abs() < 0.01,
        "Expected -224.0, got {}",
        result
    );
}

// --- fused_q8_0_q8_0_dot_scalar ---

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_zero() {
    // Q8_0 weight block: 34 bytes (2 scale + 32 quants)
    let q8_weight_data = vec![0u8; 34];
    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![0i8; 32];
    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_empty() {
    let result = fused_q8_0_q8_0_dot_scalar(&[], &[], &[], 0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_known_values() {
    let mut q8_weight_data = vec![0u8; 34];
    // weight scale = 1.0 (f16 = 0x3C00)
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C;
    // weight quants: all = 10 (as i8 bytes)
    for i in 0..32 {
        q8_weight_data[2 + i] = 10u8; // i8 value = 10
    }

    let q8_act_scales = vec![2.0f32];
    let q8_act_quants = vec![5i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // block_sum = sum(10 * 5) for 32 values = 32 * 50 = 1600
    // combined_scale = 1.0 * 2.0 = 2.0
    // result = 2.0 * 1600 = 3200.0
    assert!(
        (result - 3200.0).abs() < 1.0,
        "Expected about 3200.0, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_negative_weights() {
    let mut q8_weight_data = vec![0u8; 34];
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C; // scale = 1.0
                              // weight quants: all = -5 (0xFB as u8)
    for i in 0..32 {
        #[allow(clippy::cast_sign_loss)]
        {
            q8_weight_data[2 + i] = (-5i8) as u8;
        }
    }

    let q8_act_scales = vec![1.0f32];
    let q8_act_quants = vec![3i8; 32];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 32);
    // block_sum = 32 * (-5 * 3) = 32 * -15 = -480
    // combined_scale = 1.0 * 1.0 = 1.0
    // result = -480.0
    assert!(
        (result - (-480.0)).abs() < 1.0,
        "Expected about -480.0, got {}",
        result
    );
}

#[test]
fn test_fused_q8_0_q8_0_dot_scalar_multi_block() {
    // 2 blocks = 68 bytes, 64 elements
    let mut q8_weight_data = vec![0u8; 68];
    // Block 0: scale = 1.0, quants all = 1
    q8_weight_data[0] = 0x00;
    q8_weight_data[1] = 0x3C;
    for i in 0..32 {
        q8_weight_data[2 + i] = 1u8;
    }
    // Block 1: scale = 2.0 (f16 = 0x4000), quants all = 2
    q8_weight_data[34] = 0x00;
    q8_weight_data[35] = 0x40;
    for i in 0..32 {
        q8_weight_data[36 + i] = 2u8;
    }

    let q8_act_scales = vec![1.0f32; 2];
    let q8_act_quants = vec![1i8; 64];

    let result = fused_q8_0_q8_0_dot_scalar(&q8_weight_data, &q8_act_scales, &q8_act_quants, 64);
    // Block 0: combined = 1.0 * 1.0 = 1.0, sum = 32 * (1*1) = 32, contrib = 32.0
    // Block 1: combined = 2.0 * 1.0 = 2.0, sum = 32 * (2*1) = 64, contrib = 128.0
    // total = 160.0
    assert!(
        (result - 160.0).abs() < 1.0,
        "Expected about 160.0, got {}",
        result
    );
}

// --- fused_q4_0_q8_0_dot_simd vs scalar parity ---

#[test]
fn test_fused_q4_0_q8_0_dot_simd_vs_scalar() {
    // Build deterministic Q4_0 data: 4 blocks = 72 bytes, 128 elements
    let mut q4_data = vec![0u8; 72]; // 4 blocks * 18 bytes
    for b in 0..4 {
        let offset = b * 18;
        // scale = 0.5 (f16 = 0x3800)
        q4_data[offset] = 0x00;
        q4_data[offset + 1] = 0x38;
        for i in 0..16 {
            q4_data[offset + 2 + i] = ((b * 17 + i * 3) % 256) as u8;
        }
    }

    let activations: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let (q8_scales, q8_quants) =
        crate::quantize::activation::quantize_activations_q8_0(&activations);

    let scalar_result = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 128);
    let simd_result = fused_q4_0_q8_0_dot_simd(&q4_data, &q8_scales, &q8_quants, 128);

    let tol = if scalar_result.abs() > 1e-6 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else {
        (simd_result - scalar_result).abs()
    };
    assert!(
        tol < 0.01,
        "scalar={} simd={} rel_err={}",
        scalar_result,
        simd_result,
        tol
    );
}

// --- InterleavedQ4K::dot correctness ---

#[test]
fn test_interleaved_q4k_dot_with_nonzero_data() {
    let mut data = vec![0u8; 144];
    // d = 1.0 (f16 = 0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    // dmin = 0.0
    data[2] = 0x00;
    data[3] = 0x00;
    // Set scales: first scale byte = 1 (scale=1, min=0 for block 0)
    data[4] = 0x01;
    // qs: set to pattern 0x11 (low=1, high=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }

    let iq = InterleavedQ4K::from_q4k(&data).expect("valid data");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations).expect("dot should succeed");

    // With scale[0]=1, min=0: values are d*1*1 = 1.0 for low nibble group (32 values)
    // Other blocks have scale=0 so they produce -dmin*min which is 0
    // Should produce a positive non-zero value
    assert!(
        result.abs() > 0.0,
        "Expected non-zero result, got {}",
        result
    );
}

#[test]
fn test_interleaved_q4k_dot_activation_length_mismatch() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("valid");
    let activations = vec![1.0f32; 128]; // Should be 256
    let result = iq.dot(&activations);
    assert!(result.is_err());
}
