
// --- quantize_to_q8_blocks: varied inputs ---

#[test]
fn test_quantize_to_q8_blocks_large_values() {
    let mut values = vec![0.0f32; 32];
    for i in 0..32 {
        values[i] = (i as f32 - 16.0) * 100.0;
    }
    let blocks = quantize_to_q8_blocks(&values).expect("should work");
    assert_eq!(blocks.len(), 1);

    let dequant = dequantize_q8_blocks(&blocks);
    // Verify approximate round-trip
    for (o, d) in values.iter().zip(dequant.iter()) {
        let diff = (o - d).abs();
        assert!(
            diff < blocks[0].scale * 2.0,
            "Too large error: {} vs {}",
            o,
            d
        );
    }
}

#[test]
fn test_quantize_to_q8_blocks_zeros() {
    let values = vec![0.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("should work");
    assert_eq!(blocks.len(), 1);
    for q in &blocks[0].quants {
        assert_eq!(*q, 0);
    }
}

// --- quantize_activations_q8k_into: edge cases ---

#[test]
fn test_quantize_activations_q8k_into_large_values() {
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 100.0).collect();
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];

    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    // The max abs value should map to approximately 127
    let max_quant = quants.iter().map(|q| q.unsigned_abs()).max().unwrap_or(0);
    assert!(
        max_quant >= 126,
        "Max quant should be near 127, got {}",
        max_quant
    );
}

// --- fused_q4_0_q8_0_parallel_matvec: larger parallel path ---

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_multi_row() {
    // 4 rows of 32 elements each = 4 Q4_0 blocks (72 bytes)
    let weight_data = vec![0u8; 72]; // 4 rows * 18 bytes/row
    let activations = vec![0.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 4);
    // With zero activations, all outputs should be zero
    for &v in &output {
        assert_eq!(v, 0.0);
    }
}

// --- fused_q8_0_q8_0_parallel_matvec: success and error paths ---

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_weight_too_small() {
    let weight_data = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 2);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_activation_mismatch() {
    let weight_data = vec![0u8; 34]; // 1 row * 34 bytes
    let activations = vec![1.0f32; 64]; // Wrong size (should be 32)
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_success() {
    let weight_data = vec![0u8; 34]; // 1 row, 32 elements
    let activations = vec![0.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_multi_row() {
    // 4 rows of 32 elements each => 4 * 34 = 136 bytes
    let weight_data = vec![0u8; 136];
    let activations = vec![0.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weight_data, &activations, 32, 4);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 4);
}

// --- fused_q8_0_q8_0_parallel_matvec_into: success and error paths ---

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_weight_too_small() {
    let weight_data = vec![0u8; 10];
    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_activation_mismatch() {
    let weight_data = vec![0u8; 34];
    let activations = vec![1.0f32; 64]; // Wrong
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_output_too_small() {
    let weight_data = vec![0u8; 68]; // 2 rows * 34
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1]; // Need 2
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 2, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_success() {
    let weight_data = vec![0u8; 34];
    let activations = vec![0.0f32; 32];
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q8_0_q8_0_parallel_matvec_into(&weight_data, &activations, 32, 1, &mut output);
    assert!(result.is_ok());
}

// --- f16_to_f32_lut: additional edge cases ---

#[test]
fn test_f16_to_f32_lut_half() {
    // f16 0.5 = 0x3800
    let val = f16_to_f32_lut(0x3800);
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_two() {
    // f16 2.0 = 0x4000
    let val = f16_to_f32_lut(0x4000);
    assert!((val - 2.0).abs() < 0.001, "Expected 2.0, got {}", val);
}

#[test]
fn test_f16_to_f32_lut_infinity() {
    // f16 infinity = 0x7C00
    let val = f16_to_f32_lut(0x7C00);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn test_f16_to_f32_lut_neg_infinity() {
    // f16 neg infinity = 0xFC00
    let val = f16_to_f32_lut(0xFC00);
    assert!(val.is_infinite() && val < 0.0);
}

#[test]
fn test_f16_to_f32_lut_nan() {
    // f16 NaN = 0x7C01
    let val = f16_to_f32_lut(0x7C01);
    assert!(val.is_nan());
}

#[test]
fn test_f16_to_f32_lut_negative_zero() {
    let val = f16_to_f32_lut(0x8000);
    assert!(val == 0.0 && val.is_sign_negative());
}

// --- InterleavedQ4K: d and dmin extraction ---

#[test]
fn test_interleaved_q4k_extracts_d_dmin() {
    let mut data = vec![0u8; 144];
    // d = 2.0 (f16 = 0x4000)
    data[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    // dmin = 0.25 (f16 = 0x3400)
    data[2..4].copy_from_slice(&0x3400u16.to_le_bytes());

    let iq = InterleavedQ4K::from_q4k(&data).expect("valid data");
    assert!(
        (iq.d[0] - 2.0).abs() < 0.01,
        "d should be 2.0, got {}",
        iq.d[0]
    );
    assert!(
        (iq.dmin[0] - 0.25).abs() < 0.01,
        "dmin should be 0.25, got {}",
        iq.dmin[0]
    );
}

// --- Q8_0Block: clamping behavior ---

#[test]
fn test_q8_0_block_quantize_extreme_values() {
    // Values that require clamping
    let mut values = [0.0f32; 32];
    values[0] = 1000.0;
    values[1] = -1000.0;
    let block = Q8_0Block::quantize(&values);
    assert_eq!(block.quants[0], 127); // Clamped to max
    assert_eq!(block.quants[1], -127); // Clamped to min (symmetric)
}

// --- Q8KSuperBlock: roundtrip with varied values ---

#[test]
fn test_q8k_superblock_roundtrip_varied() {
    let mut values = [0.0f32; 256];
    for i in 0..256 {
        values[i] = (i as f32 - 128.0) * 0.5;
    }
    let block = Q8KSuperBlock::quantize(&values);
    let dequant = block.dequantize();
    for (orig, deq) in values.iter().zip(dequant.iter()) {
        let diff = (orig - deq).abs();
        assert!(
            diff < block.scale * 2.0,
            "Roundtrip error too large: orig={}, deq={}, diff={}, scale={}",
            orig,
            deq,
            diff,
            block.scale
        );
    }
}

// --- Q4_KBlock, Q5_KBlock, Q6_KBlock: clone and debug ---

#[test]
fn test_q4_k_block_clone() {
    let block = Q4_KBlock {
        d: 1.5,
        dmin: 0.3,
        scales: [7; 12],
        qs: [0xAB; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 1.5);
    assert_eq!(cloned.dmin, 0.3);
    assert_eq!(cloned.scales, [7; 12]);
    assert_eq!(cloned.qs, [0xAB; 128]);
}

#[test]
fn test_q4_k_block_debug() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0; 12],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q4_KBlock"));
}

#[test]
fn test_q5_k_block_clone() {
    let block = Q5_KBlock {
        d: 2.0,
        dmin: 0.1,
        scales: [1; 12],
        qh: [0xFF; 32],
        qs: [0x55; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 2.0);
    assert_eq!(cloned.qh, [0xFF; 32]);
}

#[test]
fn test_q5_k_block_debug() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0; 12],
        qh: [0; 32],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q5_KBlock"));
}

#[test]
fn test_q6_k_block_clone() {
    let block = Q6_KBlock {
        d: 0.5,
        scales: [3; 16],
        qh: [0xAA; 64],
        qs: [0x33; 128],
    };
    let cloned = block.clone();
    assert_eq!(cloned.d, 0.5);
    assert_eq!(cloned.scales, [3; 16]);
}

#[test]
fn test_q6_k_block_debug() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0; 16],
        qh: [0; 64],
        qs: [0; 128],
    };
    let debug = format!("{:?}", block);
    assert!(debug.contains("Q6_KBlock"));
}

// --- InterleavedQ4K: clone and debug ---

#[test]
fn test_interleaved_q4k_clone_debug() {
    let data = vec![0u8; 144];
    let iq = InterleavedQ4K::from_q4k(&data).expect("valid");
    let cloned = iq.clone();
    assert_eq!(cloned.num_super_blocks, 1);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("InterleavedQ4K"));
}

// --- Q4_0Block: clone and debug ---

#[test]
fn test_q4_0_block_clone_debug() {
    let block = Q4_0Block {
        scale: 1.0,
        quants: [0x55; 16],
    };
    let cloned = block.clone();
    assert_eq!(cloned.scale, 1.0);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("Q4_0Block"));
}

// --- DequantStats: clone, debug ---

#[test]
fn test_dequant_stats_clone_round_trip() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 756,
        simd_backend: SimdBackend::Neon,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 42);
    assert_eq!(cloned.bytes_processed, 756);
    assert_eq!(cloned.simd_backend, SimdBackend::Neon);
}

// ============================================================================
// FUSED Q4_0 × Q8_0 DOT PRODUCT — AVX2 DIRECT COVERAGE TESTS
// ============================================================================
// On AVX-512 VNNI machines, the public API dispatches to the AVX512 path,
// making these AVX2 functions unreachable. Test them directly.

/// Build a Q4_0 block: 2 bytes (f16 scale) + 16 bytes (nibbles for 32 values) = 18 bytes.
fn build_q4_0_test_block(scale: f32, nibble_val: u8) -> [u8; 18] {
    let mut block = [0u8; 18];
    let scale_bits = half::f16::from_f32(scale).to_bits();
    block[0..2].copy_from_slice(&scale_bits.to_le_bytes());
    let packed = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..16 {
        block[2 + i] = packed;
    }
    block
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 4 blocks = 128 elements (< 256, so avx2 2-block path)
    let block = build_q4_0_test_block(1.0, 5);
    let mut q4_data = Vec::with_capacity(18 * 4);
    for _ in 0..4 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 4];
    let q8_quants = vec![2i8; 128];

    let scalar = fused_q4_0_q8_0_dot_scalar(&q4_data, &q8_scales, &q8_quants, 128);
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 128) };

    let diff = (scalar - avx2).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(diff < tol, "scalar={scalar} vs avx2={avx2}, diff={diff}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4_0_avx2_dot_zero_quants() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let block = build_q4_0_test_block(1.0, 8); // nibble=8 → 8-8=0 after offset
    let mut q4_data = Vec::with_capacity(18 * 2);
    for _ in 0..2 {
        q4_data.extend_from_slice(&block);
    }
    let q8_scales = vec![1.0f32; 2];
    let q8_quants = vec![0i8; 64];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4_0_q8_0_dot_avx2(&q4_data, &q8_scales, &q8_quants, 64) };
    assert!(
        result.abs() < 1e-3,
        "zero × zero should produce ~0, got {result}"
    );
}
