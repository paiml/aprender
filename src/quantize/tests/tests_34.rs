//! T-COV-95 Phase 50: Deep coverage for quantize/mod.rs and quantize/activation.rs
//!
//! Covers:
//! - quantize_activations_q8k_into error paths
//! - quantize_to_q8_blocks and dequantize_q8_blocks roundtrip
//! - InterleavedQ4K::dot_scalar comprehensive tests
//! - fused_swiglu_scalar math correctness
//! - softmax_scalar edge cases
//! - quantize_activations_q8_0 partial block padding
//! - quantize_rmsnorm_q8_0_into zero-allocation variant
//! - quantize_rmsnorm_q8_0_scalar boundary conditions

use crate::quantize::activation::{
    fused_swiglu_scalar, quantize_activations_q8_0, quantize_rmsnorm_q8_0_into,
    quantize_rmsnorm_q8_0_scalar, softmax_scalar,
};
use crate::quantize::{
    dequantize_q8_blocks, quantize_activations_q8k_into, quantize_to_q8_blocks, InterleavedQ4K,
};

// ============================================================================
// quantize_activations_q8k_into error paths
// ============================================================================

#[test]
fn test_q8k_into_not_multiple_of_256() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("multiple of 256"),
        "Expected multiple-of-256 error, got: {}",
        err
    );
}

#[test]
fn test_q8k_into_scales_buffer_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 0]; // Too small: need 1
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small"),
        "Expected buffer-too-small error, got: {}",
        err
    );
}

#[test]
fn test_q8k_into_quants_buffer_too_small() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small: need 256
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small"),
        "Expected buffer-too-small error, got: {}",
        err
    );
}

#[test]
fn test_q8k_into_success_single_block() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    // All values are 1.0, so all quants should be the same
    let first = quants[0];
    for &q in &quants {
        assert_eq!(q, first);
    }
}

#[test]
fn test_q8k_into_success_multiple_blocks() {
    let activations: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 100.0).collect();
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0);
    assert!(scales[1] > 0.0);
}

#[test]
fn test_q8k_into_zero_activations() {
    let activations = vec![0.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
    assert!(scales[0] > 0.0); // Minimal scale to avoid div-by-zero
    for &q in &quants {
        assert_eq!(q, 0);
    }
}

// ============================================================================
// quantize_to_q8_blocks and dequantize_q8_blocks
// ============================================================================

#[test]
fn test_q8_blocks_not_multiple_of_32() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("multiple of 32"),
        "Expected multiple-of-32 error, got: {}",
        err
    );
}

#[test]
fn test_q8_blocks_roundtrip_uniform() {
    let values = vec![42.0f32; 64]; // 2 blocks
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert_eq!(blocks.len(), 2);

    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 64);

    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        assert!(
            (orig - deq).abs() < 1.0,
            "Roundtrip error: orig={}, deq={}",
            orig,
            deq
        );
    }
}

#[test]
fn test_q8_blocks_roundtrip_mixed() {
    let values: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 2.0).collect();
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert_eq!(blocks.len(), 3);

    let dequantized = dequantize_q8_blocks(&blocks);
    assert_eq!(dequantized.len(), 96);

    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let diff = (orig - deq).abs();
        // Q8 quantization should be within 2x scale
        assert!(diff < 2.0, "Roundtrip error too large: diff={}", diff);
    }
}

#[test]
fn test_q8_blocks_roundtrip_zeros() {
    let values = vec![0.0f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    let dequantized = dequantize_q8_blocks(&blocks);
    for deq in &dequantized {
        assert!((deq - 0.0).abs() < 0.01);
    }
}

#[test]
fn test_q8_blocks_empty() {
    let values: Vec<f32> = vec![];
    let blocks = quantize_to_q8_blocks(&values).expect("test value should be present");
    assert!(blocks.is_empty());
    let dequantized = dequantize_q8_blocks(&blocks);
    assert!(dequantized.is_empty());
}

// ============================================================================
// InterleavedQ4K dot_scalar
// ============================================================================

#[test]
fn test_interleaved_q4k_dot_empty() {
    let data = vec![];
    let iq = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![];
    let result = iq.dot(&activations);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0.0);
}

#[test]
fn test_interleaved_q4k_dot_mismatch() {
    let data = vec![0u8; 144]; // 1 super-block = 256 values
    let iq = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 128]; // Wrong size
    let result = iq.dot(&activations);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_dot_zero_weights() {
    let data = vec![0u8; 144]; // All zeros, d=0
    let iq = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations).expect("test value should be present");
    assert_eq!(result, 0.0); // d=0 means all values dequantize to 0
}

#[test]
fn test_interleaved_q4k_dot_nonzero() {
    let mut data = vec![0u8; 144];
    // Set d = 1.0 (f16: 0x3C00)
    data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set dmin = 0
    data[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // Set scales to 1 (all low bits)
    for i in 0..12 {
        data[4 + i] = 0x01;
    }
    // Set qs to 0x11 (low=1, high=1)
    for i in 0..128 {
        data[16 + i] = 0x11;
    }
    let iq = InterleavedQ4K::from_q4k(&data).expect("test value should be present");
    let activations = vec![1.0f32; 256];
    let result = iq.dot(&activations).expect("test value should be present");
    // Result should be non-zero since d > 0, scales > 0, qs > 0
    assert!(result.abs() > 0.0, "Expected non-zero dot product");
}

// ============================================================================
// fused_swiglu_scalar
// ============================================================================

#[test]
fn test_swiglu_scalar_zeros() {
    let mut gate = vec![0.0f32; 8];
    let up = vec![1.0f32; 8];
    fused_swiglu_scalar(&mut gate, &up);
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    for &g in &gate {
        assert!((g - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_swiglu_scalar_positive() {
    let mut gate = vec![2.0f32; 4];
    let up = vec![1.0f32; 4];
    fused_swiglu_scalar(&mut gate, &up);
    // silu(2) = 2 * sigmoid(2) = 2 / (1 + exp(-2)) ~= 1.7616
    for &g in &gate {
        assert!((g - 1.7616).abs() < 0.01, "got {}", g);
    }
}

#[test]
fn test_swiglu_scalar_negative() {
    let mut gate = vec![-5.0f32; 4];
    let up = vec![1.0f32; 4];
    fused_swiglu_scalar(&mut gate, &up);
    // silu(-5) = -5 * sigmoid(-5) = -5 / (1 + exp(5)) ~= -0.0337
    for &g in &gate {
        assert!((g - (-0.0337)).abs() < 0.01, "got {}", g);
    }
}

#[test]
fn test_swiglu_scalar_with_up_scaling() {
    let mut gate = vec![1.0f32; 4];
    let up = vec![3.0f32; 4];
    fused_swiglu_scalar(&mut gate, &up);
    // silu(1) = 1 / (1 + exp(-1)) ~= 0.7311
    // result = 0.7311 * 3.0 ~= 2.1932
    for &g in &gate {
        assert!((g - 2.1932).abs() < 0.01, "got {}", g);
    }
}

#[test]
fn test_swiglu_scalar_empty() {
    let mut gate: Vec<f32> = vec![];
    let up: Vec<f32> = vec![];
    fused_swiglu_scalar(&mut gate, &up);
    assert!(gate.is_empty());
}

// ============================================================================
// softmax_scalar
// ============================================================================

#[test]
fn test_softmax_scalar_uniform() {
    let mut x = vec![1.0f32; 4];
    softmax_scalar(&mut x);
    // All equal inputs -> uniform distribution
    for &v in &x {
        assert!((v - 0.25).abs() < 1e-5, "got {}", v);
    }
}

#[test]
fn test_softmax_scalar_sums_to_one() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    softmax_scalar(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);
}

#[test]
fn test_softmax_scalar_monotone() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_scalar(&mut x);
    // Output should be monotonically increasing
    for i in 1..x.len() {
        assert!(x[i] >= x[i - 1], "softmax should be monotone");
    }
}

#[test]
fn test_softmax_scalar_single_element() {
    let mut x = vec![42.0f32];
    softmax_scalar(&mut x);
    assert!((x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_scalar_large_values() {
    // Numerical stability test - should not overflow
    let mut x = vec![1000.0, 1001.0, 999.0];
    softmax_scalar(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {}", sum);
    assert!(x[1] > x[0]); // 1001 should have highest probability
    assert!(x[0] > x[2]); // 1000 > 999
}

#[test]
fn test_softmax_scalar_negative_values() {
    let mut x = vec![-1.0, -2.0, -3.0];
    softmax_scalar(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(x[0] > x[1]); // -1 should have highest
}

// ============================================================================
// quantize_activations_q8_0
// ============================================================================

#[test]
fn test_q8_0_activation_roundtrip() {
    let activations: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // 64 / 32 = 2 blocks
    assert_eq!(quants.len(), 64);

    // Verify roundtrip
    for block in 0..2 {
        let scale = scales[block];
        for i in 0..32 {
            let idx = block * 32 + i;
            let dequantized = quants[idx] as f32 * scale;
            let diff = (activations[idx] - dequantized).abs();
            assert!(
                diff < scale * 2.0,
                "Block {}, idx {}: diff={}",
                block,
                i,
                diff
            );
        }
    }
}

#[test]
fn test_q8_0_activation_partial_block() {
    // 40 elements = 1 full block + 1 partial (8 elements + 24 padding)
    let activations: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // ceil(40/32) = 2
    assert_eq!(quants.len(), 64); // 2 * 32

    // Last 24 elements should be zero-padded
    for i in 40..64 {
        assert_eq!(quants[i], 0, "Padding at index {} should be 0", i);
    }
}

#[test]
fn test_q8_0_activation_near_zero() {
    let activations = vec![1e-12f32; 32];
    let (scales, _quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    // Near-zero values should use minimal scale
    assert!(scales[0] > 0.0);
}

// ============================================================================
// quantize_rmsnorm_q8_0_scalar
// ============================================================================

#[test]
fn test_rmsnorm_q8_scalar_identity() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
    // RMSNorm(1, ..., 1) with weight=1 should give ~1.0 for each element
    // After quantization, we can verify dequantized values
    for (i, &q) in quants.iter().enumerate() {
        let dequantized = q as f32 * scales[0];
        assert!(
            (dequantized - 1.0).abs() < 0.1,
            "Element {}: dequant={}, expected ~1.0",
            i,
            dequantized
        );
    }
}

#[test]
fn test_rmsnorm_q8_scalar_zeros() {
    let input = vec![0.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;
    let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 1);
    for &q in &quants {
        assert_eq!(q, 0, "Zero input should give zero quants");
    }
}

include!("rmsnorm.rs");
