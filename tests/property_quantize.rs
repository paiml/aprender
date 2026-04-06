//! Property-based tests for quantization/dequantization
//!
//! These tests use proptest to verify quantization properties across all quant formats.

use proptest::prelude::*;
use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0,
    dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, f16_to_f32, fused_q4k_dot,
    fused_q4k_dot_simd, fused_q6k_dot, fused_q6k_dot_simd, softmax_simd, BLOCK_SIZE, QK_K,
};

/// Strategy for generating valid Q4_0 blocks
fn q4_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q4_0 block: 2 bytes scale (f16) + 16 bytes quants = 18 bytes
    (any::<u16>(), prop::collection::vec(any::<u8>(), 16..=16)).prop_map(|(scale_bits, quants)| {
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_bits.to_le_bytes()); // f16 as u16 bits
        block.extend(quants);
        block
    })
}

/// Strategy for generating valid Q8_0 blocks
fn q8_0_block_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q8_0 block: 2 bytes scale (f16) + 32 bytes quants = 34 bytes
    (any::<u16>(), prop::collection::vec(any::<u8>(), 32..=32)).prop_map(|(scale_bits, quants)| {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_bits.to_le_bytes()); // f16 as u16 bits
        block.extend(quants);
        block
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Q4_0 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q4_0_output_size(block in q4_0_block_strategy()) {
        let result = dequantize_q4_0(&block).expect("test");
        prop_assert_eq!(result.len(), BLOCK_SIZE);
    }

    /// Q4_0 dequantization with multiple blocks
    #[test]
    fn test_q4_0_multiple_blocks(num_blocks in 1usize..10) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // Scale: f16 1.0 = 0x3C00
            data.extend_from_slice(&0x3C00u16.to_le_bytes());
            // 16 bytes of quants
            data.extend_from_slice(&[0u8; 16]);
        }

        let result = dequantize_q4_0(&data).expect("test");
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q4_0 with zero scale produces all zeros
    #[test]
    fn test_q4_0_zero_scale(quants in prop::collection::vec(any::<u8>(), 16..=16)) {
        let mut data = Vec::new();
        // f16 zero = 0x0000
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).expect("test");
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q4_0 invalid length always errors
    #[test]
    fn test_q4_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 18)
        if len % 18 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q4_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q4_0 dequantization values are bounded by scale
    #[test]
    fn test_q4_0_values_bounded(
        // Use a finite f16 range (0x0000-0x7BFF for positive, 0x8000-0xFBFF for negative)
        // Exclude inf (0x7C00) and NaN (0x7C01+)
        scale_bits in prop::num::u16::ANY.prop_filter("finite f16", |&b| {
            let exp = (b >> 10) & 0x1F;
            exp != 0x1F // Filter out inf/nan (exponent == 31)
        }),
        quants in prop::collection::vec(any::<u8>(), 16..=16)
    ) {
        let mut data = Vec::new();
        // f16 scale as u16 bits
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q4_0(&data).expect("test");

        // Convert f16 to f32 for bound check
        let scale = half::f16::from_bits(scale_bits).to_f32();
        // Q4_0 values range from -8 to 7 after offset
        let max_abs = scale.abs() * 8.0;
        for val in result {
            // Allow some tolerance for f16 precision loss
            prop_assert!(val.abs() <= max_abs + 0.1, "val {} > max {}", val, max_abs);
        }
    }

    /// Q8_0 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q8_0_output_size(block in q8_0_block_strategy()) {
        let result = dequantize_q8_0(&block).expect("test");
        prop_assert_eq!(result.len(), BLOCK_SIZE);
    }

    /// Q8_0 dequantization with multiple blocks
    #[test]
    fn test_q8_0_multiple_blocks(num_blocks in 1usize..10) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // Scale: f16 1.0 = 0x3C00
            data.extend_from_slice(&0x3C00u16.to_le_bytes());
            // 32 bytes of quants
            data.extend_from_slice(&[0u8; 32]);
        }

        let result = dequantize_q8_0(&data).expect("test");
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q8_0 with zero scale produces all zeros
    #[test]
    fn test_q8_0_zero_scale(quants in prop::collection::vec(any::<u8>(), 32..=32)) {
        let mut data = Vec::new();
        // f16 zero = 0x0000
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).expect("test");
        for val in result {
            prop_assert!((val - 0.0).abs() < 1e-10);
        }
    }

    /// Q8_0 invalid length always errors
    #[test]
    fn test_q8_0_invalid_length(len in 1usize..100) {
        // Exclude lengths that are valid (multiples of 34)
        if len % 34 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q8_0(&data);
            prop_assert!(result.is_err());
        }
    }

    /// Q8_0 dequantization values are bounded by scale
    #[test]
    fn test_q8_0_values_bounded(
        // Use a finite f16 range (0x0000-0x7BFF for positive, 0x8000-0xFBFF for negative)
        // Exclude inf (0x7C00) and NaN (0x7C01+)
        scale_bits in prop::num::u16::ANY.prop_filter("finite f16", |&b| {
            let exp = (b >> 10) & 0x1F;
            exp != 0x1F // Filter out inf/nan (exponent == 31)
        }),
        quants in prop::collection::vec(any::<u8>(), 32..=32)
    ) {
        let mut data = Vec::new();
        // f16 scale as u16 bits
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend(quants);

        let result = dequantize_q8_0(&data).expect("test");

        // Convert f16 to f32 for bound check
        let scale = half::f16::from_bits(scale_bits).to_f32();
        // Q8_0 values range from -128 to 127
        let max_abs = scale.abs() * 128.0;
        for val in result {
            // Allow some tolerance for f16 precision loss
            prop_assert!(val.abs() <= max_abs + 0.1, "val {} > max {}", val, max_abs);
        }
    }

    /// Empty data returns empty result
    #[test]
    fn test_empty_data(() in Just(())) {
        let result_q4 = dequantize_q4_0(&[]).expect("test");
        prop_assert!(result_q4.is_empty());

        let result_q8 = dequantize_q8_0(&[]).expect("test");
        prop_assert!(result_q8.is_empty());
    }
}

/// Unit tests for fused RMSNorm + Q8_0 quantization
#[cfg(test)]
mod fused_rmsnorm_tests {
    use realizar::quantize::quantize_rmsnorm_q8_0;

    #[test]
    fn test_fused_rmsnorm_q8_0_basic() {
        // Input: [1.0, 2.0, 3.0, ...] (32 elements to fill one Q8_0 block)
        let input: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // Should produce 1 block (32 elements)
        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);

        // Verify output makes sense
        // RMSNorm should produce normalized values with unit-ish scale
        assert!(scales[0] > 0.0, "Scale should be positive");
        assert!(scales[0] < 10.0, "Scale should be reasonable");
    }

    #[test]
    fn test_fused_rmsnorm_q8_0_zeros() {
        let input: Vec<f32> = vec![0.0; 32];
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (_scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // All zeros should produce zero quants
        for &q in &quants {
            assert_eq!(q, 0, "Zero input should produce zero quants");
        }
    }

    #[test]
    fn test_fused_rmsnorm_q8_0_constant() {
        // Constant input: all same value
        let input: Vec<f32> = vec![5.0; 32];
        let norm_weight: Vec<f32> = vec![1.0; 32];
        let eps = 1e-6;

        let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // Constant input after RMSNorm should be all 1.0 (or close)
        // Then quantized to Q8_0
        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);

        // All quants should be the same for constant input
        let first = quants[0];
        for &q in &quants[1..] {
            assert_eq!(q, first, "Constant input should produce constant quants");
        }
    }
}

// ============================================================================
// K-Quant Format Property Tests
// ============================================================================

/// Strategy for generating valid Q4_K super-blocks (144 bytes each)
fn q4_k_superblock_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q4_K super-block: d (2) + dmin (2) + scales (12) + qs (128) = 144 bytes
    (
        any::<u16>(),                                  // d as f16 bits
        any::<u16>(),                                  // dmin as f16 bits
        prop::collection::vec(any::<u8>(), 12..=12),   // scales
        prop::collection::vec(any::<u8>(), 128..=128), // qs
    )
        .prop_map(|(d_bits, dmin_bits, scales, qs)| {
            let mut block = Vec::with_capacity(144);
            block.extend_from_slice(&d_bits.to_le_bytes());
            block.extend_from_slice(&dmin_bits.to_le_bytes());
            block.extend(scales);
            block.extend(qs);
            block
        })
}

/// Strategy for generating valid Q5_K super-blocks (176 bytes each)
fn q5_k_superblock_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q5_K super-block: d (2) + dmin (2) + scales (12) + qh (32) + qs (128) = 176 bytes
    (
        any::<u16>(),                                  // d as f16 bits
        any::<u16>(),                                  // dmin as f16 bits
        prop::collection::vec(any::<u8>(), 12..=12),   // scales
        prop::collection::vec(any::<u8>(), 32..=32),   // qh (high bits)
        prop::collection::vec(any::<u8>(), 128..=128), // qs (low 4 bits)
    )
        .prop_map(|(d_bits, dmin_bits, scales, qh, qs)| {
            let mut block = Vec::with_capacity(176);
            block.extend_from_slice(&d_bits.to_le_bytes());
            block.extend_from_slice(&dmin_bits.to_le_bytes());
            block.extend(scales);
            block.extend(qh);
            block.extend(qs);
            block
        })
}

/// Strategy for generating valid Q6_K super-blocks (210 bytes each)
fn q6_k_superblock_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Q6_K super-block: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
    (
        prop::collection::vec(any::<u8>(), 128..=128), // ql (low 4 bits)
        prop::collection::vec(any::<u8>(), 64..=64),   // qh (high 2 bits)
        prop::collection::vec(any::<u8>(), 16..=16),   // scales
        any::<u16>(),                                  // d as f16 bits
    )
        .prop_map(|(ql, qh, scales, d_bits)| {
            let mut block = Vec::with_capacity(210);
            block.extend(ql);
            block.extend(qh);
            block.extend(scales);
            block.extend_from_slice(&d_bits.to_le_bytes());
            block
        })
}

/// Strategy for generating finite f16 values (no inf/nan)
fn finite_f16_bits() -> impl Strategy<Value = u16> {
    prop::num::u16::ANY.prop_filter("finite f16", |&b| {
        let exp = (b >> 10) & 0x1F;
        exp != 0x1F // Filter out inf/nan (exponent == 31)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    // -------------------------------------------------------------------------
    // Q4_K Property Tests
    // -------------------------------------------------------------------------

    /// Q4_K dequantization produces exactly QK_K values per super-block
    #[test]
    fn test_q4_k_output_size(block in q4_k_superblock_strategy()) {
        let result = dequantize_q4_k(&block).expect("valid Q4_K block");
        prop_assert_eq!(result.len(), QK_K);
    }

    /// Q4_K with multiple super-blocks
    #[test]
    fn test_q4_k_multiple_superblocks(num_blocks in 1usize..5) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            // Build minimal valid Q4_K super-block
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            data.extend_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
            data.extend_from_slice(&[0u8; 12]); // scales
            data.extend_from_slice(&[0u8; 128]); // qs
        }

        let result = dequantize_q4_k(&data).expect("valid Q4_K data");
        prop_assert_eq!(result.len(), num_blocks * QK_K);
    }

    /// Q4_K with zero d produces bounded values
    #[test]
    fn test_q4_k_zero_d(
        dmin_bits in finite_f16_bits(),
        scales in prop::collection::vec(any::<u8>(), 12..=12),
        qs in prop::collection::vec(any::<u8>(), 128..=128)
    ) {
        let mut data = Vec::new();
        data.extend_from_slice(&0x0000u16.to_le_bytes()); // d = 0.0
        data.extend_from_slice(&dmin_bits.to_le_bytes());
        data.extend(scales);
        data.extend(qs);

        let result = dequantize_q4_k(&data).expect("valid Q4_K block");
        // With d=0, only dmin contributes - values should be bounded
        for val in &result {
            prop_assert!(val.is_finite(), "All values should be finite");
        }
    }

    /// Q4_K invalid length always errors
    #[test]
    fn test_q4_k_invalid_length(len in 1usize..200) {
        if len % 144 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q4_k(&data);
            prop_assert!(result.is_err());
        }
    }

    // -------------------------------------------------------------------------
    // Q5_K Property Tests
    // -------------------------------------------------------------------------

    /// Q5_K dequantization produces exactly QK_K values per super-block
    #[test]
    fn test_q5_k_output_size(block in q5_k_superblock_strategy()) {
        let result = dequantize_q5_k(&block).expect("valid Q5_K block");
        prop_assert_eq!(result.len(), QK_K);
    }

    /// Q5_K with multiple super-blocks
    #[test]
    fn test_q5_k_multiple_superblocks(num_blocks in 1usize..5) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            data.extend_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
            data.extend_from_slice(&[0u8; 12]); // scales
            data.extend_from_slice(&[0u8; 32]); // qh
            data.extend_from_slice(&[0u8; 128]); // qs
        }

        let result = dequantize_q5_k(&data).expect("valid Q5_K data");
        prop_assert_eq!(result.len(), num_blocks * QK_K);
    }

    /// Q5_K invalid length always errors
    #[test]
    fn test_q5_k_invalid_length(len in 1usize..200) {
        if len % 176 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q5_k(&data);
            prop_assert!(result.is_err());
        }
    }

    // -------------------------------------------------------------------------
    // Q6_K Property Tests
    // -------------------------------------------------------------------------

    /// Q6_K dequantization produces exactly QK_K values per super-block
    #[test]
    fn test_q6_k_output_size(block in q6_k_superblock_strategy()) {
        let result = dequantize_q6_k(&block).expect("valid Q6_K block");
        prop_assert_eq!(result.len(), QK_K);
    }

    /// Q6_K with multiple super-blocks
    #[test]
    fn test_q6_k_multiple_superblocks(num_blocks in 1usize..5) {
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            data.extend_from_slice(&[0u8; 128]); // ql
            data.extend_from_slice(&[0u8; 64]); // qh
            data.extend_from_slice(&[0u8; 16]); // scales
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        }

        let result = dequantize_q6_k(&data).expect("valid Q6_K data");
        prop_assert_eq!(result.len(), num_blocks * QK_K);
    }

    /// Q6_K invalid length always errors
    #[test]
    fn test_q6_k_invalid_length(len in 1usize..250) {
        if len % 210 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q6_k(&data);
            prop_assert!(result.is_err());
        }
    }

    // -------------------------------------------------------------------------
    // Q4_1 Property Tests
    // -------------------------------------------------------------------------

    /// Q4_1 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q4_1_output_size(num_blocks in 1usize..10) {
        // Q4_1 block: scale (2) + min (2) + quants (16) = 20 bytes
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
            data.extend_from_slice(&0x0000u16.to_le_bytes()); // min = 0.0
            data.extend_from_slice(&[0u8; 16]); // quants
        }

        let result = dequantize_q4_1(&data).expect("valid Q4_1 data");
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q4_1 invalid length always errors
    #[test]
    fn test_q4_1_invalid_length(len in 1usize..100) {
        if len % 20 != 0 {
            let data = vec![0u8; len];
            let result = dequantize_q4_1(&data);
            prop_assert!(result.is_err());
        }
    }

    // -------------------------------------------------------------------------
    // Q5_0 and Q5_1 Property Tests
    // -------------------------------------------------------------------------

    /// Q5_0 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q5_0_output_size(num_blocks in 1usize..10) {
        // Q5_0 block: scale (2) + qh (4) + quants (16) = 22 bytes
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
            data.extend_from_slice(&[0u8; 4]); // qh (high bits)
            data.extend_from_slice(&[0u8; 16]); // quants
        }

        let result = dequantize_q5_0(&data).expect("valid Q5_0 data");
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    /// Q5_1 dequantization produces exactly BLOCK_SIZE values per block
    #[test]
    fn test_q5_1_output_size(num_blocks in 1usize..10) {
        // Q5_1 block: scale (2) + min (2) + qh (4) + quants (16) = 24 bytes
        let mut data = Vec::new();
        for _ in 0..num_blocks {
            data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
            data.extend_from_slice(&0x0000u16.to_le_bytes()); // min = 0.0
            data.extend_from_slice(&[0u8; 4]); // qh (high bits)
            data.extend_from_slice(&[0u8; 16]); // quants
        }

        let result = dequantize_q5_1(&data).expect("valid Q5_1 data");
        prop_assert_eq!(result.len(), num_blocks * BLOCK_SIZE);
    }

    // -------------------------------------------------------------------------
    // F16 Property Tests
    // -------------------------------------------------------------------------

    /// f16_to_f32 produces finite output for finite input
    #[test]
    fn test_f16_to_f32_finite(bits in finite_f16_bits()) {
        let result = f16_to_f32(bits);
        prop_assert!(result.is_finite());
    }

    /// dequantize_f16 produces correct length
    #[test]
    fn test_dequantize_f16_length(num_values in 1usize..100) {
        let data: Vec<u8> = (0..num_values * 2)
            .map(|i| (i % 256) as u8)
            .collect();

        let result = dequantize_f16(&data).expect("valid F16 data");
        prop_assert_eq!(result.len(), num_values);
    }

    /// dequantize_f16 odd length always errors
    #[test]
    fn test_dequantize_f16_odd_length(len in (1usize..100).prop_filter("odd", |l| l % 2 != 0)) {
        let data = vec![0u8; len];
        let result = dequantize_f16(&data);
        prop_assert!(result.is_err());
    }
}

// ============================================================================
// SIMD Parity Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// fused_q4k_dot_simd matches fused_q4k_dot (scalar) for all valid inputs
    #[test]
    fn test_fused_q4k_dot_simd_parity(
        d_bits in finite_f16_bits(),
        dmin_bits in finite_f16_bits(),
        scales in prop::collection::vec(any::<u8>(), 12..=12),
        qs in prop::collection::vec(any::<u8>(), 128..=128)
    ) {
        // Build Q4_K super-block
        let mut q4k_data = Vec::new();
        q4k_data.extend_from_slice(&d_bits.to_le_bytes());
        q4k_data.extend_from_slice(&dmin_bits.to_le_bytes());
        q4k_data.extend(scales);
        q4k_data.extend(qs);

        // Create activations of matching size (256 elements)
        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q4k_dot(&q4k_data, &activations);
        let simd_result = fused_q4k_dot_simd(&q4k_data, &activations);

        match (scalar_result, simd_result) {
            (Ok(s), Ok(si)) => {
                // Allow numerical differences due to SIMD FMA (up to 1e-3 relative)
                let diff = (s - si).abs();
                let tolerance = s.abs().max(1.0) * 1e-3;
                prop_assert!(
                    diff < tolerance,
                    "SIMD mismatch: scalar={}, simd={}, diff={}, tolerance={}",
                    s, si, diff, tolerance
                );
            }
            (Err(_), Err(_)) => (), // Both error is fine
            (s, si) => prop_assert!(false, "Mismatch: scalar={:?}, simd={:?}", s, si),
        }
    }

    /// fused_q6k_dot_simd matches fused_q6k_dot (scalar) for all valid inputs
    #[test]
    fn test_fused_q6k_dot_simd_parity(
        ql in prop::collection::vec(any::<u8>(), 128..=128),
        qh in prop::collection::vec(any::<u8>(), 64..=64),
        scales in prop::collection::vec(any::<u8>(), 16..=16),
        d_bits in finite_f16_bits()
    ) {
        // Build Q6_K super-block
        let mut q6k_data = Vec::new();
        q6k_data.extend(ql);
        q6k_data.extend(qh);
        q6k_data.extend(scales);
        q6k_data.extend_from_slice(&d_bits.to_le_bytes());

        // Create activations of matching size (256 elements)
        let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let scalar_result = fused_q6k_dot(&q6k_data, &activations);
        let simd_result = fused_q6k_dot_simd(&q6k_data, &activations);

        match (scalar_result, simd_result) {
            (Ok(s), Ok(si)) => {
                // Allow numerical differences due to SIMD FMA (up to 1e-3 relative)
                let diff = (s - si).abs();
                let tolerance = s.abs().max(1.0) * 1e-3;
                prop_assert!(
                    diff < tolerance,
                    "SIMD mismatch: scalar={}, simd={}, diff={}, tolerance={}",
                    s, si, diff, tolerance
                );
            }
            (Err(_), Err(_)) => (), // Both error is fine
            (s, si) => prop_assert!(false, "Mismatch: scalar={:?}, simd={:?}", s, si),
        }
    }
}

// ============================================================================
// Softmax Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// softmax_simd output sums to 1.0
    #[test]
    fn test_softmax_simd_sums_to_one(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0, got {}",
            sum
        );
    }

    /// softmax_simd produces all non-negative values
    #[test]
    fn test_softmax_simd_non_negative(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        for &val in &x {
            prop_assert!(val >= 0.0, "Softmax output should be non-negative");
        }
    }

    /// softmax_simd produces all values <= 1.0
    #[test]
    fn test_softmax_simd_bounded(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        for &val in &x {
            prop_assert!(val <= 1.0 + 1e-6, "Softmax output should be <= 1.0");
        }
    }

    /// softmax_simd preserves ordering (larger input -> larger output)
    #[test]
    fn test_softmax_simd_ordering(
        values in prop::collection::vec(-10.0f32..10.0f32, 2..50)
    ) {
        let original = values.clone();
        let mut x = values;
        softmax_simd(&mut x);

        // For any pair (i, j) where original[i] > original[j], we should have x[i] >= x[j]
        for i in 0..original.len() {
            for j in (i + 1)..original.len() {
                if original[i] > original[j] + 1e-6 {
                    prop_assert!(
                        x[i] >= x[j] - 1e-6,
                        "Softmax should preserve ordering: orig[{}]={} > orig[{}]={}, but out[{}]={} < out[{}]={}",
                        i, original[i], j, original[j], i, x[i], j, x[j]
                    );
                }
            }
        }
    }
}

// ============================================================================
// Edge Case Unit Tests
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_q4_k_all_zeros() {
        // 144 bytes of zeros
        let data = vec![0u8; 144];
        let result = dequantize_q4_k(&data).expect("should succeed");
        assert_eq!(result.len(), 256);
        // With d=0 and dmin=0, all values should be 0
        for val in result {
            assert!(val.abs() < 1e-10, "Expected 0, got {}", val);
        }
    }

    #[test]
    fn test_q5_k_all_zeros() {
        let data = vec![0u8; 176];
        let result = dequantize_q5_k(&data).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_q6_k_all_zeros() {
        let data = vec![0u8; 210];
        let result = dequantize_q6_k(&data).expect("should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 1e-10);

        // Negative zero
        assert!((f16_to_f32(0x8000) - 0.0).abs() < 1e-10);

        // One
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-3);

        // Negative one
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-3);

        // Very small positive (denorm)
        let small = f16_to_f32(0x0001);
        assert!(small > 0.0 && small < 1e-4);
    }

    #[test]
    fn test_softmax_simd_single_element() {
        let mut x = vec![5.0f32];
        softmax_simd(&mut x);
        assert!(
            (x[0] - 1.0).abs() < 1e-6,
            "Single element softmax should be 1.0"
        );
    }

    #[test]
    fn test_softmax_simd_large_values() {
        // Test numerical stability with large values
        let mut x = vec![100.0f32, 101.0, 102.0];
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0 even with large values"
        );

        // The largest input should have the largest output
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_softmax_simd_negative_values() {
        let mut x = vec![-100.0f32, -101.0, -102.0];
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1.0 with negative values"
        );
    }

    // ========================================================================
    // Additional Edge Case Tests for Coverage
    // ========================================================================

    #[test]
    fn test_q4_k_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q4_k(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q4_K data should return empty vec");
    }

    #[test]
    fn test_q5_k_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q5_k(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q5_K data should return empty vec");
    }

    #[test]
    fn test_q6_k_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q6_k(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q6_K data should return empty vec");
    }

    #[test]
    fn test_q4_1_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q4_1(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q4_1 data should return empty vec");
    }

    #[test]
    fn test_q5_0_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q5_0(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q5_0 data should return empty vec");
    }

    #[test]
    fn test_q5_1_empty_data() {
        // Empty data returns empty vec
        let result = dequantize_q5_1(&[]).unwrap();
        assert!(result.is_empty(), "Empty Q5_1 data should return empty vec");
    }

    #[test]
    fn test_f16_dequantize_empty() {
        // Empty data returns empty vec
        let result = dequantize_f16(&[]).unwrap();
        assert!(result.is_empty(), "Empty F16 data should return empty vec");
    }

    #[test]
    fn test_f16_dequantize_odd_length() {
        // Odd length data should error (F16 needs 2 bytes per value)
        let result = dequantize_f16(&[0u8, 0, 0]);
        assert!(result.is_err(), "Odd-length F16 data should error");
    }

    #[test]
    fn test_fused_q4k_dot_mismatched_activation_size() {
        // Create valid Q4_K block (144 bytes for 256 values)
        let q4k_data = vec![0u8; 144];
        let activations = vec![1.0f32; 128]; // Wrong size - should be 256

        let result = fused_q4k_dot(&q4k_data, &activations);
        assert!(result.is_err(), "Mismatched activation size should error");
    }

    #[test]
    fn test_fused_q6k_dot_mismatched_activation_size() {
        // Create valid Q6_K block (210 bytes for 256 values)
        let q6k_data = vec![0u8; 210];
        let activations = vec![1.0f32; 128]; // Wrong size - should be 256

        let result = fused_q6k_dot(&q6k_data, &activations);
        assert!(result.is_err(), "Mismatched activation size should error");
    }

    #[test]
    fn test_softmax_simd_two_elements() {
        let mut x = vec![0.0f32, 0.0];
        softmax_simd(&mut x);

        // Equal inputs should produce equal outputs
        assert!(
            (x[0] - x[1]).abs() < 1e-6,
            "Equal inputs should give equal softmax outputs"
        );
        assert!((x[0] - 0.5).abs() < 1e-6, "Each should be 0.5");
    }

    #[test]
    fn test_softmax_simd_power_of_two_size() {
        // Test power-of-two sizes which might exercise SIMD paths differently
        for size in [8, 16, 32, 64, 128] {
            let mut x = vec![1.0f32; size];
            softmax_simd(&mut x);

            let sum: f32 = x.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Softmax of {} elements should sum to 1.0",
                size
            );

            // All inputs equal, so all outputs should be 1/size
            let expected = 1.0 / size as f32;
            for (i, &val) in x.iter().enumerate() {
                assert!(
                    (val - expected).abs() < 1e-5,
                    "Element {} should be {}, got {}",
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn test_f16_inf_nan_values() {
        // Test special f16 bit patterns - infinity and NaN

        // Positive infinity (0x7C00)
        let inf = f16_to_f32(0x7C00);
        assert!(inf.is_infinite() && inf > 0.0, "0x7C00 should be +inf");

        // Negative infinity (0xFC00)
        let neg_inf = f16_to_f32(0xFC00);
        assert!(
            neg_inf.is_infinite() && neg_inf < 0.0,
            "0xFC00 should be -inf"
        );

        // NaN (0x7C01 or higher with non-zero mantissa)
        let nan = f16_to_f32(0x7C01);
        assert!(nan.is_nan(), "0x7C01 should be NaN");

        // Max normal (0x7BFF)
        let max_normal = f16_to_f32(0x7BFF);
        assert!(
            max_normal > 60000.0 && max_normal < 70000.0,
            "Max f16 should be ~65504"
        );
    }
}
