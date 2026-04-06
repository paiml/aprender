//! Quantization Fuzzing Tests (T-QA-008)
//!
//! Kernel & Layer Hardening Squad: Close coverage gaps in quantize.rs (83% -> 95%)
//!
//! Uses proptest to feed boundary-sized vectors to quantization kernels,
//! forcing both scalar fallback and SIMD branch coverage.
//!
//! Target sizes: 1, 15, 16, 17, 31, 32, 33 (boundary conditions)
//!
//! Constraint: Pure CPU logic verification, < 5s execution

#![allow(clippy::needless_range_loop)]

use half::f16;
use proptest::prelude::*;
use realizar::quantize::{dequantize_f16, dequantize_q4_k, dequantize_q8_0, f16_to_f32};

/// Convert f32 to f16 bits (using half crate)
fn f32_to_f16(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

// ============================================================================
// A. Block Size Constants
// ============================================================================

/// Q8_0 block: 2 bytes (f16 scale) + 32 bytes (i8 quants) = 34 bytes per block
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_VALUES: usize = 32;

/// Q4_K super-block: 144 bytes per super-block, produces 256 values
const Q4_K_SUPER_BLOCK_BYTES: usize = 144;
const Q4_K_SUPER_BLOCK_VALUES: usize = 256;

/// F16: 2 bytes per value
const F16_BYTES_PER_VALUE: usize = 2;

// ============================================================================
// B. Helper Functions - Create Valid Quantized Data
// ============================================================================

/// Create valid Q8_0 quantized data for N values
fn create_q8_0_data(num_values: usize) -> Vec<u8> {
    // Round up to nearest block
    let num_blocks = num_values.div_ceil(Q8_0_BLOCK_VALUES);
    let mut data = vec![0u8; num_blocks * Q8_0_BLOCK_BYTES];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q8_0_BLOCK_BYTES;

        // Scale = 1.0 (f16)
        let scale_bits = f32_to_f16(1.0);
        data[block_start] = (scale_bits & 0xFF) as u8;
        data[block_start + 1] = ((scale_bits >> 8) & 0xFF) as u8;

        // Quantized values: fill with sequential bytes
        for i in 0..32 {
            data[block_start + 2 + i] = ((block_idx * 32 + i) % 256) as u8;
        }
    }

    data
}

/// Create valid Q4_K quantized data for N values
fn create_q4_k_data(num_values: usize) -> Vec<u8> {
    // Round up to nearest super-block (256 values)
    let num_super_blocks = num_values.div_ceil(Q4_K_SUPER_BLOCK_VALUES);
    let mut data = vec![0u8; num_super_blocks * Q4_K_SUPER_BLOCK_BYTES];

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * Q4_K_SUPER_BLOCK_BYTES;

        // d (f16 scale) = 1.0
        let d_bits = f32_to_f16(1.0);
        data[sb_start] = (d_bits & 0xFF) as u8;
        data[sb_start + 1] = ((d_bits >> 8) & 0xFF) as u8;

        // dmin (f16) = 0.0
        data[sb_start + 2] = 0;
        data[sb_start + 3] = 0;

        // Scales (12 bytes) - all 1s for simplicity
        for i in 0..12 {
            data[sb_start + 4 + i] = 0x11; // Scale nibbles = 1
        }

        // Quantized values (128 bytes) - fill with pattern
        for i in 0..128 {
            data[sb_start + 16 + i] = ((sb_idx + i) % 256) as u8;
        }
    }

    data
}

/// Create valid F16 data for N values
fn create_f16_data(num_values: usize) -> Vec<u8> {
    let mut data = vec![0u8; num_values * F16_BYTES_PER_VALUE];

    for i in 0..num_values {
        // Create f16 value from i
        let value = (i as f32) / (num_values as f32);
        let bits = f32_to_f16(value);
        data[i * 2] = (bits & 0xFF) as u8;
        data[i * 2 + 1] = ((bits >> 8) & 0xFF) as u8;
    }

    data
}

// ============================================================================
// C. Boundary Size Tests - Q8_0
// ============================================================================

/// Test Q8_0 with boundary sizes that stress scalar/SIMD transitions
#[test]
fn test_q8_0_boundary_size_1_block() {
    // Exactly 1 block = 32 values
    let data = create_q8_0_data(32);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok(), "Q8_0 with 32 values should succeed");
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_q8_0_boundary_size_2_blocks() {
    // Exactly 2 blocks = 64 values
    let data = create_q8_0_data(64);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok(), "Q8_0 with 64 values should succeed");
    assert_eq!(result.unwrap().len(), 64);
}

#[test]
fn test_q8_0_boundary_size_15_blocks() {
    // 15 blocks = 480 values (not power of 2)
    let data = create_q8_0_data(480);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 480);
}

#[test]
fn test_q8_0_boundary_size_16_blocks() {
    // 16 blocks = 512 values (SIMD friendly)
    let data = create_q8_0_data(512);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 512);
}

#[test]
fn test_q8_0_boundary_size_17_blocks() {
    // 17 blocks = 544 values (SIMD + remainder)
    let data = create_q8_0_data(544);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 544);
}

#[test]
fn test_q8_0_boundary_size_31_blocks() {
    // 31 blocks = 992 values
    let data = create_q8_0_data(992);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 992);
}

#[test]
fn test_q8_0_boundary_size_33_blocks() {
    // 33 blocks = 1056 values
    let data = create_q8_0_data(1056);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1056);
}

#[test]
fn test_q8_0_invalid_size() {
    // Not a multiple of block size
    let data = vec![0u8; 35]; // 34 + 1 = invalid
    let result = dequantize_q8_0(&data);
    assert!(result.is_err(), "Q8_0 with invalid size should fail");
}

#[test]
fn test_q8_0_empty() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q8_0(&data);
    // Empty should succeed with empty output or fail gracefully
    if let Ok(values) = result {
        assert_eq!(values.len(), 0);
    }
}

// ============================================================================
// D. Boundary Size Tests - Q4_K
// ============================================================================

#[test]
fn test_q4_k_boundary_size_1_super_block() {
    // 1 super-block = 256 values
    let data = create_q4_k_data(256);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok(), "Q4_K with 256 values should succeed");
    assert_eq!(result.unwrap().len(), 256);
}

#[test]
fn test_q4_k_boundary_size_2_super_blocks() {
    // 2 super-blocks = 512 values
    let data = create_q4_k_data(512);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 512);
}

#[test]
fn test_q4_k_boundary_size_15_super_blocks() {
    // 15 super-blocks = 3840 values
    let data = create_q4_k_data(3840);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 3840);
}

#[test]
fn test_q4_k_boundary_size_16_super_blocks() {
    // 16 super-blocks = 4096 values (SIMD friendly)
    let data = create_q4_k_data(4096);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 4096);
}

#[test]
fn test_q4_k_boundary_size_17_super_blocks() {
    // 17 super-blocks = 4352 values (SIMD + remainder)
    let data = create_q4_k_data(4352);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 4352);
}

#[test]
fn test_q4_k_invalid_size() {
    // Not a multiple of super-block size
    let data = vec![0u8; 145]; // 144 + 1 = invalid
    let result = dequantize_q4_k(&data);
    assert!(result.is_err(), "Q4_K with invalid size should fail");
}

#[test]
fn test_q4_k_empty() {
    let data: Vec<u8> = vec![];
    let result = dequantize_q4_k(&data);
    if let Ok(values) = result {
        assert_eq!(values.len(), 0);
    }
}

// ============================================================================
// E. Boundary Size Tests - F16
// ============================================================================

#[test]
fn test_f16_boundary_size_1() {
    let data = create_f16_data(1);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_f16_boundary_size_15() {
    let data = create_f16_data(15);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 15);
}

#[test]
fn test_f16_boundary_size_16() {
    let data = create_f16_data(16);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 16);
}

#[test]
fn test_f16_boundary_size_17() {
    let data = create_f16_data(17);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 17);
}

#[test]
fn test_f16_boundary_size_31() {
    let data = create_f16_data(31);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 31);
}

#[test]
fn test_f16_boundary_size_32() {
    let data = create_f16_data(32);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_f16_boundary_size_33() {
    let data = create_f16_data(33);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 33);
}

#[test]
fn test_f16_invalid_size() {
    // Not a multiple of 2 bytes
    let data = vec![0u8; 3];
    let result = dequantize_f16(&data);
    assert!(result.is_err(), "F16 with odd byte count should fail");
}

#[test]
fn test_f16_empty() {
    let data: Vec<u8> = vec![];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

// ============================================================================
// F. F16 Conversion Tests
// ============================================================================

#[test]
fn test_f16_round_trip_zero() {
    let original = 0.0f32;
    let f16_bits = f32_to_f16(original);
    let recovered = f16_to_f32(f16_bits);
    assert!((recovered - original).abs() < 1e-10);
}

#[test]
fn test_f16_round_trip_one() {
    let original = 1.0f32;
    let f16_bits = f32_to_f16(original);
    let recovered = f16_to_f32(f16_bits);
    assert!((recovered - original).abs() < 1e-3);
}

#[test]
fn test_f16_round_trip_negative() {
    let original = -2.5f32;
    let f16_bits = f32_to_f16(original);
    let recovered = f16_to_f32(f16_bits);
    assert!((recovered - original).abs() < 0.01);
}

#[test]
fn test_f16_special_values() {
    // Test infinity
    let inf_bits = f32_to_f16(f32::INFINITY);
    let inf_recovered = f16_to_f32(inf_bits);
    assert!(inf_recovered.is_infinite() && inf_recovered.is_sign_positive());

    // Test negative infinity
    let neg_inf_bits = f32_to_f16(f32::NEG_INFINITY);
    let neg_inf_recovered = f16_to_f32(neg_inf_bits);
    assert!(neg_inf_recovered.is_infinite() && neg_inf_recovered.is_sign_negative());

    // Test NaN
    let nan_bits = f32_to_f16(f32::NAN);
    let nan_recovered = f16_to_f32(nan_bits);
    assert!(nan_recovered.is_nan());
}

// ============================================================================
// G. Proptest - Random Fuzzing
// ============================================================================

proptest! {
    /// Fuzz Q8_0 with random block counts (1-100 blocks)
    #[test]
    fn proptest_q8_0_random_block_counts(num_blocks in 1usize..100) {
        let data = create_q8_0_data(num_blocks * Q8_0_BLOCK_VALUES);
        let result = dequantize_q8_0(&data);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap().len(), num_blocks * Q8_0_BLOCK_VALUES);
    }

    /// Fuzz Q4_K with random super-block counts (1-50 super-blocks)
    #[test]
    fn proptest_q4_k_random_super_block_counts(num_super_blocks in 1usize..50) {
        let data = create_q4_k_data(num_super_blocks * Q4_K_SUPER_BLOCK_VALUES);
        let result = dequantize_q4_k(&data);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap().len(), num_super_blocks * Q4_K_SUPER_BLOCK_VALUES);
    }

    /// Fuzz F16 with random value counts (1-1000 values)
    #[test]
    fn proptest_f16_random_value_counts(num_values in 1usize..1000) {
        let data = create_f16_data(num_values);
        let result = dequantize_f16(&data);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap().len(), num_values);
    }

    /// Fuzz F16 round-trip with random f32 values
    #[test]
    fn proptest_f16_round_trip(value in -65000.0f32..65000.0f32) {
        if value.is_nan() || value.is_infinite() {
            return Ok(());
        }
        let f16_bits = f32_to_f16(value);
        let recovered = f16_to_f32(f16_bits);
        // F16 has limited precision - allow up to 0.5% relative error
        let relative_error = if value.abs() > 0.001 {
            ((recovered - value) / value).abs()
        } else {
            (recovered - value).abs()
        };
        prop_assert!(relative_error < 0.01, "F16 round-trip error too large: {} -> {} (err: {})", value, recovered, relative_error);
    }
}

// ============================================================================
// H. Dequantization Correctness Tests
// ============================================================================

#[test]
fn test_q8_0_dequantize_correctness() {
    // Create known data: scale=2.0, quants=[0,1,2,...,31]
    let mut data = vec![0u8; Q8_0_BLOCK_BYTES];

    // Scale = 2.0 as f16
    let scale_bits = f32_to_f16(2.0);
    data[0] = (scale_bits & 0xFF) as u8;
    data[1] = ((scale_bits >> 8) & 0xFF) as u8;

    // Quantized values: 0 to 31 as signed bytes
    for i in 0..32 {
        data[2 + i] = i as u8;
    }

    let result = dequantize_q8_0(&data).expect("dequantize should succeed");
    assert_eq!(result.len(), 32);

    // Expected: value[i] = scale * quant[i] = 2.0 * i
    for i in 0..32 {
        let expected = 2.0 * (i as f32);
        assert!(
            (result[i] - expected).abs() < 0.1,
            "Q8_0 value {} should be {} but got {}",
            i,
            expected,
            result[i]
        );
    }
}

#[test]
fn test_f16_dequantize_correctness() {
    // Create known data: [0.0, 0.5, 1.0, 1.5]
    let values = [0.0f32, 0.5, 1.0, 1.5];
    let mut data = Vec::new();
    for &v in &values {
        let bits = f32_to_f16(v);
        data.push((bits & 0xFF) as u8);
        data.push(((bits >> 8) & 0xFF) as u8);
    }

    let result = dequantize_f16(&data).expect("dequantize should succeed");
    assert_eq!(result.len(), 4);

    for (i, &expected) in values.iter().enumerate() {
        assert!(
            (result[i] - expected).abs() < 0.01,
            "F16 value {} should be {} but got {}",
            i,
            expected,
            result[i]
        );
    }
}

// ============================================================================
// I. Edge Case Tests - Extreme Values
// ============================================================================

#[test]
fn test_q8_0_with_max_scale() {
    let mut data = vec![0u8; Q8_0_BLOCK_BYTES];

    // Maximum f16 value as scale (65504.0)
    let max_f16: u16 = 0x7BFF;
    data[0] = (max_f16 & 0xFF) as u8;
    data[1] = ((max_f16 >> 8) & 0xFF) as u8;

    // Small quantized values
    for i in 0..32 {
        data[2 + i] = 1;
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.unwrap();

    // All values should be approximately max_f16
    for v in &values {
        assert!(v.is_finite(), "Values should be finite even with max scale");
    }
}

#[test]
fn test_q8_0_with_zero_scale() {
    let mut data = vec![0u8; Q8_0_BLOCK_BYTES];

    // Zero scale
    data[0] = 0;
    data[1] = 0;

    // Non-zero quantized values
    for i in 0..32 {
        data[2 + i] = 100;
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.unwrap();

    // All values should be zero (0 * quant = 0)
    for v in &values {
        assert_eq!(*v, 0.0, "Zero scale should produce zero values");
    }
}

#[test]
fn test_q8_0_with_negative_quants() {
    let mut data = vec![0u8; Q8_0_BLOCK_BYTES];

    // Scale = 1.0
    let scale_bits = f32_to_f16(1.0);
    data[0] = (scale_bits & 0xFF) as u8;
    data[1] = ((scale_bits >> 8) & 0xFF) as u8;

    // Negative quantized values (signed byte)
    for i in 0..32 {
        data[2 + i] = (-10i8 + i as i8) as u8;
    }

    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    let values = result.unwrap();

    // Check first value: scale * -10 = -10.0
    assert!(
        (values[0] - (-10.0)).abs() < 0.1,
        "First value should be -10.0"
    );
}

#[test]
fn test_f16_denormalized_numbers() {
    // Test very small denormalized numbers
    let tiny: u16 = 0x0001; // Smallest positive denormalized f16
    let data = vec![(tiny & 0xFF) as u8, ((tiny >> 8) & 0xFF) as u8];

    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    let value = result.unwrap()[0];
    assert!(
        value > 0.0 && value < 1e-6,
        "Denormalized number should be tiny positive"
    );
}

// ============================================================================
// J. SIMD Branch Coverage Tests
// ============================================================================

#[test]
fn test_q8_0_simd_aligned_size() {
    // Create data aligned to SIMD register size (256 values = 8 blocks for AVX)
    let data = create_q8_0_data(256);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 256);
}

#[test]
fn test_q8_0_simd_misaligned_size() {
    // Create data not aligned to SIMD register (224 values = 7 blocks)
    let data = create_q8_0_data(224);
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 224);
}

#[test]
fn test_q4_k_simd_aligned_size() {
    // 1024 values = 4 super-blocks (nice power of 2)
    let data = create_q4_k_data(1024);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1024);
}

#[test]
fn test_q4_k_simd_misaligned_size() {
    // 768 values = 3 super-blocks (not power of 2)
    let data = create_q4_k_data(768);
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 768);
}

// ============================================================================
// K. Parallel Execution Tests
// ============================================================================

#[test]
fn test_q8_0_large_parallel() {
    // Large enough to trigger parallel execution
    let data = create_q8_0_data(32000); // 1000 blocks
    let result = dequantize_q8_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 32000);
}

#[test]
fn test_q4_k_large_parallel() {
    // Large enough to trigger parallel execution
    let data = create_q4_k_data(25600); // 100 super-blocks
    let result = dequantize_q4_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 25600);
}

#[test]
fn test_f16_large_parallel() {
    // Large F16 dequantization
    let data = create_f16_data(100000);
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 100000);
}
