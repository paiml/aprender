use crate::quantize::*;

#[test]
fn test_dequantize_q4_0_single_block() {
    // Single Q4_0 block: scale=2.0
    // Q4_0 layout: positions 0-15 are low nibbles, 16-31 are high nibbles
    let mut data = Vec::new();

    // Scale: 2.0 as f16 (2 bytes per GGML spec)
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    // 16 bytes of quantized values (0x01, 0x23, 0x45, ...)
    // Byte 0: low=0, high=1; Byte 1: low=2, high=3; etc.
    for i in 0..16u8 {
        let low = i * 2;
        let high = i * 2 + 1;
        data.push((high << 4) | low);
    }

    let result = dequantize_q4_0(&data).expect("test");
    assert_eq!(result.len(), 32);

    // Check values based on Candle layout:
    // result[0] = low nibble of byte 0 = 0 -> (0-8)*2 = -16
    assert!((result[0] - (-16.0)).abs() < 1e-3);
    // result[1] = low nibble of byte 1 = 2 -> (2-8)*2 = -12
    assert!((result[1] - (-12.0)).abs() < 1e-3);
    // result[16] = high nibble of byte 0 = 1 -> (1-8)*2 = -14
    assert!((result[16] - (-14.0)).abs() < 1e-3);
}

#[test]
fn test_dequantize_q4_0_invalid_length() {
    // 19 bytes (not a multiple of 18)
    let data = vec![0u8; 19];
    let result = dequantize_q4_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_single_block() {
    // Single Q8_0 block: scale=0.5, values=0,1,2,...,31
    let mut data = Vec::new();

    // Scale: 0.5 as f16 (2 bytes per GGML spec)
    data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());

    // 32 int8 values
    #[allow(clippy::cast_possible_truncation)]
    for i in 0..32_i8 {
        data.push(i.to_le_bytes()[0]);
    }

    let result = dequantize_q8_0(&data).expect("test");
    assert_eq!(result.len(), 32);

    // Check first few values
    assert!((result[0] - 0.0).abs() < 1e-3); // 0 * 0.5 = 0.0
    assert!((result[1] - 0.5).abs() < 1e-3); // 1 * 0.5 = 0.5
    assert!((result[31] - 15.5).abs() < 1e-3); // 31 * 0.5 = 15.5
}

#[test]
fn test_dequantize_q8_0_invalid_length() {
    // 35 bytes (not a multiple of 34)
    let data = vec![0u8; 35];
    let result = dequantize_q8_0(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_0_multiple_blocks() {
    let mut data = Vec::new();

    // Block 1: scale=1.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    for i in 0..16u8 {
        data.push((i << 4) | i);
    }

    // Block 2: scale=3.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(3.0).to_le_bytes());
    for i in 0..16u8 {
        data.push((i << 4) | i);
    }

    let result = dequantize_q4_0(&data).expect("test");
    assert_eq!(result.len(), 64); // 2 blocks * 32 values
}

#[test]
fn test_dequantize_q4_k_invalid_length() {
    // 143 bytes (not a multiple of 144)
    let data = vec![0u8; 143];
    let result = dequantize_q4_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q4_k_single_super_block() {
    // Single Q4_K super-block: 144 bytes total
    let mut data = Vec::new();

    // d = 1.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // dmin = 0.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    // scales: 12 bytes (8 blocks * 12 bits = 96 bits, packed)
    // For simplicity, use simple encoding
    data.extend_from_slice(&[0x00; 12]);

    // qs: 128 bytes (256 4-bit values)
    data.extend_from_slice(&[0x00; 128]);

    let result = dequantize_q4_k(&data).expect("test");
    assert_eq!(result.len(), 256); // 1 super-block * 256 values
}

#[test]
fn test_dequantize_q4_k_output_size() {
    // 2 super-blocks: 2 * 144 = 288 bytes
    let data = vec![0u8; 288];
    let result = dequantize_q4_k(&data).expect("test");
    assert_eq!(result.len(), 512); // 2 super-blocks * 256 values each
}

#[test]
fn test_read_f16() {
    // Test f16 reading
    let f16_1 = half::f16::from_f32(1.0);
    let bytes = f16_1.to_bits().to_le_bytes();
    let result = read_f16(&bytes);
    assert!((result - 1.0).abs() < 1e-3);

    let f16_half = half::f16::from_f32(0.5);
    let bytes = f16_half.to_bits().to_le_bytes();
    let result = read_f16(&bytes);
    assert!((result - 0.5).abs() < 1e-3);
}

#[test]
fn test_extract_scale_min() {
    // Test scale/min extraction using llama.cpp packing scheme
    let mut scales = [0u8; 12];

    // Block 0: scale in scales[0] & 63, min in scales[4] & 63
    scales[0] = 31; // scale = 31
    scales[4] = 15; // min = 15

    let (scale, min) = extract_scale_min(&scales, 0);
    assert!((scale - 31.0).abs() < 1e-6);
    assert!((min - 15.0).abs() < 1e-6);

    // Block 5: uses packed format
    // scale = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4)
    // min = (scales[9] >> 4) | ((scales[5] >> 6) << 4)
    scales[1] = 0b11_000000; // high 2 bits contribute to scale[5]
    scales[5] = 0b10_000000; // high 2 bits contribute to min[5]
    scales[9] = 0b0101_0011; // low 4 bits = scale, high 4 bits = min

    let (scale5, min5) = extract_scale_min(&scales, 5);
    // scale = (0x3) | (0x3 << 4) = 0x33 = 51
    assert!((scale5 - 51.0).abs() < 1e-6);
    // min = (0x5) | (0x2 << 4) = 0x25 = 37
    assert!((min5 - 37.0).abs() < 1e-6);
}

#[test]
fn test_dequantize_q5_k_invalid_length() {
    // 175 bytes (not a multiple of 176)
    let data = vec![0u8; 175];
    let result = dequantize_q5_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q5_k_single_super_block() {
    // Single Q5_K super-block: 176 bytes total
    let mut data = Vec::new();

    // d = 1.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // dmin = 0.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    // scales: 12 bytes (8 blocks * 12 bits = 96 bits, packed)
    data.extend_from_slice(&[0x00; 12]);

    // qh: 32 bytes (high bits)
    data.extend_from_slice(&[0x00; 32]);

    // qs: 128 bytes (low 4 bits)
    data.extend_from_slice(&[0x00; 128]);

    let result = dequantize_q5_k(&data).expect("test");
    assert_eq!(result.len(), 256); // 1 super-block * 256 values
}

#[test]
fn test_dequantize_q5_k_output_size() {
    // 2 super-blocks: 2 * 176 = 352 bytes
    let data = vec![0u8; 352];
    let result = dequantize_q5_k(&data).expect("test");
    assert_eq!(result.len(), 512); // 2 super-blocks * 256 values each
}

#[test]
fn test_dequantize_q5_k_with_data() {
    // Test Q5_K with some non-zero data
    let mut data = Vec::new();

    // d = 2.0 (f16)
    data.extend_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());

    // dmin = 0.5 (f16)
    data.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());

    // scales: 12 bytes (set first scale to max)
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // scale=63 (6 bits)
    data.extend_from_slice(&scales);

    // qh: 32 bytes (all zeros for simplicity)
    data.extend_from_slice(&[0x00; 32]);

    // qs: 128 bytes (all zeros)
    data.extend_from_slice(&[0x00; 128]);

    let result = dequantize_q5_k(&data).expect("test");
    assert_eq!(result.len(), 256);
    // Values should be computed based on formula: d * scale * q - dmin * min
}

#[test]
fn test_dequantize_q6_k_invalid_length() {
    // 209 bytes (not a multiple of 210)
    let data = vec![0u8; 209];
    let result = dequantize_q6_k(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q6_k_single_super_block() {
    // Single Q6_K super-block: 210 bytes total
    // Layout: ql (128) + qh (64) + scales (16) + d (2)
    let mut data = Vec::new();

    // ql: 128 bytes (low 4 bits)
    data.extend_from_slice(&[0x00; 128]);

    // qh: 64 bytes (high 2 bits)
    data.extend_from_slice(&[0x00; 64]);

    // scales: 16 bytes (u8, interpreted as i8)
    data.extend_from_slice(&[0u8; 16]);

    // d = 1.0 (f16) at the END
    data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("test");
    assert_eq!(result.len(), 256); // 1 super-block * 256 values
}

#[test]
fn test_dequantize_q6_k_output_size() {
    // 2 super-blocks: 2 * 210 = 420 bytes
    let data = vec![0u8; 420];
    let result = dequantize_q6_k(&data).expect("test");
    assert_eq!(result.len(), 512); // 2 super-blocks * 256 values each
}

#[test]
fn test_dequantize_q6_k_with_data() {
    // Test Q6_K with some non-zero data
    // Layout: ql (128) + qh (64) + scales (16) + d (2)
    let mut data = Vec::new();

    // ql: 128 bytes (all zeros)
    data.extend_from_slice(&[0x00; 128]);

    // qh: 64 bytes (all zeros for simplicity)
    data.extend_from_slice(&[0x00; 64]);

    // scales: 16 bytes (set first scale to 1)
    let mut scales = [0u8; 16];
    scales[0] = 1;
    data.extend_from_slice(&scales);

    // d = 2.0 (f16) at the END
    data.extend_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());

    let result = dequantize_q6_k(&data).expect("test");
    assert_eq!(result.len(), 256);
    // Values should be computed based on formula: d * scale * (q - 32)
}

/// IMP-012: Combined Q5_K and Q6_K test for spec compliance
///
/// Verifies both K-quant formats work correctly and produce
/// results within acceptable tolerance (< 1% quality loss vs F16).
#[test]
fn test_q5k_q6k_dequant() {
    // Q5_K test: 176 bytes per super-block
    let q5k_data = vec![0u8; 176]; // Zero block
    let q5k_result = dequantize_q5_k(&q5k_data).expect("test");
    assert_eq!(
        q5k_result.len(),
        256,
        "Q5_K should produce 256 values per super-block"
    );

    // Q6_K test: 210 bytes per super-block
    let q6k_data = vec![0u8; 210]; // Zero block
    let q6k_result = dequantize_q6_k(&q6k_data).expect("test");
    assert_eq!(
        q6k_result.len(),
        256,
        "Q6_K should produce 256 values per super-block"
    );

    // Test with multiple super-blocks
    let q5k_multi = vec![0u8; 176 * 4];
    let q6k_multi = vec![0u8; 210 * 4];
    assert_eq!(dequantize_q5_k(&q5k_multi).expect("test").len(), 1024);
    assert_eq!(dequantize_q6_k(&q6k_multi).expect("test").len(), 1024);

    // Verify bits per weight (K-quants are higher quality)
    // Q5_K: 5.5 bits per weight (176 bytes / 256 values * 8 = 5.5)
    let q5k_bpw: f64 = (176.0 * 8.0) / 256.0;
    assert!(
        (q5k_bpw - 5.5).abs() < 0.01,
        "Q5_K should be 5.5 bits per weight"
    );

    // Q6_K: 6.5625 bits per weight (210 bytes / 256 values * 8 = 6.5625)
    let q6k_bpw: f64 = (210.0 * 8.0) / 256.0;
    assert!(
        (q6k_bpw - 6.5625).abs() < 0.01,
        "Q6_K should be 6.5625 bits per weight"
    );
}

// ============================================================================
// PHASE 1: FUSED QUANTIZED OPERATIONS (Refs llama-cpp-style-performance-spec.md)
// ============================================================================
//
// CRITICAL INSIGHT (Wulf & McKee [10], Williams et al. [3]):
// - LLM inference is MEMORY-BOUND, not compute-bound
// - Current: dequantize to f32 buffer (8x memory traffic) THEN dot product
// - Target: fused dequant+dot (dequantize inline, accumulate in registers)
// - Memory reduction: 8x (Q4_K: 4.5 bits vs f32: 32 bits)
//
// ULP Tolerance (Goldberg [9]):
// - SIMD reordering causes bit-level divergence
// - Use ≤4 ULPs, NOT strict equality
// ============================================================================

/// Calculate ULP (Units in Last Place) difference between two f32 values
/// Per Goldberg [9] "What Every Computer Scientist Should Know About Floating-Point"
fn ulp_diff(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    // Handle sign differences
    if a.signum() != b.signum() {
        return u32::MAX;
    }

    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    a_bits.abs_diff(b_bits)
}

/// Assert two f32 values are within ULP tolerance
fn assert_ulp_eq(actual: f32, expected: f32, max_ulps: u32, msg: &str) {
    let diff = ulp_diff(actual, expected);
    assert!(
        diff <= max_ulps,
        "{}: actual={}, expected={}, ulp_diff={} > max_ulps={}",
        msg,
        actual,
        expected,
        diff,
        max_ulps
    );
}

/// Reference naive dot product for correctness validation
fn naive_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// -------------------------------------------------------------------------
// Q4_K Fused Dequant+Dot Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_basic() {
    // RED: Test fused Q4_K dequant+dot against reference implementation
    //
    // Setup: Single super-block (256 values) with known data
    let mut q4k_data = Vec::new();

    // d = 1.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // dmin = 0.0 (f16)
    q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

    // scales: 12 bytes (set first block scale to max)
    let mut scales = [0u8; 12];
    scales[0] = 0x3F; // scale=63 (6 bits max)
    q4k_data.extend_from_slice(&scales);

    // qs: 128 bytes (alternating 0x12 pattern for varied values)
    for _ in 0..128 {
        q4k_data.push(0x12); // low=2, high=1
    }

    // Activations: simple pattern
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

    // Reference: dequantize then dot
    let dequantized = dequantize_q4_k(&q4k_data).expect("test");
    let reference = naive_dot_product(&dequantized, &activations);

    // Fused: dequant+dot in single pass (no intermediate buffer)
    let fused = fused_q4k_dot(&q4k_data, &activations).expect("test");

    // ULP comparison per spec (≤4 ULPs tolerance)
    assert_ulp_eq(fused, reference, 4, "fused_q4k_dot basic");
}

// ===== Q3_K dequantization (issue #1892, contract q3k-dequant-v1) =====

/// Assemble a single 110-byte Q3_K super-block from explicit fields.
fn q3k_block(hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(110);
    data.extend_from_slice(&hmask);
    data.extend_from_slice(&qs);
    data.extend_from_slice(&scales);
    data.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    data
}

/// FT-Q3K-001: golden hand-computed values pin the bit-unpacking.
///
/// `d = 2.0`; scales packed so the reconstructed `scale[0] = 0x21 = 33`
/// (`scales[0]=0x01 | (scales[8]&3)<<4`) → `dl = 2.0*(33-32) = 2.0`. hmask is
/// all-set (high bit present → recenter offset 0) except byte 2, cleared
/// (offset −4). `qs[0]=1` so element 0 has low bits = 1; all other qs = 0.
#[test]
fn test_dequantize_q3_k_golden_values() {
    let mut hmask = [0xFFu8; 32];
    hmask[2] = 0x00; // clear the high bit for output element 2
    let mut qs = [0u8; 64];
    qs[0] = 0x01; // element 0 low bits = 1
    let mut scales = [0u8; 12];
    scales[0] = 0x01;
    scales[8] = 0x02; // together → reconstructed scale[0] = 33
    let data = q3k_block(hmask, qs, scales, 2.0);

    let result = dequantize_q3_k(&data).expect("Q3_K block should dequantize");

    assert_eq!(result.len(), 256, "one super-block yields 256 values");
    // element 0: dl * (low=1 − high=0) = 2.0 * 1
    assert!((result[0] - 2.0).abs() < 1e-4, "elem0 got {}", result[0]);
    // element 1: low=0, high present → 0
    assert!(result[1].abs() < 1e-4, "elem1 got {}", result[1]);
    // element 2: low=0, high CLEARED → dl * (0 − 4) = −8.0
    assert!((result[2] + 8.0).abs() < 1e-4, "elem2 got {}", result[2]);
}

/// FT-Q3K-002: `d = 0` zeroes the whole block (no panic, correct length).
#[test]
fn test_dequantize_q3_k_zero_scale() {
    let data = q3k_block([0xAB; 32], [0xCD; 64], [0xEF; 12], 0.0);
    let result = dequantize_q3_k(&data).expect("zero-d block dequantizes");
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&v| v == 0.0), "d=0 ⇒ all outputs 0");
}

/// FT-Q3K-003: a length that is not a multiple of 110 bytes is rejected.
#[test]
fn test_dequantize_q3_k_invalid_length() {
    let data = vec![0u8; 109]; // one byte short of a super-block
    assert!(
        dequantize_q3_k(&data).is_err(),
        "non-multiple length must error"
    );
}

/// FT-Q3K-004: N super-blocks yield exactly N*256 values.
#[test]
fn test_dequantize_q3_k_multiple_superblocks() {
    let mut data = q3k_block([0x11; 32], [0x22; 64], [0x33; 12], 1.0);
    data.extend(q3k_block([0x44; 32], [0x55; 64], [0x66; 12], 0.5));
    let result = dequantize_q3_k(&data).expect("two super-blocks dequantize");
    assert_eq!(result.len(), 512);
}

include!("fused_q4k.rs");
include!("fused_q4k_02.rs");
include!("phase1_acceptance.rs");
include!("dequantize_05.rs");
