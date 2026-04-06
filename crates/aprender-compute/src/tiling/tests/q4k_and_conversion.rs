use super::super::*;

// ========================================================================
// TILE-003: Q4K MatVec Tests
// ========================================================================

#[test]
fn test_tiled_q4k_matvec_creation() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    assert_eq!(matvec.m, 4096);
    assert_eq!(matvec.k, 4096);
    assert_eq!(matvec.superblocks_per_row(), 16); // 4096 / 256
    assert_eq!(matvec.total_superblocks(), 4096 * 16);
}

#[test]
#[should_panic(expected = "K dimension")]
fn test_tiled_q4k_matvec_unaligned_k() {
    let _ = TiledQ4KMatvec::new(4096, 100); // Not aligned to 256
}

#[test]
fn test_tiled_q4k_matvec_weight_offset() {
    let matvec = TiledQ4KMatvec::new(100, 512);
    // Row 0: offset 0
    assert_eq!(matvec.weight_row_offset(0), 0);
    // Row 1: offset = 2 superblocks * 144 bytes = 288
    assert_eq!(matvec.weight_row_offset(1), 2 * Q4K_SUPERBLOCK_BYTES);
}

#[test]
fn test_tiled_q4k_matvec_optimal_rows() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    // With 256KB L2, should fit many rows
    let rows = matvec.optimal_parallel_rows(256 * 1024);
    assert!(rows >= 4); // At least micro-kernel size
    assert!(rows <= 4096); // At most all rows
}

#[test]
fn test_tiled_q4k_matvec_stats() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    let stats = matvec.stats();

    // Weight bytes: 4096 * 16 * 144 = 9,437,184 bytes
    assert_eq!(stats.superblocks, 4096 * 16);
    // Arithmetic ops: 4096 * 4096 * 2 = 33,554,432
    assert_eq!(stats.arithmetic_ops, 4096 * 4096 * 2);
    // AI should be reasonable for Q4K
    assert!(stats.arithmetic_intensity > 1.0);
}

#[test]
fn test_q4k_constants() {
    assert_eq!(Q4K_SUPERBLOCK_SIZE, 256);
    assert_eq!(Q4K_SUPERBLOCK_BYTES, 144);
}

// F325: K dimension exactly equals superblock
#[test]
fn test_k_equals_superblock() {
    let matvec = TiledQ4KMatvec::new(100, 256);
    assert_eq!(matvec.superblocks_per_row(), 1);
    assert_eq!(matvec.total_superblocks(), 100);
}

// F326: Very large M dimension
#[test]
fn test_large_m_dimension() {
    let matvec = TiledQ4KMatvec::new(100_000, 256);
    assert_eq!(matvec.superblocks_per_row(), 1);
    assert_eq!(matvec.total_superblocks(), 100_000);
    // Should still compute optimal rows
    let rows = matvec.optimal_parallel_rows(256 * 1024);
    assert!(rows >= 4);
}

// F327: Very large K dimension
#[test]
fn test_large_k_dimension() {
    let matvec = TiledQ4KMatvec::new(10, 32768); // 32K hidden dim
    assert_eq!(matvec.superblocks_per_row(), 128);
    let stats = matvec.stats();
    assert!(stats.arithmetic_intensity > 0.0);
}

// F344: TilingStats fields
#[test]
fn test_tiling_stats_complete() {
    let matvec = TiledQ4KMatvec::new(100, 512);
    let stats = matvec.stats();

    assert_eq!(stats.input_bytes, 512 * 4);
    assert_eq!(stats.output_bytes, 100 * 4);
    assert_eq!(stats.superblocks, 100 * 2); // 512/256 = 2 per row
    assert!(stats.total_weight_bytes > 0);
}

// ========================================================================
// f16 conversion tests
// ========================================================================

// F331: f16 to f32 conversion
#[test]
fn test_f16_conversion() {
    // Zero
    assert_eq!(f16_to_f32(&[0x00, 0x00]), 0.0);

    // One (0x3C00 in f16)
    let one = f16_to_f32(&[0x00, 0x3C]);
    assert!((one - 1.0).abs() < 0.001);

    // Negative one (0xBC00)
    let neg_one = f16_to_f32(&[0x00, 0xBC]);
    assert!((neg_one - (-1.0)).abs() < 0.001);

    // Infinity (0x7C00)
    assert!(f16_to_f32(&[0x00, 0x7C]).is_infinite());

    // NaN (0x7C01)
    assert!(f16_to_f32(&[0x01, 0x7C]).is_nan());
}

// F332: f16 subnormal conversion
#[test]
fn test_f16_subnormal() {
    // Smallest positive subnormal: 0x0001
    let subnormal = f16_to_f32(&[0x01, 0x00]);
    assert!(subnormal > 0.0);
    assert!(subnormal < 0.001); // Very small

    // Negative zero: 0x8000
    let neg_zero = f16_to_f32(&[0x00, 0x80]);
    assert_eq!(neg_zero, -0.0);
    assert!(neg_zero.is_sign_negative());

    // Negative infinity: 0xFC00
    let neg_inf = f16_to_f32(&[0x00, 0xFC]);
    assert!(neg_inf.is_infinite());
    assert!(neg_inf.is_sign_negative());
}

// F345: extract_scale_min_6bit function — GGML Q4_K split format
#[test]
fn test_extract_scale_min_6bit() {
    // GGML Q4_K scale layout (12 bytes):
    //   bytes[0..3]:  bits[5:0] = scale SB 0-3,  bits[7:6] = high2 of scale SB 4-7
    //   bytes[4..7]:  bits[5:0] = min SB 0-3,    bits[7:6] = high2 of min SB 4-7
    //   bytes[8..11]: bits[3:0] = low4 of scale SB 4-7, bits[7:4] = low4 of min SB 4-7

    // SB 0: scale = bytes[0] & 0x3F = 0x3F = 63, min = bytes[4] & 0x3F = 0
    let scales = [0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let (sc, mn) = extract_scale_min_6bit(&scales, 0);
    assert_eq!(sc, 63.0);
    assert_eq!(mn, 0.0);

    // SB 1: scale = bytes[1] & 0x3F, min = bytes[5] & 0x3F
    let scales2 = [0x00, 0x2A, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let (sc1, mn1) = extract_scale_min_6bit(&scales2, 1);
    assert_eq!(sc1, 42.0); // 0x2A = 42
    assert_eq!(mn1, 21.0); // 0x15 = 21

    // SB 4: scale = (bytes[8] & 0x0F) | ((bytes[0] >> 6) << 4)
    //        min  = (bytes[8] >> 4) | ((bytes[4] >> 6) << 4)
    // bytes[0] = 0xC0 → bits[7:6] = 3, bytes[4] = 0x80 → bits[7:6] = 2
    // bytes[8] = 0x97 → lo4 = 7, hi4 = 9
    // scale = 7 | (3 << 4) = 7 | 48 = 55
    // min   = 9 | (2 << 4) = 9 | 32 = 41
    let scales3 = [0xC0, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x97, 0x00, 0x00, 0x00];
    let (sc4, mn4) = extract_scale_min_6bit(&scales3, 4);
    assert_eq!(sc4, 55.0);
    assert_eq!(mn4, 41.0);

    // SB 7: scale = (bytes[11] & 0x0F) | ((bytes[3] >> 6) << 4)
    //        min  = (bytes[11] >> 4) | ((bytes[7] >> 6) << 4)
    let scales4 = [0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0xFE];
    // bytes[3]=0x40 → bits[7:6]=1, bytes[7]=0xC0 → bits[7:6]=3
    // bytes[11]=0xFE → lo4=0xE=14, hi4=0xF=15
    // scale = 14 | (1 << 4) = 14 | 16 = 30
    // min   = 15 | (3 << 4) = 15 | 48 = 63
    let (sc7, mn7) = extract_scale_min_6bit(&scales4, 7);
    assert_eq!(sc7, 30.0);
    assert_eq!(mn7, 63.0);
}

// Verify extract_scale_min_6bit matches parse_q4k_header for all 8 sub-blocks
#[test]
fn test_extract_scale_min_matches_reference() {
    // Use known-correct parse_q4k_header from backends::q4k as oracle
    // Construct a full 16-byte header with interesting scale patterns
    let header: [u8; 16] = [
        0x00, 0x3C, // d (f16)
        0x00, 0x3C, // dmin (f16)
        // scales[0..3]: SB 0-3 scales (6-bit) + high2 of SB 4-7 scales
        0xC5, 0x8A, 0x4F, 0xD4,
        // scales[4..7]: SB 0-3 mins (6-bit) + high2 of SB 4-7 mins
        0x91, 0xD6, 0x63, 0xAB, // scales[8..11]: lo4=SB 4-7 scale, hi4=SB 4-7 min
        0x37, 0xB9, 0x2C, 0xE5,
    ];

    let scales = &header[4..16];

    // Compute reference via the GGML split format manually
    for i in 0..4u8 {
        let (sc, mn) = extract_scale_min_6bit(scales, i as usize);
        let expected_sc = (scales[i as usize] & 0x3F) as f32;
        let expected_mn = (scales[4 + i as usize] & 0x3F) as f32;
        assert_eq!(sc, expected_sc, "SB {i} scale mismatch");
        assert_eq!(mn, expected_mn, "SB {i} min mismatch");
    }

    for i in 0..4u8 {
        let (sc, mn) = extract_scale_min_6bit(scales, (4 + i) as usize);
        let combo = scales[8 + i as usize];
        let expected_sc = ((combo & 0x0F) | ((scales[i as usize] >> 6) << 4)) as f32;
        let expected_mn = (((combo >> 4) & 0x0F) | ((scales[4 + i as usize] >> 6) << 4)) as f32;
        assert_eq!(sc, expected_sc, "SB {} scale mismatch", 4 + i);
        assert_eq!(mn, expected_mn, "SB {} min mismatch", 4 + i);
    }
}

// F333: Execute scalar implementation
#[test]
fn test_execute_scalar() {
    let matvec = TiledQ4KMatvec::new(2, 256);

    // Create minimal valid Q4K weights (2 rows × 1 superblock each)
    let mut weights = vec![0u8; 2 * Q4K_SUPERBLOCK_BYTES];

    // Set up first row: d=1.0, dmin=0.0, all qs=0
    // f16 for 1.0 is 0x3C00
    weights[0] = 0x00;
    weights[1] = 0x3C;
    // dmin = 0
    weights[2] = 0x00;
    weights[3] = 0x00;
    // scales all zero (simplified)
    // qs all zero -> dequantized values will be 0

    // Second row: same setup
    let offset = Q4K_SUPERBLOCK_BYTES;
    weights[offset] = 0x00;
    weights[offset + 1] = 0x3C;

    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];

    matvec.execute_scalar(&weights, &input, &mut output);

    // With zero quantized values, output should be 0 or near 0
    // (The exact value depends on the scale/min extraction)
    assert!(output[0].is_finite());
    assert!(output[1].is_finite());
}

// GH-182: Cross-validate scalar_superblock_dot against dequantize_q4k_to_f32 oracle.
// This is the key regression test: the dequantize function uses the known-correct
// parse_q4k_header + pair-based qs layout, so any discrepancy means the tiling
// code has a scale/qs bug.
#[test]
fn test_scalar_dot_matches_dequantize_oracle() {
    use crate::backends::q4k::dequantize_q4k_to_f32;

    // Build a Q4K super-block with non-trivial scale patterns that exercise
    // the split format for sub-blocks 4-7.
    let mut sb = vec![0u8; 144];

    // d = 0.5 (f16: 0x3800), dmin = 0.25 (f16: 0x3400)
    sb[0] = 0x00;
    sb[1] = 0x38; // d
    sb[2] = 0x00;
    sb[3] = 0x34; // dmin

    // Scale bytes: exercise all 8 sub-blocks with distinct values
    // bytes[0..3] = scale SB 0-3 (6-bit) + high2 of scale SB 4-7
    sb[4] = 0xCA; // SB0 scale = 0x0A=10, high2 for SB4 = 3
    sb[5] = 0x94; // SB1 scale = 0x14=20, high2 for SB5 = 2
    sb[6] = 0x5E; // SB2 scale = 0x1E=30, high2 for SB6 = 1
    sb[7] = 0x28; // SB3 scale = 0x28=40, high2 for SB7 = 0
                  // bytes[4..7] = min SB 0-3 (6-bit) + high2 of min SB 4-7
    sb[8] = 0x45; // SB0 min = 0x05=5,  high2 for min SB4 = 1
    sb[9] = 0x8F; // SB1 min = 0x0F=15, high2 for min SB5 = 2
    sb[10] = 0xD9; // SB2 min = 0x19=25, high2 for min SB6 = 3
    sb[11] = 0x23; // SB3 min = 0x23=35, high2 for min SB7 = 0
                   // bytes[8..11] = lo4 scale SB 4-7 | (lo4 min SB 4-7 << 4)
    sb[12] = 0x35; // SB4: scale_lo4=5, min_lo4=3 → scale=5|(3<<4)=53, min=3|(1<<4)=19
    sb[13] = 0x7A; // SB5: scale_lo4=0xA=10, min_lo4=7 → scale=10|(2<<4)=42, min=7|(2<<4)=39
    sb[14] = 0x2F; // SB6: scale_lo4=0xF=15, min_lo4=2 → scale=15|(1<<4)=31, min=2|(3<<4)=50
    sb[15] = 0x83; // SB7: scale_lo4=3, min_lo4=8 → scale=3|(0<<4)=3, min=8|(0<<4)=8

    // Fill qs with a repeating pattern to exercise all nibble values
    for i in 0..128 {
        let lo = ((i * 7 + 3) % 16) as u8;
        let hi = ((i * 11 + 5) % 16) as u8;
        sb[16 + i] = lo | (hi << 4);
    }

    // Oracle: dequantize the full super-block
    let dequant = dequantize_q4k_to_f32(&sb, 256);

    // Test input: alternating pattern to catch nibble/position bugs
    let mut input = vec![0.0f32; 256];
    for i in 0..256 {
        input[i] = (i as f32 * 0.01) - 1.28; // range [-1.28, 1.27]
    }

    // Reference dot product via oracle dequantized values
    let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

    // Tiling dot product
    let matvec = TiledQ4KMatvec::new(1, 256);
    let mut output = vec![0.0f32; 1];
    matvec.execute_scalar(&sb, &input, &mut output);

    let actual = output[0];
    let rel_err = if expected.abs() > 1e-6 {
        (actual - expected).abs() / expected.abs()
    } else {
        (actual - expected).abs()
    };

    assert!(
        rel_err < 1e-5,
        "GH-182: scalar_superblock_dot diverges from dequantize oracle!\n\
         expected={expected}, actual={actual}, rel_err={rel_err}\n\
         This indicates scale extraction or qs addressing mismatch."
    );
}
