use super::*;

/// Standard test data: 256 floats centered around zero.
fn test_data_256() -> Vec<f32> {
    (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect()
}

/// Compute max absolute error between original and dequantized data.
fn max_abs_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Compute data range (max - min).
fn data_range(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b))
}

/// Assert roundtrip error is within a fraction of data range.
fn assert_roundtrip_within_range(
    original: &[f32],
    dequantized: &[f32],
    fraction: f32,
    label: &str,
) {
    let error = max_abs_error(original, dequantized);
    let threshold = data_range(original) * fraction;
    assert!(
        error < threshold,
        "{label} roundtrip error {error} exceeds threshold {threshold}"
    );
}

#[test]
fn test_q4k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q4_k(&data);
    assert_eq!(quantized.len(), 144);
    let dequantized = dequantize_q4_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.5, "Q4K");
}

#[test]
fn test_q5k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q5_k(&data);
    assert_eq!(quantized.len(), 176);
    let dequantized = dequantize_q5_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.4, "Q5K");
}

#[test]
fn test_q6k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q6_k(&data);
    assert_eq!(quantized.len(), 210);
    let dequantized = dequantize_q6_k_to_f32(&quantized, 256);
    assert!(
        max_abs_error(&data, &dequantized) < 1.0,
        "Q6K roundtrip error too high"
    );
}

#[test]
fn test_q4k_matrix() {
    let data: Vec<f32> = (0..512).map(|i| i as f32 / 100.0).collect();
    let shape = vec![2, 256];
    let quantized = quantize_q4_k_matrix(&data, &shape);
    assert_eq!(quantized.len(), 2 * 144);
}

#[test]
fn test_transpose_q4k() {
    let cols = 256;
    let rows = 2;
    let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 / 10.0).collect();
    let quantized = quantize_q4_k(&data);
    let shape = vec![cols, rows];
    let (transposed_data, new_shape) = transpose_q4k_for_matmul(&quantized, &shape);
    assert_eq!(new_shape, vec![rows, cols]);
    assert!(!transposed_data.is_empty());
}

#[test]
fn test_f16_min_normal() {
    let f16_val = half::f16::from_f32(F16_MIN_NORMAL);
    let roundtrip = f16_val.to_f32();
    assert!(
        roundtrip > 0.0,
        "F16_MIN_NORMAL should be positive after f16 roundtrip"
    );
    assert!(roundtrip < 1e-4, "F16_MIN_NORMAL should be small");
}

#[test]
fn test_constants() {
    assert_eq!(Q4_K_BLOCK_SIZE, 256);
    assert_eq!(Q4_K_BLOCK_BYTES, 144);
    assert_eq!(Q5_K_BLOCK_SIZE, 256);
    assert_eq!(Q5_K_BLOCK_BYTES, 176);
    assert_eq!(Q6_K_BLOCK_SIZE, 256);
    assert_eq!(Q6_K_BLOCK_BYTES, 210);
}

// ===== Dequantize f16 scale sanitization tests =====
// Regression: Q5K and Q6K dequantize did not sanitize f16 scale values,
// unlike Q4K which guards against NaN/Inf/subnormal via sanitize_f16_scale().
// In clean-room containers (no SIMD flags), subnormal f16 values can propagate
// through dequantization and produce incorrect or non-finite results.

/// Construct a Q6K block with a hand-crafted f16 scale value and verify
/// dequantization produces only finite outputs.
#[test]
fn test_q6k_dequantize_subnormal_scale() {
    // Build a minimal Q6K block (210 bytes) with a subnormal f16 scale.
    // Layout: ql[128] + qh[64] + scales[16] + d(f16)[2]
    let mut block = vec![0u8; Q6_K_BLOCK_BYTES];

    // Set ql to non-zero pattern so dequant produces non-trivial values
    for i in 0..128 {
        block[i] = 0x12;
    }
    // Set scales to non-zero
    for i in 192..208 {
        block[i] = 1;
    }
    // Set d (f16) to a subnormal value: 0x0001 is the smallest positive subnormal f16
    block[208] = 0x01;
    block[209] = 0x00;

    let result = dequantize_q6_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Q6K dequant produced non-finite value at index {}: {}",
            i,
            v
        );
    }
}

/// Construct a Q6K block with NaN f16 scale and verify dequantization
/// returns zeros (not NaN propagation).
#[test]
fn test_q6k_dequantize_nan_scale() {
    let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
    for i in 0..128 {
        block[i] = 0x55;
    }
    for i in 192..208 {
        block[i] = 2;
    }
    // f16 NaN: exponent all 1s, non-zero mantissa. 0x7C01 is a NaN.
    block[208] = 0x01;
    block[209] = 0x7C;

    let result = dequantize_q6_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Q6K NaN scale propagated to index {}: {}",
            i,
            v
        );
        assert!(
            v.abs() < f32::EPSILON,
            "Q6K with NaN scale should produce 0.0 at index {}, got {}",
            i,
            v
        );
    }
}

/// Construct a Q5K block with a subnormal f16 scale and verify finite results.
#[test]
fn test_q5k_dequantize_subnormal_scale() {
    let mut block = vec![0u8; Q5_K_BLOCK_BYTES];
    // Set scales (bytes 4..16) to non-zero
    for i in 4..16 {
        block[i] = 0x21;
    }
    // Set qh (bytes 16..48) to non-zero
    for i in 16..48 {
        block[i] = 0x55;
    }
    // Set qs (bytes 48..176) to non-zero
    for i in 48..176 {
        block[i] = 0x33;
    }
    // d (f16) subnormal at bytes 0..1
    block[0] = 0x01;
    block[1] = 0x00;
    // dmin (f16) subnormal at bytes 2..3
    block[2] = 0x01;
    block[3] = 0x00;

    let result = dequantize_q5_k_to_f32(&block, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Q5K dequant produced non-finite value at index {}: {}",
            i,
            v
        );
    }
}

/// Q6K roundtrip with SIMD-boundary-crossing data: values that span
/// SIMD lane widths (8-wide, 16-wide) to catch scaling mismatches.
#[test]
fn test_q6k_simd_scaling_roundtrip() {
    // Data with sharp transitions at SIMD lane boundaries (every 8 and 16 elements)
    let data: Vec<f32> = (0..256)
        .map(|i| {
            let base = (i as f32 - 128.0) / 10.0;
            // Introduce sharp scaling change at lane boundary
            if i % 16 < 8 {
                base * 0.01
            } else {
                base * 100.0
            }
        })
        .collect();

    let quantized = quantize_q6_k(&data);
    let dequantized = dequantize_q6_k_to_f32(&quantized, 256);

    // All values must be finite
    for (i, &v) in dequantized.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Q6K SIMD scaling roundtrip: non-finite at index {}: {}",
            i,
            v
        );
    }

    // Roundtrip error should be bounded
    let max_err = data
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let range = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        max_err < range * 0.15,
        "Q6K SIMD scaling roundtrip error {} exceeds 15% of range {}",
        max_err,
        range
    );
}

// ============================================================================
// OBLIG-QUANT-F32-F16-RNE: f32_to_f16 IEEE round-to-nearest-even
//
// `aprender_quant::f32_to_f16` delegates to `half::f16::from_f32`, so it is
// IEEE RNE by construction. This falsifier LOCKS that delegation: if anyone
// re-introduces a hand-rolled round-toward-zero impl (the bug fixed in the
// trueno / aprender-solve paths), these known-divergence cases go RED.
// ============================================================================

#[test]
fn falsify_quant_f32_to_f16_rne_known_divergences() {
    use crate::f32_to_f16;
    // 255.99 rounds UP across the mantissa-carry boundary (truncation gives 0x5BFF).
    assert_eq!(
        f32_to_f16(255.99),
        0x5C00,
        "255.99 must round up (mantissa carry)"
    );
    // 65520.0 rounds the tie UP to +Inf (truncation gives max-finite 0x7BFF).
    assert_eq!(f32_to_f16(65520.0), 0x7C00, "65520.0 must round to +Inf");
    // Smallest subnormal half-way case rounds to nearest even (truncation gives 0).
    assert_eq!(
        f32_to_f16(5.960_464_5e-8),
        0x0001,
        "subnormal half-way rounds to 0x0001"
    );
    // Max finite f16 preserved.
    assert_eq!(f32_to_f16(65504.0), 0x7BFF, "max finite f16 preserved");
}

#[test]
fn falsify_quant_f32_to_f16_bit_identical_to_half() {
    use crate::f32_to_f16;
    let step = 0x29u32;
    let mut u: u32 = 0;
    loop {
        let v = f32::from_bits(u);
        let got = f32_to_f16(v);
        let want = half::f16::from_f32(v).to_bits();
        if got != want {
            let both_nan =
                half::f16::from_bits(got).is_nan() && half::f16::from_bits(want).is_nan();
            assert!(
                both_nan,
                "quant f32_to_f16 diverges from half at bits={u:#010x}: got={got:#06x} want={want:#06x}"
            );
        }
        let (next, overflow) = u.overflowing_add(step);
        if overflow {
            break;
        }
        u = next;
    }
}

/// 176 bytes: one ggml `block_q5_K` super-block, verbatim from a llama.cpp-quantized GGUF
/// (`Qwen3.5-0.8B-Q4_K_M.gguf`, tensor `blk.0.attn_qkv.weight`, super-block 1).
const GGML_Q5K_BLOCK: [u8; 176] = [
    0x2c, 0x03, 0x12, 0x14, 0xf3, 0xff, 0xf5, 0xf6, 0xb1, 0xbd, 0x7f, 0xa5, 0x45, 0x50, 0xef, 0x89,
    0x85, 0xbf, 0x07, 0x27, 0xef, 0xae, 0x4b, 0xaf, 0x87, 0x25, 0x96, 0x97, 0x36, 0x0e, 0x27, 0xd1,
    0x9f, 0x1f, 0x0e, 0x8f, 0x97, 0x05, 0x27, 0x06, 0x0d, 0x36, 0x9e, 0x3c, 0x4f, 0x25, 0x87, 0xaf,
    0xd5, 0x45, 0xf7, 0x38, 0x16, 0x0e, 0x08, 0x45, 0x75, 0xd6, 0x4a, 0x53, 0x4e, 0x10, 0x23, 0xfa,
    0x45, 0x67, 0x73, 0x46, 0x51, 0xe9, 0x34, 0x3f, 0x08, 0x2e, 0x4e, 0xcd, 0x6e, 0xc9, 0x2f, 0x54,
    0x00, 0xe7, 0x5b, 0x75, 0x9a, 0x4b, 0x30, 0xdc, 0xdf, 0xc1, 0xfd, 0xe1, 0x0a, 0x2a, 0xd1, 0x6d,
    0x2a, 0x62, 0x46, 0x8b, 0xbc, 0x83, 0xd6, 0xf6, 0xc7, 0xc8, 0x08, 0x09, 0x1b, 0x85, 0xa4, 0x88,
    0xe0, 0x10, 0x6a, 0x5e, 0x5f, 0x3e, 0x8f, 0x0c, 0xba, 0xed, 0x30, 0x53, 0x44, 0xdd, 0x8d, 0xf1,
    0xc2, 0xc1, 0x68, 0xe8, 0x7b, 0x09, 0x3a, 0x6e, 0xd9, 0x53, 0x58, 0x1f, 0x4d, 0x56, 0x3b, 0x2d,
    0x0d, 0xdc, 0xea, 0xdf, 0xef, 0x1f, 0x22, 0x47, 0x2c, 0x07, 0xe9, 0x2c, 0xeb, 0xca, 0x80, 0x02,
    0x1b, 0xb0, 0xe9, 0x79, 0x2e, 0xea, 0xc2, 0xfd, 0xfb, 0xb9, 0x2d, 0xcb, 0xe8, 0xe8, 0x98, 0x2b,
];
/// gguf-py `dequantize(block, Q5_K)` of `GGML_Q5K_BLOCK`: the 256 values in ggml order.
const GGML_Q5K_EXPECTED: [f32; 256] = [
    0.003_142_595_3,
    0.003_142_595_3,
    0.008_079_29,
    0.010_547_638,
    0.005_610_943,
    -0.014_135_838,
    0.010_547_638,
    0.003_142_595_3,
    0.003_142_595_3,
    0.005_610_943,
    -0.024_009_228,
    -0.001_794_099_8,
    -0.014_135_838,
    -0.048_692_703,
    -0.001_794_099_8,
    0.015_484_333,
    0.003_142_595_3,
    0.008_079_29,
    -0.041_287_66,
    0.005_610_943,
    -0.006_730_795,
    0.013_015_985_5,
    0.000_674_247_74,
    -0.011_667_49,
    0.010_547_638,
    -0.014_135_838,
    -0.014_135_838,
    -0.016_604_185,
    0.025_357_723,
    0.013_015_985_5,
    0.027_826_07,
    0.000_674_247_74,
    -0.020_978_69,
    0.000_365_257_26,
    0.033_905_745,
    -0.002_683_878,
    -0.008_782_148,
    -0.011_831_284,
    -0.011_831_284,
    0.000_365_257_26,
    0.009_512_663,
    -0.020_978_69,
    0.000_365_257_26,
    0.003_414_392_5,
    0.000_365_257_26,
    -0.008_782_148,
    -0.005_733_013,
    -0.014_880_419,
    0.000_365_257_26,
    0.006_463_527_7,
    0.009_512_663,
    0.000_365_257_26,
    0.003_414_392_5,
    -0.017_929_554,
    -0.002_683_878,
    -0.002_683_878,
    -0.060_617_447,
    -0.005_733_013,
    0.000_365_257_26,
    -0.024_027_824,
    0.006_463_527_7,
    -0.024_027_824,
    -0.005_733_013,
    0.003_414_392_5,
    -0.021_562_576,
    -0.003_606_557_8,
    0.006_654_024,
    -0.008_736_849,
    0.004_088_878_6,
    0.006_654_024,
    -0.062_604_904,
    0.009_219_17,
    0.016_914_606,
    -0.018_997_43,
    0.011_784_315,
    -0.018_997_43,
    0.004_088_878_6,
    0.004_088_878_6,
    -0.018_997_43,
    -0.029_258_013,
    0.004_088_878_6,
    -0.016_432_285,
    -0.006_171_703_3,
    0.006_654_024,
    0.009_219_17,
    -0.013_867_14,
    -0.006_171_703_3,
    -0.006_171_703_3,
    -0.003_606_557_8,
    -0.001_041_412_4,
    -0.001_041_412_4,
    0.001_523_733_1,
    0.006_654_024,
    -0.008_736_849,
    -0.011_301_994,
    -0.001_041_412_4,
    -0.036_767_96,
    0.041_638_374,
    -0.023_700_237,
    -0.018_473_148,
    0.028_570_652,
    0.015_502_93,
    0.012_889_385,
    0.039_024_83,
    -0.002_791_881_6,
    -0.005_405_426,
    0.002_435_207_4,
    -0.000_178_337_1,
    -0.036_767_96,
    0.010_275_841,
    -0.002_791_881_6,
    -0.021_086_693,
    0.010_275_841,
    0.020_730_019,
    0.015_502_93,
    0.025_957_108,
    -0.008_018_970_5,
    -0.015_859_604,
    -0.002_791_881_6,
    0.002_435_207_4,
    0.036_411_285,
    -0.005_405_426,
    0.005_048_752,
    0.005_048_752,
    0.007_662_296_3,
    -0.015_859_604,
    -0.010_632_515,
    0.025_957_108,
    -0.035_774_23,
    0.005_268_097,
    -0.010_122_776,
    0.000_137_805_94,
    0.002_702_951_4,
    0.000_137_805_94,
    0.002_702_951_4,
    -0.004_992_485,
    -0.010_122_776,
    -0.002_427_339_6,
    0.005_268_097,
    0.012_963_533,
    0.015_528_679,
    -0.002_427_339_6,
    -0.002_427_339_6,
    0.007_833_242,
    0.010_398_388,
    0.007_833_242,
    -0.015_253_067,
    -0.015_253_067,
    0.033_484_697,
    -0.012_687_921_5,
    -0.010_122_776,
    0.000_137_805_94,
    -0.012_687_921_5,
    0.012_963_533,
    0.025_789_26,
    0.043_745_28,
    -0.002_427_339_6,
    -0.020_383_358,
    -0.007_557_630_5,
    -0.002_427_339_6,
    -0.004_243_850_7,
    0.002_725_601_2,
    -0.022_829_056,
    0.012_018_204,
    0.012_018_204,
    0.007_371_902_5,
    -0.018_182_755,
    0.000_402_450_56,
    -0.011_213_303,
    0.032_926_56,
    -0.029_798_508,
    -0.025_152_206,
    0.009_695_053,
    -0.006_567_001_3,
    0.018_987_656,
    -0.001_920_700_1,
    -0.008_890_152,
    -0.008_890_152,
    -0.022_829_056,
    -0.004_243_850_7,
    -0.020_505_905,
    -0.036_767_96,
    0.007_371_902_5,
    -0.022_829_056,
    -0.006_567_001_3,
    0.012_018_204,
    -0.025_152_206,
    0.002_725_601_2,
    -0.027_475_357,
    0.012_018_204,
    -0.029_798_508,
    0.005_048_752,
    0.009_826_899,
    0.006_777_763_4,
    0.000_679_492_95,
    0.015_925_169,
    0.064_711_33,
    0.015_925_169,
    0.025_072_575,
    -0.008_467_913,
    0.006_777_763_4,
    -0.008_467_913,
    -0.002_369_642_3,
    0.006_777_763_4,
    0.003_728_628_2,
    0.000_679_492_95,
    -0.029_811_86,
    0.025_072_575,
    0.003_728_628_2,
    -0.029_811_86,
    -0.002_369_642_3,
    -0.002_369_642_3,
    0.012_876_034,
    0.000_679_492_95,
    -0.023_713_589,
    0.009_826_899,
    0.003_728_628_2,
    -0.002_369_642_3,
    0.009_826_899,
    0.003_728_628_2,
    0.043_367_386,
    -0.005_418_777_5,
    -0.005_418_777_5,
    0.003_728_628_2,
    0.004_390_716_6,
    0.040_254_354,
    -0.001_126_766_2,
    -0.003_885_507_6,
    0.043_013_096,
    0.007_149_458,
    -0.034_231_663,
    0.015_425_682,
    0.009_908_199,
    -0.039_749_146,
    0.043_013_096,
    0.009_908_199,
    -0.001_126_766_2,
    -0.006_644_249,
    -0.017_679_214,
    0.004_390_716_6,
    0.007_149_458,
    -0.009_402_99,
    -0.001_126_766_2,
    0.023_701_906,
    0.009_908_199,
    -0.001_126_766_2,
    -0.006_644_249,
    0.001_631_975_2,
    0.001_631_975_2,
    -0.009_402_99,
    0.009_908_199,
    -0.006_644_249,
    -0.001_126_766_2,
    -0.001_126_766_2,
    0.029_219_389,
    0.009_908_199,
];

/// FALSIFY-QDOT-007 (PMAT-1101): the `Q5_K` reader behind `apr import` decodes ggml's
/// `block_q5_K`. The oracle is gguf-py, llama.cpp's own reader, on a super-block llama.cpp
/// quantized (the fixture realizar's `quantize::tests::q5k_ggml` also uses). The quantizer is
/// pinned to this reader by `test_q5k_roundtrip`.
#[test]
fn test_q5k_ggml_dequantize_matches_gguf_py_bit_exact() {
    let got = dequantize_q5_k_to_f32(&GGML_Q5K_BLOCK, 256);
    assert_eq!(got.len(), GGML_Q5K_EXPECTED.len());
    for (i, (g, e)) in got.iter().zip(GGML_Q5K_EXPECTED).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "value {i}: aprender-quant {g}, gguf-py {e}"
        );
    }
}

/// FALSIFY-QDOT-007 (PMAT-1101), packer half: `quantize_q5_k` must pack ggml's layout, i.e.
/// round-trip through the gguf-py-pinned reader above. The ramp in `test_q5k_roundtrip` gives
/// every sub-block the same 5-bit codes, so a packer that swaps sub-blocks survives it (a
/// mutant did). Pseudo-random values give every sub-block its own codes.
#[test]
fn test_q5k_ggml_quantize_roundtrip_distinct_sub_blocks() {
    let mut state = 0x9E37_79B9_u32;
    let data: Vec<f32> = (0..512)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        })
        .collect();
    let dequantized = dequantize_q5_k_to_f32(&quantize_q5_k(&data), data.len());
    let err = max_abs_error(&data, &dequantized);
    assert!(
        err < 0.15,
        "Q5_K round-trip max error {err}: packer and reader disagree on the layout"
    );
}
