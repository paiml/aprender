//! T-COV-95 Phase 52: Additional coverage for apr_transformer/mod.rs
//!
//! Covers:
//! - dequant_perrow: multi-block rows, edge cases
//! - dequant_q4k_block: detailed nibble extraction, dmin interaction
//! - dequant_q6k_block: scale/offset interactions, all-zero ql/qh
//! - ActivationStats: mixed NaN/Inf/zero, two-element variance
//! - AprTransformer: from_apr_bytes extended paths, serialization edge cases

use crate::apr_transformer::{
    ActivationStats, AprTransformerConfig, ForwardTrace, LayerActivation,
};

use super::super::{dequant_perrow, dequant_q4k_block, dequant_q6k_block};

// ============================================================================
// dequant_q4k_block: detailed tests
// ============================================================================

#[test]
fn test_dequant_q4k_block_with_dmin() {
    // Test that dmin actually subtracts from the result
    let mut block = vec![0u8; 144];
    // d = 1.0 (f16: 0x3C00)
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 1.0 (f16: 0x3C00)
    block[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set scales: scale[0] = 1, min[4] = 1 (m1 for first 64 values)
    block[4] = 0x01; // scale[0] = 1 & 63 = 1
    block[8] = 0x01; // min = scales[4] = 1 & 63 = 1

    // All qs = 0 -> d1 * (0 & 0xF) - dm1 = 0 - dmin*m1 = -1.0
    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // First 32 values should be d*scale*(0) - dmin*min = 0 - 1.0 = -1.0
    assert!(
        (out[0] - (-1.0)).abs() < 0.01,
        "Expected -1.0, got {}",
        out[0]
    );
}

#[test]
fn test_dequant_q4k_block_nibble_extraction() {
    // Test that low and high nibbles are extracted correctly
    let mut block = vec![0u8; 144];
    // d = 1.0
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // dmin = 0
    block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // scale[0] = 1 (for first 64 values)
    block[4] = 0x01;
    // Set first qs byte to 0xAB: low nibble = 0xB = 11, high nibble = 0xA = 10
    block[16] = 0xAB;
    // Rest of qs = 0

    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);

    // out[0] = d * sc1 * (0xAB & 0xF) - dm1 = 1.0 * 1.0 * 11 - 0 = 11.0
    assert!(
        (out[0] - 11.0).abs() < 0.01,
        "Expected 11.0 for low nibble, got {}",
        out[0]
    );
}

#[test]
fn test_dequant_q4k_block_all_ones_qs() {
    let mut block = vec![0u8; 144];
    // d = 0.5 (f16: 0x3800)
    block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
    // dmin = 0
    block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // scale[0] = 2
    block[4] = 0x02;
    // All qs bytes = 0xFF (low nibble = 15, high nibble = 15)
    for i in 0..128 {
        block[16 + i] = 0xFF;
    }
    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // First value: d * sc1 * 15 = 0.5 * 2 * 15 = 15.0
    assert!(
        (out[0] - 15.0).abs() < 0.01,
        "Expected 15.0, got {}",
        out[0]
    );
}

#[test]
fn test_dequant_q4k_block_d_negative() {
    // Test with negative d (f16 for -1.0 = 0xBC00)
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&0xBC00u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    block[4] = 0x01; // scale = 1
    block[16] = 0x01; // qs[0] low nibble = 1

    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // out[0] = -1.0 * 1.0 * 1 - 0 = -1.0
    assert!(
        (out[0] - (-1.0)).abs() < 0.01,
        "Expected -1.0, got {}",
        out[0]
    );
}

// ============================================================================
// dequant_q6k_block: detailed tests
// ============================================================================

#[test]
fn test_dequant_q6k_block_with_scale_and_ql() {
    let mut block = vec![0u8; 210];
    // d = 0.5 (f16: 0x3800) at offset 208
    block[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
    // scale[0] = 2
    block[192] = 2;
    // ql[0] = 0x11 -> low nibble = 1 for q1, high nibble = 1 for q3
    block[0] = 0x11;
    // qh[0] = 0x00 -> no high bits
    block[128] = 0x00;

    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    // q1 = ((0x11 & 0xF) | ((0 & 3) << 4)) - 32 = (1 | 0) - 32 = -31
    // out[0] = 0.5 * 2.0 * (-31) = -31.0
    assert!(
        (out[0] - (-31.0)).abs() < 0.1,
        "Expected -31.0, got {}",
        out[0]
    );
}

#[test]
fn test_dequant_q6k_block_negative_d() {
    let mut block = vec![0u8; 210];
    // d = -1.0 (f16: 0xBC00) at offset 208
    block[208..210].copy_from_slice(&0xBC00u16.to_le_bytes());
    // scale[0] = 1
    block[192] = 1;
    // ql[0] = 0 -> q1 = (0 | 0) - 32 = -32
    // out[0] = -1.0 * 1 * (-32) = 32.0
    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    assert!((out[0] - 32.0).abs() < 0.1, "Expected 32.0, got {}", out[0]);
}

#[test]
fn test_dequant_q6k_block_high_bits_contribution() {
    let mut block = vec![0u8; 210];
    // d = 1.0 at offset 208
    block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    // scale[0] = 1
    block[192] = 1;
    // ql[0] = 0x0F (low nibble = 15, for q1)
    block[0] = 0x0F;
    // qh[0] = 0x03 (bits 0-1 = 3, contributes to q1)
    block[128] = 0x03;

    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    // q1 = ((0x0F & 0xF) | ((3 & 3) << 4)) - 32 = (15 | 48) - 32 = 63 - 32 = 31
    // out[0] = 1.0 * 1 * 31 = 31.0
    assert!((out[0] - 31.0).abs() < 0.1, "Expected 31.0, got {}", out[0]);
}

#[test]
fn test_dequant_q6k_block_second_half() {
    // Test the n=128 iteration (second half of the super-block)
    let mut block = vec![0u8; 210];
    // d = 1.0
    block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    // scale[8] = 3 (for the second 128-value half, idx=1, sc = scales[8..])
    block[200] = 3;
    // ql at offset 64 (for idx=1): ql[64] = 0x02
    block[64] = 0x02;
    // qh at offset 160 (for idx=1): qh[32] = 0x00
    block[160] = 0x00;

    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    // For n=128, l=0: q1 = ((ql[64] & 0xF) | ((qh[32] & 3) << 4)) - 32 = (2 | 0) - 32 = -30
    // out[128] = 1.0 * sc[is=0] * (-30) = 1.0 * 3 * (-30) = -90.0
    assert!(
        (out[128] - (-90.0)).abs() < 0.1,
        "Expected -90.0, got {}",
        out[128]
    );
}

// ============================================================================
// dequant_perrow: additional edge cases
// ============================================================================

#[test]
fn test_dequant_perrow_multi_block_row() {
    // 1 row, 512 cols -> 2 blocks per row for Q4K (256 elems/block)
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 1;
    let cols: usize = 512;
    let blocks_per_row = cols.div_ceil(block_elems); // 2
    let data = vec![0u8; rows * blocks_per_row * block_bytes];
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        // Fill each block output with a distinct pattern based on index
        for (i, v) in out.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
    });

    assert_eq!(result.len(), 512);
    // Both blocks use same closure, so pattern repeats
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[256] - 1.0).abs() < 0.001);
}

#[test]
fn test_dequant_perrow_single_row() {
    let block_bytes = 210; // Q6K
    let block_elems = 256;
    let rows = 1;
    let cols = 256;
    let data = vec![0u8; block_bytes];
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = i as f32;
        }
    });

    assert_eq!(result.len(), 256);
    assert!((result[0] - 0.0).abs() < 0.001);
    assert!((result[255] - 255.0).abs() < 0.001);
}

#[test]
fn test_dequant_perrow_zero_data() {
    // Empty data -> should return zeros (fill path)
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 2;
    let cols = 256;
    let data: Vec<u8> = Vec::new(); // No data at all
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        for v in out.iter_mut() {
            *v = 42.0;
        }
    });

    // Should have filled remaining with zeros since data is insufficient
    assert_eq!(result.len(), rows * cols);
    // All values should be 0.0 (early fill)
    for &v in &result {
        assert!(
            (v - 0.0).abs() < 0.001,
            "Expected 0.0 for insufficient data, got {}",
            v
        );
    }
}

#[test]
fn test_dequant_perrow_cols_not_multiple_of_block_elems() {
    // cols = 100 (not a multiple of 256), 1 block per row still needed
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 2;
    let cols: usize = 100;
    let blocks_per_row = cols.div_ceil(block_elems); // 1
    let data = vec![0u8; rows * blocks_per_row * block_bytes];
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
    });

    // Should have exactly rows * cols values (padding stripped)
    assert_eq!(result.len(), rows * cols);
    // Values should be 1..=100 for first row (not extending to 256)
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[99] - 100.0).abs() < 0.001);
}

// ============================================================================
// ActivationStats: additional edge cases
// ============================================================================

#[test]
fn test_activation_stats_mixed_nan_inf_zero() {
    let data = vec![
        0.0,
        f32::NAN,
        f32::INFINITY,
        1.0,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        2.0,
    ];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 8);
    assert_eq!(stats.nan_count, 2);
    assert_eq!(stats.inf_count, 2);
    assert_eq!(stats.zero_count, 2);
    // Valid values: 0.0, 1.0, 0.0, 2.0 -> mean = 3.0/4 = 0.75
    assert!((stats.mean - 0.75).abs() < 0.01);
}

#[test]
fn test_activation_stats_two_elements_variance() {
    let data = vec![0.0, 10.0];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 2);
    assert!((stats.min - 0.0).abs() < 0.001);
    assert!((stats.max - 10.0).abs() < 0.001);
    assert!((stats.mean - 5.0).abs() < 0.001);
    // std_dev of [0, 10] = sqrt((25+25)/1) = sqrt(50) ~= 7.071
    assert!((stats.std_dev - 7.071).abs() < 0.1);
}

#[test]
fn test_activation_stats_all_inf() {
    let data = vec![f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 3);
    assert_eq!(stats.inf_count, 3);
    assert_eq!(stats.mean, 0.0); // No valid values
    assert_eq!(stats.std_dev, 0.0);
}

#[test]
fn test_activation_stats_negative_only() {
    let data = vec![-10.0, -20.0, -30.0];
    let stats = ActivationStats::from_slice(&data);
    assert!((stats.min - (-30.0)).abs() < 0.001);
    assert!((stats.max - (-10.0)).abs() < 0.001);
    assert!((stats.mean - (-20.0)).abs() < 0.001);
}

// ============================================================================
// AprTransformerConfig tests
// ============================================================================

#[test]
fn test_apr_transformer_config_serde_roundtrip() {
    let config = AprTransformerConfig {
        architecture: "qwen2".to_string(),
        hidden_dim: 1536,
        num_layers: 28,
        num_heads: 12,
        num_kv_heads: 2,
        vocab_size: 151936,
        intermediate_dim: 8960,
        context_length: 32768,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
            eos_token_id: None,
    ..Default::default()
    };

    let json = serde_json::to_string(&config).expect("serialize failed");
    let deserialized: AprTransformerConfig =
        serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(deserialized.architecture, "qwen2");
    assert_eq!(deserialized.hidden_dim, 1536);
    assert_eq!(deserialized.num_layers, 28);
    assert_eq!(deserialized.num_heads, 12);
    assert_eq!(deserialized.num_kv_heads, 2);
    assert_eq!(deserialized.vocab_size, 151936);
    assert_eq!(deserialized.intermediate_dim, 8960);
    assert_eq!(deserialized.context_length, 32768);
    assert!((deserialized.rope_theta - 1_000_000.0).abs() < 1.0);
    assert!((deserialized.eps - 1e-6).abs() < 1e-9);
}

#[test]
fn test_apr_transformer_config_small_model() {
    let config = AprTransformerConfig {
        architecture: "tiny".to_string(),
        hidden_dim: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 8,
        intermediate_dim: 8,
        context_length: 16,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let json = serde_json::to_string(&config).expect("serialize failed");
    assert!(json.contains("\"tiny\""));
    assert!(json.contains("\"hidden_dim\":4"));
}

// ============================================================================
// AprTransformer from_apr_bytes error cases
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    use crate::apr_transformer::AprTransformer;
    let data = vec![0u8; 32]; // Less than 64 bytes minimum
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("too small") || err.contains("64"),
        "Expected size error, got: {}",
        err
    );
}

#[test]
fn test_from_apr_bytes_bad_magic() {
    use crate::apr_transformer::AprTransformer;
    let mut data = vec![0u8; 128];
    // Set first bytes to something that's NOT "APR"
    data[0] = b'X';
    data[1] = b'Y';
    data[2] = b'Z';
    data[3] = b'0';
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("magic") || err.contains("Invalid APR") || err.contains("APR"),
        "Expected magic error, got: {}",
        err
    );
}

include!("apr_02.rs");
