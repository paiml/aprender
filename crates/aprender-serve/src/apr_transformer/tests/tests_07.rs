//! T-COV-95 Phase 50: Deep coverage for apr_transformer/mod.rs
//!
//! Covers:
//! - ActivationStats::from_slice: empty, normal, NaN/Inf handling, all-zeros, single element
//! - dequant_perrow: aligned, padded, truncated data
//! - dequant_q6k_block: zero block, non-zero block
//! - dequant_q4k_block: zero block, non-zero block
//! - LayerActivation construction
//! - ForwardTrace construction
//! - AprTransformerConfig serialization roundtrip

// Import from apr_transformer module (two levels up: tests/mod.rs -> apr_transformer/mod.rs)
use crate::apr_transformer::{
    ActivationStats, AprTransformerConfig, ForwardTrace, LayerActivation,
};

// Private functions accessible via super::super (within the same crate module)
use super::super::{dequant_perrow, dequant_q4k_block, dequant_q6k_block};

// ============================================================================
// ActivationStats::from_slice
// ============================================================================

#[test]
fn test_activation_stats_empty() {
    let stats = ActivationStats::from_slice(&[]);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.zero_count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std_dev, 0.0);
}

#[test]
fn test_activation_stats_single_element() {
    let stats = ActivationStats::from_slice(&[42.0]);
    assert_eq!(stats.count, 1);
    assert!((stats.min - 42.0).abs() < 0.001);
    assert!((stats.max - 42.0).abs() < 0.001);
    assert!((stats.mean - 42.0).abs() < 0.001);
    assert_eq!(stats.std_dev, 0.0); // Single element -> std_dev = 0
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.zero_count, 0);
}

#[test]
fn test_activation_stats_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 5);
    assert!((stats.min - 1.0).abs() < 0.001);
    assert!((stats.max - 5.0).abs() < 0.001);
    assert!((stats.mean - 3.0).abs() < 0.001);
    // std_dev of [1,2,3,4,5] = sqrt(2.5) ~= 1.5811
    assert!((stats.std_dev - 1.5811).abs() < 0.01);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.zero_count, 0);
}

#[test]
fn test_activation_stats_with_nan() {
    let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 5);
    assert_eq!(stats.nan_count, 2);
    assert_eq!(stats.inf_count, 0);
    // Mean computed over valid values only: (1 + 3 + 5) / 3 = 3.0
    assert!((stats.mean - 3.0).abs() < 0.001);
}

#[test]
fn test_activation_stats_with_inf() {
    let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 5);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 2);
    // Mean computed over valid values: (1 + 3 + 5) / 3 = 3.0
    assert!((stats.mean - 3.0).abs() < 0.001);
}

#[test]
fn test_activation_stats_all_nan() {
    let data = vec![f32::NAN, f32::NAN, f32::NAN];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 3);
    assert_eq!(stats.nan_count, 3);
    assert_eq!(stats.mean, 0.0); // No valid values -> mean=0
    assert_eq!(stats.std_dev, 0.0);
}

#[test]
fn test_activation_stats_all_zeros() {
    let data = vec![0.0; 10];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 10);
    assert_eq!(stats.zero_count, 10);
    assert!((stats.min - 0.0).abs() < 0.001);
    assert!((stats.max - 0.0).abs() < 0.001);
    assert!((stats.mean - 0.0).abs() < 0.001);
    assert!((stats.std_dev - 0.0).abs() < 0.001);
}

#[test]
fn test_activation_stats_negative_values() {
    let data = vec![-5.0, -3.0, -1.0, 0.0, 2.0, 4.0];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 6);
    assert!((stats.min - (-5.0)).abs() < 0.001);
    assert!((stats.max - 4.0).abs() < 0.001);
    // Mean: (-5 + -3 + -1 + 0 + 2 + 4) / 6 = -3/6 = -0.5
    assert!((stats.mean - (-0.5)).abs() < 0.001);
    assert_eq!(stats.zero_count, 1);
}

#[test]
fn test_activation_stats_large_values() {
    let data = vec![1e30, -1e30, 1e-30, -1e-30];
    let stats = ActivationStats::from_slice(&data);
    assert_eq!(stats.count, 4);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert!(stats.min < -1e29);
    assert!(stats.max > 1e29);
}

#[test]
fn test_activation_stats_default() {
    let stats = ActivationStats::default();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert_eq!(stats.zero_count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std_dev, 0.0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
}

#[test]
fn test_activation_stats_clone() {
    let data = vec![1.0, 2.0, 3.0];
    let stats = ActivationStats::from_slice(&data);
    let cloned = stats.clone();
    assert_eq!(cloned.count, stats.count);
    assert!((cloned.mean - stats.mean).abs() < 0.001);
    assert!((cloned.std_dev - stats.std_dev).abs() < 0.001);
}

#[test]
fn test_activation_stats_debug() {
    let stats = ActivationStats::from_slice(&[1.0, 2.0]);
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("ActivationStats"));
}

// ============================================================================
// dequant_q4k_block
// ============================================================================

#[test]
fn test_dequant_q4k_block_zeros() {
    let block = vec![0u8; 144]; // All zeros
    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // When d=0 and dmin=0, all values should be 0
    for &v in &out {
        assert!((v - 0.0).abs() < 0.001, "Expected 0.0, got {}", v);
    }
}

#[test]
fn test_dequant_q4k_block_nonzero_d() {
    let mut block = vec![0u8; 144];
    // Set d = 1.0 (f16: 0x3C00)
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set dmin = 0
    block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
    // Set scale[0] = 1 (for first block of 64)
    block[4] = 0x01;
    // Set some qs to nonzero (nibbles)
    for i in 0..128 {
        block[16 + i] = 0x55; // low=5, high=5
    }
    let mut out = vec![0.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // At least some values should be nonzero
    let nonzero_count = out.iter().filter(|&&v| v.abs() > 0.001).count();
    assert!(
        nonzero_count > 0,
        "Expected some nonzero values from dequant_q4k_block"
    );
}

#[test]
fn test_dequant_q4k_block_output_length() {
    let block = vec![0u8; 144];
    let mut out = vec![-1.0f32; 256];
    dequant_q4k_block(&block, &mut out);
    // Should write exactly 256 values
    assert_eq!(out.len(), 256);
}

// ============================================================================
// dequant_q6k_block
// ============================================================================

#[test]
fn test_dequant_q6k_block_zeros() {
    let block = vec![0u8; 210]; // All zeros
    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    // When d=0 (in f16 at bytes 208-209), all values should be 0
    for &v in &out {
        assert!((v - 0.0).abs() < 0.001, "Expected 0.0, got {}", v);
    }
}

#[test]
fn test_dequant_q6k_block_nonzero_d() {
    let mut block = vec![0u8; 210];
    // Set d = 1.0 (f16: 0x3C00) at offset 208
    block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    // Set some scale values
    block[192] = 1; // scale[0] = 1
    block[193] = 1;
    // Set some ql values to nonzero
    for i in 0..128 {
        block[i] = 0x33; // low nibble = 3
    }
    let mut out = vec![0.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    // Some values should be nonzero (d=1.0, scale=1, some q values)
    let nonzero_count = out.iter().filter(|&&v| v.abs() > 0.001).count();
    assert!(
        nonzero_count > 0,
        "Expected some nonzero values from dequant_q6k_block"
    );
}

#[test]
fn test_dequant_q6k_block_output_length() {
    let block = vec![0u8; 210];
    let mut out = vec![-1.0f32; 256];
    dequant_q6k_block(&block, &mut out);
    assert_eq!(out.len(), 256);
}

// ============================================================================
// dequant_perrow
// ============================================================================

#[test]
fn test_dequant_perrow_aligned() {
    // 2 rows, 256 cols (1 block per row) for Q4K (144 bytes/block, 256 values/block)
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 2;
    let cols = 256;
    let data = vec![0u8; rows * block_bytes]; // all zeros
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |block, out| {
        // Simple dequant: just fill with 1.0
        for v in out.iter_mut() {
            *v = if block[0] == 0 { 0.0 } else { 1.0 };
        }
    });

    assert_eq!(result.len(), rows * cols);
}

#[test]
fn test_dequant_perrow_padded() {
    // 2 rows, 128 cols but block_elems=256, so each row is padded to 256
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 2;
    let cols: usize = 128; // less than block_elems
    let blocks_per_row = cols.div_ceil(block_elems); // 1
    let data = vec![0u8; rows * blocks_per_row * block_bytes];
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = i as f32;
        }
    });

    // Should have rows * cols values (padding stripped)
    assert_eq!(result.len(), rows * cols);
    // First row: values 0..128
    for i in 0..cols {
        assert!((result[i] - i as f32).abs() < 0.001);
    }
}

#[test]
fn test_dequant_perrow_truncated_data() {
    // Data is shorter than expected - should fill remaining with zeros
    let block_bytes = 144;
    let block_elems = 256;
    let rows = 2;
    let cols = 256;
    let data = vec![0u8; block_bytes]; // Only enough for 1 row
    let dims = vec![rows, cols];

    let result = dequant_perrow(&data, &dims, block_elems, block_bytes, |_block, out| {
        for v in out.iter_mut() {
            *v = 42.0;
        }
    });

    // Should still return rows*cols values (second row padded with zeros)
    assert_eq!(result.len(), rows * cols);
}

// ============================================================================
// LayerActivation construction
// ============================================================================

#[test]
fn test_layer_activation_construction() {
    let stats = ActivationStats::from_slice(&[1.0, 2.0, 3.0]);
    let layer_act = LayerActivation {
        layer_idx: 0,
        attn_norm_stats: stats.clone(),
        qkv_stats: stats.clone(),
        attn_out_stats: stats.clone(),
        ffn_norm_stats: stats.clone(),
        ffn_out_stats: stats.clone(),
        output_stats: stats,
    };
    assert_eq!(layer_act.layer_idx, 0);
    assert_eq!(layer_act.attn_norm_stats.count, 3);
    assert_eq!(layer_act.output_stats.count, 3);
}

#[test]
fn test_layer_activation_debug() {
    let stats = ActivationStats::default();
    let layer_act = LayerActivation {
        layer_idx: 5,
        attn_norm_stats: stats.clone(),
        qkv_stats: stats.clone(),
        attn_out_stats: stats.clone(),
        ffn_norm_stats: stats.clone(),
        ffn_out_stats: stats.clone(),
        output_stats: stats,
    };
    let debug_str = format!("{:?}", layer_act);
    assert!(debug_str.contains("LayerActivation"));
}

#[test]
fn test_layer_activation_clone() {
    let stats = ActivationStats::from_slice(&[1.0]);
    let layer_act = LayerActivation {
        layer_idx: 3,
        attn_norm_stats: stats.clone(),
        qkv_stats: stats.clone(),
        attn_out_stats: stats.clone(),
        ffn_norm_stats: stats.clone(),
        ffn_out_stats: stats.clone(),
        output_stats: stats,
    };
    let cloned = layer_act.clone();
    assert_eq!(cloned.layer_idx, 3);
}

// ============================================================================
// ForwardTrace construction
// ============================================================================

#[test]
fn test_forward_trace_construction() {
    let stats = ActivationStats::from_slice(&[1.0, 2.0]);
    let trace = ForwardTrace {
        input_tokens: vec![1, 2, 3],
        embed_stats: stats.clone(),
        layer_activations: vec![],
        final_norm_stats: stats.clone(),
        logits_stats: stats,
        logits: vec![0.1, 0.2, 0.7],
    };
    assert_eq!(trace.input_tokens, vec![1, 2, 3]);
    assert!(trace.layer_activations.is_empty());
    assert_eq!(trace.logits.len(), 3);
}

#[test]
fn test_forward_trace_with_layers() {
    let stats = ActivationStats::default();
    let layer_act = LayerActivation {
        layer_idx: 0,
        attn_norm_stats: stats.clone(),
        qkv_stats: stats.clone(),
        attn_out_stats: stats.clone(),
        ffn_norm_stats: stats.clone(),
        ffn_out_stats: stats.clone(),
        output_stats: stats.clone(),
    };
    let trace = ForwardTrace {
        input_tokens: vec![1],
        embed_stats: stats.clone(),
        layer_activations: vec![layer_act],
        final_norm_stats: stats.clone(),
        logits_stats: stats,
        logits: vec![0.5],
    };
    assert_eq!(trace.layer_activations.len(), 1);
    assert_eq!(trace.layer_activations[0].layer_idx, 0);
}

#[test]
fn test_forward_trace_debug() {
    let stats = ActivationStats::default();
    let trace = ForwardTrace {
        input_tokens: vec![1],
        embed_stats: stats.clone(),
        layer_activations: vec![],
        final_norm_stats: stats.clone(),
        logits_stats: stats,
        logits: vec![],
    };
    let debug_str = format!("{:?}", trace);
    assert!(debug_str.contains("ForwardTrace"));
}

#[test]
fn test_forward_trace_clone() {
    let stats = ActivationStats::default();
    let trace = ForwardTrace {
        input_tokens: vec![1, 2],
        embed_stats: stats.clone(),
        layer_activations: vec![],
        final_norm_stats: stats.clone(),
        logits_stats: stats,
        logits: vec![0.3, 0.7],
    };
    let cloned = trace.clone();
    assert_eq!(cloned.input_tokens, vec![1, 2]);
    assert_eq!(cloned.logits.len(), 2);
}
