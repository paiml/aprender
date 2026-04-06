//! Tests for AprTransformer::forward() with Q4K fused kernel paths (PARITY-027)
//!
//! Coverage target: src/apr_transformer/mod.rs lines 1428-1654 — Q4K branches
//! These branches are only reached when `q4k_layers` is populated.

use crate::apr_transformer::{
    AprTransformer, AprTransformerConfig, AprTransformerLayer, Q4KLayerWeights,
};

/// Build a valid Q4K super-block (144 bytes) covering 256 values.
/// Layout: [d:f16(2), dmin:f16(2), scales:12, quants:128]
fn build_q4k_block(d: f32, dmin: f32, nibble_val: u8) -> [u8; 144] {
    let mut block = [0u8; 144];
    // d as f16
    let d_bits = half::f16::from_f32(d).to_bits();
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    // dmin as f16
    let dmin_bits = half::f16::from_f32(dmin).to_bits();
    block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    // scales: set scale=1, min=0 in packed 6-bit format
    for i in 0..12 {
        block[4 + i] = 0x01;
    }
    // quants: 128 bytes, each byte has lo and hi nibble
    let packed = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..128 {
        block[16 + i] = packed;
    }
    block
}

/// Build Q4K weight data for a matrix [out_dim, in_dim].
/// Each row gets ceil(in_dim/256) super-blocks.
fn build_q4k_weight(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let block = build_q4k_block(0.5, 0.0, 3);
    let mut data = Vec::with_capacity(out_dim * bytes_per_row);
    for _ in 0..out_dim {
        for _ in 0..super_blocks_per_row {
            data.extend_from_slice(&block);
        }
    }
    data
}

/// Create a pygmy SwiGLU model WITH Q4K layers for fused kernel testing.
/// Same dimensions as make_pygmy_model (hidden=8, intermediate=16) but with
/// q4k_layers populated to exercise the Q4K forward branches.
fn make_pygmy_model_with_q4k_swiglu() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();
    let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();
    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.01)
        .collect();

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight,
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight),
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let lm_head_weight: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();

    // Build Q4K layers for fused kernel paths
    let q4k_layer = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None,
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(build_q4k_weight(hidden_dim, hidden_dim)),
        ffn_gate_weight: Some(build_q4k_weight(intermediate_dim, hidden_dim)),
        ffn_up_weight: Some(build_q4k_weight(intermediate_dim, hidden_dim)),
        ffn_down_weight: Some(build_q4k_weight(hidden_dim, intermediate_dim)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    AprTransformer {
        config,
        token_embedding,

        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k_layer]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// Create a pygmy GELU model WITH Q4K layers for standard MLP path.
fn make_pygmy_model_with_q4k_gelu() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 16;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok + d) as f32) * 0.01;
        }
    }

    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| if i % (hidden_dim + 1) == 0 { 0.1 } else { 0.01 })
        .collect();
    // No gate weight = GELU path
    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.01)
        .collect();

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: Some(vec![0.0; hidden_dim]),
        qkv_weight,
        qkv_bias: Some(vec![0.01; qkv_out_dim]),
        attn_output_weight,
        attn_output_bias: Some(vec![0.01; hidden_dim]),
        ffn_gate_weight: None, // No gate = GELU path
        ffn_gate_bias: None,
        ffn_up_weight,
        ffn_up_bias: Some(vec![0.01; intermediate_dim]),
        ffn_down_weight,
        ffn_down_bias: Some(vec![0.01; hidden_dim]),
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let lm_head_weight: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();

    // Q4K layers for GELU standard MLP path
    let q4k_layer = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: None,
        attn_k_weight: None,
        attn_v_weight: None,
        attn_v_weight_q6k: None,
        attn_output_weight: Some(build_q4k_weight(hidden_dim, hidden_dim)),
        ffn_gate_weight: None, // No gate for GELU
        ffn_up_weight: Some(build_q4k_weight(intermediate_dim, hidden_dim)),
        ffn_down_weight: Some(build_q4k_weight(hidden_dim, intermediate_dim)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    AprTransformer {
        config,
        token_embedding,

        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: Some(vec![0.0; hidden_dim]),
        lm_head_weight,
        lm_head_bias: Some(vec![0.01; vocab_size]),
        q4k_layers: Some(vec![q4k_layer]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// SwiGLU path with Q4K fused kernels
// ============================================================================

#[test]
fn test_forward_q4k_swiglu_produces_logits() {
    let model = make_pygmy_model_with_q4k_swiglu();
    let logits = model.forward(&[1]).expect("Q4K forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_q4k_swiglu_multi_token() {
    let model = make_pygmy_model_with_q4k_swiglu();
    let logits = model
        .forward(&[0, 1, 2])
        .expect("Q4K multi-token forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_q4k_swiglu_differs_from_f32() {
    // Q4K path should produce different results from F32 path
    // because the quantized weights are different random data
    let q4k_model = make_pygmy_model_with_q4k_swiglu();
    let mut f32_model = make_pygmy_model_with_q4k_swiglu();
    f32_model.q4k_layers = None; // Force F32 path

    let q4k_logits = q4k_model.forward(&[5]).expect("Q4K forward");
    let f32_logits = f32_model.forward(&[5]).expect("F32 forward");

    // They should differ (different weight data paths)
    let max_diff: f32 = q4k_logits
        .iter()
        .zip(f32_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // At least some difference expected
    assert!(
        max_diff > 1e-6,
        "Q4K and F32 should produce different logits"
    );
}

#[test]
fn test_forward_q4k_swiglu_empty_tokens_error() {
    let model = make_pygmy_model_with_q4k_swiglu();
    let result = model.forward(&[]);
    assert!(result.is_err());
}

// ============================================================================
// GELU standard MLP path with Q4K fused kernels
// ============================================================================

#[test]
fn test_forward_q4k_gelu_produces_logits() {
    let model = make_pygmy_model_with_q4k_gelu();
    let logits = model
        .forward(&[3])
        .expect("Q4K GELU forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_q4k_gelu_multi_token() {
    let model = make_pygmy_model_with_q4k_gelu();
    let logits = model
        .forward(&[0, 5, 10])
        .expect("Q4K GELU multi-token forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_q4k_gelu_with_biases() {
    // This model has all biases (qkv, attn_output, ffn_up, ffn_down, output_norm, lm_head)
    let model = make_pygmy_model_with_q4k_gelu();
    let logits = model
        .forward(&[7])
        .expect("Q4K GELU with biases should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Q4K forward with partial Q4K layers (some weights Q4K, others F32)
// ============================================================================

#[test]
fn test_forward_q4k_partial_only_attn_output() {
    // Only attn_output uses Q4K, FFN falls back to F32
    let mut model = make_pygmy_model_with_q4k_swiglu();
    if let Some(ref mut layers) = model.q4k_layers {
        layers[0].ffn_gate_weight = None;
        layers[0].ffn_up_weight = None;
        layers[0].ffn_down_weight = None;
    }
    let logits = model
        .forward(&[2])
        .expect("partial Q4K forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_q4k_partial_only_ffn() {
    // Only FFN uses Q4K, attn_output falls back to F32
    let mut model = make_pygmy_model_with_q4k_swiglu();
    if let Some(ref mut layers) = model.q4k_layers {
        layers[0].attn_output_weight = None;
    }
    let logits = model
        .forward(&[4])
        .expect("partial Q4K (FFN only) forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Q6K fallback path for ffn_down_weight (SwiGLU)
// ============================================================================

/// Build Q6K weight data for a matrix [out_dim, in_dim].
/// Q6K super-block: 210 bytes per 256 values.
fn build_q6k_weight(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 210;
    // Build a minimal valid Q6K super-block (210 bytes)
    // Layout: [ql:128, qh:64, scales:16, d:f16(2)] = 210
    let mut block = [0u8; 210];
    // d (scale) as f16 at offset 208
    let d_bits = half::f16::from_f32(0.5).to_bits();
    block[208..210].copy_from_slice(&d_bits.to_le_bytes());
    // scales: set to 1
    for i in 0..16 {
        block[192 + i] = 1;
    }
    // ql and qh: zero (all quantized values = 0)

    let mut data = Vec::with_capacity(out_dim * bytes_per_row);
    for _ in 0..out_dim {
        for _ in 0..super_blocks_per_row {
            data.extend_from_slice(&block);
        }
    }
    data
}

#[test]
fn test_forward_q6k_ffn_down_swiglu() {
    // SwiGLU model with Q6K for ffn_down instead of Q4K
    // This exercises the ffn_down_weight_q6k fallback path
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let mut model = make_pygmy_model_with_q4k_swiglu();
    if let Some(ref mut layers) = model.q4k_layers {
        layers[0].ffn_down_weight = None; // Remove Q4K down
        layers[0].ffn_down_weight_q6k = Some(build_q6k_weight(hidden_dim, intermediate_dim));
        // Add Q6K down
    }
    let logits = model
        .forward(&[6])
        .expect("Q6K ffn_down forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// AprTransformer::forward_with_cache with Q4K (PMAT-103 cached path)
// ============================================================================

#[test]
fn test_forward_with_cache_q4k_swiglu() {
    use crate::apr_transformer::AprKVCache;

    let model = make_pygmy_model_with_q4k_swiglu();
    let max_seq = 32;
    let mut cache = AprKVCache::new(&model.config);

    // Process 3 tokens sequentially using cached path
    for (pos, &tok) in [1u32, 3, 7].iter().enumerate() {
        let logits = model
            .forward_with_cache(tok, &mut cache, pos)
            .expect("Q4K cached forward should succeed");
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_forward_with_cache_q4k_separate_qkv() {
    use crate::apr_transformer::AprKVCache;

    let hidden_dim = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / 2; // num_heads=2
    let kv_size = num_kv_heads * head_dim; // =8
    let intermediate_dim = 16;

    let mut model = make_pygmy_model_with_q4k_swiglu();
    // Add separate Q, K, V weights
    if let Some(ref mut layers) = model.q4k_layers {
        layers[0].attn_q_weight = Some(build_q4k_weight(hidden_dim, hidden_dim));
        layers[0].attn_k_weight = Some(build_q4k_weight(kv_size, hidden_dim));
        layers[0].attn_v_weight = Some(build_q4k_weight(kv_size, hidden_dim));
        // Also keep FFN Q4K weights
        layers[0].attn_output_weight = Some(build_q4k_weight(hidden_dim, hidden_dim));
        layers[0].ffn_gate_weight = Some(build_q4k_weight(intermediate_dim, hidden_dim));
        layers[0].ffn_up_weight = Some(build_q4k_weight(intermediate_dim, hidden_dim));
        layers[0].ffn_down_weight = Some(build_q4k_weight(hidden_dim, intermediate_dim));
    }

    let mut cache = AprKVCache::new(&model.config);
    for (pos, &tok) in [2u32, 5, 9].iter().enumerate() {
        let logits = model
            .forward_with_cache(tok, &mut cache, pos)
            .expect("Q4K separate QKV cached forward should succeed");
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_forward_with_cache_q6k_v_weight() {
    use crate::apr_transformer::AprKVCache;

    let hidden_dim = 8;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / 2;
    let kv_size = num_kv_heads * head_dim;

    let mut model = make_pygmy_model_with_q4k_swiglu();
    if let Some(ref mut layers) = model.q4k_layers {
        // Use Q4K for Q and K, but Q6K for V (exercises attn_v_weight_q6k path)
        layers[0].attn_q_weight = Some(build_q4k_weight(hidden_dim, hidden_dim));
        layers[0].attn_k_weight = Some(build_q4k_weight(kv_size, hidden_dim));
        layers[0].attn_v_weight = None; // No Q4K V
        layers[0].attn_v_weight_q6k = Some(build_q6k_weight(kv_size, hidden_dim));
        // Q6K V
    }

    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(3, &mut cache, 0)
        .expect("Q6K V cached forward should succeed");
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}
