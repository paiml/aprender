//! T-COV-95 Phase 54: Q4K Fused Kernel Dispatch & Force-F32 Paths
//!
//! Covers uncovered Q4K/Q6K fused kernel dispatch in forward_with_cache and
//! forward (batch), plus APR_FORCE_F32 paths and from_apr_bytes error edges.

use crate::apr_transformer::config::Q4KLayerWeights;
use crate::apr_transformer::{
    AprKVCache, AprTransformer, AprTransformerConfig, AprTransformerLayer,
};

// Q4K: each row = ceil(in_dim/256) blocks × 144 bytes/block, total = out_dim rows
fn q4k_bytes(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let blocks_per_row = (in_dim + 255) / 256;
    vec![0u8; out_dim * blocks_per_row * 144]
}

// Q6K: each row = ceil(in_dim/256) blocks × 210 bytes/block, total = out_dim rows
fn q6k_bytes(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let blocks_per_row = (in_dim + 255) / 256;
    vec![0u8; out_dim * blocks_per_row * 210]
}

/// Build a minimal AprTransformer with populated q4k_layers for fused kernel paths.
/// Uses SwiGLU (has gate weight, ffn_norm) with separate Q/K/V Q4K weights.
fn build_apr_with_q4k_fused(
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
) -> AprTransformer {
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    // F32 layer weights (fallback path, also used for norm/bias)
    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        // QKV as flat F32 (used when Q4K separate weights not found)
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
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

    // Q4K layer with separate weights for fused kernel dispatch
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight_q6k: None,
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_up_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_down_weight: Some(q4k_bytes(hidden, intermediate)),
        ffn_down_weight_q6k: None,
        ffn_up_weight_q6k: None,
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// Build AprTransformer with Q6K variant weights (V=Q6K, up=Q6K, down=Q6K).
fn build_apr_with_q6k_variants(hidden: usize, intermediate: usize, vocab: usize) -> AprTransformer {
    let heads = 4;
    let kv_heads = 4;
    let head_dim = hidden / heads;
    let kv_size = kv_heads * head_dim;

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden],
        attn_norm_bias: None,
        qkv_weight: vec![0.001; (hidden + 2 * kv_size) * hidden],
        qkv_bias: None,
        attn_output_weight: vec![0.001; hidden * hidden],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.001; intermediate * hidden]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.001; intermediate * hidden],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.001; hidden * intermediate],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden]),
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

    // Q4K layer with Q6K fallback variants
    let q4k = Q4KLayerWeights {
        qkv_weight: None,
        attn_q_weight: Some(q4k_bytes(hidden, hidden)),
        attn_k_weight: Some(q4k_bytes(kv_size, hidden)),
        attn_v_weight: None, // No Q4K V — forces Q6K fallback
        attn_v_weight_q6k: Some(q6k_bytes(kv_size, hidden)),
        attn_output_weight: Some(q4k_bytes(hidden, hidden)),
        ffn_gate_weight: Some(q4k_bytes(intermediate, hidden)),
        ffn_up_weight: None,   // No Q4K up — forces Q6K fallback
        ffn_down_weight: None, // No Q4K down — forces Q6K fallback
        ffn_down_weight_q6k: Some(q6k_bytes(hidden, intermediate)),
        ffn_up_weight_q6k: Some(q6k_bytes(intermediate, hidden)),
    };

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: hidden,
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: vocab,
            intermediate_dim: intermediate,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.01; vocab * hidden],
        layers: vec![layer],
        output_norm_weight: vec![1.0; hidden],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab * hidden],
        lm_head_bias: None,
        q4k_layers: Some(vec![q4k]),
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

// ============================================================================
// forward_with_cache: Q4K fused kernel paths
// ============================================================================

#[test]
fn test_fwc_q4k_fused_first_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K fused first token: {}",
        result.unwrap_err()
    );
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_fwc_q4k_fused_multi_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q4K fused second token: {}",
        result.unwrap_err()
    );
    assert_eq!(result.expect("test value should be present").len(), 16);
}

#[test]
fn test_fwc_q4k_fused_three_tokens() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(0, &mut cache, 0).expect("test value should be present");
    let _ = apr.forward_with_cache(1, &mut cache, 1).expect("test value should be present");
    let result = apr.forward_with_cache(2, &mut cache, 2);
    assert!(
        result.is_ok(),
        "Q4K fused third token: {}",
        result.unwrap_err()
    );
    assert_eq!(result.expect("test value should be present").len(), 16);
}

#[test]
fn test_fwc_q4k_fused_gqa() {
    // GQA: num_kv_heads < num_heads
    let apr = build_apr_with_q4k_fused(32, 64, 4, 2, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Q4K fused GQA first token: {}",
        result.unwrap_err()
    );

    let result2 = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result2.is_ok(),
        "Q4K fused GQA second token: {}",
        result2.unwrap_err()
    );
}

// ============================================================================
// forward_with_cache: Q6K variant paths (V=Q6K, up=Q6K, down=Q6K)
// ============================================================================

#[test]
fn test_fwc_q6k_v_fallback() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K V fallback: {}", result.unwrap_err());
    assert_eq!(result.expect("test value should be present").len(), 16);
}

#[test]
fn test_fwc_q6k_variants_multi_token() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let mut cache = AprKVCache::new(&apr.config);

    let _ = apr.forward_with_cache(1, &mut cache, 0).expect("test value should be present");
    let result = apr.forward_with_cache(2, &mut cache, 1);
    assert!(
        result.is_ok(),
        "Q6K variants multi: {}",
        result.unwrap_err()
    );
}

// ============================================================================
// forward_with_cache: lm_head Q4K and Q6K paths
// ============================================================================

#[test]
fn test_fwc_lm_head_q4k() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Add Q4K lm_head: [vocab_size, hidden_dim]
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q4K lm_head: {}", result.unwrap_err());
    assert_eq!(result.expect("test value should be present").len(), 16);
}

#[test]
fn test_fwc_lm_head_q6k() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    // Q6K lm_head (no Q4K lm_head)
    apr.lm_head_weight_q6k = Some(q6k_bytes(16, 32));
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "Q6K lm_head: {}", result.unwrap_err());
    assert_eq!(result.expect("test value should be present").len(), 16);
}

// ============================================================================
// forward (batch): Q4K fused kernel paths
// ============================================================================

#[test]
fn test_forward_batch_q4k_fused_single_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K batch single: {}", result.unwrap_err());
    assert_eq!(result.expect("test value should be present").len(), 16);
}

#[test]
fn test_forward_batch_q4k_fused_multi_token() {
    let apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    let result = apr.forward(&[1, 2, 3]);
    assert!(result.is_ok(), "Q4K batch multi: {}", result.unwrap_err());
    let logits = result.expect("test value should be present");
    // forward() returns vocab_size logits (last token position)
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_batch_q6k_variants() {
    let apr = build_apr_with_q6k_variants(32, 64, 16);
    let result = apr.forward(&[1, 2]);
    assert!(result.is_ok(), "Q6K batch: {}", result.unwrap_err());
}

#[test]
fn test_forward_batch_q4k_lm_head() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.lm_head_weight_q4k = Some(q4k_bytes(16, 32));
    let result = apr.forward(&[1]);
    assert!(result.is_ok(), "Q4K lm_head batch: {}", result.unwrap_err());
}

// ============================================================================
// forward_with_cache: no q4k_layers (F32 fallback for all)
// ============================================================================

#[test]
fn test_fwc_f32_fallback_explicit() {
    let mut apr = build_apr_with_q4k_fused(32, 64, 4, 4, 16);
    apr.q4k_layers = None; // Force F32 fallback
    let mut cache = AprKVCache::new(&apr.config);

    let result = apr.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok(), "F32 fallback: {}", result.unwrap_err());
}

// ============================================================================
// from_apr_bytes: Error path — no lm_head AND no matching embed for tying
// ============================================================================

// APR v2 constants
const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"
const HEADER_SIZE: usize = 64;

struct TensorDef {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    data: Vec<u8>,
}

/// Build APR v2 binary data using BINARY tensor index format (matching from_apr_bytes parser).
fn build_apr_v2(metadata_json: &str, tensors: &[TensorDef]) -> Vec<u8> {
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Build tensor index entries in BINARY format
    let mut index_bytes = Vec::new();
    let mut current_offset = 0u64;

    for t in tensors {
        // name_len (2) + name + dtype (1) + ndim (1) + dims (8 each) + offset (8) + size (8)
        index_bytes.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(t.name.as_bytes());
        index_bytes.push(t.dtype);
        index_bytes.push(t.dims.len() as u8);
        for &dim in &t.dims {
            index_bytes.extend_from_slice(&dim.to_le_bytes());
        }
        index_bytes.extend_from_slice(&current_offset.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
        current_offset += t.data.len() as u64;
    }

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + index_bytes.len() as u64;
    let total_data_size: usize = tensors.iter().map(|t| t.data.len()).sum();
    let total_size = data_offset as usize + total_data_size;

    let mut buf = vec![0u8; total_size];

    // Write header
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = 2; // version major
    buf[5] = 0;
    buf[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
    buf[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    buf[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    buf[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    buf[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    buf[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    buf[idx_start..idx_start + index_bytes.len()].copy_from_slice(&index_bytes);

    // Write tensor data
    let mut pos = data_offset as usize;
    for t in tensors {
        buf[pos..pos + t.data.len()].copy_from_slice(&t.data);
        pos += t.data.len();
    }

    buf
}

fn make_f32_data(n: usize, val: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

include!("apr_04.rs");
include!("fwc_q4k.rs");
include!("fwc_q6k.rs");
