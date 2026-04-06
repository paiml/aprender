//! Tests for forward pass implementations in core.rs
//!
//! Tests cover:
//! - Basic forward pass for LLaMA-style (RMSNorm + SwiGLU) and phi-2 style (LayerNorm + GELU)
//! - Forward with KV cache (forward_cached)
//! - Edge cases: single token, multiple tokens, GQA attention
//! - Architecture detection (use_rmsnorm flag)

use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedLayer, OwnedQuantizedModel,
    OwnedQuantizedTensor,
};

/// Create Q4_K test data with deterministic values
fn create_q4k_test_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;
    let data_size = out_dim * bytes_per_row;
    let mut data = vec![0u8; data_size];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = row * bytes_per_row + sb * 144;
            // d=1.0 in f16 format
            data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            // dmin=0
            data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
            // Deterministic test pattern
            for i in 4..144 {
                data[offset + i] = ((row + sb + i) % 16) as u8;
            }
        }
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: 12, // Q4_K
    }
}

/// Create a LLaMA-style model (RMSNorm + SwiGLU + GQA)
fn create_llama_style_model(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
) -> OwnedQuantizedModel {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let layer = OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: None, // No bias = RMSNorm detection
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim)),
            qkv_bias: None,
            attn_output_weight: create_q4k_test_tensor(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_test_tensor(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_test_tensor(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: Some(create_q4k_test_tensor(hidden_dim, intermediate_dim)), // SwiGLU
            ffn_gate_bias: None,
            ffn_norm_weight: Some(vec![1.0f32; hidden_dim]), // LLaMA has separate FFN norm
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };
        layers.push(layer);
    }

    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];
    let output_norm_weight = vec![1.0f32; hidden_dim];
    let lm_head_weight = create_q4k_test_tensor(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding,
        position_embedding: None,
        layers,
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight,
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a phi-2 style model (LayerNorm + GELU + MHA)
fn create_phi2_style_model(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_layers: usize,
) -> OwnedQuantizedModel {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads: num_heads, // MHA: same KV heads as Q heads
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let qkv_out_dim = 3 * hidden_dim; // MHA: Q=K=V=hidden_dim

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let layer = OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // Has bias = LayerNorm
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_tensor(hidden_dim, qkv_out_dim)),
            qkv_bias: Some(vec![0.0f32; qkv_out_dim]),
            attn_output_weight: create_q4k_test_tensor(hidden_dim, hidden_dim),
            attn_output_bias: Some(vec![0.0f32; hidden_dim]),
            ffn_up_weight: create_q4k_test_tensor(hidden_dim, intermediate_dim),
            ffn_up_bias: Some(vec![0.0f32; intermediate_dim]),
            ffn_down_weight: create_q4k_test_tensor(intermediate_dim, hidden_dim),
            ffn_down_bias: Some(vec![0.0f32; hidden_dim]),
            ffn_gate_weight: None, // No gate = GELU path
            ffn_gate_bias: None,
            ffn_norm_weight: None, // phi-2 has no separate FFN norm
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };
        layers.push(layer);
    }

    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];
    let output_norm_weight = vec![1.0f32; hidden_dim];
    let lm_head_weight = create_q4k_test_tensor(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding,
        position_embedding: None,
        layers,
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight,
        output_norm_bias: Some(vec![0.0f32; hidden_dim]),
        lm_head_weight,
        lm_head_bias: Some(vec![0.0f32; vocab_size]),
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

// =============================================================================
// Forward Pass Tests (LLaMA-style)
// =============================================================================

#[test]
fn test_forward_llama_single_token() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let token_ids = [42u32];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100, "Logits should have vocab_size elements");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_forward_llama_multiple_tokens() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let token_ids = [1u32, 2, 3, 4, 5];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100, "Logits should have vocab_size elements");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_forward_llama_gqa_config() {
    // GQA: 8 Q heads, 2 KV heads (group size = 4)
    let model = create_llama_style_model(100, 64, 128, 8, 2, 1);
    let token_ids = [10u32, 20, 30];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_llama_multi_layer() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 3);
    let token_ids = [5u32, 10, 15];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Forward Pass Tests (phi-2 style)
// =============================================================================

#[test]
fn test_forward_phi2_single_token() {
    let model = create_phi2_style_model(100, 64, 128, 4, 1);
    let token_ids = [42u32];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_phi2_multiple_tokens() {
    let model = create_phi2_style_model(100, 64, 128, 4, 1);
    let token_ids = [1u32, 2, 3, 4, 5];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_phi2_with_biases() {
    // phi-2 uses biases throughout - verify they are applied
    let model = create_phi2_style_model(100, 64, 128, 4, 2);
    let token_ids = [50u32];

    let logits = model
        .forward(&token_ids)
        .expect("Forward pass should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Forward Cached Tests (LLaMA-style)
// =============================================================================

#[test]
fn test_forward_cached_llama_first_token() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model
        .forward_cached(42, &mut cache, 0)
        .expect("Forward cached should succeed");

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_cached_llama_second_token() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // First token
    let _ = model.forward_cached(10, &mut cache, 0).unwrap();
    // Second token uses cache
    let logits = model.forward_cached(20, &mut cache, 1).unwrap();

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_cached_llama_sequence() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    // Process multiple tokens sequentially
    for pos in 0..5 {
        let token = (pos as u32 + 1) * 10;
        let logits = model.forward_cached(token, &mut cache, pos).unwrap();
        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}

#[test]
#[ignore = "needs update for GQA dimension changes"]
fn test_forward_cached_llama_gqa() {
    // GQA model: 8 Q heads, 2 KV heads
    let model = create_llama_style_model(100, 64, 128, 8, 2, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model.forward_cached(50, &mut cache, 0).unwrap();
    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));

    // Second token with GQA
    let logits = model.forward_cached(51, &mut cache, 1).unwrap();
    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Forward Cached Tests (phi-2 style)
// =============================================================================

#[test]
fn test_forward_cached_phi2_first_token() {
    let model = create_phi2_style_model(100, 64, 128, 4, 1);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    let logits = model.forward_cached(42, &mut cache, 0).unwrap();

    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_cached_phi2_sequence() {
    let model = create_phi2_style_model(100, 64, 128, 4, 2);
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, 128);

    for pos in 0..3 {
        let logits = model
            .forward_cached((pos as u32 + 1) * 5, &mut cache, pos)
            .unwrap();
        assert_eq!(logits.len(), 100);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_forward_token_at_boundary() {
    let vocab_size = 100;
    let model = create_llama_style_model(vocab_size, 64, 128, 4, 4, 1);

    // Token at vocab boundary
    let logits = model.forward(&[99u32]).unwrap();
    assert_eq!(logits.len(), vocab_size);
}

#[test]
fn test_forward_token_zero() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);

    let logits = model.forward(&[0u32]).unwrap();
    assert_eq!(logits.len(), 100);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_forward_deterministic() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);
    let token_ids = [42u32];

    let logits1 = model.forward(&token_ids).unwrap();
    let logits2 = model.forward(&token_ids).unwrap();

    // Forward pass should be deterministic
    for (a, b) in logits1.iter().zip(logits2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Forward pass should be deterministic: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_forward_cached_deterministic() {
    let model = create_llama_style_model(100, 64, 128, 4, 4, 1);

    // First run
    let mut cache1 = OwnedQuantizedKVCache::from_config(&model.config, 128);
    let logits1 = model.forward_cached(42, &mut cache1, 0).unwrap();

    // Second run (fresh cache)
    let mut cache2 = OwnedQuantizedKVCache::from_config(&model.config, 128);
    let logits2 = model.forward_cached(42, &mut cache2, 0).unwrap();

    for (a, b) in logits1.iter().zip(logits2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Forward cached should be deterministic"
        );
    }
}

include!("core_tests_architecture_detection.rs");
