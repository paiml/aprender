
#[test]
fn test_forward_with_cache_single_token() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    let result = transformer.forward_with_cache(&[0], &mut cache);

    assert!(result.is_ok());
    let logits = result.expect("test value should be present");
    assert_eq!(logits.len(), 100);
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_forward_with_cache_sequential_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Process tokens one at a time (autoregressive)
    for (i, &token) in [1u32, 2, 3, 4, 5].iter().enumerate() {
        let result = transformer.forward_with_cache(&[token], &mut cache);
        assert!(result.is_ok());
        assert_eq!(cache.len(), i + 1);
    }
}

#[test]
fn test_forward_with_cache_batch_tokens() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Process multiple tokens at once (prefill)
    let result = transformer.forward_with_cache(&[1, 2, 3, 4], &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 4);
}

#[test]
fn test_forward_with_cache_multi_layer() {
    let transformer = create_minimal_q4_transformer(4);
    let mut cache = transformer.create_kv_cache();

    // Process tokens one at a time (autoregressive style)
    // Multi-token batch processing isn't supported with current layer-major loop
    for &token in &[10u32, 20, 30] {
        let result = transformer.forward_with_cache(&[token], &mut cache);
        assert!(result.is_ok());
    }

    assert_eq!(cache.len(), 3);
}

#[test]
fn test_forward_with_cache_swiglu() {
    let transformer = create_q4_transformer_with_gate(2);
    let mut cache = transformer.create_kv_cache();

    // Process tokens one at a time (autoregressive style)
    for &token in &[1u32, 2, 3] {
        let _ = transformer.forward_with_cache(&[token], &mut cache);
    }

    // Generate one more token
    let result = transformer.forward_with_cache(&[4], &mut cache);

    assert!(result.is_ok());
    assert_eq!(cache.len(), 4);
}

#[test]
fn test_forward_with_cache_vs_no_cache() {
    let transformer = create_minimal_q4_transformer(1);

    // Without cache
    let no_cache_result = transformer.forward(&[1, 2, 3]);

    // With cache (prefill all at once)
    let mut cache = transformer.create_kv_cache();
    let cache_result = transformer.forward_with_cache(&[1, 2, 3], &mut cache);

    assert!(no_cache_result.is_ok());
    assert!(cache_result.is_ok());

    // Both should produce same-size output
    assert_eq!(no_cache_result.expect("test value should be present").len(), cache_result.expect("test value should be present").len());
}

// ============================================================================
// Part 7: Memory and Utility Tests
// ============================================================================

#[test]
fn test_memory_size_basic() {
    let transformer = create_minimal_q4_transformer(1);
    let size = transformer.memory_size();

    // Should include embedding, norm, lm_head, and layer weights
    assert!(size > 0);

    // Rough calculation:
    // token_embedding: 100 * 64 * 4 = 25600 bytes
    // output_norm: 64 * 4 = 256 bytes
    // lm_head: Q4_0 bytes for 64 * 100
    // Plus layer weights
    assert!(size > 25000);
}

#[test]
fn test_memory_size_scales_with_layers() {
    let transformer1 = create_minimal_q4_transformer(1);
    let transformer2 = create_minimal_q4_transformer(4);

    let size1 = transformer1.memory_size();
    let size2 = transformer2.memory_size();

    // More layers = more memory
    assert!(size2 > size1);
}

#[test]
fn test_memory_size_with_gate() {
    let transformer_no_gate = create_minimal_q4_transformer(1);
    let transformer_with_gate = create_q4_transformer_with_gate(1);

    let size_no_gate = transformer_no_gate.memory_size();
    let size_with_gate = transformer_with_gate.memory_size();

    // Gate adds extra FFN weights
    assert!(size_with_gate > size_no_gate);
}

// ============================================================================
// Part 8: Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_forward_large_sequence() {
    let transformer = create_minimal_q4_transformer(1);

    // Process a longer sequence
    let tokens: Vec<u32> = (0..20).collect();
    let result = transformer.forward(&tokens);

    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 100);
}

#[test]
fn test_forward_with_cache_long_generation() {
    let transformer = create_minimal_q4_transformer(1);
    let mut cache = transformer.create_kv_cache();

    // Prefill
    let _ = transformer.forward_with_cache(&[1, 2, 3, 4, 5], &mut cache);

    // Generate 20 more tokens
    for i in 0..20u32 {
        let result = transformer.forward_with_cache(&[i + 10], &mut cache);
        assert!(result.is_ok());
    }

    assert_eq!(cache.len(), 25); // 5 prefill + 20 generation
}

#[test]
fn test_gqa_different_kv_heads() {
    // Test with different num_kv_heads (GQA configuration)
    let transformer = create_gqa_transformer(4, 2); // 4 Q heads, 2 KV heads

    let result = transformer.forward(&[0, 1]);

    assert!(result.is_ok());
}

#[test]
fn test_small_head_dim() {
    // Test with small head dimension
    let hidden_dim = 32;
    let num_heads = 4; // head_dim = 8

    let transformer = create_custom_transformer(hidden_dim, 64, num_heads, num_heads, 50, 1);

    let result = transformer.forward(&[0]);
    assert!(result.is_ok());
}

#[test]
fn test_rope_at_various_positions() {
    let transformer = create_minimal_q4_transformer(1);

    // Test RoPE at different sequence positions
    for i in 0..10 {
        let tokens: Vec<u32> = (0..=i).map(|x| x as u32).collect();
        let mut cache = transformer.create_kv_cache(); // Fresh cache each iteration
        let result = transformer.forward_with_cache(&tokens, &mut cache);
        assert!(result.is_ok());
    }
}

#[test]
fn test_parallel_attention_threshold() {
    // Test with >=4 heads (should use parallel path)
    let transformer_parallel = create_custom_transformer(64, 128, 8, 8, 100, 1);
    let result = transformer_parallel.forward(&[0, 1]);
    assert!(result.is_ok());

    // Test with <4 heads (sequential path)
    let transformer_sequential = create_custom_transformer(64, 128, 2, 2, 100, 1);
    let result2 = transformer_sequential.forward(&[0, 1]);
    assert!(result2.is_ok());
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_config(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
) -> AprTransformerConfig {
    AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
    ..Default::default()
    }
}

fn create_test_layer_q4(
    hidden_dim: usize,
    intermediate_dim: usize,
    with_gate: bool,
) -> QuantizedAprLayerQ4 {
    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: if with_gate {
            Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
        } else {
            None
        },
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}

fn create_minimal_q4_transformer(num_layers: usize) -> QuantizedAprTransformerQ4 {
    create_custom_transformer(64, 128, 4, 4, 100, num_layers)
}

fn create_q4_transformer_with_gate(num_layers: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test_swiglu".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

fn create_gqa_transformer(num_heads: usize, num_kv_heads: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = num_heads * 16; // head_dim = 16
    create_custom_transformer(hidden_dim, hidden_dim * 2, num_heads, num_kv_heads, 100, 1)
}

fn create_custom_transformer(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    num_layers: usize,
) -> QuantizedAprTransformerQ4 {
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + 2 * kv_dim;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_dim),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test_custom".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

// ============================================================================
// Part 9: from_gguf conversion tests (coverage for q4_simd.rs:189-268)
// ============================================================================

#[test]
fn test_from_gguf_basic_fused_qkv() {
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let gguf_model = create_test_model_with_config(&config);
    let q4_model = QuantizedAprTransformerQ4::from_gguf(&gguf_model);

    assert_eq!(q4_model.config.hidden_dim, 64);
    assert_eq!(q4_model.config.intermediate_dim, 128);
    assert_eq!(q4_model.config.num_heads, 4);
    assert_eq!(q4_model.config.num_kv_heads, 4);
    assert_eq!(q4_model.config.vocab_size, 100);
    assert_eq!(q4_model.config.num_layers, 1);
    assert_eq!(q4_model.layers.len(), 1);
    assert_eq!(q4_model.token_embedding.len(), 100 * 64);
    assert_eq!(q4_model.output_norm_weight.len(), 64);
    assert_eq!(q4_model.lm_head_weight.in_dim, 64);
    assert_eq!(q4_model.lm_head_weight.out_dim, 100);
}
