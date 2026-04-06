
// ============================================================================
// LayerNorms Tests
// ============================================================================

/// Test LayerNorms struct basic creation
#[test]
fn test_layer_norms_create() {
    let norms = LayerNorms {
        attn_norm: vec![1.0, 2.0, 3.0],
        ffn_norm: vec![4.0, 5.0, 6.0],
    };

    assert_eq!(norms.attn_norm.len(), 3);
    assert_eq!(norms.ffn_norm.len(), 3);
}

/// Test LayerNorms Clone
#[test]
fn test_layer_norms_clone_deep() {
    let norms = LayerNorms {
        attn_norm: vec![1.0, 2.0],
        ffn_norm: vec![3.0, 4.0],
    };

    let cloned = norms.clone();

    assert_eq!(cloned.attn_norm, norms.attn_norm);
    assert_eq!(cloned.ffn_norm, norms.ffn_norm);

    // Verify it's a deep clone
    let mut modified = cloned;
    modified.attn_norm[0] = 99.0;
    assert!((norms.attn_norm[0] - 1.0).abs() < 1e-6);
}

/// Test LayerNorms Debug
#[test]
fn test_layer_norms_debug_format() {
    let norms = LayerNorms {
        attn_norm: vec![1.0],
        ffn_norm: vec![2.0],
    };

    let debug = format!("{norms:?}");
    assert!(debug.contains("LayerNorms"));
    assert!(debug.contains("attn_norm"));
    assert!(debug.contains("ffn_norm"));
}

// ============================================================================
// GpuModelQ4 Tests
// ============================================================================

/// Test GpuModelQ4 Clone
#[test]
fn test_gpu_model_q4_clone_deep() {
    let model = create_tiny_model();
    let cloned = model.clone();

    assert_eq!(cloned.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(cloned.num_layers, model.num_layers);
    assert_eq!(cloned.has_gate, model.has_gate);
    assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
}

/// Test GpuModelQ4 Debug
#[test]
fn test_gpu_model_q4_debug_format() {
    let model = create_tiny_model();
    let debug = format!("{model:?}");

    assert!(debug.contains("GpuModelQ4"));
    assert!(debug.contains("config"));
    assert!(debug.contains("num_layers"));
}

// ============================================================================
// QuantizedAprTensorQ4 Tests
// ============================================================================

/// Test QuantizedAprTensorQ4::new
#[test]
fn test_quantized_tensor_new() {
    let data = vec![1u8, 2, 3, 4];
    let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 64);

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);
}

/// Test QuantizedAprTensorQ4::zeros
#[test]
fn test_quantized_tensor_zeros() {
    let tensor = QuantizedAprTensorQ4::zeros(32, 64);

    // 32 * 64 = 2048 elements
    // 2048 / 32 = 64 blocks
    // 64 * 18 = 1152 bytes
    assert_eq!(tensor.data.len(), 1152);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);

    // All bytes should be 0
    for &b in &tensor.data {
        assert_eq!(b, 0);
    }
}

/// Test QuantizedAprTensorQ4::expected_bytes
#[test]
fn test_quantized_tensor_expected_bytes() {
    // 32 elements = 1 block = 18 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);

    // 64 elements = 2 blocks = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);

    // 33 elements = 2 blocks (ceil) = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);

    // 0 elements = 0 blocks = 0 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_tiny_model() -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 10,
            intermediate_dim: 8,
            context_length: 16,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; 10 * 4],
        output_norm_weight: vec![1.0; 4],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; 4],
            ffn_norm: vec![1.0; 4],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_model_for_rope(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size: 10,
            intermediate_dim: hidden_dim * 2,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; 10 * hidden_dim],
        output_norm_weight: vec![1.0; hidden_dim],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_model_for_attention(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> GpuModelQ4 {
    GpuModelQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size: 10,
            intermediate_dim: hidden_dim * 2,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; 10 * hidden_dim],
        output_norm_weight: vec![1.0; hidden_dim],
        layer_norms: vec![LayerNorms {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
        }],
        num_layers: 1,
        has_gate: false,
    }
}

fn create_test_apr_model(with_gate: bool, with_ffn_norm: bool) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let intermediate_dim = 128;
    let vocab_size = 100;

    let layer = QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: if with_gate {
            Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
        } else {
            None
        },
        ffn_norm_weight: if with_ffn_norm {
            Some(vec![1.0; hidden_dim])
        } else {
            None
        },
    };

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 2,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers: vec![layer.clone(), layer],
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

// Make the activation functions available for testing via super import
#[cfg(test)]
mod tests {
    // Re-export activation functions using crate path
}
