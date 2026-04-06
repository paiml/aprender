//! EXTREME TDD: GGUFTransformer and Forward Pass Coverage Tests
//!
//! Part 1 coverage tests for:
//! - GGUFTransformer struct methods
//! - Forward pass functions
//! - Layer processing functions
//! - Helper functions (embed, layer_norm, matmul, etc.)

use realizar::gguf::{
    ArchConstraints, GGUFConfig, GGUFModel, GGUFTransformer, GGUFValue, InferenceScratchBuffer,
    OwnedQuantizedKVCache, QKVWeights, QuantizedGenerateConfig, QuantizedTensorRef, GGUF_ALIGNMENT,
    GGUF_MAGIC, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// ============================================================================
// HELPER FUNCTIONS FOR BUILDING TEST GGUF DATA
// ============================================================================

/// Add string metadata to GGUF data
fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value.as_bytes());
}

/// Add u32 metadata to GGUF data
fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add tensor info to GGUF data
fn add_tensor_info(data: &mut Vec<u8>, name: &str, dims: &[u64], qtype: u32, offset: u64) {
    // Name
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    // n_dims
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    // Dims (in reverse order - GGUF stores reversed)
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    // qtype
    data.extend_from_slice(&qtype.to_le_bytes());
    // offset
    data.extend_from_slice(&offset.to_le_bytes());
}

/// Build a minimal transformer GGUF with F32 weights
fn build_minimal_transformer_gguf(
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
) -> Vec<u8> {
    let mut data = Vec::new();

    // Count tensors: token_embd + output_norm + output + per-layer weights
    // Per layer: attn_norm, attn_qkv, attn_output, ffn_up, ffn_down
    let tensor_count = 3 + num_layers * 5;

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&(tensor_count as u64).to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // metadata_count

    // Metadata
    add_string_meta(&mut data, "general.architecture", "test");
    add_u32_meta(&mut data, "test.embedding_length", hidden_dim as u32);
    add_u32_meta(&mut data, "test.block_count", num_layers as u32);
    add_u32_meta(
        &mut data,
        "test.attention.head_count",
        (hidden_dim / 64) as u32,
    );

    // Calculate sizes
    let embed_size = vocab_size * hidden_dim;
    let qkv_size = hidden_dim * hidden_dim * 3;
    let attn_out_size = hidden_dim * hidden_dim;
    let ffn_up_size = hidden_dim * hidden_dim * 4;
    let ffn_down_size = hidden_dim * 4 * hidden_dim;

    // Track offset for tensor data
    let mut offset = 0u64;

    // Add tensor info
    add_tensor_info(
        &mut data,
        "token_embd.weight",
        &[vocab_size as u64, hidden_dim as u64],
        GGUF_TYPE_F32,
        offset,
    );
    offset += (embed_size * 4) as u64;

    for layer_idx in 0..num_layers {
        add_tensor_info(
            &mut data,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            &[hidden_dim as u64],
            GGUF_TYPE_F32,
            offset,
        );
        offset += (hidden_dim * 4) as u64;

        add_tensor_info(
            &mut data,
            &format!("blk.{}.attn_qkv.weight", layer_idx),
            &[hidden_dim as u64 * 3, hidden_dim as u64],
            GGUF_TYPE_F32,
            offset,
        );
        offset += (qkv_size * 4) as u64;

        add_tensor_info(
            &mut data,
            &format!("blk.{}.attn_output.weight", layer_idx),
            &[hidden_dim as u64, hidden_dim as u64],
            GGUF_TYPE_F32,
            offset,
        );
        offset += (attn_out_size * 4) as u64;

        add_tensor_info(
            &mut data,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            &[hidden_dim as u64 * 4, hidden_dim as u64],
            GGUF_TYPE_F32,
            offset,
        );
        offset += (ffn_up_size * 4) as u64;

        add_tensor_info(
            &mut data,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            &[hidden_dim as u64, hidden_dim as u64 * 4],
            GGUF_TYPE_F32,
            offset,
        );
        offset += (ffn_down_size * 4) as u64;
    }

    add_tensor_info(
        &mut data,
        "output_norm.weight",
        &[hidden_dim as u64],
        GGUF_TYPE_F32,
        offset,
    );
    offset += (hidden_dim * 4) as u64;

    add_tensor_info(
        &mut data,
        "output.weight",
        &[vocab_size as u64, hidden_dim as u64],
        GGUF_TYPE_F32,
        offset,
    );

    // Pad to alignment
    while data.len() % GGUF_ALIGNMENT != 0 {
        data.push(0);
    }

    // Add tensor data (initialized to small values for numerical stability)
    let total_tensor_bytes = embed_size * 4
        + num_layers
            * (hidden_dim * 4
                + qkv_size * 4
                + attn_out_size * 4
                + ffn_up_size * 4
                + ffn_down_size * 4)
        + hidden_dim * 4
        + embed_size * 4;

    // Initialize with small random-ish values
    for i in 0..total_tensor_bytes {
        let val = 0.01 * ((i % 100) as f32 - 50.0) / 50.0;
        data.extend_from_slice(&val.to_le_bytes());
    }

    data
}

// ============================================================================
// GGUFTransformer STRUCT TESTS
// ============================================================================

#[test]
fn test_cov_gguf_transformer_layer_structure() {
    let data = build_minimal_transformer_gguf(64, 100, 2);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("load");

    // Verify layer structure
    assert_eq!(transformer.layers.len(), 2);
    assert_eq!(transformer.config.num_layers, 2);
    assert_eq!(transformer.config.hidden_dim, 64);
}

#[test]
fn test_cov_gguf_transformer_config_values() {
    let data = build_minimal_transformer_gguf(128, 200, 3);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("load");

    assert_eq!(transformer.config.hidden_dim, 128);
    assert_eq!(transformer.config.vocab_size, 200);
    assert_eq!(transformer.config.num_layers, 3);
}

// ============================================================================
// GGUFTransformerLayer TESTS
// ============================================================================

#[test]
fn test_cov_gguf_transformer_layer_weights() {
    let data = build_minimal_transformer_gguf(64, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("load");

    let layer = &transformer.layers[0];

    // Check weight shapes
    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert_eq!(layer.qkv_weight.len(), 64 * 64 * 3);
    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 64 * 4);
    assert_eq!(layer.ffn_down_weight.len(), 64 * 4 * 64);
}

#[test]
fn test_cov_gguf_transformer_layer_optional_bias() {
    let data = build_minimal_transformer_gguf(64, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("load");

    let layer = &transformer.layers[0];

    // Our minimal model doesn't have biases
    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.qkv_bias.is_none());
    assert!(layer.attn_output_bias.is_none());
    assert!(layer.ffn_up_bias.is_none());
    assert!(layer.ffn_down_bias.is_none());
}

#[test]
fn test_cov_gguf_transformer_layer_ffn_gate_optional() {
    let data = build_minimal_transformer_gguf(64, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let transformer = GGUFTransformer::from_gguf(&model, &data).expect("load");

    let layer = &transformer.layers[0];

    // Our minimal model doesn't have SwiGLU gate
    assert!(layer.ffn_gate_weight.is_none());
    assert!(layer.ffn_gate_bias.is_none());
}

// ============================================================================
// QUANTIZED TRANSFORMER TESTS
// ============================================================================

#[test]
fn test_cov_quantized_tensor_ref_fields() {
    let tensor_ref = QuantizedTensorRef {
        offset: 1024,
        byte_size: 4096,
        num_elements: 2048,
        qtype: GGUF_TYPE_Q4_0,
    };

    assert_eq!(tensor_ref.offset, 1024);
    assert_eq!(tensor_ref.byte_size, 4096);
    assert_eq!(tensor_ref.num_elements, 2048);
    assert_eq!(tensor_ref.qtype, GGUF_TYPE_Q4_0);
}

#[test]
fn test_cov_qkv_weights_fused_out_dim() {
    let tensor_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 1000,
        num_elements: 64 * 192, // hidden_dim * 3 * hidden_dim
        qtype: GGUF_TYPE_F32,
    };

    let qkv = QKVWeights::Fused(tensor_ref);
    assert_eq!(qkv.out_dim(64), 192); // 3 * hidden_dim
}

#[test]
fn test_cov_qkv_weights_fused_q_dim() {
    let tensor_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 1000,
        num_elements: 64 * 192,
        qtype: GGUF_TYPE_F32,
    };

    let qkv = QKVWeights::Fused(tensor_ref);
    assert_eq!(qkv.q_dim(64), 64); // hidden_dim
}

#[test]
fn test_cov_qkv_weights_separate_out_dim() {
    let q_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 500,
        num_elements: 64 * 64,
        qtype: GGUF_TYPE_F32,
    };
    let k_ref = QuantizedTensorRef {
        offset: 500,
        byte_size: 500,
        num_elements: 64 * 32,
        qtype: GGUF_TYPE_F32,
    };
    let v_ref = QuantizedTensorRef {
        offset: 1000,
        byte_size: 500,
        num_elements: 64 * 32,
        qtype: GGUF_TYPE_F32,
    };

    let qkv = QKVWeights::Separate {
        q: q_ref,
        k: k_ref,
        v: v_ref,
    };
    assert_eq!(qkv.out_dim(64), 64 + 32 + 32); // q_dim + k_dim + v_dim
}

#[test]
fn test_cov_qkv_weights_separate_q_dim() {
    let q_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 500,
        num_elements: 64 * 64,
        qtype: GGUF_TYPE_F32,
    };
    let k_ref = QuantizedTensorRef {
        offset: 500,
        byte_size: 500,
        num_elements: 64 * 32,
        qtype: GGUF_TYPE_F32,
    };
    let v_ref = QuantizedTensorRef {
        offset: 1000,
        byte_size: 500,
        num_elements: 64 * 32,
        qtype: GGUF_TYPE_F32,
    };

    let qkv = QKVWeights::Separate {
        q: q_ref,
        k: k_ref,
        v: v_ref,
    };
    assert_eq!(qkv.q_dim(64), 64); // q projection output dim
}

// ============================================================================
// INFERENCE SCRATCH BUFFER TESTS
// ============================================================================

#[test]
fn test_cov_inference_scratch_buffer_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = InferenceScratchBuffer::from_config(&config);

    assert_eq!(scratch.hidden.len(), 128);
    assert_eq!(scratch.normed.len(), 128);
    assert_eq!(scratch.q.len(), 128);
    assert_eq!(scratch.k.len(), 128);
    assert_eq!(scratch.v.len(), 128);
    assert_eq!(scratch.attn_out.len(), 128);
    assert_eq!(scratch.ffn_up.len(), 512);
    assert_eq!(scratch.ffn_gate.len(), 512);
    assert_eq!(scratch.ffn_down.len(), 128);
    assert_eq!(scratch.logits.len(), 1000);
}

#[test]
fn test_cov_inference_scratch_buffer_reset() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let mut scratch = InferenceScratchBuffer::from_config(&config);

    // Modify some values
    scratch.hidden[0] = 1.0;
    scratch.normed[0] = 2.0;

    // Reset
    scratch.reset();

    // hidden and normed should be zeroed
    assert_eq!(scratch.hidden[0], 0.0);
    assert_eq!(scratch.normed[0], 0.0);
}

#[test]
fn test_cov_inference_scratch_buffer_q8k_buffers() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 512, // Must be multiple of 256 for Q8K
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        intermediate_dim: 1024,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = InferenceScratchBuffer::from_config(&config);

    // Q8K uses 256-element super-blocks
    assert_eq!(scratch.q8k_hidden_scales.len(), 512 / 256);
    assert_eq!(scratch.q8k_hidden_quants.len(), 512);
    assert_eq!(scratch.q8k_inter_scales.len(), 1024 / 256);
    assert_eq!(scratch.q8k_inter_quants.len(), 1024);
}

// ============================================================================
// QUANTIZED GENERATE CONFIG TESTS
// ============================================================================

#[test]
fn test_cov_quantized_generate_config_default() {
    let config = QuantizedGenerateConfig::default();

    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.0).abs() < 0.001);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_cov_quantized_generate_config_custom() {
    let config = QuantizedGenerateConfig {
        max_tokens: 50,
        temperature: 0.7,
        top_k: 20,
        stop_tokens: vec![1, 2, 3],
        trace: false,
        ..Default::default()
    };

    assert_eq!(config.max_tokens, 50);
    assert!((config.temperature - 0.7).abs() < 0.001);
    assert_eq!(config.top_k, 20);
    assert_eq!(config.stop_tokens, vec![1, 2, 3]);
}

// ============================================================================
// GGUF MODEL METADATA TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_architecture() {
    let data = build_minimal_transformer_gguf(64, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.architecture(), Some("test"));
}

#[test]
fn test_cov_gguf_model_embedding_dim() {
    let data = build_minimal_transformer_gguf(128, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.embedding_dim(), Some(128));
}

#[test]
fn test_cov_gguf_model_num_layers() {
    let data = build_minimal_transformer_gguf(64, 100, 4);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.num_layers(), Some(4));
}

#[test]
fn test_cov_gguf_model_num_heads() {
    let data = build_minimal_transformer_gguf(128, 100, 1);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // hidden_dim / 64 = 2 heads
    assert_eq!(model.num_heads(), Some(2));
}

// ============================================================================
// OWNED QUANTIZED KV CACHE TESTS
// ============================================================================

#[test]
fn test_cov_owned_quantized_kv_cache_new() {
    let cache = OwnedQuantizedKVCache::new(4, 64, 512); // 4 layers, 64 kv_dim, 512 max_len

    assert_eq!(cache.len(), 0);
    assert!(cache.get_k(0).is_empty());
    assert!(cache.get_v(0).is_empty());
}

#[test]
fn test_cov_owned_quantized_kv_cache_append() {
    let mut cache = OwnedQuantizedKVCache::new(2, 32, 100);

    let k = vec![1.0f32; 32];
    let v = vec![2.0f32; 32];

    cache.append(0, &k, &v);

    assert!(!cache.get_k(0).is_empty());
    assert!(!cache.get_v(0).is_empty());
}

#[test]
fn test_cov_owned_quantized_kv_cache_advance() {
    let mut cache = OwnedQuantizedKVCache::new(2, 32, 100);

    assert_eq!(cache.len(), 0);
    cache.advance();
    assert_eq!(cache.len(), 1);
    cache.advance();
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_cov_owned_quantized_kv_cache_reset() {
    let mut cache = OwnedQuantizedKVCache::new(2, 32, 100);

    let k = vec![1.0f32; 32];
    let v = vec![2.0f32; 32];

    cache.append(0, &k, &v);
    cache.advance();
    cache.reset();

    assert_eq!(cache.len(), 0);
    assert!(cache.get_k(0).is_empty());
    assert!(cache.get_v(0).is_empty());
}

#[test]
fn test_cov_owned_quantized_kv_cache_multiple_layers() {
    let mut cache = OwnedQuantizedKVCache::new(3, 64, 100);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];

    cache.append(0, &k, &v);
    cache.append(1, &k, &v);
    cache.append(2, &k, &v);

    assert!(!cache.get_k(0).is_empty());
    assert!(!cache.get_k(1).is_empty());
    assert!(!cache.get_k(2).is_empty());
}

// ============================================================================
// GGUF VALUE TYPE TESTS
// ============================================================================

#[test]
fn test_cov_gguf_value_uint8() {
    let val = GGUFValue::UInt8(255);
    if let GGUFValue::UInt8(v) = val {
        assert_eq!(v, 255);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_int8() {
    let val = GGUFValue::Int8(-128);
    if let GGUFValue::Int8(v) = val {
        assert_eq!(v, -128);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_float32() {
    let val = GGUFValue::Float32(1.23456);
    if let GGUFValue::Float32(v) = val {
        assert!((v - 1.23456).abs() < 0.0001);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_bool() {
    let val_true = GGUFValue::Bool(true);
    let val_false = GGUFValue::Bool(false);

    if let GGUFValue::Bool(v) = val_true {
        assert!(v);
    }
    if let GGUFValue::Bool(v) = val_false {
        assert!(!v);
    }
}

#[test]
fn test_cov_gguf_value_string() {
    let val = GGUFValue::String("hello".to_string());
    if let GGUFValue::String(s) = val {
        assert_eq!(s, "hello");
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_array() {
    let val = GGUFValue::Array(vec![
        GGUFValue::UInt32(1),
        GGUFValue::UInt32(2),
        GGUFValue::UInt32(3),
    ]);
    if let GGUFValue::Array(arr) = val {
        assert_eq!(arr.len(), 3);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_uint64() {
    let val = GGUFValue::UInt64(u64::MAX);
    if let GGUFValue::UInt64(v) = val {
        assert_eq!(v, u64::MAX);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_int64() {
    let val = GGUFValue::Int64(i64::MIN);
    if let GGUFValue::Int64(v) = val {
        assert_eq!(v, i64::MIN);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_cov_gguf_value_float64() {
    let val = GGUFValue::Float64(1.234567890123);
    if let GGUFValue::Float64(v) = val {
        assert!((v - 1.234567890123).abs() < 0.0000001);
    } else {
        panic!("Wrong variant");
    }
}

// ============================================================================
// GGUF CONSTANTS TESTS
// ============================================================================

#[test]
fn test_cov_gguf_magic_constant() {
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    // "GGUF" in ASCII little-endian
    let magic_bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&magic_bytes, b"GGUF");
}

#[test]
fn test_cov_gguf_version_constant() {
    assert_eq!(GGUF_VERSION_V3, 3);
}

#[test]
fn test_cov_gguf_alignment_constant() {
    assert_eq!(GGUF_ALIGNMENT, 32);
}

#[test]
fn test_cov_gguf_type_constants() {
    assert_eq!(GGUF_TYPE_F32, 0);
    assert_eq!(GGUF_TYPE_Q4_0, 2);
    assert_eq!(GGUF_TYPE_Q8_0, 8);
    assert_eq!(GGUF_TYPE_Q4_K, 12);
}

// ============================================================================
// GPU-DEPENDENT TESTS (with feature flag)
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCached};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cov_owned_quantized_model_cached_new() {
        // Create a temp file with GGUF data
        let data = build_minimal_transformer_gguf(64, 100, 1);
        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let mapped = MappedGGUFModel::from_path(temp.path()).expect("load mapped");
        let owned = OwnedQuantizedModel::from_mapped(&mapped).expect("load");
        let cached = OwnedQuantizedModelCached::new(owned);

        // Should have model accessible via method
        assert_eq!(cached.model().config().hidden_dim, 64);
    }
}

// ============================================================================
// CUDA-DEPENDENT TESTS (with feature flag)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cov_owned_quantized_model_cuda_disabled_by_default() {
        // Create a temp file with GGUF data
        let data = build_minimal_transformer_gguf(64, 100, 1);
        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let mapped = MappedGGUFModel::from_path(temp.path()).expect("load mapped");
        let owned = OwnedQuantizedModel::from_mapped(&mapped).expect("load");

        // CUDA should not be enabled by default (needs explicit enable_cuda call)
        assert!(!owned.cuda_enabled());
    }

    #[test]
    fn test_cov_owned_quantized_model_clone_preserves_config() {
        // Create a temp file with GGUF data
        let data = build_minimal_transformer_gguf(64, 100, 1);
        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let mapped = MappedGGUFModel::from_path(temp.path()).expect("load mapped");
        let owned = OwnedQuantizedModel::from_mapped(&mapped).expect("load");

        // Clone and verify config preserved
        let cloned = Clone::clone(&owned);

        // Verify config is preserved in clone
        assert_eq!(cloned.config().hidden_dim, owned.config().hidden_dim);
        assert_eq!(cloned.config().vocab_size, owned.config().vocab_size);
        assert_eq!(cloned.config().num_layers, owned.config().num_layers);
        // Cloned model should not have CUDA enabled (not copied)
        assert!(!cloned.cuda_enabled());
    }
}
