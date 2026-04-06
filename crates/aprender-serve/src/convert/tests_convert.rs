//! T-COV-95 Coverage Bridge (Part 13 - GgufToAprConverter::convert full pipeline,
//! from_gguf_transformer edge cases, to_apr_bytes/from_apr_bytes roundtrip with layers)
//!
//! Targets uncovered lines in convert/mod.rs:
//!   - GgufToAprConverter::convert() (lines 88-97) - full GGUF->APR pipeline
//!   - from_gguf_transformer with biases, optional fields
//!   - to_apr_bytes with metadata alignment padding
//!   - from_apr_bytes successful path with real transformer data

use super::*;
use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};
use crate::gguf::{GGUFConfig, GGUFModel, GGUFTransformer, GGUFTransformerLayer};

// ============================================================================
// GgufToAprConverter::convert() - Full GGUF->APR pipeline
// ============================================================================

#[test]
fn test_convert_from_gguf_bytes_llama() {
    // Build a valid GGUF model and convert it
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let result = GgufToAprConverter::convert(&gguf_data);
    assert!(
        result.is_ok(),
        "convert() should succeed for valid LLaMA GGUF: {:?}",
        result.err()
    );

    let transformer = result.expect("should convert");
    assert_eq!(transformer.config.architecture, "llama");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);
    assert_eq!(transformer.config.num_kv_heads, 4);
    assert!(!transformer.token_embedding.is_empty());
    assert!(!transformer.layers.is_empty());
    assert!(!transformer.output_norm_weight.is_empty());
}

#[test]
fn test_convert_from_gguf_bytes_phi2() {
    // Build a valid Phi-2 style GGUF (fused QKV)
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let result = GgufToAprConverter::convert(&gguf_data);
    assert!(
        result.is_ok(),
        "convert() should succeed for valid Phi-2 GGUF: {:?}",
        result.err()
    );

    let transformer = result.expect("should convert");
    assert_eq!(transformer.config.architecture, "phi2");
}

#[test]
fn test_convert_invalid_gguf_bytes() {
    // Invalid GGUF data should fail
    let bad_data = vec![0xFF; 100];
    let result = GgufToAprConverter::convert(&bad_data);
    assert!(result.is_err(), "convert() should fail for invalid GGUF");
}

#[test]
fn test_convert_empty_data() {
    let result = GgufToAprConverter::convert(&[]);
    assert!(result.is_err(), "convert() should fail for empty data");
}

// ============================================================================
// GgufToAprConverter::convert -> stats roundtrip
// ============================================================================

#[test]
fn test_convert_then_stats() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let transformer = GgufToAprConverter::convert(&gguf_data).expect("should convert");
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.architecture, "llama");
    assert_eq!(stats.hidden_dim, 64);
    assert_eq!(stats.num_layers, 1);
    assert!(stats.total_parameters > 0);
    assert!(stats.memory_bytes_f32 > 0);
}

// ============================================================================
// GgufToAprConverter::convert -> to_apr_bytes -> from_apr_bytes roundtrip
// ============================================================================

#[test]
fn test_convert_full_roundtrip() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let transformer = GgufToAprConverter::convert(&gguf_data).expect("should convert");

    // Serialize to APR bytes
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("should serialize");

    // Deserialize back
    let restored = GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("should deserialize");

    // Verify config preserved
    assert_eq!(restored.config.architecture, "llama");
    assert_eq!(restored.config.hidden_dim, 64);
    assert_eq!(restored.config.num_layers, 1);
    assert_eq!(restored.config.num_heads, 4);
    assert_eq!(restored.config.num_kv_heads, 4);
    // intermediate_dim may differ due to GGUF tensor inference vs metadata
    assert!(restored.config.intermediate_dim > 0);
}

// ============================================================================
// from_gguf_transformer: Layer field preservation
// ============================================================================

#[test]
fn test_from_gguf_transformer_preserves_layer_biases() {
    // Create GGUFTransformer with biases set
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: Some(vec![0.1; 8]),
        qkv_weight: vec![0.0; 8 * 24], // hidden * 3 * hidden
        qkv_bias: Some(vec![0.01; 24]),
        attn_output_weight: vec![0.0; 8 * 8],
        attn_output_bias: Some(vec![0.02; 8]),
        ffn_gate_weight: Some(vec![0.0; 16 * 8]),
        ffn_gate_bias: Some(vec![0.03; 16]),
        ffn_up_weight: vec![0.0; 16 * 8],
        ffn_up_bias: Some(vec![0.04; 16]),
        ffn_down_weight: vec![0.0; 8 * 16],
        ffn_down_bias: Some(vec![0.05; 8]),
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: Some(vec![0.06; 8]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.0; 80],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: Some(vec![0.07; 8]),
        lm_head_weight: vec![0.0; 80],
        lm_head_bias: Some(vec![0.08; 10]),
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    // Verify biases were preserved
    assert!(apr.layers[0].attn_norm_bias.is_some());
    assert!(apr.layers[0].qkv_bias.is_some());
    assert!(apr.layers[0].attn_output_bias.is_some());
    assert!(apr.layers[0].ffn_gate_bias.is_some());
    assert!(apr.layers[0].ffn_up_bias.is_some());
    assert!(apr.layers[0].ffn_down_bias.is_some());
    assert!(apr.layers[0].ffn_norm_bias.is_some());
    assert!(apr.output_norm_bias.is_some());
    assert!(apr.lm_head_bias.is_some());

    // Verify q4k_layers and quantized lm_head are None
    assert!(apr.q4k_layers.is_none());
    assert!(apr.lm_head_weight_q6k.is_none());
    assert!(apr.lm_head_weight_q4k.is_none());
}

#[test]
fn test_from_gguf_transformer_no_biases() {
    // Create GGUFTransformer with no biases
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.0; 8 * 24],
        qkv_bias: None,
        attn_output_weight: vec![0.0; 64],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; 128],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.0; 128],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.0; 80],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 80],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    // Verify biases are None
    assert!(apr.layers[0].attn_norm_bias.is_none());
    assert!(apr.layers[0].qkv_bias.is_none());
    assert!(apr.layers[0].attn_output_bias.is_none());
    assert!(apr.layers[0].ffn_gate_weight.is_none());
    assert!(apr.layers[0].ffn_gate_bias.is_none());
    assert!(apr.output_norm_bias.is_none());
    assert!(apr.lm_head_bias.is_none());
}

#[test]
fn test_from_gguf_transformer_multi_layer() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 3,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let make_layer = || GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.0; 8 * 24],
        qkv_bias: None,
        attn_output_weight: vec![0.0; 64],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.0; 128]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.0; 128],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.0; 128],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.0; 80],
        position_embedding: None,
        layers: vec![make_layer(), make_layer(), make_layer()],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 80],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert_eq!(apr.layers.len(), 3);
    assert_eq!(apr.config.num_layers, 3);
}

// ============================================================================
// to_apr_bytes: Header structure validation
// ============================================================================

#[test]
fn test_to_apr_bytes_header_offsets_consistent() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 20,
            intermediate_dim: 32,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; 320],

        layers: vec![AprTransformerLayer::empty(16, 32)],
        output_norm_weight: vec![1.0; 16],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 320],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("should serialize");

    // Parse offsets from header
    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().expect("slice")) as usize;
    let metadata_size = u32::from_le_bytes(bytes[20..24].try_into().expect("slice")) as usize;
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().expect("slice")) as usize;
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().expect("slice")) as usize;

    // metadata_offset should be HEADER_SIZE (64)
    assert_eq!(metadata_offset, 64);

    // metadata_size should fit between metadata_offset and tensor_index_offset
    let metadata_padded = metadata_size.div_ceil(64) * 64;
    assert_eq!(
        tensor_index_offset,
        metadata_offset + metadata_padded,
        "tensor_index should follow padded metadata"
    );

    // data_offset should follow tensor index
    assert!(
        data_offset >= tensor_index_offset,
        "data_offset should follow tensor_index_offset"
    );

    // Total bytes should cover all sections
    assert!(
        bytes.len() >= data_offset,
        "File should be large enough to contain data section"
    );
}

// ============================================================================
// to_apr_bytes: Metadata padding to 64-byte boundary
// ============================================================================

#[test]
fn test_to_apr_bytes_metadata_padded_to_alignment() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "x".to_string(), // Very short arch name -> small metadata
            hidden_dim: 4,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 2,
            intermediate_dim: 4,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.0; 8],

        layers: vec![],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 8],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("should serialize");

    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().expect("slice")) as usize;

    // tensor_index_offset should be 64 + metadata_padded_len
    // metadata_padded_len should be a multiple of 64
    let metadata_padded_len = tensor_index_offset - 64;
    assert_eq!(
        metadata_padded_len % 64,
        0,
        "Metadata should be padded to 64-byte boundary"
    );
}

include!("tests_apr.rs");
