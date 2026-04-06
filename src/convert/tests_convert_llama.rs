//! T-COV-95 Synthetic Falsification: convert/mod.rs via Pygmy GGUF models
//!
//! Tests GGUF to APR conversion using GGUFBuilder synthetic models.
//! A 1KB GGUF file exercises the same conversion logic as a 100GB one.

use crate::gguf::test_factory::{
    build_minimal_llama_gguf, build_minimal_phi2_gguf, create_f32_embedding_data,
    create_f32_norm_weights, create_q4_k_data, create_q6_k_data, create_q8_0_data, GGUFBuilder,
};

use super::GgufToAprConverter;
use crate::apr_transformer::AprTransformer;
use crate::gguf::{GGUFModel, GGUFTransformer};

// ============================================================================
// GgufToAprConverter::convert with Pygmy Models
// ============================================================================

#[test]
fn test_convert_llama_pygmy() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let result = GgufToAprConverter::convert(&gguf_data);
    assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

    let apr = result.unwrap();
    assert_eq!(apr.config.architecture, "llama");
    assert_eq!(apr.config.hidden_dim, 64);
    assert_eq!(apr.config.num_layers, 1);
    assert_eq!(apr.config.num_heads, 4);
}

#[test]
fn test_convert_phi2_pygmy() {
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let result = GgufToAprConverter::convert(&gguf_data);
    assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

    let apr = result.unwrap();
    assert_eq!(apr.config.architecture, "phi2");
    assert_eq!(apr.config.hidden_dim, 64);
}

#[test]
fn test_convert_empty_vocab() {
    // Minimal GGUF with vocab_size=1
    let embed_data = create_f32_embedding_data(1, 32);
    let norm_data = create_f32_norm_weights(32);
    let q_data = create_q4_k_data(32 * 32);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 2)
        .num_kv_heads("llama", 2)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 64)
        .add_f32_tensor("token_embd.weight", &[1, 32], &embed_data)
        .add_f32_tensor("blk.0.attn_norm.weight", &[32], &norm_data)
        .add_q4_k_tensor("blk.0.attn_q.weight", &[32, 32], &q_data)
        .add_q4_k_tensor("blk.0.attn_k.weight", &[32, 32], &q_data)
        .add_q4_k_tensor("blk.0.attn_v.weight", &[32, 32], &q_data)
        .add_q4_k_tensor("blk.0.attn_output.weight", &[32, 32], &q_data)
        .add_f32_tensor("blk.0.ffn_norm.weight", &[32], &norm_data)
        .add_q4_k_tensor("blk.0.ffn_up.weight", &[32, 64], &q_data)
        .add_q4_k_tensor("blk.0.ffn_down.weight", &[64, 32], &q_data)
        .add_q4_k_tensor("blk.0.ffn_gate.weight", &[32, 64], &q_data)
        .add_f32_tensor("output_norm.weight", &[32], &norm_data)
        .build();

    let result = GgufToAprConverter::convert(&gguf_data);
    assert!(result.is_ok(), "Conversion failed: {:?}", result.err());
}

#[test]
fn test_convert_truncated_data() {
    // Truncated GGUF data should fail
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let truncated = &gguf_data[..gguf_data.len() / 2];

    let result = GgufToAprConverter::convert(truncated);
    assert!(result.is_err());
}

#[test]
fn test_convert_invalid_magic() {
    // Invalid GGUF magic
    let result = GgufToAprConverter::convert(b"INVALID_MAGIC_12345678");
    assert!(result.is_err());
}

// ============================================================================
// GgufToAprConverter::from_gguf_transformer
// ============================================================================

#[test]
fn test_from_gguf_transformer_llama() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let gguf_model = GGUFModel::from_bytes(&gguf_data).unwrap();
    let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data).unwrap();

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf_transformer);

    assert_eq!(
        apr.config.architecture,
        gguf_transformer.config.architecture
    );
    assert_eq!(apr.config.hidden_dim, gguf_transformer.config.hidden_dim);
    assert_eq!(apr.config.num_layers, gguf_transformer.config.num_layers);
    assert_eq!(apr.layers.len(), gguf_transformer.layers.len());
    assert_eq!(
        apr.token_embedding.len(),
        gguf_transformer.token_embedding.len()
    );
}

#[test]
fn test_from_gguf_transformer_phi2() {
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let gguf_model = GGUFModel::from_bytes(&gguf_data).unwrap();
    let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data).unwrap();

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf_transformer);
    assert_eq!(apr.config.architecture, "phi2");
}

#[test]
fn test_from_gguf_transformer_preserves_weights() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let gguf_model = GGUFModel::from_bytes(&gguf_data).unwrap();
    let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data).unwrap();

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf_transformer);

    // Check weight sizes match
    assert_eq!(
        apr.output_norm_weight.len(),
        gguf_transformer.output_norm_weight.len()
    );
    assert_eq!(
        apr.lm_head_weight.len(),
        gguf_transformer.lm_head_weight.len()
    );

    // Check first layer weights match
    if !apr.layers.is_empty() {
        assert_eq!(
            apr.layers[0].attn_norm_weight.len(),
            gguf_transformer.layers[0].attn_norm_weight.len()
        );
        assert_eq!(
            apr.layers[0].qkv_weight.len(),
            gguf_transformer.layers[0].qkv_weight.len()
        );
    }
}

// ============================================================================
// GgufToAprConverter::to_apr_bytes and round-trip
// ============================================================================

#[test]
fn test_to_apr_bytes_basic() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();

    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).unwrap();

    // Check APR header magic
    assert!(apr_bytes.len() >= 64);
    assert_eq!(&apr_bytes[0..3], b"APR");
}

#[test]
fn test_to_apr_bytes_header_layout() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();

    let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).unwrap();

    // Parse header fields
    let version_major = apr_bytes[4];
    let version_minor = apr_bytes[5];
    let tensor_count =
        u32::from_le_bytes([apr_bytes[8], apr_bytes[9], apr_bytes[10], apr_bytes[11]]);

    assert_eq!(version_major, 2);
    assert_eq!(version_minor, 0);
    assert_eq!(tensor_count, 1); // Single "weights" tensor
}

#[test]
fn test_apr_round_trip() {
    // Convert GGUF to APR
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let original = GgufToAprConverter::convert(&gguf_data).unwrap();

    // Serialize to bytes
    let apr_bytes = GgufToAprConverter::to_apr_bytes(&original).unwrap();

    // Deserialize back
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).unwrap();

    // Verify config matches
    assert_eq!(original.config.architecture, loaded.config.architecture);
    assert_eq!(original.config.hidden_dim, loaded.config.hidden_dim);
    assert_eq!(original.config.num_layers, loaded.config.num_layers);
    assert_eq!(original.config.num_heads, loaded.config.num_heads);
    assert_eq!(original.config.vocab_size, loaded.config.vocab_size);
}

#[test]
fn test_apr_round_trip_weights_preserved() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let original = GgufToAprConverter::convert(&gguf_data).unwrap();

    let apr_bytes = GgufToAprConverter::to_apr_bytes(&original).unwrap();
    let loaded = GgufToAprConverter::from_apr_bytes(&apr_bytes).unwrap();

    // Check embedding weights
    assert_eq!(original.token_embedding.len(), loaded.token_embedding.len());
    for (o, l) in original
        .token_embedding
        .iter()
        .zip(loaded.token_embedding.iter())
    {
        assert!((o - l).abs() < 1e-6, "Embedding mismatch: {} vs {}", o, l);
    }

    // Check layer weights
    assert_eq!(original.layers.len(), loaded.layers.len());
}

// ============================================================================
// GgufToAprConverter::from_apr_bytes error cases
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let result = GgufToAprConverter::from_apr_bytes(&[0; 32]);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"NOAP");
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_metadata() {
    // Valid header but truncated metadata
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    data[4] = 2; // version
    data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size > file size

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// Mixed tensor type conversions
// ============================================================================

#[test]
fn test_convert_with_q4_k_tensors() {
    let vocab_size = 32;
    let hidden_dim = 64;
    let kv_dim = 64;
    let intermediate_dim = 128;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert_eq!(apr.config.hidden_dim, hidden_dim);
    assert!(!apr.layers.is_empty());
}

#[test]
fn test_convert_with_q8_0_tensors() {
    let vocab_size = 32;
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q8_data = create_q8_0_data(hidden_dim * hidden_dim);
    let ffn_data = create_q8_0_data(hidden_dim * intermediate_dim);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q8_0_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q8_0_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_data,
        )
        .add_q8_0_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_data,
        )
        .add_q8_0_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert_eq!(apr.config.hidden_dim, hidden_dim);
}

include!("tests_convert_02.rs");
