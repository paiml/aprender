//! GGUF Tests Part 26: T-COV-95 Coverage Bridge (B6)
//!
//! Tests for loader.rs uncovered paths:
//! - GGUFModel::from_bytes: truncated data at various points, wrong magic, wrong version
//! - Metadata accessors: architecture, embedding_dim, num_layers, etc.
//! - decode/encode: various token edge cases
//! - GGUFTransformer::from_gguf: various model configurations
//!
//! Refs PMAT-802: Protocol T-COV-95 Batch B6

use crate::gguf::test_factory::*;
use crate::gguf::GGUFModel;

// ============================================================================
// GGUFModel::from_bytes - Error paths
// ============================================================================

#[test]
fn test_from_bytes_empty() {
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_4_bytes_wrong_magic() {
    let data = [0x00, 0x00, 0x00, 0x00];
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_4_bytes_correct_magic() {
    // GGUF magic but no version/counts
    let data = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err()); // Too short for version
}

#[test]
fn test_from_bytes_8_bytes_truncated() {
    let mut data = vec![0x47, 0x47, 0x55, 0x46]; // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err()); // Too short for counts
}

#[test]
fn test_from_bytes_16_bytes_truncated() {
    let mut data = vec![0x47, 0x47, 0x55, 0x46]; // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err()); // Too short for metadata_count
}

#[test]
fn test_from_bytes_24_bytes_truncated() {
    let mut data = vec![0x47, 0x47, 0x55, 0x46]; // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0
    let result = GGUFModel::from_bytes(&data);
    // Should parse successfully with 0 tensors and 0 metadata
    assert!(result.is_ok());
}

// ============================================================================
// GGUFBuilder-based valid model tests
// ============================================================================

#[test]
fn test_from_bytes_valid_minimal() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_from_bytes_valid_with_all_metadata() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 2)
        .num_heads("llama", 8)
        .num_kv_heads("llama", 4)
        .context_length("llama", 512)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 128)
        .vocab_size("llama", 100)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.header.version, 3);
}

// ============================================================================
// Metadata Accessors
// ============================================================================

#[test]
fn test_metadata_architecture() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let arch = model.architecture();
    assert!(arch.is_some());
    assert_eq!(arch.expect("test value should be present"), "llama");
}

#[test]
fn test_metadata_embedding_dim() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 128)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let dim = model.embedding_dim();
    assert!(dim.is_some());
    assert_eq!(dim.expect("test value should be present"), 128);
}

#[test]
fn test_metadata_num_layers() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 4)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let layers = model.num_layers();
    assert!(layers.is_some());
    assert_eq!(layers.expect("test value should be present"), 4);
}

#[test]
fn test_metadata_num_heads() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 8)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let heads = model.num_heads();
    assert!(heads.is_some());
    assert_eq!(heads.expect("test value should be present"), 8);
}

#[test]
fn test_metadata_num_kv_heads() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 8)
        .num_kv_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let kv_heads = model.num_kv_heads();
    assert!(kv_heads.is_some());
    assert_eq!(kv_heads.expect("test value should be present"), 4);
}

#[test]
fn test_metadata_context_length() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .context_length("llama", 2048)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let ctx = model.context_length();
    assert!(ctx.is_some());
    assert_eq!(ctx.expect("test value should be present"), 2048);
}

#[test]
fn test_metadata_rope_freq_base() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .rope_freq_base("llama", 500000.0)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let base = model.rope_freq_base();
    assert!(base.is_some());
    assert!((base.expect("test value should be present") - 500000.0).abs() < 1.0);
}

#[test]
fn test_metadata_rms_epsilon() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .rms_epsilon("llama", 1e-6)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let eps = model.rms_epsilon();
    assert!(eps.is_some());
    assert!((eps.expect("test value should be present") - 1e-6).abs() < 1e-8);
}

#[test]
fn test_metadata_missing_returns_none() {
    // Build with minimal metadata - kv_heads, context_length, etc. not set
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    // These should be None since we didn't set them
    // (Some may have defaults - just verify they don't panic)
    let _ = model.rope_type();
    let _ = model.bos_token_id();
    let _ = model.eos_token_id();
}

// ============================================================================
// decode/encode
// ============================================================================

#[test]
fn test_decode_empty_tokens() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let text = model.decode(&[]);
    assert!(text.is_empty());
}

#[test]
fn test_encode_empty_string() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let tokens = model.encode("");
    // encode may return None or Some(empty) for empty string
    if let Some(tokens) = tokens {
        assert!(tokens.is_empty() || tokens.len() == 1);
    }
}

#[test]
fn test_vocabulary() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    // May return None if no vocab metadata
    let vocab = model.vocabulary();
    let _ = vocab; // Just verify no panic
}

// ============================================================================
// GGUFTransformer::from_gguf
// ============================================================================

#[test]
fn test_transformer_from_gguf_minimal() {
    use crate::gguf::GGUFTransformer;

    let vocab = 8;
    let hidden = 4;

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .num_kv_heads("llama", 1)
        .context_length("llama", 32)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 8)
        .vocab_size("llama", vocab as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[hidden as u64, vocab as u64],
            &create_f32_embedding_data(vocab, hidden),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "blk.0.attn_q.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_k.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_v.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_output.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden as u64, 8],
            &vec![0.01f32; hidden * 8],
        )
        .add_f32_tensor(
            "blk.0.ffn_up.weight",
            &[hidden as u64, 8],
            &vec![0.01f32; hidden * 8],
        )
        .add_f32_tensor(
            "blk.0.ffn_down.weight",
            &[8, hidden as u64],
            &vec![0.01f32; 8 * hidden],
        )
        .add_f32_tensor(
            "blk.0.ffn_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "output_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "output.weight",
            &[hidden as u64, vocab as u64],
            &vec![0.01f32; hidden * vocab],
        )
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let transformer = GGUFTransformer::from_gguf(&model, &data);
    assert!(
        transformer.is_ok(),
        "from_gguf failed: {:?}",
        transformer.err()
    );
    let t = transformer.expect("test value should be present");
    assert_eq!(t.config.hidden_dim, hidden);
    assert_eq!(t.config.num_layers, 1);
    assert_eq!(t.layers.len(), 1);
}

#[test]
fn test_transformer_from_gguf_no_tensors() {
    use crate::gguf::GGUFTransformer;

    // Model with metadata but no tensors
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let transformer = GGUFTransformer::from_gguf(&model, &data);
    // Should fail because required tensors are missing
    assert!(transformer.is_err());
}

// ============================================================================
// Header struct fields
// ============================================================================

#[test]
fn test_header_version_and_counts() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.header.version, 3);
    // We set architecture, hidden_dim, num_layers, num_heads = 4 metadata entries
    assert!(model.header.metadata_count >= 4);
}

#[test]
fn test_tensor_count_zero() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.header.tensor_count, 0);
    assert!(model.tensors.is_empty());
}

#[test]
fn test_tensor_count_with_tensors() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 4)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("test.weight", &[4, 4], &vec![0.1f32; 16])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.header.tensor_count, 1);
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "test.weight");
}
