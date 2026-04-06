//! Phase 34: Matmul coverage tests
//!
//! These lib tests illuminate gguf/inference/matmul.rs:
//! - embed() - Token embedding lookup
//! - embed_into() - Pre-allocated embedding lookup
//! - fused_matmul() - Quantized matrix multiplication
//!
//! Uses proptest for empirical saturation across quantization types.
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::types::{
    GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_0,
};
use crate::gguf::{GGUFConfig, OwnedQuantizedTensor};

// =============================================================================
// Embed Tests
// =============================================================================

#[test]
fn test_phase34_embed_single_token() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let embeddings = model.embed(&[0]);

    assert_eq!(embeddings.len(), config.hidden_dim);
    assert!(embeddings.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phase34_embed_multiple_tokens() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let embeddings = model.embed(&[0, 1, 2, 3, 4]);

    assert_eq!(embeddings.len(), 5 * config.hidden_dim);
    assert!(embeddings.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phase34_embed_out_of_vocab() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    // Token 9999 is beyond vocab size 100
    let embeddings = model.embed(&[9999]);

    // Should return zeros for out-of-vocab tokens
    assert_eq!(embeddings.len(), config.hidden_dim);
    assert!(embeddings.iter().all(|&x| x == 0.0));
}

#[test]
fn test_phase34_embed_boundary_token() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    // Token 99 is the last valid token (vocab_size - 1)
    let embeddings = model.embed(&[99]);

    assert_eq!(embeddings.len(), config.hidden_dim);
    assert!(embeddings.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phase34_embed_empty() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let embeddings = model.embed(&[]);

    assert!(embeddings.is_empty());
}

// =============================================================================
// Embed Into Tests
// =============================================================================

#[test]
fn test_phase34_embed_into_single() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let mut output = vec![0.0f32; config.hidden_dim];

    model.embed_into(0, &mut output);

    // Output should be filled with embedding values
    assert_eq!(output.len(), config.hidden_dim);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phase34_embed_into_out_of_vocab() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let mut output = vec![999.0f32; config.hidden_dim]; // Pre-fill with non-zero

    model.embed_into(9999, &mut output);

    // Should be zeroed for out-of-vocab
    assert!(output.iter().all(|&x| x == 0.0));
}

// =============================================================================
// Fused Matmul Q4_0 Tests
// =============================================================================

fn create_q4_0_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q4_0: 18 bytes per 32 elements
    let num_elements = in_dim * out_dim;
    let num_blocks = num_elements.div_ceil(32);
    let byte_size = num_blocks * 18;

    // Create valid Q4_0 data
    let mut data = Vec::with_capacity(byte_size);
    for _ in 0..num_blocks {
        // f16 scale (2 bytes)
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 16 bytes of quants (32 4-bit values)
        data.extend([0x88u8; 16]); // Mid-range values
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_0,
    }
}

#[test]
fn test_phase34_fused_matmul_q4_0_single_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);

    let in_dim = 64;
    let out_dim = 128;
    let input = vec![1.0f32; in_dim]; // Single sequence position
    let weight = create_q4_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q4_0 fused_matmul failed: {:?}",
        result.err()
    );

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_phase34_fused_matmul_q4_0_multi_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);

    let in_dim = 64;
    let out_dim = 128;
    let seq_len = 4;
    let input = vec![1.0f32; in_dim * seq_len]; // 4 sequence positions
    let weight = create_q4_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q4_0 multi-seq fused_matmul failed: {:?}",
        result.err()
    );

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim * seq_len);
    assert!(output.iter().all(|x| x.is_finite()));
}

// =============================================================================
// Fused Matmul Q8_0 Tests
// =============================================================================

fn create_q8_0_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q8_0: 34 bytes per 32 elements (2 f16 scale + 32 i8 quants)
    let num_elements = in_dim * out_dim;
    let num_blocks = num_elements.div_ceil(32);
    let byte_size = num_blocks * 34;

    let mut data = Vec::with_capacity(byte_size);
    for _ in 0..num_blocks {
        // f16 scale (2 bytes)
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        // 32 i8 quants
        data.extend([0i8 as u8; 32]);
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q8_0,
    }
}

#[test]
fn test_phase34_fused_matmul_q8_0_single_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);

    let in_dim = 64;
    let out_dim = 128;
    let input = vec![1.0f32; in_dim];
    let weight = create_q8_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q8_0 fused_matmul failed: {:?}",
        result.err()
    );

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_phase34_fused_matmul_q8_0_multi_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);

    let in_dim = 64;
    let out_dim = 128;
    let seq_len = 3;
    let input = vec![0.5f32; in_dim * seq_len];
    let weight = create_q8_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    assert_eq!(
        result.expect("test value should be present").len(),
        out_dim * seq_len
    );
}

// =============================================================================
// Fused Matmul Q4_1 Tests
// =============================================================================

fn create_q4_1_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q4_1: 20 bytes per 32 elements (2 scale + 2 min + 16 quants)
    let num_elements = in_dim * out_dim;
    let num_blocks = num_elements.div_ceil(32);
    let byte_size = num_blocks * 20;

    let mut data = Vec::with_capacity(byte_size);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        let min = half::f16::from_f32(0.0);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend_from_slice(&min.to_le_bytes());
        data.extend([0x00u8; 16]);
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_1,
    }
}

include!("phase34_fused.rs");
include!("phase34_embed.rs");
