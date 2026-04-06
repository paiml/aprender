//! GGUF Loader Tests Part 03: to_apr_bytes + from_apr Coverage
//!
//! This module provides comprehensive test coverage for:
//! - `OwnedQuantizedModel::to_apr_bytes()` - serialize quantized model to APR v2
//! - `OwnedQuantizedModel::from_apr()` - load from MappedAprModel
//! - Internal helpers: qtype_to_dtype, dtype_to_byte, write_tensor_entry
//! - Separate vs Fused QKV paths in serialization
//! - FFN gate/norm presence/absence branches
//! - Roundtrip: to_apr_bytes -> write to disk -> MappedAprModel -> from_apr

use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
};

// ============================================================================
// Helper: Build a minimal OwnedQuantizedModel for testing
// ============================================================================

/// Create a tiny OwnedQuantizedModel with F32 weights (qtype=0) and separate QKV
fn build_minimal_owned_quantized_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    // Token embedding (F32): vocab_size * hidden_dim floats -> 4 bytes each
    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    // Create F32 data (qtype 0) for weight tensors
    fn make_f32_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let data: Vec<u8> = (0..in_dim * out_dim)
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
            .collect();
        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: 0, // F32
        }
    }

    // Separate Q, K, V
    let q_weight = make_f32_tensor(hidden_dim, hidden_dim);
    let k_weight = make_f32_tensor(hidden_dim, kv_dim);
    let v_weight = make_f32_tensor(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: make_f32_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_f32_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_f32_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(make_f32_tensor(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_f32_tensor(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a model with Fused QKV (phi2 style)
fn build_fused_qkv_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;

    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 2,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    fn make_f32_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let data: Vec<u8> = (0..in_dim * out_dim)
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
            .collect();
        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype: 0,
        }
    }

    // Fused QKV: hidden -> 3 * hidden
    let qkv_dim = 3 * hidden_dim;
    let fused_qkv = OwnedQuantizedTensor {
        data: vec![0u8; hidden_dim * qkv_dim * 4], // F32
        in_dim: hidden_dim,
        out_dim: qkv_dim,
        qtype: 0,
    };

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(fused_qkv),
        qkv_bias: None,
        attn_output_weight: make_f32_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_f32_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_f32_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: None, // No gate for phi2-style GELU
        ffn_gate_bias: None,
        ffn_norm_weight: None, // Some models skip FFN norm
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: make_f32_tensor(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Create a model with Q4_K quantized weights (qtype=12)
fn build_q4k_model() -> OwnedQuantizedModel {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let vocab_size = 10;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "qwen2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        rope_type: 2,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    // Q4_K: 144 bytes per 256 elements
    fn make_q4k_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let num_elements = in_dim * out_dim;
        let num_super_blocks = num_elements.div_ceil(256);
        OwnedQuantizedTensor {
            data: vec![0u8; num_super_blocks * 144],
            in_dim,
            out_dim,
            qtype: 12, // Q4_K
        }
    }

    let q_weight = make_q4k_tensor(hidden_dim, hidden_dim);
    let k_weight = make_q4k_tensor(hidden_dim, kv_dim);
    let v_weight = make_q4k_tensor(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: make_q4k_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: make_q4k_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: make_q4k_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(make_q4k_tensor(hidden_dim, intermediate_dim)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    // Q6_K lm_head: 210 bytes per 256 elements
    let lm_head_elements = hidden_dim * vocab_size;
    let lm_head_sb = lm_head_elements.div_ceil(256);
    let lm_head = OwnedQuantizedTensor {
        data: vec![0u8; lm_head_sb * 210],
        in_dim: hidden_dim,
        out_dim: vocab_size,
        qtype: 14, // Q6_K
    };

    OwnedQuantizedModel {
        config,
        token_embedding: embed_data,
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: lm_head,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

// ============================================================================
// to_apr_bytes: Basic tests
// ============================================================================

#[test]
fn test_to_apr_bytes_produces_valid_header() {
    let model = build_minimal_owned_quantized_model();
    let result = model.to_apr_bytes();
    assert!(result.is_ok(), "to_apr_bytes failed: {:?}", result.err());

    let bytes = result.expect("should produce bytes");

    // Check APR magic
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00], "Magic bytes");
    // Check version
    assert_eq!(bytes[4], 2, "Major version");
    assert_eq!(bytes[5], 0, "Minor version");
    // Must be at least 64 bytes (header)
    assert!(bytes.len() >= 64, "Must include full header");
}

#[test]
fn test_to_apr_bytes_tensor_count() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // tensor_count is at offset [8..12]
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

    // Separate QKV = 3 tensors per layer, plus:
    // token_embd, attn_norm, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down, output_norm, output
    // = 1 + (1 + 3 + 1 + 1 + 1 + 1 + 1) * 1 + 1 + 1 = 12 tensors for 1 layer
    assert!(
        tensor_count > 0,
        "Should have at least one tensor, got {}",
        tensor_count
    );
}

#[test]
fn test_to_apr_bytes_metadata_offset() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // metadata_offset at [12..20] should be 64 (HEADER_SIZE)
    let metadata_offset = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    assert_eq!(metadata_offset, 64, "metadata_offset should be HEADER_SIZE");
}

#[test]
fn test_to_apr_bytes_data_after_index() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // tensor_index_offset at [24..32]
    let tensor_index_offset = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);
    // data_offset at [32..40]
    let data_offset = u64::from_le_bytes([
        bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
    ]);

    // data must come after tensor index
    assert!(
        data_offset >= tensor_index_offset,
        "data_offset ({data_offset}) must be >= tensor_index_offset ({tensor_index_offset})"
    );
}

// ============================================================================
// to_apr_bytes: Separate vs Fused QKV paths
// ============================================================================

#[test]
fn test_to_apr_bytes_separate_qkv() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Should contain separate Q, K, V tensor names in the binary index
    // Tensor names are written as: 2-byte len + name bytes
    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_q.weight"),
        "Should have separate Q tensor"
    );
    assert!(
        bytes_str.contains("blk.0.attn_k.weight"),
        "Should have separate K tensor"
    );
    assert!(
        bytes_str.contains("blk.0.attn_v.weight"),
        "Should have separate V tensor"
    );
}

#[test]
fn test_to_apr_bytes_fused_qkv() {
    let model = build_fused_qkv_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_qkv.weight"),
        "Should have fused QKV tensor"
    );
    // Should NOT have separate Q/K/V
    assert!(
        !bytes_str.contains("blk.0.attn_q.weight"),
        "Should NOT have separate Q tensor when fused"
    );
}

// ============================================================================
// to_apr_bytes: FFN gate and norm presence/absence
// ============================================================================

#[test]
fn test_to_apr_bytes_with_ffn_gate_and_norm() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.ffn_gate.weight"),
        "Should have FFN gate weight when present"
    );
    assert!(
        bytes_str.contains("blk.0.ffn_norm.weight"),
        "Should have FFN norm weight when present"
    );
}

#[test]
fn test_to_apr_bytes_without_ffn_gate_or_norm() {
    let model = build_fused_qkv_model(); // No gate or FFN norm
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        !bytes_str.contains("blk.0.ffn_gate.weight"),
        "Should NOT have FFN gate when None"
    );
    assert!(
        !bytes_str.contains("blk.0.ffn_norm.weight"),
        "Should NOT have FFN norm when None"
    );
}

// ============================================================================
// to_apr_bytes: Different quantization types in dtype mapping
// ============================================================================

#[test]
fn test_to_apr_bytes_q4k_model() {
    let model = build_q4k_model();
    let result = model.to_apr_bytes();
    assert!(
        result.is_ok(),
        "Q4K model should serialize: {:?}",
        result.err()
    );

    let bytes = result.expect("should produce bytes");
    // Should have valid APR magic
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00]);
}

include!("loader_tests_apr.rs");
