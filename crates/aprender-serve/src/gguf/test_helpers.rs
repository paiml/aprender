//! Test helpers for GGUF module testing
//!
//! This module contains shared test utilities that are used across
//! multiple test files in the GGUF module shatter.
//!
//! ## Contents
//!
//! - `create_test_model_with_config`: Creates a minimal OwnedQuantizedModel for testing
//! - `create_q4k_test_data`: Creates Q4_K quantized tensor data for testing
//!
//! ## Usage
//!
//! ```rust,ignore
//! #[cfg(test)]
//! use crate::gguf::test_helpers::{create_test_model_with_config, create_q4k_test_data};
//! ```

use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
};

/// Create a test model with specific configuration
///
/// This helper creates a minimal `OwnedQuantizedModel` with deterministic
/// test weights for verifying attention, FFN, and other model operations.
///
/// # Arguments
///
/// * `config` - The model configuration to use
///
/// # Returns
///
/// An `OwnedQuantizedModel` with test weights
pub(crate) fn create_test_model_with_config(config: &GGUFConfig) -> OwnedQuantizedModel {
    // Create minimal model weights for testing.
    // GH-278: Model structure is contract-consistent — gate weights, biases,
    // position embeddings, and FFN norms are created based on ArchConstraints.
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let constraints = &config.constraints;

    // QKV projection: hidden_dim -> hidden_dim + 2*kv_dim (Q + K + V)
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight = create_q4k_test_data(hidden_dim, qkv_out_dim);

    // Output projection: hidden_dim -> hidden_dim
    let attn_output_weight = create_q4k_test_data(hidden_dim, hidden_dim);

    // FFN weights
    let ffn_up_weight = create_q4k_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_q4k_test_data(intermediate_dim, hidden_dim);

    // Gate weight: present for gated FFN architectures (SwiGLU, GatedMLP)
    let ffn_gate_weight = if constraints.has_gate_ffn() {
        Some(create_q4k_test_data(hidden_dim, intermediate_dim))
    } else {
        None
    };

    // Layer norm weights and biases (contract-driven)
    let attn_norm_weight = vec![1.0f32; hidden_dim];
    let attn_norm_bias = if !constraints.uses_rmsnorm() {
        Some(vec![0.0f32; hidden_dim])
    } else {
        None
    };

    let layer = OwnedQuantizedLayer {
        attn_norm_weight,
        attn_norm_bias,
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let token_embedding = vec![0.1f32; vocab_size * hidden_dim];
    let output_norm_weight = vec![1.0f32; hidden_dim];
    let lm_head_weight = create_q4k_test_data(hidden_dim, vocab_size);

    // Position embedding: present for absolute position encoding (GPT-2, BERT, whisper)
    let position_embedding = if constraints.uses_absolute_positions() {
        Some(vec![0.01f32; config.context_length * hidden_dim])
    } else {
        None
    };

    // Output norm bias for LayerNorm models
    let output_norm_bias = if !constraints.uses_rmsnorm() {
        Some(vec![0.0f32; hidden_dim])
    } else {
        None
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding,
        position_embedding,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight,
        output_norm_bias,
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

/// Create Q4_K test data for given dimensions
///
/// Q4_K uses row-major storage where each row has ceil(in_dim/256) super-blocks.
/// Each super-block is 144 bytes and covers 256 values.
///
/// # Arguments
///
/// * `in_dim` - Input dimension (number of columns)
/// * `out_dim` - Output dimension (number of rows)
///
/// # Returns
///
/// An `OwnedQuantizedTensor` with Q4_K quantized test data
pub(crate) fn create_q4k_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Row-major storage: each row needs ceil(in_dim/256) super-blocks
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
            // Fill scales and quantized values with deterministic test pattern
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_q4k_test_data_basic() {
        let tensor = create_q4k_test_data(256, 64);
        assert_eq!(tensor.in_dim, 256);
        assert_eq!(tensor.out_dim, 64);
        assert_eq!(tensor.qtype, 12); // Q4_K
                                      // 1 super-block per row, 144 bytes each, 64 rows
        assert_eq!(tensor.data.len(), 64 * 144);
    }

    #[test]
    fn test_create_q4k_test_data_multi_superblock() {
        // 512 values needs 2 super-blocks per row
        let tensor = create_q4k_test_data(512, 32);
        assert_eq!(tensor.in_dim, 512);
        assert_eq!(tensor.out_dim, 32);
        // 2 super-blocks per row, 144 bytes each, 32 rows
        assert_eq!(tensor.data.len(), 32 * 2 * 144);
    }

    #[test]
    fn test_create_test_model_with_config_basic() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.token_embedding.len(), 100 * 64);
    }

    #[test]
    fn test_create_q4k_test_data_small_input() {
        // Less than 256 still needs 1 super-block
        let tensor = create_q4k_test_data(64, 16);
        assert_eq!(tensor.in_dim, 64);
        assert_eq!(tensor.out_dim, 16);
        // ceil(64/256) = 1 super-block per row
        assert_eq!(tensor.data.len(), 16 * 144);
    }

    #[test]
    fn test_create_q4k_test_data_large_dimensions() {
        let tensor = create_q4k_test_data(1024, 128);
        assert_eq!(tensor.in_dim, 1024);
        assert_eq!(tensor.out_dim, 128);
        // ceil(1024/256) = 4 super-blocks per row
        assert_eq!(tensor.data.len(), 128 * 4 * 144);
    }

    #[test]
    fn test_create_q4k_test_data_d_value() {
        // Check that d=1.0 is properly encoded in f16
        let tensor = create_q4k_test_data(256, 1);
        // First 2 bytes of first super-block should be 0x3C00 (1.0 in f16)
        assert_eq!(tensor.data[0], 0x00);
        assert_eq!(tensor.data[1], 0x3C);
    }

    #[test]
    fn test_create_q4k_test_data_dmin_value() {
        // Check that dmin=0 is properly encoded
        let tensor = create_q4k_test_data(256, 1);
        // Bytes 2-3 of first super-block should be 0x0000
        assert_eq!(tensor.data[2], 0x00);
        assert_eq!(tensor.data[3], 0x00);
    }

    #[test]
    fn test_create_test_model_with_config_gqa() {
        // Test with GQA (different num_heads and num_kv_heads)
        let config = GGUFConfig {
            architecture: "gqa_test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("gqa_test"),
            hidden_dim: 128,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4 Q heads per KV head
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        assert_eq!(model.config.num_heads, 8);
        assert_eq!(model.config.num_kv_heads, 2);
    }

    #[test]
    fn test_create_test_model_output_norm() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 32,
            intermediate_dim: 64,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            vocab_size: 50,
            rope_theta: 10000.0,
            context_length: 256,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        assert_eq!(model.output_norm_weight.len(), 32);
        // Output norm should be initialized to 1.0
        assert!(model
            .output_norm_weight
            .iter()
            .all(|&w| (w - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_create_test_model_lm_head() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        // LM head maps hidden_dim -> vocab_size
        assert_eq!(model.lm_head_weight.in_dim, 64);
        assert_eq!(model.lm_head_weight.out_dim, 100);
    }

    #[test]
    fn test_create_test_model_layer_attn_norm() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 48,
            intermediate_dim: 96,
            num_heads: 3,
            num_kv_heads: 3,
            num_layers: 1,
            vocab_size: 50,
            rope_theta: 10000.0,
            context_length: 256,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        assert_eq!(model.layers[0].attn_norm_weight.len(), 48);
        // Attention norm should be initialized to 1.0
        assert!(model.layers[0]
            .attn_norm_weight
            .iter()
            .all(|&w| (w - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_create_q4k_test_data_boundary() {
        // Test exactly at super-block boundary (256)
        let tensor = create_q4k_test_data(256, 1);
        assert_eq!(tensor.in_dim, 256);
        let expected_size = 144; // 1 row * 1 super-block * 144 bytes
        assert_eq!(tensor.data.len(), expected_size);
    }

    #[test]
    fn test_create_q4k_test_data_just_over_boundary() {
        // Test just over super-block boundary (257 needs 2 super-blocks)
        let tensor = create_q4k_test_data(257, 1);
        assert_eq!(tensor.in_dim, 257);
        let expected_size = 2 * 144; // 1 row * 2 super-blocks * 144 bytes
        assert_eq!(tensor.data.len(), expected_size);
    }
}
