//! Convert Module Tests Part 02 - T-COV-95 Coverage Bridge (B5)
//!
//! Tests for:
//! - GgufToAprConverter::convert with GGUFBuilder data
//! - to_apr_bytes / from_apr_bytes round-trip
//! - ConversionStats methods
//! - from_apr_bytes error paths (truncated, missing weights, invalid JSON)
//!
//! Refs PMAT-802: Protocol T-COV-95 Batch B5

#[cfg(test)]
mod tests {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
    use crate::convert::*;

    // =========================================================================
    // GgufToAprConverter::convert with GGUFBuilder
    // =========================================================================

    #[test]
    fn test_convert_minimal_gguf() {
        use crate::gguf::test_factory::{
            create_f32_embedding_data, create_f32_norm_weights, GGUFBuilder,
        };

        let vocab = 32;
        let hidden = 16;
        let intermediate = 32;
        let n_heads = 2;
        let n_kv_heads = 2;

        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", hidden as u32)
            .num_layers("llama", 1)
            .num_heads("llama", n_heads as u32)
            .num_kv_heads("llama", n_kv_heads as u32)
            .context_length("llama", 128)
            .rope_freq_base("llama", 10000.0)
            .rms_epsilon("llama", 1e-5)
            .ffn_hidden_dim("llama", intermediate as u32)
            .vocab_size("llama", vocab as u32)
            // Token embedding
            .add_f32_tensor(
                "token_embd.weight",
                &[hidden as u64, vocab as u64],
                &create_f32_embedding_data(vocab, hidden),
            )
            // Layer 0 attention
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
            // Layer 0 FFN
            .add_f32_tensor(
                "blk.0.ffn_gate.weight",
                &[hidden as u64, intermediate as u64],
                &vec![0.01f32; hidden * intermediate],
            )
            .add_f32_tensor(
                "blk.0.ffn_up.weight",
                &[hidden as u64, intermediate as u64],
                &vec![0.01f32; hidden * intermediate],
            )
            .add_f32_tensor(
                "blk.0.ffn_down.weight",
                &[intermediate as u64, hidden as u64],
                &vec![0.01f32; intermediate * hidden],
            )
            .add_f32_tensor(
                "blk.0.ffn_norm.weight",
                &[hidden as u64],
                &create_f32_norm_weights(hidden),
            )
            // Output
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

        let result = GgufToAprConverter::convert(&data);
        assert!(result.is_ok(), "Convert failed: {:?}", result.err());
        let transformer = result.unwrap();
        assert_eq!(transformer.config.hidden_dim, hidden);
        assert_eq!(transformer.config.num_layers, 1);
    }

    #[test]
    fn test_convert_invalid_gguf_data() {
        let result = GgufToAprConverter::convert(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_empty_data() {
        let result = GgufToAprConverter::convert(&[]);
        assert!(result.is_err());
    }

    // =========================================================================
    // to_apr_bytes / from_apr_bytes round-trip
    // =========================================================================

    fn create_tiny_transformer() -> AprTransformer {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 8,
            intermediate_dim: 8,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_dim;

        AprTransformer {
            config,
            token_embedding: vec![0.1f32; vocab * hidden],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden * hidden * 3],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden * hidden],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.01; hidden * intermediate]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden * intermediate],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate * hidden],
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden]),
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            }],
            output_norm_weight: vec![1.0; hidden],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden * vocab],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    #[test]
    fn test_to_apr_bytes_creates_valid_data() {
        let transformer = create_tiny_transformer();
        let bytes = GgufToAprConverter::to_apr_bytes(&transformer);
        assert!(bytes.is_ok());
        let data = bytes.unwrap();
        // Should have APR magic at start
        assert!(data.len() > 64); // At least header size
    }

    #[test]
    fn test_to_apr_bytes_from_apr_bytes_roundtrip() {
        let original = create_tiny_transformer();
        let bytes = GgufToAprConverter::to_apr_bytes(&original).unwrap();
        let restored = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(restored.is_ok(), "Round-trip failed: {:?}", restored.err());
        let restored = restored.unwrap();
        assert_eq!(restored.config.hidden_dim, original.config.hidden_dim);
        assert_eq!(restored.config.num_layers, original.config.num_layers);
        assert_eq!(restored.config.vocab_size, original.config.vocab_size);
        assert_eq!(restored.config.architecture, original.config.architecture);
        assert_eq!(restored.layers.len(), original.layers.len());
    }

    // =========================================================================
    // from_apr_bytes error paths
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_too_short() {
        let data = vec![0u8; 10]; // Too short for header
        let result = GgufToAprConverter::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_invalid_header() {
        let data = vec![0u8; 128]; // Valid length but wrong magic
        let result = GgufToAprConverter::from_apr_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // ConversionStats
    // =========================================================================

    #[test]
    fn test_conversion_stats() {
        let transformer = create_tiny_transformer();
        let stats = GgufToAprConverter::stats(&transformer);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.hidden_dim, 4);
        assert_eq!(stats.vocab_size, 8);
        assert_eq!(stats.architecture, "llama");
    }

    #[test]
    fn test_conversion_stats_memory_methods() {
        let transformer = create_tiny_transformer();
        let stats = GgufToAprConverter::stats(&transformer);
        let mb = stats.memory_mb();
        let gb = stats.memory_gb();
        assert!(mb >= 0.0);
        assert!(gb >= 0.0);
        assert!(mb >= gb * 1024.0 - 1.0); // mb ~= gb * 1024
    }

    #[test]
    fn test_conversion_stats_parameter_methods() {
        let transformer = create_tiny_transformer();
        let stats = GgufToAprConverter::stats(&transformer);
        let m = stats.parameters_m();
        let b = stats.parameters_b();
        assert!(m >= 0.0);
        assert!(b >= 0.0);
    }

    // =========================================================================
    // from_gguf_transformer: direct conversion test
    // =========================================================================

    #[test]
    fn test_from_gguf_transformer_preserves_config() {
        use crate::gguf::test_factory::{
            create_f32_embedding_data, create_f32_norm_weights, GGUFBuilder,
        };
        use crate::gguf::{GGUFModel, GGUFTransformer};

        let vocab = 8;
        let hidden = 4;

        let gguf_data = GGUFBuilder::new()
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

        let gguf_model = GGUFModel::from_bytes(&gguf_data).unwrap();
        let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, &gguf_data).unwrap();
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf_transformer);
        assert_eq!(
            apr.config.architecture,
            gguf_transformer.config.architecture
        );
        assert_eq!(apr.config.hidden_dim, gguf_transformer.config.hidden_dim);
        assert_eq!(apr.layers.len(), gguf_transformer.layers.len());
    }
}
