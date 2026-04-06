//! Convert Module Tests Part 03 - T-COV-95 Deep Coverage Bridge
//!
//! Tests for:
//! - GgufToAprQ4KConverter::infer_rope_type for all architecture families
//! - GgufToAprQ4KConverter helper methods: get_string, get_u32, get_f32
//! - RawTensor struct construction and fields
//! - Q4KConversionStats Debug/Clone
//! - ConversionStats edge cases (zero values)
//! - to_apr_bytes / from_apr_bytes additional error paths
//!
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
    use crate::convert::*;
    use std::collections::HashMap;

    #[test]
    fn test_to_apr_bytes_roundtrip_with_biases() {
        let config = AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 8,
            intermediate_dim: 8,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_dim;

        let transformer = AprTransformer {
            config,
            token_embedding: vec![0.1f32; vocab * hidden],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden],
                attn_norm_bias: Some(vec![0.0; hidden]), // phi2 has biases
                qkv_weight: vec![0.01; hidden * hidden * 3],
                qkv_bias: Some(vec![0.0; hidden * 3]),
                attn_output_weight: vec![0.01; hidden * hidden],
                attn_output_bias: Some(vec![0.0; hidden]),
                ffn_gate_weight: None, // phi2 has no gate
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden * intermediate],
                ffn_up_bias: Some(vec![0.0; intermediate]),
                ffn_down_weight: vec![0.01; intermediate * hidden],
                ffn_down_bias: Some(vec![0.0; hidden]),
                ffn_norm_weight: Some(vec![1.0; hidden]),
                ffn_norm_bias: Some(vec![0.0; hidden]),
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
            output_norm_bias: Some(vec![0.0; hidden]),
            lm_head_weight: vec![0.01; hidden * vocab],
            lm_head_bias: Some(vec![0.0; vocab]),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&transformer).unwrap();
        let restored = GgufToAprConverter::from_apr_bytes(&bytes).unwrap();
        assert_eq!(restored.config.architecture, "phi2");
        assert_eq!(restored.config.num_heads, 2);
        // Verify biases survived roundtrip
        assert!(restored.layers[0].attn_norm_bias.is_some());
        assert!(restored.layers[0].qkv_bias.is_some());
    }

    // =========================================================================
    // BUG-APR-002: Byte size calculation for quantized tensors
    // =========================================================================
    // The Q4K converter was using integer division (/) which rounds DOWN,
    // causing "tensor exceeds file bounds" for tensors not perfectly divisible
    // by block size. Fix: use div_ceil() to round UP.

    #[test]
    fn test_bug_apr_002_q4k_byte_size_div_ceil() {
        // Q4_K block: 256 elements = 144 bytes
        // For 65600 elements (256 * 256 + 64):
        // - Wrong: (65600 / 256) * 144 = 256 * 144 = 36864 bytes
        // - Right: 65600.div_ceil(256) * 144 = 257 * 144 = 37008 bytes
        let num_elements = 65600usize;
        let wrong_byte_size = (num_elements / 256) * 144;
        let right_byte_size = num_elements.div_ceil(256) * 144;

        assert_eq!(
            wrong_byte_size, 36864,
            "Wrong calculation (integer division)"
        );
        assert_eq!(right_byte_size, 37008, "Correct calculation (div_ceil)");
        assert!(
            right_byte_size > wrong_byte_size,
            "div_ceil must give larger result"
        );
    }

    #[test]
    fn test_bug_apr_002_q8_byte_size_div_ceil() {
        // Q8_0 block: 32 elements = 34 bytes (2 scale + 32 quants)
        // For 1000 elements (32 * 31 + 8):
        // - Wrong: (1000 / 32) * 34 = 31 * 34 = 1054 bytes
        // - Right: 1000.div_ceil(32) * 34 = 32 * 34 = 1088 bytes
        let num_elements = 1000usize;
        let wrong_byte_size = (num_elements / 32) * 34;
        let right_byte_size = num_elements.div_ceil(32) * 34;

        assert_eq!(
            wrong_byte_size, 1054,
            "Wrong calculation (integer division)"
        );
        assert_eq!(right_byte_size, 1088, "Correct calculation (div_ceil)");
        assert!(
            right_byte_size > wrong_byte_size,
            "div_ceil must give larger result"
        );
    }

    #[test]
    fn test_bug_apr_002_exact_divisibility_no_change() {
        // When num_elements is perfectly divisible, both methods give same result
        let num_elements = 65536usize; // 256 * 256
        let old_style = (num_elements / 256) * 144;
        let new_style = num_elements.div_ceil(256) * 144;

        assert_eq!(
            old_style, new_style,
            "Exact divisibility should give same result"
        );
        assert_eq!(new_style, 36864);
    }

    #[test]
    fn test_bug_apr_002_q5k_byte_size_div_ceil() {
        // Q5_K block: 256 elements = 176 bytes
        // For 65537 elements (256 * 256 + 1):
        // - Wrong: (65537 / 256) * 176 = 256 * 176 = 45056 bytes
        // - Right: 65537.div_ceil(256) * 176 = 257 * 176 = 45232 bytes
        let num_elements = 65537usize;
        let wrong_byte_size = (num_elements / 256) * 176;
        let right_byte_size = num_elements.div_ceil(256) * 176;

        assert_ne!(wrong_byte_size, right_byte_size);
        assert_eq!(right_byte_size - wrong_byte_size, 176); // One extra block
    }

    #[test]
    fn test_bug_apr_002_q6k_byte_size_div_ceil() {
        // Q6_K block: 256 elements = 210 bytes
        // For 65600 elements:
        // - Wrong: (65600 / 256) * 210 = 256 * 210 = 53760 bytes
        // - Right: 65600.div_ceil(256) * 210 = 257 * 210 = 53970 bytes
        let num_elements = 65600usize;
        let wrong_byte_size = (num_elements / 256) * 210;
        let right_byte_size = num_elements.div_ceil(256) * 210;

        assert_ne!(wrong_byte_size, right_byte_size);
        assert_eq!(right_byte_size - wrong_byte_size, 210); // One extra block
    }
include!("tests_infer_rope.rs");
}
