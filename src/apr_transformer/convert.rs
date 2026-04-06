//! APR Transformer Format Conversion (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.

/// Convert from `GGUFTransformer` to APR format
///
/// This dequantizes all GGUF weights to F32 for WASM compatibility.
#[cfg(feature = "default")]
use super::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
#[cfg(feature = "default")]
impl From<&crate::gguf::GGUFTransformer> for AprTransformer {
    fn from(gguf: &crate::gguf::GGUFTransformer) -> Self {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
            eos_token_id: gguf.config.eos_token_id,
            ..Default::default()
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
                attn_q_norm_weight: l.attn_q_norm_weight.clone(),
                attn_k_norm_weight: l.attn_k_norm_weight.clone(),
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
            })
            .collect();

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }
}
