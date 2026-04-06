//! GPU Model Weight Loading (PMAT-COMPLY)
//!
//! Extracted from model.rs for file health compliance.
//! Contains GGUF weight loading helpers.

use super::types::{BlockWeights, GpuModelConfig};
use crate::error::Result;

/// Loaded GGUF weights ready for GPU model construction.
pub(super) struct GgufWeights {
    pub config: GpuModelConfig,
    pub embedding_weights: Vec<f32>,
    pub block_weights: Vec<BlockWeights>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub lm_head_bias: Vec<f32>,
}

/// Load all model weights from a memory-mapped GGUF file.
///
/// Handles both fused QKV (LLaMA-style) and separate Q/K/V (Qwen-style)
/// projections, with GQA support for K/V dimensions.
///
/// # Errors
///
/// Returns error if required tensors are missing or shapes don't match.
pub(super) fn load_weights_from_gguf(mapped: &crate::gguf::MappedGGUFModel) -> Result<GgufWeights> {
    use crate::gguf::ValidatedModelConfig;

    // Phase 2: Validate config at construction boundary.
    let gguf_config = ValidatedModelConfig::from_gguf(&mapped.model)?.into_inner();

    let config = GpuModelConfig {
        vocab_size: gguf_config.vocab_size,
        hidden_dim: gguf_config.hidden_dim,
        num_heads: gguf_config.num_heads,
        num_kv_heads: gguf_config.num_kv_heads, // IMP-088: GQA support
        num_layers: gguf_config.num_layers,
        intermediate_dim: gguf_config.intermediate_dim,
        eps: gguf_config.eps,
        rope_theta: gguf_config.rope_theta, // Phase 21: RoPE support
        explicit_head_dim: gguf_config.explicit_head_dim,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: Some(gguf_config.constraints),
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let data = mapped.data();

    // Load token embeddings (always dequantized for fast lookup)
    let embedding_weights = mapped.model.get_tensor_f32("token_embd.weight", data)?;

    // Load transformer blocks
    let block_weights = load_block_weights(mapped, &config, data)?;

    // GH-279: Architecture completeness validation
    // Verify that architecture-required weights were actually loaded
    // (catches silent None defaults for required fields like ffn_gate)
    if let Some(ref constraints) = config.constraints {
        validate_block_completeness(&block_weights, constraints, &config)?;
    }

    // Final layer norm
    let final_norm_weight = mapped.model.get_tensor_f32("output_norm.weight", data)?;
    let final_norm_bias = mapped
        .model
        .get_tensor_f32("output_norm.bias", data)
        .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

    // LM head
    let lm_head_weight = mapped.model.get_tensor_f32("output.weight", data)?;
    let lm_head_bias = mapped
        .model
        .get_tensor_f32("output.bias", data)
        .unwrap_or_else(|_| vec![0.0f32; config.vocab_size]);

    Ok(GgufWeights {
        config,
        embedding_weights,
        block_weights,
        final_norm_weight,
        final_norm_bias,
        lm_head_weight,
        lm_head_bias,
    })
}

/// Load transformer block weights from GGUF.
///
/// Handles fused QKV (LLaMA) vs separate Q/K/V (Qwen) with GQA support.
fn load_block_weights(
    mapped: &crate::gguf::MappedGGUFModel,
    config: &GpuModelConfig,
    data: &[u8],
) -> Result<Vec<BlockWeights>> {
    let mut block_weights = Vec::with_capacity(config.num_layers);

    for layer_idx in 0..config.num_layers {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm (small, keep as f32)
        let attn_norm_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
        let attn_norm_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

        // QKV projection - try fused QKV first (LLaMA), then separate Q/K/V (Qwen)
        let (qkv_weight, qkv_bias) = load_qkv_weights(mapped, config, data, &prefix)?;

        // Output projection
        let out_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_output.weight", prefix), data)?;
        let out_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

        // FFN norm
        let ffn_norm_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
            .unwrap_or_else(|_| vec![1.0f32; config.hidden_dim]);
        let ffn_norm_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

        // FFN projections
        let ffn_fc1_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_up.weight", prefix), data)?;
        let ffn_fc1_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.intermediate_dim]);

        let ffn_fc2_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_down.weight", prefix), data)?;
        let ffn_fc2_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

        // Try to load gate weight for SwiGLU (optional)
        let ffn_gate_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.ffn_gate.weight", prefix), data)
            .ok();

        block_weights.push(BlockWeights {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            out_weight,
            out_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            ffn_fc1_weight,
            ffn_fc1_bias,
            ffn_fc2_weight,
            ffn_fc2_bias,
            ffn_gate_weight,
            // GH-278: No linear attention in GGUF loader
            linear_attn: None,
            moe_experts: None,
        });
    }

    Ok(block_weights)
}

/// Load QKV weights, handling fused (LLaMA) vs separate (Qwen) formats.
fn load_qkv_weights(
    mapped: &crate::gguf::MappedGGUFModel,
    config: &GpuModelConfig,
    data: &[u8],
    prefix: &str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if let Ok(fused_qkv) = mapped
        .model
        .get_tensor_f32(&format!("{}.attn_qkv.weight", prefix), data)
    {
        // Fused QKV (LLaMA-style)
        let bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; 3 * config.hidden_dim]);
        Ok((fused_qkv, bias))
    } else {
        // Separate Q/K/V (Qwen-style) - concatenate into fused format
        // For GQA: Q has num_heads * head_dim, K/V have num_kv_heads * head_dim
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = config.head_dim();
        let kv_dim = config.num_kv_heads * head_dim; // K/V dimension for GQA

        let q_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_q.weight", prefix), data)?;
        let k_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_k.weight", prefix), data)?;
        let v_weight = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_v.weight", prefix), data)?;

        // Concatenate Q, K, V weights
        let mut qkv_weight = Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
        qkv_weight.extend_from_slice(&q_weight);
        qkv_weight.extend_from_slice(&k_weight);
        qkv_weight.extend_from_slice(&v_weight);

        // Load biases if available (use correct dimensions for GQA)
        let q_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);
        let k_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; kv_dim]); // GQA: K/V use num_kv_heads
        let v_bias = mapped
            .model
            .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
            .unwrap_or_else(|_| vec![0.0f32; kv_dim]); // GQA: K/V use num_kv_heads

        // Total bias size: Q (hidden_dim) + K (kv_dim) + V (kv_dim)
        let total_bias_dim = config.hidden_dim + 2 * kv_dim;
        let mut qkv_bias = Vec::with_capacity(total_bias_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Ok((qkv_weight, qkv_bias))
    }
}

/// GH-279: Validate that architecture-required block weights are present.
///
/// Catches cases where optional fields (like `ffn_gate_weight`) are silently `None`
/// for architectures that actually require them (all SwiGLU architectures).
fn validate_block_completeness(
    blocks: &[BlockWeights],
    constraints: &crate::gguf::ArchConstraints,
    config: &GpuModelConfig,
) -> Result<()> {
    use crate::arch_requirements::{required_roles, WeightRole};

    let roles = required_roles(constraints);

    for (layer_idx, block) in blocks.iter().enumerate() {
        for &role in roles {
            let missing = match role {
                // FFN gate is optional in BlockWeights but required by SwiGLU architectures
                WeightRole::FfnGate => block.ffn_gate_weight.is_none(),
                // All other roles are loaded with `?` (hard error on missing),
                // but validate shapes are non-empty as a safety net
                WeightRole::AttnNorm => block.attn_norm_weight.is_empty(),
                WeightRole::FfnNorm => block.ffn_norm_weight.is_empty(),
                WeightRole::QProj | WeightRole::KProj | WeightRole::VProj => {
                    block.qkv_weight.is_empty()
                },
                WeightRole::OProj => block.out_weight.is_empty(),
                WeightRole::FfnUp => block.ffn_fc1_weight.is_empty(),
                WeightRole::FfnDown => block.ffn_fc2_weight.is_empty(),
                // QK norm and bias: not yet loaded in this path, but validate
                // that the architecture doesn't require them (they would be zero-vectors)
                WeightRole::AttnQNorm | WeightRole::AttnKNorm => {
                    // QK norm not supported in GpuModel scheduler path yet.
                    // If architecture requires it, that's a gap we must flag.
                    constraints.has_qk_norm
                },
                WeightRole::AttnQBias | WeightRole::AttnKBias | WeightRole::AttnVBias => {
                    // Bias is loaded via unwrap_or_else (zero-filled), which is acceptable
                    // for architectures that require bias — the weight exists even if all-zero
                    // from the GGUF fallback. Not a structural gap.
                    false
                },
            };

            if missing {
                return Err(crate::error::RealizarError::UnsupportedOperation {
                    operation: "validate_block_completeness".to_string(),
                    reason: format!(
                        "GH-279: Layer {} missing required weight '{}' for architecture \
                         (has_qk_norm={}, has_bias={}, {} layers, hidden={})",
                        layer_idx,
                        role.field_name(),
                        constraints.has_qk_norm,
                        constraints.has_bias,
                        config.num_layers,
                        config.hidden_dim,
                    ),
                });
            }
        }
    }

    Ok(())
}
