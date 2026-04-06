//! APR to GpuModel Adapter (PMAT-106)
//!
//! Converts APR transformers to `GpuModel` for GPU inference.
//!
//! # Overview
//!
//! This module provides adapters for both F32 and Q4 APR formats:
//! - [`AprF32ToGpuAdapter`] - For `.apr` files with F32 weights (direct copy)
//! - [`AprToGpuAdapter`] - For GGUF Q4_0 models (dequantizes to F32)
//!
//! # Coverage Impact
//!
//! Testing these adapters exercises:
//! - `apr_transformer/mod.rs` - F32 weight extraction
//! - `apr_transformer/q4_simd.rs` - Q4 weight extraction
//! - `gpu/scheduler/model.rs` - GpuModel creation
//! - `quantize/dequant.rs` - Q4_0 dequantization

use crate::apr_transformer::{
    AprTransformer, AprTransformerConfig, AprTransformerLayer, QuantizedAprLayerQ4,
    QuantizedAprTransformerQ4,
};
use crate::error::Result;
use crate::gpu::scheduler::{
    BlockWeights, GpuModel, GpuModelConfig, LinearAttnWeights, LmHeadWeight,
    LmHeadWeightTransposed, MoeExpertWeights, ValidatedGpuWeights,
};
use crate::quantize::dequantize_q4_0;
use thiserror::Error;

/// Errors during APR to GPU conversion
#[derive(Debug, Error)]
pub enum AprGpuError {
    /// Dequantization failed
    #[error("Failed to dequantize Q4_0 weights: {0}")]
    DequantError(String),

    /// Weight dimension mismatch
    #[error("Weight dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of elements
        expected: usize,
        /// Actual number of elements
        actual: usize,
    },

    /// GpuModel creation failed
    #[error("Failed to create GpuModel: {0}")]
    GpuModelError(String),
}

/// Adapter for converting F32 APR models to GPU format
///
/// Used for `.apr` files which contain F32 weights. No dequantization needed.
pub struct AprF32ToGpuAdapter;

impl AprF32ToGpuAdapter {
    /// Convert F32 APR transformer to GpuModel
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with F32 weights
    ///
    /// # Returns
    ///
    /// `GpuModel` ready for GPU inference
    ///
    /// # Example
    ///
    /// ```ignore
    /// use realizar::apr_transformer::AprTransformer;
    /// use realizar::gpu::adapters::AprF32ToGpuAdapter;
    ///
    /// let apr = AprTransformer::from_apr_bytes(&data)?;
    /// let gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr)?;
    /// ```
    pub fn to_gpu_model(apr: &AprTransformer) -> Result<GpuModel> {
        let config = AprToGpuAdapter::config_to_gpu(&apr.config);
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;

        // Embedding weights (already F32)
        let embedding_weights = apr.token_embedding.clone();

        // LM head weights (already F32)
        // APR get_f32() with transpose_cublas_weights produces [vocab_size, hidden_dim] (HF convention).
        // GpuModel expects lm_head_weight as [hidden_dim, vocab_size] (same as GGUF loader).
        // So the APR weight is the "transposed" form and vice versa.
        let apr_lm_head = apr.lm_head_weight.clone(); // [vocab_size, hidden_dim]
        let apr_lm_head_t = transpose_matrix(&apr_lm_head, config.vocab_size, hidden_dim); // [hidden_dim, vocab_size]
        let lm_head_weight = apr_lm_head_t; // [hidden_dim, vocab_size] — GPU matmul layout (matches GGUF)
        let lm_head_weight_t = apr_lm_head; // [vocab_size, hidden_dim] — CPU matmul layout

        // Convert each layer
        let mut block_weights = Vec::with_capacity(apr.layers.len());
        for (block_idx, layer) in apr.layers.iter().enumerate() {
            block_weights.push(Self::convert_layer(
                layer,
                hidden_dim,
                intermediate_dim,
                config.num_heads,
                config.num_kv_heads,
                &config,
                block_idx,
            ));
        }

        // Final norm
        let final_norm_weight = apr.output_norm_weight.clone();
        let final_norm_bias = apr
            .output_norm_bias
            .clone()
            .unwrap_or_else(|| vec![0.0; hidden_dim]);

        // LM head bias
        let lm_head_bias = apr
            .lm_head_bias
            .clone()
            .unwrap_or_else(|| vec![0.0; config.vocab_size]);

        // PMAT-284: Validated weights — newtypes prevent lm_head swap at compile time
        let validated = ValidatedGpuWeights::new(
            config,
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            LmHeadWeight(lm_head_weight),
            LmHeadWeightTransposed(lm_head_weight_t),
            lm_head_bias,
        )
        .map_err(|e| crate::error::RealizarError::InvalidShape {
            reason: e.to_string(),
        })?;
        GpuModel::from_validated_weights(validated)
    }

    /// Convert a single F32 layer to BlockWeights
    fn convert_layer(
        layer: &AprTransformerLayer,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        config: &GpuModelConfig,
        block_idx: usize,
    ) -> BlockWeights {
        let is_linear = config.is_linear_layer(block_idx);

        // Phase 21 FIX: APR stores weights as [out_dim, in_dim] row-major,
        // but GPU gemm expects [in_dim, out_dim]. Transpose all projection weights.

        // GH-278: Linear layers have different QKV dimensions
        let (qkv_out_dim, out_rows, out_cols) = if is_linear {
            let key_dim = config.linear_key_dim();
            let value_dim = config.linear_value_dim();
            // QKV = Q(key_dim) + K(key_dim) + V(value_dim)
            (key_dim + key_dim + value_dim, hidden_dim, value_dim)
        } else {
            // Use config.head_dim() to handle explicit_head_dim (Qwen3-Coder: 128 vs 2048/32=64)
            let head_dim = config.head_dim();
            let kv_dim = num_kv_heads * head_dim;
            let q_dim = num_heads * head_dim;
            // Output proj: [hidden_dim, q_dim] — maps attention space back to residual
            (q_dim + 2 * kv_dim, hidden_dim, q_dim)
        };

        // Transpose QKV: [qkv_out_dim, hidden_dim] -> [hidden_dim, qkv_out_dim]
        let qkv_weight_t = transpose_matrix(&layer.qkv_weight, qkv_out_dim, hidden_dim);

        // Transpose output projection: [out_rows, out_cols] -> [out_cols, out_rows]
        let out_weight_t = transpose_matrix(&layer.attn_output_weight, out_rows, out_cols);

        // Transpose FFN up (fc1): [intermediate_dim, hidden_dim] -> [hidden_dim, intermediate_dim]
        let fc1_weight_t = transpose_matrix(&layer.ffn_up_weight, intermediate_dim, hidden_dim);

        // Transpose FFN down (fc2): [hidden_dim, intermediate_dim] -> [intermediate_dim, hidden_dim]
        let fc2_weight_t = transpose_matrix(&layer.ffn_down_weight, hidden_dim, intermediate_dim);

        // Transpose gate weight if present: [intermediate_dim, hidden_dim] -> [hidden_dim, intermediate_dim]
        let gate_weight_t = layer
            .ffn_gate_weight
            .as_ref()
            .map(|w| transpose_matrix(w, intermediate_dim, hidden_dim));

        // GH-278: Build LinearAttnWeights when layer has Gated Delta Net weights
        let linear_attn = if is_linear {
            Self::build_linear_attn_weights(layer, config)
        } else {
            None
        };

        BlockWeights {
            attn_norm_weight: layer.attn_norm_weight.clone(),
            attn_norm_bias: layer
                .attn_norm_bias
                .clone()
                .unwrap_or_else(|| vec![0.0; hidden_dim]),
            qkv_weight: qkv_weight_t,
            qkv_bias: layer.qkv_bias.clone().unwrap_or_default(),
            out_weight: out_weight_t,
            out_bias: layer
                .attn_output_bias
                .clone()
                .unwrap_or_else(|| vec![0.0; hidden_dim]),
            // Use actual FFN norm if available, otherwise identity (Phase 21 fix)
            ffn_norm_weight: layer
                .ffn_norm_weight
                .clone()
                .unwrap_or_else(|| vec![1.0; hidden_dim]),
            ffn_norm_bias: layer
                .ffn_norm_bias
                .clone()
                .unwrap_or_else(|| vec![0.0; hidden_dim]),
            ffn_fc1_weight: fc1_weight_t,
            ffn_fc1_bias: layer
                .ffn_up_bias
                .clone()
                .unwrap_or_else(|| vec![0.0; intermediate_dim]),
            ffn_fc2_weight: fc2_weight_t,
            ffn_fc2_bias: layer
                .ffn_down_bias
                .clone()
                .unwrap_or_else(|| vec![0.0; hidden_dim]),
            // SwiGLU gate weight - critical for Qwen/LLaMA models
            ffn_gate_weight: gate_weight_t,
            linear_attn,
            moe_experts: Self::build_moe_weights(layer, config),
        }
    }

    /// ALB-010: Build MoeExpertWeights from AprTransformerLayer fields
    fn build_moe_weights(
        layer: &AprTransformerLayer,
        config: &GpuModelConfig,
    ) -> Option<MoeExpertWeights> {
        let gate_weight = layer.moe_gate_weight.as_ref()?;
        let num_experts = config.num_experts?;
        let num_experts_per_tok = config.num_experts_per_tok?;
        let expert_intermediate = config.expert_intermediate_size?;

        Some(MoeExpertWeights {
            gate_weight: gate_weight.clone(),
            expert_gate_up: layer.moe_expert_gate_up.clone().unwrap_or_default(),
            expert_down: layer.moe_expert_down.clone().unwrap_or_default(),
            shared_gate: layer.moe_shared_gate.clone().unwrap_or_default(),
            shared_up: layer.moe_shared_up.clone().unwrap_or_default(),
            shared_down: layer.moe_shared_down.clone().unwrap_or_default(),
            shared_expert_gate_weight: layer
                .moe_shared_expert_gate_weight
                .clone()
                .unwrap_or_default(),
            num_experts,
            num_experts_per_tok,
            expert_intermediate,
        })
    }

    /// GH-278: Build LinearAttnWeights from AprTransformerLayer fields
    ///
    /// Transposes weight matrices from APR [out_dim, in_dim] to GPU [in_dim, out_dim].
    fn build_linear_attn_weights(
        layer: &AprTransformerLayer,
        config: &GpuModelConfig,
    ) -> Option<LinearAttnWeights> {
        let z = layer.linear_attn_z_weight.as_ref()?;
        let b = layer.linear_attn_b_weight.as_ref()?;
        let a = layer.linear_attn_a_weight.as_ref()?;

        let hidden_dim = config.hidden_dim;
        let value_dim = config.linear_value_dim();
        let num_v_heads = config.linear_num_value_heads.unwrap_or(0);

        Some(LinearAttnWeights {
            // z: [value_dim, hidden_dim] → transpose → [hidden_dim, value_dim]
            z_weight: transpose_matrix(z, value_dim, hidden_dim),
            // b: [num_v_heads, hidden_dim] → transpose → [hidden_dim, num_v_heads]
            b_weight: transpose_matrix(b, num_v_heads, hidden_dim),
            // a: [num_v_heads, hidden_dim] → transpose → [hidden_dim, num_v_heads]
            a_weight: transpose_matrix(a, num_v_heads, hidden_dim),
            // conv1d: [conv_dim, kernel_size] — no transpose needed (1D)
            conv1d_weight: layer.linear_attn_conv1d_weight.clone().unwrap_or_default(),
            // a_log: [num_v_heads] — scalar per head
            a_log: layer.linear_attn_a_log.clone().unwrap_or_default(),
            // dt_bias: [num_v_heads] — scalar per head
            dt_bias: layer.linear_attn_dt_bias.clone().unwrap_or_default(),
            // norm: [value_dim or head_v_dim]
            norm_weight: layer.linear_attn_norm_weight.clone().unwrap_or_default(),
        })
    }
}

/// Adapter for converting Q4 APR models to GPU format
pub struct AprToGpuAdapter;

impl AprToGpuAdapter {
    /// Convert APR config to GPU config
    #[must_use]
    pub fn config_to_gpu(apr_config: &AprTransformerConfig) -> GpuModelConfig {
        GpuModelConfig {
            vocab_size: apr_config.vocab_size,
            hidden_dim: apr_config.hidden_dim,
            num_heads: apr_config.num_heads,
            num_kv_heads: apr_config.num_kv_heads,
            num_layers: apr_config.num_layers,
            intermediate_dim: apr_config.intermediate_dim,
            eps: apr_config.eps,
            rope_theta: apr_config.rope_theta,
            // GH-278: Pass hybrid attention fields through
            explicit_head_dim: apr_config.explicit_head_dim,
            layer_types: apr_config.layer_types.clone(),
            linear_key_head_dim: apr_config.linear_key_head_dim,
            linear_value_head_dim: apr_config.linear_value_head_dim,
            linear_num_key_heads: apr_config.linear_num_key_heads,
            linear_num_value_heads: apr_config.linear_num_value_heads,
            linear_conv_kernel_dim: apr_config.linear_conv_kernel_dim,
            constraints: None,
            // ALB-010: MoE fields
            num_experts: apr_config.num_experts,
            num_experts_per_tok: apr_config.num_experts_per_tok,
            expert_intermediate_size: apr_config.expert_intermediate_size,
        }
    }

    /// Dequantize a Q4_0 tensor to F32
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Q4_0 quantized bytes
    /// * `expected_elements` - Expected number of output elements
    ///
    /// # Returns
    ///
    /// Dequantized F32 vector
    pub fn dequantize_tensor(data: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
        let result = dequantize_q4_0(data)?;

        // Validate dimensions
        if result.len() < expected_elements {
            // Pad with zeros if needed (can happen with block alignment)
            let mut padded = result;
            padded.resize(expected_elements, 0.0);
            Ok(padded)
        } else {
            // Truncate to expected size
            Ok(result.into_iter().take(expected_elements).collect())
        }
    }

    /// Extract QKV weights from APR layer
    ///
    /// APR stores QKV as a single tensor, which matches GpuModel format.
    /// Uses config's head_dim() to handle explicit_head_dim (Qwen3-Coder: 128 vs 2048/32=64).
    pub fn extract_qkv_weights(
        layer: &QuantizedAprLayerQ4,
        config: &GpuModelConfig,
    ) -> Result<Vec<f32>> {
        let qkv_out_dim = config.qkv_dim();
        let expected = config.hidden_dim * qkv_out_dim;

        Self::dequantize_tensor(&layer.qkv_weight.data, expected)
    }

    /// Extract output projection weights
    /// Output proj shape: [hidden_dim, q_dim] where q_dim = num_heads * head_dim
    pub fn extract_out_weights(
        layer: &QuantizedAprLayerQ4,
        config: &GpuModelConfig,
    ) -> Result<Vec<f32>> {
        let expected = config.hidden_dim * config.q_dim();
        Self::dequantize_tensor(&layer.attn_output_weight.data, expected)
    }

    /// Extract FFN weights (fc1 = up, fc2 = down)
    ///
    /// Note: APR uses SwiGLU with separate gate/up, but GpuModel combines them.
    /// For compatibility, we return up weights as fc1.
    pub fn extract_ffn_weights(
        layer: &QuantizedAprLayerQ4,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // FC1 (up projection): [hidden_dim, intermediate_dim]
        let fc1_expected = hidden_dim * intermediate_dim;
        let fc1 = Self::dequantize_tensor(&layer.ffn_up_weight.data, fc1_expected)?;

        // FC2 (down projection): [intermediate_dim, hidden_dim]
        let fc2_expected = intermediate_dim * hidden_dim;
        let fc2 = Self::dequantize_tensor(&layer.ffn_down_weight.data, fc2_expected)?;

        Ok((fc1, fc2))
    }

    /// Convert full APR transformer to GpuModel
    ///
    /// # Arguments
    ///
    /// * `apr` - Source APR transformer with Q4_0 weights
    ///
    /// # Returns
    ///
    /// `GpuModel` ready for GPU inference
    ///
    /// # Example
    ///
    /// ```ignore
    /// use realizar::apr_transformer::QuantizedAprTransformerQ4;
    /// use realizar::gpu::adapters::AprToGpuAdapter;
    ///
    /// let apr = QuantizedAprTransformerQ4::from_gguf(&gguf_model);
    /// let gpu_model = AprToGpuAdapter::to_gpu_model(&apr)?;
    /// ```
    pub fn to_gpu_model(apr: &QuantizedAprTransformerQ4) -> Result<GpuModel> {
        let config = Self::config_to_gpu(&apr.config);
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        // Use config methods for head_dim to handle explicit_head_dim
        // (e.g., Qwen3-Coder: head_dim=128, hidden=2048, num_heads=32 → 2048/32=64 is WRONG)
        let qkv_out_dim = config.qkv_dim();
        let q_dim = config.q_dim();

        // Embedding weights (already F32 in APR)
        let embedding_weights = apr.token_embedding.clone();

        // Dequantize LM head
        let lm_head_expected = hidden_dim * config.vocab_size;
        let lm_head_weight = Self::dequantize_tensor(&apr.lm_head_weight.data, lm_head_expected)?;

        // Phase 22 FIX: Transpose LM head from APR [vocab_size, hidden_dim] to GPU [hidden_dim, vocab_size]
        // APR stores weights as [out_dim, in_dim], GPU matmul expects [in_dim, out_dim]
        let lm_head_weight_t = transpose_matrix(&lm_head_weight, config.vocab_size, hidden_dim);

        // Convert each layer
        let mut block_weights = Vec::with_capacity(apr.layers.len());
        for (block_idx, layer) in apr.layers.iter().enumerate() {
            let qkv = Self::extract_qkv_weights(layer, &config)?;
            let out = Self::extract_out_weights(layer, &config)?;
            let (fc1, fc2) = Self::extract_ffn_weights(layer, hidden_dim, intermediate_dim)?;

            // Extract gate weight for SwiGLU (optional)
            let ffn_gate_weight = if let Some(ref gate) = layer.ffn_gate_weight {
                let gate_expected = hidden_dim * intermediate_dim;
                Some(Self::dequantize_tensor(&gate.data, gate_expected)?)
            } else {
                None
            };

            // F-REGR-231 FIX: Transpose all weight matrices from [out_dim, in_dim] to [in_dim, out_dim]
            // This matches the F32 path in AprF32ToGpuAdapter::convert_layer
            // APR/GGUF stores weights as [out_dim, in_dim] row-major, but GpuModel matmul expects [in_dim, out_dim]
            let qkv_weight_t = transpose_matrix(&qkv, qkv_out_dim, hidden_dim);
            // Output proj: [hidden_dim, q_dim] where q_dim = num_heads * head_dim (may != hidden_dim)
            let out_weight_t = transpose_matrix(&out, hidden_dim, q_dim);
            let fc1_weight_t = transpose_matrix(&fc1, intermediate_dim, hidden_dim);
            let fc2_weight_t = transpose_matrix(&fc2, hidden_dim, intermediate_dim);
            let gate_weight_t = ffn_gate_weight
                .as_ref()
                .map(|w| transpose_matrix(w, intermediate_dim, hidden_dim));

            // GH-278: Q4 models don't carry linear attention weights (F32-only path)
            // Linear attention layers in GGUF/Q4 are not yet supported
            let _ = block_idx; // Acknowledge block_idx for future GH-278 Q4 linear attn support

            block_weights.push(BlockWeights {
                attn_norm_weight: layer.attn_norm_weight.clone(),
                attn_norm_bias: vec![0.0; hidden_dim], // APR doesn't use bias
                qkv_weight: qkv_weight_t,
                qkv_bias: vec![], // No bias in APR
                out_weight: out_weight_t,
                out_bias: vec![0.0; hidden_dim],
                ffn_norm_weight: layer
                    .ffn_norm_weight
                    .clone()
                    .unwrap_or_else(|| vec![1.0; hidden_dim]),
                ffn_norm_bias: vec![0.0; hidden_dim],
                ffn_fc1_weight: fc1_weight_t,
                ffn_fc1_bias: vec![0.0; intermediate_dim],
                ffn_fc2_weight: fc2_weight_t,
                ffn_fc2_bias: vec![0.0; hidden_dim],
                ffn_gate_weight: gate_weight_t,
                // GH-278: Q4 linear attention support deferred (SafeTensors F32 path only)
                linear_attn: None,
                moe_experts: None,
            });
        }

        // Final norm
        let final_norm_weight = apr.output_norm_weight.clone();
        let final_norm_bias = vec![0.0; hidden_dim];

        // LM head bias
        let lm_head_bias = vec![0.0; config.vocab_size];

        // PMAT-284: Validated weights — newtypes prevent lm_head swap at compile time
        let validated = ValidatedGpuWeights::new(
            config,
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            LmHeadWeight(lm_head_weight),
            LmHeadWeightTransposed(lm_head_weight_t),
            lm_head_bias,
        )
        .map_err(|e| crate::error::RealizarError::InvalidShape {
            reason: e.to_string(),
        })?;
        GpuModel::from_validated_weights(validated)
    }
}

/// Transpose a row-major matrix.
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
#[must_use]
pub fn transpose_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(data, rows, cols)
}

include!("apr_config.rs");
