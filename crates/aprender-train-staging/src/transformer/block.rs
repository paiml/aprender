//! Transformer block module
//!
//! This module provides complete transformer blocks combining attention, FFN, and normalization.

use crate::autograd::add;
use crate::Tensor;
use std::collections::HashMap;

use super::attention::MultiHeadAttention;
use super::config::TransformerConfig;
use super::feedforward::FeedForward;
use super::norm::RMSNorm;

/// Complete transformer block
pub struct TransformerBlock {
    /// Configuration
    config: TransformerConfig,
    /// Layer index
    layer_idx: usize,
    /// Input layer normalization
    pub input_norm: RMSNorm,
    /// Self-attention
    pub self_attn: MultiHeadAttention,
    /// Post-attention layer normalization
    pub post_attn_norm: RMSNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
}

impl TransformerBlock {
    /// Create new transformer block with initialized weights
    pub fn new(config: &TransformerConfig, layer_idx: usize) -> Self {
        Self {
            config: config.clone(),
            layer_idx,
            input_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            self_attn: MultiHeadAttention::new(config),
            post_attn_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            ffn: FeedForward::new(config),
        }
    }

    /// Create transformer block from parameter map
    ///
    /// Expected parameter names (following HuggingFace LLaMA convention):
    /// - `model.layers.{layer_idx}.input_layernorm.weight`
    /// - `model.layers.{layer_idx}.self_attn.*`
    /// - `model.layers.{layer_idx}.post_attention_layernorm.weight`
    /// - `model.layers.{layer_idx}.mlp.*`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Option<Self> {
        let prefix = format!("model.layers.{layer_idx}");

        let input_norm = RMSNorm::from_params(
            params,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
            config.hidden_size,
        )?;

        let self_attn =
            MultiHeadAttention::from_params(config, params, &format!("{prefix}.self_attn"))?;

        let post_attn_norm = RMSNorm::from_params(
            params,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
            config.hidden_size,
        )?;

        let ffn = FeedForward::from_params(config, params, &format!("{prefix}.mlp"))?;

        Some(Self { config: config.clone(), layer_idx, input_norm, self_attn, post_attn_norm, ffn })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;

        // Pre-norm + attention + residual
        let norm1 = self.input_norm.forward_batched(x, seq_len, hidden_size);
        let attn_out = self.self_attn.forward(&norm1, seq_len);
        let residual1 = add(x, &attn_out);

        // Pre-norm + FFN + residual
        let norm2 = self.post_attn_norm.forward_batched(&residual1, seq_len, hidden_size);
        let ffn_out = self.ffn.forward(&norm2, seq_len);
        add(&residual1, &ffn_out)
    }

    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.input_norm.weight, &self.post_attn_norm.weight];
        params.extend(self.self_attn.parameters());
        params.extend(self.ffn.parameters());
        params
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        params.push(&mut self.input_norm.weight);
        params.push(&mut self.post_attn_norm.weight);
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params
    }

    /// Get named parameters for checkpoint serialization.
    /// Returns (name, tensor) pairs matching HuggingFace weight conventions.
    pub fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let prefix = format!("model.layers.{}", self.layer_idx);
        let mut params = vec![
            (format!("{prefix}.input_layernorm.weight"), &self.input_norm.weight),
            (format!("{prefix}.post_attention_layernorm.weight"), &self.post_attn_norm.weight),
        ];
        params.extend(self.self_attn.named_parameters(&format!("{prefix}.self_attn")));
        params.push((format!("{prefix}.mlp.gate_proj.weight"), &self.ffn.w_gate));
        params.push((format!("{prefix}.mlp.up_proj.weight"), &self.ffn.w_up));
        params.push((format!("{prefix}.mlp.down_proj.weight"), &self.ffn.w_down));
        params
    }

    /// ENT-282: Set a named parameter by suffix (after "model.layers.{idx}.").
    pub fn set_named_parameter(&mut self, suffix: &str, value: Tensor) -> bool {
        match suffix {
            "input_layernorm.weight" => {
                self.input_norm.weight = value;
                true
            }
            "post_attention_layernorm.weight" => {
                self.post_attn_norm.weight = value;
                true
            }
            "mlp.gate_proj.weight" => {
                self.ffn.w_gate = value;
                true
            }
            "mlp.up_proj.weight" => {
                self.ffn.w_up = value;
                true
            }
            "mlp.down_proj.weight" => {
                self.ffn.w_down = value;
                true
            }
            _ => self.self_attn.set_named_parameter(suffix, value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block_tiny() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 0);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = block.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_transformer_block_layer_idx() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 5);
        assert_eq!(block.layer_idx(), 5);
    }

    #[test]
    fn test_transformer_block_parameters() {
        let config = TransformerConfig::tiny();
        let block = TransformerBlock::new(&config, 0);
        let params = block.parameters();
        // input_norm + post_attn_norm + 4 attn weights + 3 ffn = 9 (no biases by default)
        assert_eq!(params.len(), 9);
    }

    #[test]
    fn test_transformer_block_from_params_success() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Input norm
        params.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        // Self-attention weights
        params.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; q_dim * hidden_size], true),
        );
        params.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; kv_hidden_size * hidden_size], true),
        );
        params.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; kv_hidden_size * hidden_size], true),
        );
        params.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * q_dim], true),
        );

        // Post-attention norm
        params.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        // MLP weights
        params.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        params.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        params.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
        );

        let block = TransformerBlock::from_params(&config, &params, 0);
        assert!(block.is_some());
        let block = block.expect("operation should succeed");
        assert_eq!(block.layer_idx(), 0);
    }

    #[test]
    fn test_transformer_block_from_params_missing_norm() {
        let config = TransformerConfig::tiny();
        let params = HashMap::new();
        // Empty params - should fail

        let block = TransformerBlock::from_params(&config, &params, 0);
        assert!(block.is_none());
    }
}
