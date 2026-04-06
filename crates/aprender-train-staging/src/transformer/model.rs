//! Complete transformer model module
//!
//! This module provides the full transformer model for language modeling.

use crate::autograd::matmul_nt;
use crate::error::{Error, Result};
use crate::Tensor;
use provable_contracts_macros::{ensures, requires};
use std::collections::HashMap;
use std::path::Path;

use super::block::TransformerBlock;
use super::config::TransformerConfig;
use super::embedding::Embedding;
use super::norm::RMSNorm;
use super::weights::{load_safetensors_weights, validate_weights, Architecture};

/// Complete transformer model
pub struct Transformer {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding layer
    pub embed_tokens: Embedding,
    /// Transformer layers
    pub layers: Vec<TransformerBlock>,
    /// Final layer normalization
    pub norm: RMSNorm,
    /// Language model head (tied to embeddings or separate)
    pub lm_head: Option<Tensor>,
}

impl Transformer {
    /// Create new transformer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let layers: Vec<TransformerBlock> =
            (0..config.num_hidden_layers).map(|i| TransformerBlock::new(config, i)).collect();

        Self {
            config: config.clone(),
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            lm_head: None, // Use tied embeddings by default
        }
    }

    /// Create transformer from parameter map
    ///
    /// Expected parameter names (following HuggingFace LLaMA convention):
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.*`
    /// - `model.norm.weight`
    /// - `lm_head.weight` (optional, uses tied embeddings if not present)
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
    ) -> Option<Self> {
        let embed_tokens = Embedding::from_params(
            params,
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;

        let layers: Option<Vec<TransformerBlock>> = (0..config.num_hidden_layers)
            .map(|i| TransformerBlock::from_params(config, params, i))
            .collect();
        let layers = layers?;

        let norm =
            RMSNorm::from_params(params, "model.norm", config.rms_norm_eps, config.hidden_size)?;

        // PMAT-329: Validate lm_head shape if present
        let lm_head = if let Some(tensor) = params.get("lm_head.weight") {
            let expected = config.hidden_size * config.vocab_size;
            if tensor.len() != expected {
                eprintln!(
                    "[PMAT-329] lm_head.weight: shape mismatch — got {} elements, expected {expected} ({hidden}x{vocab})",
                    tensor.len(),
                    hidden = config.hidden_size,
                    vocab = config.vocab_size,
                );
                return None;
            }
            Some(tensor.clone())
        } else {
            None
        };

        Some(Self { config: config.clone(), embed_tokens, layers, norm, lm_head })
    }

    /// Load transformer from SafeTensors file(s)
    ///
    /// Reads SafeTensors weights from `model_path`, converts BF16/F16 to F32,
    /// validates shapes against `config`, checks for NaN/Inf, and constructs
    /// the complete `Transformer`.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model directory or single SafeTensors file
    /// * `config` - Transformer configuration specifying model dimensions
    ///
    /// # Errors
    ///
    /// Returns `Error::ConfigError` if:
    /// - No SafeTensors files found
    /// - Required weight tensors are missing
    /// - Weight shapes do not match config dimensions
    /// - Weights contain NaN or Inf values
    /// - Layer count does not match config
    pub fn from_safetensors(
        model_path: impl AsRef<Path>,
        config: &TransformerConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Load and convert all weights from SafeTensors files
        let weights = load_safetensors_weights(model_path, Architecture::Auto)?;

        // Structural validation: all required keys present
        validate_weights(&weights, config.num_hidden_layers)?;

        // Shape validation against config dimensions
        Self::validate_weight_shapes(&weights, config)?;

        // NaN/Inf validation
        Self::validate_weight_values(&weights)?;

        // Build transformer from validated weights
        Self::from_params(config, &weights).ok_or_else(|| {
            Error::ConfigError(
                "Failed to construct Transformer from loaded weights \
                 (internal from_params returned None after validation passed)"
                    .into(),
            )
        })
    }

    /// Load transformer from APR file (.apr format)
    ///
    /// Reads tensor data from an APR binary file, dequantizing from any stored
    /// dtype (F16, Q4K, etc.) to F32. Uses the same validation pipeline as
    /// `from_safetensors`: structural, shape, and NaN/Inf checks.
    ///
    /// # Arguments
    /// * `apr_path` - Path to the .apr model file
    /// * `config` - Transformer configuration (typically read from APR metadata)
    ///
    /// # Errors
    /// Returns `Error::ConfigError` if tensors are missing, shapes mismatch, or
    /// weights contain NaN/Inf values.
    pub fn from_apr(apr_path: impl AsRef<Path>, config: &TransformerConfig) -> Result<Self> {
        use aprender::serialization::apr::AprReader;

        let apr_path = apr_path.as_ref();
        let reader = AprReader::open(apr_path).map_err(|e| {
            Error::ConfigError(format!("Failed to open APR file '{}': {e}", apr_path.display()))
        })?;

        // Build weight map from APR tensors — map GGUF names to HF convention (PMAT-489)
        let is_gguf_names = reader.tensors.iter().any(|t| t.name == "token_embd.weight");
        if is_gguf_names {
            eprintln!(
                "[PMAT-489] Detected GGUF tensor names in APR file, mapping to HF convention"
            );
        }
        let mut weights = HashMap::new();
        for desc in &reader.tensors {
            let data = reader.read_tensor_as_f32(&desc.name).map_err(|e| {
                Error::ConfigError(format!("Failed to read tensor '{}': {e}", desc.name))
            })?;
            let mapped_name = if is_gguf_names {
                super::weights::mapping::map_weight_name(
                    &desc.name,
                    super::weights::Architecture::Gguf,
                )
            } else {
                desc.name.clone()
            };
            weights.insert(mapped_name, Tensor::from_vec(data, false));
        }

        // Same validation pipeline as from_safetensors
        validate_weights(&weights, config.num_hidden_layers)?;
        Self::validate_weight_shapes(&weights, config)?;
        Self::validate_weight_values(&weights)?;

        Self::from_params(config, &weights).ok_or_else(|| {
            Error::ConfigError(
                "Failed to construct Transformer from APR weights \
                 (from_params returned None after validation passed)"
                    .into(),
            )
        })
    }

    /// Validate that all weight tensor shapes match the config dimensions
    fn validate_weight_shapes(
        weights: &HashMap<String, Tensor>,
        config: &TransformerConfig,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_hidden = config.num_kv_heads * config.head_dim();
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;

        // Helper closure for shape checking
        let check = |name: &str, expected: usize| -> Result<()> {
            if let Some(tensor) = weights.get(name) {
                if tensor.len() != expected {
                    return Err(Error::ConfigError(format!(
                        "Shape mismatch for '{name}': expected {expected} elements, got {}",
                        tensor.len()
                    )));
                }
            }
            // Missing keys are caught by validate_weights
            Ok(())
        };

        // Global weights
        check("model.embed_tokens.weight", vocab * hidden)?;
        check("model.norm.weight", hidden)?;

        // Optional lm_head
        if weights.contains_key("lm_head.weight") {
            check("lm_head.weight", vocab * hidden)?;
        }

        // Per-layer weights
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");

            // Layer norms
            check(&format!("{p}.input_layernorm.weight"), hidden)?;
            check(&format!("{p}.post_attention_layernorm.weight"), hidden)?;

            // Attention projections: Q/O use q_dim, K/V use kv_hidden
            check(&format!("{p}.self_attn.q_proj.weight"), q_dim * hidden)?;
            check(&format!("{p}.self_attn.k_proj.weight"), kv_hidden * hidden)?;
            check(&format!("{p}.self_attn.v_proj.weight"), kv_hidden * hidden)?;
            check(&format!("{p}.self_attn.o_proj.weight"), hidden * q_dim)?;

            // Optional attention biases (Qwen2 etc.)
            check(&format!("{p}.self_attn.q_proj.bias"), q_dim)?;
            check(&format!("{p}.self_attn.k_proj.bias"), kv_hidden)?;
            check(&format!("{p}.self_attn.v_proj.bias"), kv_hidden)?;

            // MLP projections
            check(&format!("{p}.mlp.gate_proj.weight"), hidden * intermediate)?;
            check(&format!("{p}.mlp.up_proj.weight"), hidden * intermediate)?;
            check(&format!("{p}.mlp.down_proj.weight"), intermediate * hidden)?;
        }

        Ok(())
    }

    /// Validate that no weight tensors contain NaN or Inf values
    fn validate_weight_values(weights: &HashMap<String, Tensor>) -> Result<()> {
        for (name, tensor) in weights {
            let data = tensor.data();
            for (i, &val) in data.iter().enumerate() {
                if val.is_nan() {
                    return Err(Error::ConfigError(format!(
                        "NaN detected in weight '{name}' at index {i}"
                    )));
                }
                if val.is_infinite() {
                    return Err(Error::ConfigError(format!(
                        "Inf detected in weight '{name}' at index {i}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Forward pass for language modeling
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Logits tensor (seq_len * vocab_size, flattened)
    #[requires(!token_ids.is_empty())]
    #[ensures(ret.len() == token_ids.len() * self.config.vocab_size)]
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(token_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final normalization
        let normalized = self.norm.forward_batched(&hidden, seq_len, hidden_size);

        // Language model head
        let lm_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens.weight);

        // lm_head / tied embed_tokens is [vocab_size, hidden_size] in HF (ENT-269)
        let result =
            matmul_nt(&normalized, lm_weight, seq_len, hidden_size, self.config.vocab_size);
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Forward pass returning hidden states (before lm_head)
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size, flattened)
    #[requires(!token_ids.is_empty())]
    #[ensures(ret.len() == token_ids.len() * self.config.hidden_size)]
    pub fn forward_hidden(&self, token_ids: &[u32]) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(token_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final normalization
        let result = self.norm.forward_batched(&hidden, seq_len, hidden_size);
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Forward pass returning hidden states with LoRA adjusts (KAIZEN-011)
    ///
    /// Like `forward_hidden` but applies LoRA adapters to Q/V projections in
    /// each transformer layer's attention. Enables non-CUDA LoRA training by
    /// putting LoRA parameters into the autograd graph.
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    /// * `lora_layers` - LoRA layers in [Q_0, V_0, Q_1, V_1, ...] order
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size, flattened)
    pub fn forward_hidden_with_lora(
        &self,
        token_ids: &[u32],
        lora_layers: &[crate::lora::LoRALayer],
    ) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        let mut hidden = self.embed_tokens.forward(token_ids);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let norm1 = layer.input_norm.forward_batched(&hidden, seq_len, hidden_size);

            // KAIZEN-011: Apply LoRA to attention Q/V projections
            let q_idx = layer_idx * 2;
            let v_idx = layer_idx * 2 + 1;
            let attn_out = if v_idx < lora_layers.len() {
                layer.self_attn.forward_with_lora(
                    &norm1,
                    seq_len,
                    lora_layers[q_idx].lora_a(),
                    lora_layers[q_idx].lora_b(),
                    lora_layers[v_idx].lora_a(),
                    lora_layers[v_idx].lora_b(),
                    lora_layers[q_idx].rank(),
                    lora_layers[q_idx].scale(),
                )
            } else {
                layer.self_attn.forward(&norm1, seq_len)
            };

            let residual = crate::autograd::add(&hidden, &attn_out);
            let norm2 = layer.post_attn_norm.forward_batched(&residual, seq_len, hidden_size);
            let ffn_out = layer.ffn.forward(&norm2, seq_len);
            hidden = crate::autograd::add(&residual, &ffn_out);
        }

        let result = self.norm.forward_batched(&hidden, seq_len, hidden_size);
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Forward pass with LoRA adapters (ENT-LoRA-001)
    ///
    /// Like `forward` but applies LoRA adapters to Q/V projections.
    /// Returns full logits (seq_len * vocab_size).
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    /// * `lora_layers` - LoRA layers in [Q_0, V_0, Q_1, V_1, ...] order
    pub fn forward_with_lora(
        &self,
        token_ids: &[u32],
        lora_layers: &[crate::lora::LoRALayer],
    ) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        let hidden = self.forward_hidden_with_lora(token_ids, lora_layers);
        let lm_weight = self.lm_head.as_ref().unwrap_or(&self.embed_tokens.weight);
        let result = matmul_nt(&hidden, lm_weight, seq_len, hidden_size, self.config.vocab_size);
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Get the last token's logits (for generation)
    pub fn forward_last(&self, token_ids: &[u32]) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let logits = self.forward(token_ids);
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;

        // Extract last position
        let start = (seq_len - 1) * vocab_size;
        let end = start + vocab_size;
        let last_logits: Vec<f32> =
            logits.data().as_slice().expect("logits must be contiguous")[start..end].to_vec();

        let result = Tensor::from_vec(last_logits, logits.requires_grad());
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embed_tokens.weight, &self.norm.weight];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(lm_head) = &self.lm_head {
            params.push(lm_head);
        }
        params
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        params.push(&mut self.embed_tokens.weight);
        params.push(&mut self.norm.weight);
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(lm_head) = &mut self.lm_head {
            params.push(lm_head);
        }
        params
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Embed a single token, returning hidden_size floats.
    pub fn embed_token(&self, token_id: u32) -> Vec<f32> {
        let w = self.embed_tokens.weight.data();
        let data = w.as_slice().expect("contiguous embedding");
        let h = self.config.hidden_size;
        let offset = (token_id as usize) * h;
        data[offset..offset + h].to_vec()
    }

    /// Get the output norm weight as a slice.
    pub fn output_norm_weight_slice(&self) -> &[f32] {
        self.norm.weight.data().as_slice().expect("contiguous norm weight")
    }

    /// Get the lm_head weight as a slice (vocab_size × hidden_size, row-major).
    pub fn lm_head_weight_slice(&self) -> &[f32] {
        let w = self.lm_head.as_ref().unwrap_or(&self.embed_tokens.weight);
        w.data().as_slice().expect("contiguous lm_head")
    }

    /// Get the language model head weight tensor.
    ///
    /// Returns the dedicated `lm_head` weight if present, otherwise falls back
    /// to tied embedding weights.
    pub fn lm_head_weight(&self) -> &Tensor {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens.weight)
    }

    /// Get named parameters for checkpoint serialization.
    ///
    /// Returns (name, tensor) pairs matching HuggingFace weight conventions.
    /// This handles variable parameter counts (e.g., models with/without attention biases)
    /// correctly, unlike the hardcoded 9-params-per-layer assumption.
    pub fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![
            ("model.embed_tokens.weight".to_string(), &self.embed_tokens.weight),
            ("model.norm.weight".to_string(), &self.norm.weight),
        ];
        for layer in &self.layers {
            params.extend(layer.named_parameters());
        }
        if let Some(ref lm_head) = self.lm_head {
            params.push(("lm_head.weight".to_string(), lm_head));
        }
        params
    }

    /// ENT-282: Set a named parameter by name (for delta checkpoint overlay).
    ///
    /// Returns true if the parameter was found and set.
    pub fn set_named_parameter(&mut self, name: &str, value: Tensor) -> bool {
        if name == "model.embed_tokens.weight" {
            self.embed_tokens.weight = value;
            return true;
        }
        if name == "model.norm.weight" {
            self.norm.weight = value;
            return true;
        }
        if name == "lm_head.weight" {
            self.lm_head = Some(value);
            return true;
        }
        // Per-layer parameters: model.layers.{idx}.{suffix}
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                    if idx < self.layers.len() {
                        let suffix = &rest[dot_pos + 1..];
                        return self.layers[idx].set_named_parameter(suffix, value);
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_tiny_forward() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size);
    }

    #[test]
    fn test_transformer_tiny_forward_last() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3];
        let logits = transformer.forward_last(&tokens);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_transformer_parameters() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let params = transformer.parameters();
        // embed_tokens + norm + (layers * (input_norm + post_attn_norm + 4 attn weights + 3 ffn weights))
        // = 2 + 2 * (2 + 4 + 3) = 2 + 2 * 9 = 20
        assert_eq!(params.len(), 20);
    }

    #[test]
    fn test_transformer_config_accessor() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        assert_eq!(transformer.config().hidden_size, config.hidden_size);
        assert_eq!(transformer.config().vocab_size, config.vocab_size);
    }

    #[test]
    fn test_transformer_single_token() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![42];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_output_finite_values() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![1, 2, 3, 4, 5];
        let logits = transformer.forward(&tokens);
        // All outputs should be finite (no NaN or Inf)
        assert!(logits.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_transformer_empty_lm_head_uses_tied_weights() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        // Default transformer should have no separate lm_head
        assert!(transformer.lm_head.is_none());
        // But should still produce valid logits
        let tokens = vec![1, 2];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 2 * config.vocab_size);
    }

    #[test]
    fn test_from_params_returns_none_on_missing() {
        let config = TransformerConfig::tiny();
        let params: HashMap<String, Tensor> = HashMap::new();
        let result = Transformer::from_params(&config, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_transformer_from_params_with_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Embedding
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );

        // All layers
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
            );
        }

        // Final norm
        params.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        // LM head (separate, not tied)
        params.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * vocab_size], true),
        );

        let transformer = Transformer::from_params(&config, &params);
        assert!(transformer.is_some());
        let transformer = transformer.expect("operation should succeed");
        assert!(transformer.lm_head.is_some());
        assert_eq!(transformer.layers.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_transformer_from_params_without_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Embedding
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );

        // All layers
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
            );
        }

        // Final norm - no lm_head
        params.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        let transformer = Transformer::from_params(&config, &params);
        assert!(transformer.is_some());
        let transformer = transformer.expect("operation should succeed");
        assert!(transformer.lm_head.is_none()); // Should use tied embeddings
    }

    #[test]
    fn test_transformer_parameters_with_lm_head() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Add a separate lm_head
        transformer.lm_head =
            Some(Tensor::from_vec(vec![0.1; config.hidden_size * config.vocab_size], true));

        let params = transformer.parameters();
        // embed_tokens + norm + (layers * 9) + lm_head
        // = 1 + 1 + (2 * 9) + 1 = 21
        assert_eq!(params.len(), 21);
    }

    #[test]
    fn test_transformer_forward_with_lm_head() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Add a separate lm_head
        transformer.lm_head =
            Some(Tensor::from_vec(vec![0.1; config.hidden_size * config.vocab_size], true));

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 3 * config.vocab_size);
        assert!(logits.data().iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // FALSIFY-L: §2.1.2 LM Head Contract — Five-Whys Gap Analysis (Refs PMAT-329)
    //
    // Contract: tensor-layout-v1.yaml §tensors.lm_head
    //   critical: "true"
    //   note: "GH-202 root cause - wrong shape caused [PAD] garbage output"
    //
    // Five-Whys:
    //   Why 1: entrenar-trained model's lm_head could corrupt inference
    //   Why 2: lm_head save/load has no shape validation
    //   Why 3: from_params accepts ANY tensor for lm_head (like embedding)
    //   Why 4: entrenar predates ValidatedWeight contract
    //   Why 5: No cross-crate contract enforcement for trained models
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's lm_head handling prevents garbage output after training."
    // =========================================================================

    /// FALSIFY-L1e: from_params rejects wrong-shape lm_head (PMAT-329 fix)
    ///
    /// from_params now validates lm_head shape against vocab*hidden.
    /// A tensor of 50 elements is rejected when vocab*hidden is expected.
    #[test]
    fn falsify_l1e_from_params_rejects_wrong_shape_lm_head() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();

        // Valid embedding + layers + norm
        params.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab_size * hidden_size], true),
        );
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_idx}");
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
            );
        }
        params.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden_size], true),
        );

        // WRONG-SHAPE lm_head: 50 elements for hidden*vocab expected
        params.insert("lm_head.weight".to_string(), Tensor::from_vec(vec![0.1; 50], true));

        let transformer = Transformer::from_params(&config, &params);
        // FIXED (PMAT-329): now rejected
        assert!(
            transformer.is_none(),
            "FALSIFY-L1e: PMAT-329 fix — from_params MUST reject wrong-shape lm_head"
        );
    }

    /// FALSIFY-L2e: Tied embeddings produce valid logit dimensions
    ///
    /// When lm_head is None, the embedding weight [vocab, hidden] is used as lm_head.
    /// The matmul must produce [seq_len, vocab_size] logits.
    #[test]
    fn falsify_l2e_tied_embeddings_produce_correct_logit_dims() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        assert!(transformer.lm_head.is_none(), "Default should use tied embeddings");

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(
            logits.len(),
            3 * config.vocab_size,
            "FALSIFY-L2e: Tied embedding logits must be seq_len * vocab_size"
        );

        // All logits must be finite (not NaN/Inf)
        let data = logits.data();
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "FALSIFY-L2e: Tied logits must not contain NaN");
        assert_eq!(inf_count, 0, "FALSIFY-L2e: Tied logits must not contain Inf");
    }

    /// FALSIFY-L3e: Separate lm_head produces valid logit dimensions
    #[test]
    fn falsify_l3e_separate_lm_head_produces_correct_logit_dims() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);
        transformer.lm_head =
            Some(Tensor::from_vec(vec![0.1; config.hidden_size * config.vocab_size], true));

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(
            logits.len(),
            3 * config.vocab_size,
            "FALSIFY-L3e: Separate lm_head logits must be seq_len * vocab_size"
        );
        let data = logits.data();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "FALSIFY-L3e: Separate lm_head logits must all be finite"
        );
    }

    /// FALSIFY-L4e: lm_head is included in parameters() and parameters_mut()
    ///
    /// If lm_head is present but not returned by parameters(), the optimizer
    /// won't update it during training → frozen lm_head → garbage after finetuning.
    #[test]
    fn falsify_l4e_lm_head_in_parameter_list() {
        let config = TransformerConfig::tiny();
        let mut transformer = Transformer::new(&config);

        // Without lm_head: N params
        let n_without = transformer.parameters().len();

        // With lm_head: N+1 params
        transformer.lm_head =
            Some(Tensor::from_vec(vec![0.1; config.hidden_size * config.vocab_size], true));
        let n_with = transformer.parameters().len();
        assert_eq!(
            n_with,
            n_without + 1,
            "FALSIFY-L4e: lm_head must be included in parameters() — optimizer needs it"
        );

        // Also check parameters_mut
        let n_mut = transformer.parameters_mut().len();
        assert_eq!(
            n_mut, n_with,
            "FALSIFY-L4e: parameters_mut() must include lm_head for gradient updates"
        );
    }

    /// FALSIFY-L5e: forward_last returns exactly vocab_size logits
    ///
    /// The last token's logits are used for next-token prediction.
    /// Off-by-one in the slice extraction → wrong token generated.
    #[test]
    fn falsify_l5e_forward_last_correct_size() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        let tokens = vec![1, 2, 3, 4, 5];
        let logits = transformer.forward_last(&tokens);
        assert_eq!(
            logits.len(),
            config.vocab_size,
            "FALSIFY-L5e: forward_last must return exactly vocab_size logits"
        );
        let data = logits.data();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "FALSIFY-L5e: forward_last logits must all be finite"
        );
    }

    #[test]
    fn test_causal_lm_loss_backward() {
        use crate::train::CausalLMLoss;
        use crate::train::LossFn;

        let vocab_size = 100;
        let seq_len = 3;
        let loss_fn = CausalLMLoss::new(vocab_size);

        // Create some logits
        let logits = Tensor::from_vec(
            (0..seq_len * vocab_size).map(|i| (i as f32 * 0.01).sin()).collect(),
            true,
        );

        // Target token IDs
        let targets = Tensor::from_vec(vec![5.0, 10.0, 15.0], false);

        let mut loss = loss_fn.forward(&logits, &targets);

        // Backward
        crate::autograd::backward(&mut loss, None);

        // Loss should be positive
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());

        // Logits should have gradient
        assert!(logits.grad().is_some());
        let grad = logits.grad().expect("gradient should be available");
        assert!(grad.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // FALSIFY-EMB-003 / FALSIFY-TE-001..004: Tied Embeddings Contract
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had L-series lm_head tests but no EMB-003/TE-* tagged tests
    //   Why 2: L-series validates shape, not tied-weight CONTRACT claims
    //   Why 3: no mapping from tied-embeddings-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: tied weights were assumed correct because code path is "just fallback"
    //
    // References:
    //   - provable-contracts/contracts/embedding-algebra-v1.yaml (EMB-003)
    //   - provable-contracts/contracts/tied-embeddings-v1.yaml (TE-001..004)
    //   - Press & Wolf (2017) "Using the Output Embedding to Improve Language Models"
    // =========================================================================

    /// FALSIFY-EMB-003: Tied weight sharing — lm_head uses embed_tokens.weight
    ///
    /// Contract: when lm_head is None, forward() uses embed_tokens.weight directly
    /// (pointer/identity sharing, not a copy)
    #[test]
    fn falsify_emb_003_tied_weight_sharing() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        // Default: lm_head is None → tied
        assert!(transformer.lm_head.is_none());

        // The weight used for lm_head projection IS embed_tokens.weight
        let lm_weight = transformer.lm_head.as_ref().unwrap_or(&transformer.embed_tokens.weight);
        let embed_weight = &transformer.embed_tokens.weight;

        // They must be the same Tensor (same data pointer, not just equal values)
        assert!(
            std::ptr::eq(lm_weight, embed_weight),
            "FALSIFIED EMB-003: tied lm_head must be same object as embed_tokens.weight"
        );
    }

    /// FALSIFY-TE-001: Output shape = (seq_len, vocab_size)
    #[test]
    fn falsify_te_001_output_shape() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        for seq_len in [1, 3, 10] {
            let tokens: Vec<u32> = (0..seq_len).collect();
            let logits = transformer.forward(&tokens);
            assert_eq!(
                logits.len(),
                seq_len as usize * config.vocab_size,
                "FALSIFIED TE-001: output shape for seq_len={seq_len}"
            );
        }
    }

    /// FALSIFY-TE-002: Tied equivalence — tied output == explicit matmul with cloned W
    ///
    /// Contract: forward() with tied lm_head must produce bit-identical output
    /// to manually computing matmul(hidden, W_embed) with a separate copy of the
    /// embedding weight matrix. If they diverge, the tied path silently aliases
    /// or transposes incorrectly.
    #[test]
    fn falsify_te_002_tied_equivalence() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        // Tied path: forward() uses embed_tokens.weight as lm_head
        let tokens = vec![0u32, 3, 7, 15, 42];
        let tied_logits = transformer.forward(&tokens);

        // Explicit path: clone embed weight, run hidden states, matmul manually
        let hidden = transformer.forward_hidden(&tokens);
        let w_clone = transformer.embed_tokens.weight.clone();
        let explicit_logits =
            matmul_nt(&hidden, &w_clone, tokens.len(), config.hidden_size, config.vocab_size);

        let tied_data = tied_logits.data();
        let explicit_data = explicit_logits.data();

        assert_eq!(
            tied_data.len(),
            explicit_data.len(),
            "FALSIFIED TE-002: output lengths differ: {} vs {}",
            tied_data.len(),
            explicit_data.len()
        );

        for (i, (&t, &e)) in tied_data.iter().zip(explicit_data.iter()).enumerate() {
            assert!(
                (t - e).abs() < 1e-6,
                "FALSIFIED TE-002: tied[{i}] = {t} != explicit[{i}] = {e}"
            );
        }
    }

    /// FALSIFY-TE-003: No extra parameters for tied embeddings
    ///
    /// Contract: tied model has exactly N params, untied has N+1 (the separate lm_head)
    #[test]
    fn falsify_te_003_no_extra_params() {
        let config = TransformerConfig::tiny();
        let tied = Transformer::new(&config);
        let tied_count = tied.parameters().len();

        let mut untied = Transformer::new(&config);
        untied.lm_head =
            Some(Tensor::from_vec(vec![0.1; config.hidden_size * config.vocab_size], true));
        let untied_count = untied.parameters().len();

        assert_eq!(
            untied_count,
            tied_count + 1,
            "FALSIFIED TE-003: tied model must have exactly 1 fewer param than untied"
        );
    }

    /// FALSIFY-TE-004: Finite output for tied embeddings
    #[test]
    fn falsify_te_004_finite_output() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);
        let tokens = vec![0u32, 5, 10, 50, 99];
        let logits = transformer.forward(&tokens);
        let data = logits.data();

        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        assert_eq!(
            nan_count, 0,
            "FALSIFIED TE-004: tied embedding output contains {nan_count} NaN values"
        );
        assert_eq!(
            inf_count, 0,
            "FALSIFIED TE-004: tied embedding output contains {inf_count} Inf values"
        );
    }

    // =========================================================================
    // PROPTEST FALSIFY-TE: Tied embeddings property-based falsification
    //
    // Five-Whys (PMAT-354, Phase 9):
    //   Why 1: YAML tied-embeddings-v1.yaml calls for "proptest with seq_len in [1,128]"
    //   Why 2: All 4 TE tests use fixed token sequences
    //   Why 3: TE proptest had ZERO coverage across the entire stack
    //   Why 4: Transformer construction is expensive, discouraging property testing
    //   Why 5: Fixed tokens miss edge cases in arbitrary token→logit paths
    //
    // References:
    //   - tied-embeddings-v1.yaml FALSIFY-TE-001: "proptest with seq_len in [1,128]"
    //   - tied-embeddings-v1.yaml FALSIFY-TE-002: "clone W_embed, compare tied vs explicit"
    //   - tied-embeddings-v1.yaml FALSIFY-TE-004: "proptest with finite x, check is_finite()"
    // =========================================================================

    mod te_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // TE-001-prop: Output shape for random seq_len
        // Construct transformer once per test run, vary only the token sequence
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]
            #[test]
            fn falsify_te_001_prop_output_shape(
                seq_len in 1_usize..32,
            ) {
                let config = TransformerConfig::tiny();
                let transformer = Transformer::new(&config);
                let tokens: Vec<u32> = (0..seq_len).map(|i| (i % config.vocab_size) as u32).collect();
                let logits = transformer.forward(&tokens);
                prop_assert_eq!(
                    logits.len(),
                    seq_len * config.vocab_size,
                    "FALSIFIED TE-001-prop: seq_len={}, got len={}", seq_len, logits.len()
                );
            }
        }

        // TE-002-prop: Tied equivalence for random tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(20))]
            #[test]
            fn falsify_te_002_prop_tied_equivalence(
                token_ids in proptest::collection::vec(0_u32..999, 1..8),
            ) {
                let config = TransformerConfig::tiny();
                let transformer = Transformer::new(&config);

                let tied_logits = transformer.forward(&token_ids);
                let hidden = transformer.forward_hidden(&token_ids);
                let w_clone = transformer.embed_tokens.weight.clone();
                let explicit_logits = matmul_nt(
                    &hidden, &w_clone,
                    token_ids.len(), config.hidden_size, config.vocab_size,
                );

                let tied_data = tied_logits.data();
                let explicit_data = explicit_logits.data();
                prop_assert_eq!(tied_data.len(), explicit_data.len());

                for (i, (&t, &e)) in tied_data.iter().zip(explicit_data.iter()).enumerate() {
                    prop_assert!(
                        (t - e).abs() < 1e-5,
                        "FALSIFIED TE-002-prop: tied[{}]={} != explicit[{}]={}",
                        i, t, i, e
                    );
                }
            }
        }

        // TE-004-prop: All outputs finite for random tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(30))]
            #[test]
            fn falsify_te_004_prop_finite(
                token_ids in proptest::collection::vec(0_u32..999, 1..16),
            ) {
                let config = TransformerConfig::tiny();
                let transformer = Transformer::new(&config);
                let logits = transformer.forward(&token_ids);
                let data = logits.data();

                for (i, &v) in data.iter().enumerate() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED TE-004-prop: logits[{}]={} non-finite (n_tokens={})",
                        i, v, token_ids.len()
                    );
                }
            }
        }
    }

    // =========================================================================
    // FALSIFY-PIPE-001: Cross-contract pipeline test
    //
    // Five-Whys (PMAT-354, Phase 8):
    //   Why 1: no test exercises the full §2.1.1 pipeline as a single chain
    //   Why 2: EM, TE, SM tests each validate one contract in isolation
    //   Why 3: bugs can hide at contract boundaries (shape mismatch between stages)
    //   Why 4: the embed→tied_lm_head→softmax chain is the critical inference path
    //   Why 5: cross-contract pipeline faults would only show in integration
    //
    // Pipeline: embed(token_ids) → transformer_layers → norm → tied_matmul → softmax
    // Claims verified:
    //   EM-001: embed output shape = (seq_len, d_model)
    //   TE-001: tied logits shape = (seq_len, vocab_size)
    //   SM-001: softmax(logits) sums to 1.0 per row
    //   SM-002: all probabilities positive
    //   SM-003: argmax preserved through softmax
    // =========================================================================

    /// FALSIFY-PIPE-001: Full embed → tied_lm_head → softmax pipeline
    #[test]
    fn falsify_pipe_001_embed_tied_softmax_pipeline() {
        let config = TransformerConfig::tiny();
        let transformer = Transformer::new(&config);

        let tokens = vec![0u32, 3, 7, 15, 42];
        let seq_len = tokens.len();
        let vocab_size = config.vocab_size;

        // Stage 1: Full forward pass (embed → layers → norm → tied matmul)
        let logits = transformer.forward(&tokens);
        let logits_data = logits.data();

        // TE-001: logits shape = (seq_len, vocab_size)
        assert_eq!(
            logits_data.len(),
            seq_len * vocab_size,
            "FALSIFIED PIPE-001/TE-001: logits len={} != seq_len({seq_len}) * vocab({vocab_size})",
            logits_data.len()
        );

        // TE-004: all logits finite
        for (i, &l) in logits_data.iter().enumerate() {
            assert!(l.is_finite(), "FALSIFIED PIPE-001/TE-004: logits[{i}] = {l} not finite");
        }

        // Stage 2: Apply softmax per row (the sampling step)
        let logits_slice = logits_data.as_slice().expect("operation should succeed");
        for row in 0..seq_len {
            let start = row * vocab_size;
            let end = start + vocab_size;
            let row_logits = &logits_slice[start..end];

            // Compute softmax for this row
            let max_val = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row_logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

            // SM-001: sums to 1.0
            let prob_sum: f32 = probs.iter().sum();
            assert!(
                (prob_sum - 1.0).abs() < 1e-4,
                "FALSIFIED PIPE-001/SM-001: row {row} prob sum={prob_sum}"
            );

            // SM-002: all positive
            for (i, &p) in probs.iter().enumerate() {
                assert!(p >= 0.0, "FALSIFIED PIPE-001/SM-002: row {row} prob[{i}]={p} negative");
            }

            // SM-003: argmax preserved
            let logit_argmax = row_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
                .expect("operation should succeed")
                .0;
            let prob_argmax = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
                .expect("operation should succeed")
                .0;
            assert_eq!(
                logit_argmax, prob_argmax,
                "FALSIFIED PIPE-001/SM-003: row {row} argmax changed {logit_argmax} → {prob_argmax}"
            );
        }
    }

    // =========================================================================
    // SSC-024: Transformer::from_safetensors() tests
    //
    // Tests for loading pretrained weights from SafeTensors files.
    // Uses synthetic SafeTensors with the tiny config to avoid needing
    // real 500MB model files in CI.
    // =========================================================================

    mod safetensors_tests {
        use super::*;
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};
        use tempfile::TempDir;

        /// Helper: create a synthetic SafeTensors file with all weights
        /// matching the tiny config (hidden=64, 2 layers, vocab=1000).
        fn create_tiny_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut tensors_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            // Helper to create f32 bytes
            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            // Embedding
            tensors_data.push((
                "model.embed_tokens.weight".to_string(),
                make_f32(vocab * hidden, 0.01),
                vec![vocab, hidden],
            ));

            // Final norm
            tensors_data.push((
                "model.norm.weight".to_string(),
                make_f32(hidden, 1.0),
                vec![hidden],
            ));

            // Per-layer weights
            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");

                // Layer norms
                tensors_data.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                tensors_data.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));

                // Attention projections
                tensors_data.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));

                // MLP projections
                tensors_data.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                tensors_data.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                tensors_data.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            // Build TensorViews from owned data and serialize
            let views: Vec<TensorView<'_>> = tensors_data
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid tensor view")
                })
                .collect();

            let named_views: Vec<(&str, &TensorView<'_>)> = tensors_data
                .iter()
                .zip(views.iter())
                .map(|((name, _, _), view)| (name.as_str(), view))
                .collect();

            let file_path = dir.join("model.safetensors");
            let serialized =
                serialize(named_views, None::<std::collections::HashMap<String, String>>)
                    .expect("serialize safetensors");
            std::fs::write(&file_path, serialized).expect("write safetensors file");
            file_path
        }

        /// Helper: create a SafeTensors file with bf16 weights (like real HF models)
        fn create_tiny_bf16_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut tensors_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            // Helper to create bf16 bytes
            let make_bf16 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(half::bf16::from_f32(val), n)
                    .flat_map(half::bf16::to_le_bytes)
                    .collect()
            };

            // Embedding
            tensors_data.push((
                "model.embed_tokens.weight".to_string(),
                make_bf16(vocab * hidden, 0.01),
                vec![vocab, hidden],
            ));

            // Final norm
            tensors_data.push((
                "model.norm.weight".to_string(),
                make_bf16(hidden, 1.0),
                vec![hidden],
            ));

            // Per-layer weights
            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");

                tensors_data.push((
                    format!("{p}.input_layernorm.weight"),
                    make_bf16(hidden, 1.0),
                    vec![hidden],
                ));
                tensors_data.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_bf16(hidden, 1.0),
                    vec![hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_bf16(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_bf16(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_bf16(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_bf16(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                tensors_data.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_bf16(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                tensors_data.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_bf16(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                tensors_data.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_bf16(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            let views: Vec<TensorView<'_>> = tensors_data
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).expect("valid tensor view")
                })
                .collect();

            let named_views: Vec<(&str, &TensorView<'_>)> = tensors_data
                .iter()
                .zip(views.iter())
                .map(|((name, _, _), view)| (name.as_str(), view))
                .collect();

            let file_path = dir.join("model.safetensors");
            let serialized =
                serialize(named_views, None::<std::collections::HashMap<String, String>>)
                    .expect("serialize safetensors");
            std::fs::write(&file_path, serialized).expect("write safetensors file");
            file_path
        }

        // -----------------------------------------------------------------
        // Happy path tests
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_f32_success() {
            let dir = TempDir::new().expect("create temp dir");
            create_tiny_safetensors(dir.path());
            let config = TransformerConfig::tiny();

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(
                result.is_ok(),
                "from_safetensors should succeed: {}",
                result.as_ref().err().map_or(String::new(), std::string::ToString::to_string)
            );

            let transformer = result.expect("validated above");
            assert_eq!(transformer.layers.len(), config.num_hidden_layers);
            assert!(transformer.lm_head.is_none()); // tiny config has no lm_head
        }

        #[test]
        fn test_ssc024_from_safetensors_bf16_conversion() {
            let dir = TempDir::new().expect("create temp dir");
            create_tiny_bf16_safetensors(dir.path());
            let config = TransformerConfig::tiny();

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(
                result.is_ok(),
                "BF16 loading should succeed: {}",
                result.as_ref().err().map_or(String::new(), std::string::ToString::to_string)
            );

            let transformer = result.expect("validated above");
            assert_eq!(transformer.layers.len(), config.num_hidden_layers);

            // Verify forward pass produces finite output
            let tokens = vec![1u32, 2, 3];
            let logits = transformer.forward(&tokens);
            assert_eq!(logits.len(), 3 * config.vocab_size);
            assert!(
                logits.data().iter().all(|v| v.is_finite()),
                "BF16-loaded model should produce finite outputs"
            );
        }

        #[test]
        fn test_ssc024_from_safetensors_single_file_path() {
            let dir = TempDir::new().expect("create temp dir");
            let file_path = create_tiny_safetensors(dir.path());
            let config = TransformerConfig::tiny();

            // Pass the file path directly, not the directory
            let result = Transformer::from_safetensors(&file_path, &config);
            assert!(
                result.is_ok(),
                "Direct file path should work: {}",
                result.as_ref().err().map_or(String::new(), std::string::ToString::to_string)
            );
        }

        #[test]
        fn test_ssc024_loaded_model_forward_produces_finite() {
            let dir = TempDir::new().expect("create temp dir");
            create_tiny_safetensors(dir.path());
            let config = TransformerConfig::tiny();

            let transformer =
                Transformer::from_safetensors(dir.path(), &config).expect("loading should succeed");

            // Run forward pass
            let tokens = vec![0u32, 5, 42, 99];
            let logits = transformer.forward(&tokens);

            assert_eq!(logits.len(), tokens.len() * config.vocab_size);
            let data = logits.data();
            let nan_count = data.iter().filter(|v| v.is_nan()).count();
            let inf_count = data.iter().filter(|v| v.is_infinite()).count();
            assert_eq!(nan_count, 0, "Loaded model output must not contain NaN");
            assert_eq!(inf_count, 0, "Loaded model output must not contain Inf");
        }

        // -----------------------------------------------------------------
        // Error case: no SafeTensors files
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_no_files() {
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err());
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(
                err_msg.contains("No SafeTensors files"),
                "Error should mention missing files: {err_msg}"
            );
        }

        // -----------------------------------------------------------------
        // Error case: shape mismatch
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_wrong_embedding_shape() {
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;

            // Create a file with wrong embedding shape
            let wrong_embed_bytes: Vec<u8> =
                std::iter::repeat_n(0.01_f32, 42).flat_map(f32::to_le_bytes).collect();

            // We need at least embedding + norm + 2 layers to pass validate_weights.
            // But the embedding shape is wrong, so validate_weight_shapes should catch it.
            // Actually, we need ALL required keys for validate_weights to pass first.
            // Let's create a full set but with wrong embedding size.
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;

            let mut td: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            // WRONG: embedding has 42 elements instead of vocab * hidden
            td.push(("model.embed_tokens.weight".to_string(), wrong_embed_bytes, vec![42]));
            td.push(("model.norm.weight".to_string(), make_f32(hidden, 1.0), vec![hidden]));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                td.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            let views: Vec<TensorView<'_>> = td
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view")
                })
                .collect();
            let named: Vec<(&str, &TensorView<'_>)> =
                td.iter().zip(views.iter()).map(|((n, _, _), v)| (n.as_str(), v)).collect();

            let file_path = dir.path().join("model.safetensors");
            let serialized =
                serialize(named, None::<std::collections::HashMap<String, String>>).expect("ser");
            std::fs::write(&file_path, serialized).expect("write");

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err(), "Wrong embedding shape should fail");
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(
                err_msg.contains("Shape mismatch") || err_msg.contains("embed_tokens"),
                "Error should indicate shape issue: {err_msg}"
            );
        }

        // -----------------------------------------------------------------
        // Error case: NaN in weights
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_nan_detection() {
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut td: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            // Embedding with NaN injected
            let mut embed_vals: Vec<f32> = vec![0.01; vocab * hidden];
            embed_vals[42] = f32::NAN;
            let embed_bytes: Vec<u8> = embed_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

            td.push(("model.embed_tokens.weight".to_string(), embed_bytes, vec![vocab, hidden]));
            td.push(("model.norm.weight".to_string(), make_f32(hidden, 1.0), vec![hidden]));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                td.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            let views: Vec<TensorView<'_>> = td
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view")
                })
                .collect();
            let named: Vec<(&str, &TensorView<'_>)> =
                td.iter().zip(views.iter()).map(|((n, _, _), v)| (n.as_str(), v)).collect();

            let file_path = dir.path().join("model.safetensors");
            let serialized =
                serialize(named, None::<std::collections::HashMap<String, String>>).expect("ser");
            std::fs::write(&file_path, serialized).expect("write");

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err(), "NaN in weights should fail");
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(err_msg.contains("NaN"), "Error should mention NaN: {err_msg}");
        }

        // -----------------------------------------------------------------
        // Error case: Inf in weights
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_inf_detection() {
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut td: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            // norm with Inf injected
            let mut norm_vals: Vec<f32> = vec![1.0; hidden];
            norm_vals[0] = f32::INFINITY;
            let norm_bytes: Vec<u8> = norm_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

            td.push((
                "model.embed_tokens.weight".to_string(),
                make_f32(vocab * hidden, 0.01),
                vec![vocab, hidden],
            ));
            td.push(("model.norm.weight".to_string(), norm_bytes, vec![hidden]));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                td.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            let views: Vec<TensorView<'_>> = td
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view")
                })
                .collect();
            let named: Vec<(&str, &TensorView<'_>)> =
                td.iter().zip(views.iter()).map(|((n, _, _), v)| (n.as_str(), v)).collect();

            let file_path = dir.path().join("model.safetensors");
            let serialized =
                serialize(named, None::<std::collections::HashMap<String, String>>).expect("ser");
            std::fs::write(&file_path, serialized).expect("write");

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err(), "Inf in weights should fail");
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(err_msg.contains("Inf"), "Error should mention Inf: {err_msg}");
        }

        // -----------------------------------------------------------------
        // Error case: missing layer weights (wrong layer count)
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_missing_layer() {
            let dir = TempDir::new().expect("create temp dir");
            // Create a file with 2 layers of weights
            create_tiny_safetensors(dir.path());

            // But try to load with config expecting 3 layers
            let mut config = TransformerConfig::tiny();
            config.num_hidden_layers = 3;

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err(), "Missing layer 2 should fail");
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(
                err_msg.contains("Missing") || err_msg.contains("layers.2"),
                "Error should mention missing layer: {err_msg}"
            );
        }

        // -----------------------------------------------------------------
        // Error case: wrong attention projection shape
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_wrong_q_proj_shape() {
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let q_dim = config.q_dim();
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut td: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            td.push((
                "model.embed_tokens.weight".to_string(),
                make_f32(vocab * hidden, 0.01),
                vec![vocab, hidden],
            ));
            td.push(("model.norm.weight".to_string(), make_f32(hidden, 1.0), vec![hidden]));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                td.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));

                // WRONG: q_proj has 7 elements instead of q_dim*hidden
                if i == 0 {
                    td.push((format!("{p}.self_attn.q_proj.weight"), make_f32(7, 0.01), vec![7]));
                } else {
                    td.push((
                        format!("{p}.self_attn.q_proj.weight"),
                        make_f32(q_dim * hidden, 0.01),
                        vec![q_dim, hidden],
                    ));
                }
                td.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(kv_hidden * hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(kv_hidden * hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * q_dim, 0.01),
                    vec![hidden, q_dim],
                ));
                td.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));
            }

            let views: Vec<TensorView<'_>> = td
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view")
                })
                .collect();
            let named: Vec<(&str, &TensorView<'_>)> =
                td.iter().zip(views.iter()).map(|((n, _, _), v)| (n.as_str(), v)).collect();

            let file_path = dir.path().join("model.safetensors");
            let serialized =
                serialize(named, None::<std::collections::HashMap<String, String>>).expect("ser");
            std::fs::write(&file_path, serialized).expect("write");

            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(result.is_err(), "Wrong q_proj shape should fail");
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(
                err_msg.contains("Shape mismatch") && err_msg.contains("q_proj"),
                "Error should mention q_proj shape mismatch: {err_msg}"
            );
        }

        // -----------------------------------------------------------------
        // Validate weight_shapes helper independently
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_validate_weight_shapes_success() {
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut weights = HashMap::new();
            weights.insert(
                "model.embed_tokens.weight".to_string(),
                Tensor::from_vec(vec![0.1; vocab * hidden], true),
            );
            weights
                .insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                weights.insert(
                    format!("{p}.input_layernorm.weight"),
                    Tensor::from_vec(vec![1.0; hidden], true),
                );
                weights.insert(
                    format!("{p}.post_attention_layernorm.weight"),
                    Tensor::from_vec(vec![1.0; hidden], true),
                );
                weights.insert(
                    format!("{p}.self_attn.q_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * hidden], true),
                );
                weights.insert(
                    format!("{p}.self_attn.k_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
                );
                weights.insert(
                    format!("{p}.self_attn.v_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
                );
                weights.insert(
                    format!("{p}.self_attn.o_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * hidden], true),
                );
                weights.insert(
                    format!("{p}.mlp.gate_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * intermediate], true),
                );
                weights.insert(
                    format!("{p}.mlp.up_proj.weight"),
                    Tensor::from_vec(vec![0.1; hidden * intermediate], true),
                );
                weights.insert(
                    format!("{p}.mlp.down_proj.weight"),
                    Tensor::from_vec(vec![0.1; intermediate * hidden], true),
                );
            }

            let result = Transformer::validate_weight_shapes(&weights, &config);
            assert!(
                result.is_ok(),
                "Valid shapes should pass: {}",
                result.as_ref().err().map_or(String::new(), std::string::ToString::to_string)
            );
        }

        #[test]
        fn test_ssc024_validate_weight_shapes_wrong_norm() {
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let vocab = config.vocab_size;

            let mut weights = HashMap::new();
            weights.insert(
                "model.embed_tokens.weight".to_string(),
                Tensor::from_vec(vec![0.1; vocab * hidden], true),
            );
            // Wrong norm size: 3 instead of hidden
            weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; 3], true));

            let result = Transformer::validate_weight_shapes(&weights, &config);
            assert!(result.is_err());
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(err_msg.contains("model.norm.weight"));
        }

        // -----------------------------------------------------------------
        // Validate weight_values helper independently
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_validate_weight_values_clean() {
            let mut weights = HashMap::new();
            weights.insert("a".to_string(), Tensor::from_vec(vec![0.1, 0.2, 0.3], true));
            weights.insert("b".to_string(), Tensor::from_vec(vec![1.0, -1.0, 0.0], true));

            let result = Transformer::validate_weight_values(&weights);
            assert!(result.is_ok());
        }

        #[test]
        fn test_ssc024_validate_weight_values_nan() {
            let mut weights = HashMap::new();
            weights.insert("clean".to_string(), Tensor::from_vec(vec![0.1, 0.2], true));
            weights
                .insert("poisoned".to_string(), Tensor::from_vec(vec![0.1, f32::NAN, 0.3], true));

            let result = Transformer::validate_weight_values(&weights);
            assert!(result.is_err());
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(err_msg.contains("NaN"));
            assert!(err_msg.contains("poisoned"));
        }

        #[test]
        fn test_ssc024_validate_weight_values_inf() {
            let mut weights = HashMap::new();
            weights.insert("w".to_string(), Tensor::from_vec(vec![f32::NEG_INFINITY, 0.2], true));

            let result = Transformer::validate_weight_values(&weights);
            assert!(result.is_err());
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error"),
            };
            assert!(err_msg.contains("Inf"));
        }

        /// GH-262: Qwen3-4B shape validation with q_dim != hidden_size.
        ///
        /// Uses a 1-layer config mimicking Qwen3-4B dimensions to verify
        /// validate_weight_shapes accepts the correct q_dim-based shapes.
        #[test]
        fn test_gh262_qwen3_4b_weight_shapes_q_dim_ne_hidden() {
            // Minimal Qwen3-like config: q_dim (128) != hidden_size (80)
            let config = TransformerConfig {
                hidden_size: 80,
                num_attention_heads: 4,
                num_kv_heads: 2,
                intermediate_size: 128,
                num_hidden_layers: 1,
                vocab_size: 256,
                max_position_embeddings: 512,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                use_bias: false,
                head_dim_override: Some(32), // head_dim=32, so q_dim = 4*32 = 128, hidden=80
                architecture: crate::transformer::config::ModelArchitecture::Decoder,
                hf_architecture: None,
                hf_model_type: None,
                tie_word_embeddings: false,
            };

            let hidden = config.hidden_size; // 80
            let q_dim = config.q_dim(); // 4 * 32 = 128
            let kv_hidden = config.num_kv_heads * config.head_dim(); // 2 * 32 = 64
            let intermediate = config.intermediate_size; // 128
            let vocab = config.vocab_size; // 256

            // Verify q_dim != hidden (the Qwen3-4B characteristic)
            assert_ne!(q_dim, hidden, "test requires q_dim != hidden_size");

            let mut weights = HashMap::new();
            weights.insert(
                "model.embed_tokens.weight".to_string(),
                Tensor::from_vec(vec![0.1; vocab * hidden], true),
            );
            weights
                .insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));

            let p = "model.layers.0";
            weights.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden], true),
            );
            weights.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden], true),
            );
            // Q: [q_dim, hidden] — NOT [hidden, hidden]
            weights.insert(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; q_dim * hidden], true),
            );
            // K: [kv_hidden, hidden]
            weights.insert(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; kv_hidden * hidden], true),
            );
            // V: [kv_hidden, hidden]
            weights.insert(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; kv_hidden * hidden], true),
            );
            // O: [hidden, q_dim] — NOT [hidden, hidden]
            weights.insert(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * q_dim], true),
            );
            weights.insert(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * intermediate], true),
            );
            weights.insert(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * intermediate], true),
            );
            weights.insert(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate * hidden], true),
            );

            // Should pass: shapes use q_dim for Q/O projections
            let result = Transformer::validate_weight_shapes(&weights, &config);
            assert!(
                result.is_ok(),
                "Qwen3-like shapes (q_dim={q_dim} != hidden={hidden}) should validate: {:?}",
                result.err()
            );

            // Should also construct successfully via from_params
            let model = Transformer::from_params(&config, &weights);
            assert!(model.is_some(), "Qwen3-like model with q_dim != hidden should construct");
        }

        /// GH-262: Using hidden_size instead of q_dim for q_proj must fail.
        #[test]
        fn test_gh262_wrong_q_proj_size_hidden_instead_of_q_dim() {
            let config = TransformerConfig {
                hidden_size: 80,
                num_attention_heads: 4,
                num_kv_heads: 2,
                intermediate_size: 128,
                num_hidden_layers: 1,
                vocab_size: 256,
                max_position_embeddings: 512,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                use_bias: false,
                head_dim_override: Some(32), // q_dim=128, hidden=80
                architecture: crate::transformer::config::ModelArchitecture::Decoder,
                hf_architecture: None,
                hf_model_type: None,
                tie_word_embeddings: false,
            };

            let hidden = config.hidden_size; // 80
            let kv_hidden = config.num_kv_heads * config.head_dim(); // 64
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut weights = HashMap::new();
            weights.insert(
                "model.embed_tokens.weight".to_string(),
                Tensor::from_vec(vec![0.1; vocab * hidden], true),
            );
            weights
                .insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));

            let p = "model.layers.0";
            weights.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden], true),
            );
            weights.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![1.0; hidden], true),
            );
            // BUG: q_proj uses hidden*hidden (6400) instead of q_dim*hidden (10240)
            weights.insert(
                format!("{p}.self_attn.q_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * hidden], true),
            );
            weights.insert(
                format!("{p}.self_attn.k_proj.weight"),
                Tensor::from_vec(vec![0.1; kv_hidden * hidden], true),
            );
            weights.insert(
                format!("{p}.self_attn.v_proj.weight"),
                Tensor::from_vec(vec![0.1; kv_hidden * hidden], true),
            );
            weights.insert(
                format!("{p}.self_attn.o_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * hidden], true),
            );
            weights.insert(
                format!("{p}.mlp.gate_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * intermediate], true),
            );
            weights.insert(
                format!("{p}.mlp.up_proj.weight"),
                Tensor::from_vec(vec![0.1; hidden * intermediate], true),
            );
            weights.insert(
                format!("{p}.mlp.down_proj.weight"),
                Tensor::from_vec(vec![0.1; intermediate * hidden], true),
            );

            // Must fail: q_proj has wrong size
            let result = Transformer::validate_weight_shapes(&weights, &config);
            assert!(result.is_err(), "hidden*hidden q_proj should fail when q_dim != hidden");
            let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(
                err_msg.contains("q_proj") && err_msg.contains("Shape mismatch"),
                "Error should mention q_proj shape mismatch, got: {err_msg}"
            );
        }

        // -----------------------------------------------------------------
        // Name mapping integration: Qwen2 bias tensors are preserved
        // -----------------------------------------------------------------

        #[test]
        fn test_ssc024_from_safetensors_with_extra_bias_tensors() {
            // Qwen2 models have bias tensors that are loaded alongside weights.
            // from_params ignores them (doesn't look for bias keys), but they
            // should not cause errors.
            let dir = TempDir::new().expect("create temp dir");
            let config = TransformerConfig::tiny();
            let hidden = config.hidden_size;
            let kv_hidden = config.num_kv_heads * config.head_dim();
            let intermediate = config.intermediate_size;
            let vocab = config.vocab_size;

            let mut td: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

            let make_f32 = |n: usize, val: f32| -> Vec<u8> {
                std::iter::repeat_n(val, n).flat_map(f32::to_le_bytes).collect()
            };

            td.push((
                "model.embed_tokens.weight".to_string(),
                make_f32(vocab * hidden, 0.01),
                vec![vocab, hidden],
            ));
            td.push(("model.norm.weight".to_string(), make_f32(hidden, 1.0), vec![hidden]));

            for i in 0..config.num_hidden_layers {
                let p = format!("model.layers.{i}");
                td.push((
                    format!("{p}.input_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.post_attention_layernorm.weight"),
                    make_f32(hidden, 1.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.self_attn.q_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.k_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.weight"),
                    make_f32(hidden * kv_hidden, 0.01),
                    vec![kv_hidden, hidden],
                ));
                td.push((
                    format!("{p}.self_attn.o_proj.weight"),
                    make_f32(hidden * hidden, 0.01),
                    vec![hidden, hidden],
                ));
                td.push((
                    format!("{p}.mlp.gate_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.up_proj.weight"),
                    make_f32(hidden * intermediate, 0.01),
                    vec![intermediate, hidden],
                ));
                td.push((
                    format!("{p}.mlp.down_proj.weight"),
                    make_f32(intermediate * hidden, 0.01),
                    vec![hidden, intermediate],
                ));

                // Qwen2-style bias tensors
                td.push((
                    format!("{p}.self_attn.q_proj.bias"),
                    make_f32(hidden, 0.0),
                    vec![hidden],
                ));
                td.push((
                    format!("{p}.self_attn.k_proj.bias"),
                    make_f32(kv_hidden, 0.0),
                    vec![kv_hidden],
                ));
                td.push((
                    format!("{p}.self_attn.v_proj.bias"),
                    make_f32(kv_hidden, 0.0),
                    vec![kv_hidden],
                ));
            }

            let views: Vec<TensorView<'_>> = td
                .iter()
                .map(|(_, bytes, shape)| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("view")
                })
                .collect();
            let named: Vec<(&str, &TensorView<'_>)> =
                td.iter().zip(views.iter()).map(|((n, _, _), v)| (n.as_str(), v)).collect();

            let file_path = dir.path().join("model.safetensors");
            let serialized =
                serialize(named, None::<std::collections::HashMap<String, String>>).expect("ser");
            std::fs::write(&file_path, serialized).expect("write");

            // Should succeed even with extra bias tensors
            let result = Transformer::from_safetensors(dir.path(), &config);
            assert!(
                result.is_ok(),
                "Extra bias tensors should not cause failure: {}",
                result.as_ref().err().map_or(String::new(), std::string::ToString::to_string)
            );
        }
    }
}
