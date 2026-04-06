//! Complete encoder model (BERT/RoBERTa/CodeBERT)
//!
//! Provides the full encoder pipeline:
//! ```text
//! token_ids → Embedding + PositionEmbedding + TokenTypeEmbedding → LayerNorm
//!           → N × EncoderBlock → [seq_len, hidden_size]
//! ```
//!
//! For classification, use CLS pooling on the output (position 0).

use crate::autograd::add;
use crate::error::{Error, Result};
use crate::Tensor;
use std::collections::HashMap;
use std::path::Path;

use super::config::TransformerConfig;
use super::embedding::{Embedding, LearnedPositionEmbedding};
use super::encoder_block::EncoderBlock;
use super::norm::LayerNorm;
use super::weights::{load_safetensors_weights, Architecture};

/// Complete encoder model (BERT/RoBERTa/CodeBERT).
pub struct EncoderModel {
    /// Configuration
    pub config: TransformerConfig,
    /// Token embedding
    pub embed_tokens: Embedding,
    /// Position embedding (learned absolute)
    pub position_embeddings: LearnedPositionEmbedding,
    /// Token type embedding (segment A/B, optional for RoBERTa but present in weights)
    pub token_type_embeddings: Option<Embedding>,
    /// Post-embedding LayerNorm
    pub embeddings_layernorm: LayerNorm,
    /// Encoder layers
    pub layers: Vec<EncoderBlock>,
}

impl EncoderModel {
    /// Create new encoder with default initialization.
    pub fn new(config: &TransformerConfig) -> Self {
        let max_positions = config.max_position_embeddings;
        let eps = config.rms_norm_eps;
        let layers = (0..config.num_hidden_layers).map(|i| EncoderBlock::new(config, i)).collect();

        Self {
            config: config.clone(),
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            position_embeddings: LearnedPositionEmbedding::new(max_positions, config.hidden_size),
            token_type_embeddings: Some(Embedding::new(2, config.hidden_size)),
            embeddings_layernorm: LayerNorm::new(config.hidden_size, eps),
            layers,
        }
    }

    /// Create encoder from pre-trained parameters (after RoBERTa name mapping).
    ///
    /// Expected parameter names:
    /// - `encoder.embed_tokens.weight`
    /// - `encoder.position_embeddings.weight`
    /// - `encoder.token_type_embeddings.weight` (optional)
    /// - `encoder.embeddings_layernorm.{weight,bias}`
    /// - `encoder.layers.{i}.*`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
    ) -> Option<Self> {
        let max_positions = config.max_position_embeddings;

        let embed_tokens = Embedding::from_params(
            params,
            "encoder.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;

        let position_embeddings = LearnedPositionEmbedding::from_params(
            params,
            "encoder.position_embeddings.weight",
            max_positions,
            config.hidden_size,
        )?;

        // Token type embeddings are optional (RoBERTa has them but sets to zero)
        // CodeBERT has type_vocab_size=1, standard BERT/RoBERTa has 2.
        // Infer from actual tensor shape rather than hardcoding.
        let token_type_embeddings =
            params.get("encoder.token_type_embeddings.weight").and_then(|tensor| {
                let type_vocab_size = tensor.len() / config.hidden_size;
                if type_vocab_size == 0 || tensor.len() != type_vocab_size * config.hidden_size {
                    return None;
                }
                Embedding::from_params(
                    params,
                    "encoder.token_type_embeddings.weight",
                    type_vocab_size,
                    config.hidden_size,
                )
            });

        let embeddings_layernorm = LayerNorm::from_params(
            params,
            "encoder.embeddings_layernorm",
            config.rms_norm_eps,
            config.hidden_size,
        )?;

        let layers: Option<Vec<EncoderBlock>> = (0..config.num_hidden_layers)
            .map(|i| EncoderBlock::from_params(config, params, i))
            .collect();
        let layers = layers?;

        Some(Self {
            config: config.clone(),
            embed_tokens,
            position_embeddings,
            token_type_embeddings,
            embeddings_layernorm,
            layers,
        })
    }

    /// Load encoder from SafeTensors file(s).
    pub fn from_safetensors(config: &TransformerConfig, model_path: &Path) -> Result<Self> {
        let weights = load_safetensors_weights(model_path, Architecture::RoBERTa)?;
        Self::from_params(config, &weights).ok_or_else(|| {
            Error::ConfigError("Failed to construct encoder from loaded weights".into())
        })
    }

    /// Forward pass: token_ids → hidden states [seq_len × hidden_size].
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let h = self.config.hidden_size;

        // Token embeddings
        let token_emb = self.embed_tokens.forward(token_ids);

        // Position embeddings
        let pos_emb = self.position_embeddings.forward(seq_len);

        // Add token + position embeddings
        let mut combined = add(&token_emb, &pos_emb);

        // Add token type embeddings (all zeros = segment A)
        if let Some(ref tte) = self.token_type_embeddings {
            let type_ids: Vec<u32> = vec![0; seq_len];
            let type_emb = tte.forward(&type_ids);
            combined = add(&combined, &type_emb);
        }

        // Post-embedding LayerNorm
        let mut hidden = self.embeddings_layernorm.forward_batched(&combined, seq_len, h);

        // Pass through encoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        hidden
    }

    /// Extract [CLS] embedding (position 0) from hidden states.
    ///
    /// For classification, the [CLS] token at position 0 attends bidirectionally
    /// to all other tokens, making it a summary representation.
    pub fn cls_embedding(&self, token_ids: &[u32]) -> Tensor {
        let hidden = self.forward(token_ids);
        let h = self.config.hidden_size;
        let data = hidden.data();
        let slice = data.as_slice().expect("hidden contiguous");
        Tensor::from_vec(slice[..h].to_vec(), false)
    }

    /// Get total parameter count.
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.embed_tokens.vocab_size() * self.embed_tokens.hidden_size();
        count += self.position_embeddings.weight.len();
        if let Some(ref tte) = self.token_type_embeddings {
            count += tte.vocab_size() * tte.hidden_size();
        }
        count += self.embeddings_layernorm.weight.len() * 2; // weight + bias
        for layer in &self.layers {
            count += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }
        count
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::transformer::ModelArchitecture;

    fn tiny_encoder_config() -> TransformerConfig {
        // Minimal config for testing
        TransformerConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 4,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            architecture: ModelArchitecture::Encoder,
            ..TransformerConfig::tiny()
        }
    }

    #[test]
    fn clf_001_encoder_model_forward_shape() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![1, 2, 3, 4];
        let output = model.forward(&token_ids);
        assert_eq!(output.len(), 4 * config.hidden_size);
    }

    #[test]
    fn clf_001_encoder_model_forward_finite() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![10, 20, 30];
        let output = model.forward(&token_ids);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn clf_001_encoder_cls_embedding_shape() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![5, 10, 15];
        let cls = model.cls_embedding(&token_ids);
        assert_eq!(cls.len(), config.hidden_size);
    }

    #[test]
    fn clf_001_encoder_cls_embedding_deterministic() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let token_ids = vec![1, 2, 3];
        let cls1 = model.cls_embedding(&token_ids);
        let cls2 = model.cls_embedding(&token_ids);
        let d1 = cls1.data();
        let d2 = cls2.data();
        let s1 = d1.as_slice().unwrap();
        let s2 = d2.as_slice().unwrap();
        assert_eq!(s1, s2, "CLS embedding must be deterministic");
    }

    #[test]
    fn clf_001_encoder_num_parameters() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let count = model.num_parameters();
        // Should be > 0 and reasonable
        assert!(count > 1000, "encoder should have substantial params, got {count}");
    }

    #[test]
    fn test_encoder_forward_single_token() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let output = model.forward(&[42]);
        assert_eq!(output.len(), config.hidden_size);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encoder_cls_embedding_finite() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let cls = model.cls_embedding(&[1, 2, 3, 4, 5]);
        let data = cls.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encoder_config_stored() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert_eq!(model.config.hidden_size, 32);
        assert_eq!(model.config.num_hidden_layers, 2);
        assert_eq!(model.config.vocab_size, 100);
    }

    #[test]
    fn test_encoder_layers_count() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_encoder_token_type_embeddings_present() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert!(model.token_type_embeddings.is_some());
    }

    #[test]
    fn test_encoder_from_params_missing_weights() {
        let config = tiny_encoder_config();
        let empty_params: HashMap<String, Tensor> = HashMap::new();
        let result = EncoderModel::from_params(&config, &empty_params);
        assert!(result.is_none(), "from_params should return None with empty params");
    }

    #[test]
    fn test_encoder_from_safetensors_missing_file() {
        let config = tiny_encoder_config();
        let result = EncoderModel::from_safetensors(&config, std::path::Path::new("/nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_forward_different_seq_lens() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);

        for seq_len in [1, 2, 4, 8, 16] {
            let token_ids: Vec<u32> = (0..seq_len as u32).collect();
            let output = model.forward(&token_ids);
            assert_eq!(
                output.len(),
                seq_len * config.hidden_size,
                "Output mismatch for seq_len={seq_len}"
            );
        }
    }

    #[test]
    fn test_encoder_num_params_includes_all_components() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let total = model.num_parameters();

        // Embedding: vocab_size * hidden_size = 100 * 32 = 3200
        let embed_params = config.vocab_size * config.hidden_size;
        // Position: max_pos * hidden = 32 * 32 = 1024
        let pos_params = config.max_position_embeddings * config.hidden_size;
        // Token type: 2 * hidden = 64
        let tte_params = 2 * config.hidden_size;
        // LayerNorm: hidden * 2 = 64 (weight + bias)
        let ln_params = config.hidden_size * 2;

        let non_layer_params = embed_params + pos_params + tte_params + ln_params;
        assert!(
            total > non_layer_params,
            "Total params ({total}) should exceed non-layer params ({non_layer_params})"
        );
    }

    #[test]
    fn test_encoder_forward_max_token_id() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        // Use token ID at the edge of vocab
        let output = model.forward(&[99]); // vocab_size = 100, max valid = 99
        assert_eq!(output.len(), config.hidden_size);
    }

    #[test]
    fn test_encoder_deterministic_across_calls() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let ids = vec![10, 20, 30, 40];

        let out1 = model.forward(&ids);
        let out2 = model.forward(&ids);

        let d1 = out1.data();
        let d2 = out2.data();
        let s1 = d1.as_slice().unwrap();
        let s2 = d2.as_slice().unwrap();
        assert_eq!(s1, s2);
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_encoder_forward_varying_vocab_ids() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        // Use a range of token IDs
        let ids: Vec<u32> = (0..20).collect();
        let output = model.forward(&ids);
        assert_eq!(output.len(), 20 * config.hidden_size);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encoder_from_params_partial_weights() {
        let config = tiny_encoder_config();
        let h = config.hidden_size;
        let v = config.vocab_size;
        let mut params: HashMap<String, Tensor> = HashMap::new();

        // Add only embed_tokens, not position_embeddings → should return None
        let embed_data = vec![0.0_f32; v * h];
        params
            .insert("encoder.embed_tokens.weight".to_string(), Tensor::from_vec(embed_data, false));

        let result = EncoderModel::from_params(&config, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_encoder_cls_embedding_different_inputs_differ() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let cls1 = model.cls_embedding(&[1, 2, 3]);
        let cls2 = model.cls_embedding(&[10, 20, 30]);
        let d1 = cls1.data();
        let d2 = cls2.data();
        let s1 = d1.as_slice().unwrap();
        let s2 = d2.as_slice().unwrap();
        // Different inputs should (with overwhelming probability) produce different embeddings
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_encoder_position_embeddings_present() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert_eq!(
            model.position_embeddings.weight.len(),
            config.max_position_embeddings * config.hidden_size
        );
    }

    #[test]
    fn test_encoder_embeddings_layernorm_present() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert_eq!(model.embeddings_layernorm.weight.len(), config.hidden_size);
    }

    #[test]
    fn test_encoder_num_parameters_varies_with_config() {
        let config1 = tiny_encoder_config();
        let model1 = EncoderModel::new(&config1);

        let config2 = TransformerConfig {
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_kv_heads: 8,
            intermediate_size: 128,
            vocab_size: 200,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            architecture: ModelArchitecture::Encoder,
            ..TransformerConfig::tiny()
        };
        let model2 = EncoderModel::new(&config2);

        // Larger model should have more parameters
        assert!(model2.num_parameters() > model1.num_parameters());
    }

    #[test]
    fn test_encoder_forward_two_tokens() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        let output = model.forward(&[5, 10]);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_encoder_forward_at_max_position() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        // Use max_position_embeddings tokens
        let ids: Vec<u32> = (0..config.max_position_embeddings as u32).collect();
        let output = model.forward(&ids);
        assert_eq!(output.len(), config.max_position_embeddings * config.hidden_size);
    }

    #[test]
    fn test_encoder_no_token_type_embeddings() {
        let config = tiny_encoder_config();
        let mut model = EncoderModel::new(&config);
        // Remove token type embeddings
        model.token_type_embeddings = None;
        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.len(), 3 * config.hidden_size);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encoder_num_parameters_without_tte() {
        let config = tiny_encoder_config();
        let mut model = EncoderModel::new(&config);
        let with_tte = model.num_parameters();
        model.token_type_embeddings = None;
        let without_tte = model.num_parameters();
        assert!(with_tte > without_tte);
        // Difference should be 2 * hidden_size (token type embedding for 2 types)
        assert_eq!(with_tte - without_tte, 2 * config.hidden_size);
    }

    #[test]
    fn test_encoder_config_is_encoder() {
        let config = tiny_encoder_config();
        let model = EncoderModel::new(&config);
        assert!(model.config.is_encoder());
    }
}
