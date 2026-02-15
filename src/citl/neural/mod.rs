//! Neural encoder for multi-language error embeddings.
//!
//! Uses transformer architecture with contrastive learning to create
//! embeddings that cluster similar errors across languages.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │ Tokenized   │────►│  Embedding  │────►│ Transformer │────►│   Pooled    │
//! │   Input     │     │   Layer     │     │   Encoder   │     │  Embedding  │
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//! ```
//!
//! # GPU Support
//!
//! The encoder automatically uses GPU when available via trueno's backend:
//! - Build with `--features gpu` for GPU acceleration
//! - Falls back to SIMD CPU when GPU unavailable
//!
//! # References
//!
//! - Chen, T., et al. (2020). A simple framework for contrastive learning. ICML.
//! - Gao, T., et al. (2021). `SimCSE`: Simple contrastive learning of sentence embeddings.

use crate::autograd::{no_grad, Tensor};
use crate::nn::{Dropout, Linear, Module};
use std::collections::HashMap;
#[allow(unused_imports)]
use trueno::Vector;

/// Configuration for the neural encoder.
#[derive(Debug, Clone)]
pub struct NeuralEncoderConfig {
    /// Vocabulary size for tokenizer
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension in transformer FFN
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Output embedding dimension
    pub output_dim: usize,
}

impl Default for NeuralEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            embed_dim: 256,
            hidden_dim: 512,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            dropout: 0.1,
            output_dim: 256,
        }
    }
}

impl NeuralEncoderConfig {
    /// Create a minimal config for testing.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            vocab_size: 1000,
            embed_dim: 64,
            hidden_dim: 128,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 128,
            dropout: 0.0,
            output_dim: 64,
        }
    }

    /// Create a small config for development.
    #[must_use]
    pub fn small() -> Self {
        Self {
            vocab_size: 4096,
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 256,
            dropout: 0.1,
            output_dim: 128,
        }
    }
}

/// Neural error encoder using transformer architecture.
///
/// Encodes compiler errors and source context into dense embeddings
/// suitable for similarity search and pattern matching.
#[derive(Debug)]
pub struct NeuralErrorEncoder {
    config: NeuralEncoderConfig,
    /// Token embedding layer
    token_embedding: Embedding,
    /// Position embedding layer
    position_embedding: Embedding,
    /// Transformer encoder layers
    encoder_layers: Vec<TransformerLayer>,
    /// Output projection to final embedding dimension
    output_projection: Linear,
    /// Dropout for regularization
    dropout: Dropout,
    /// Vocabulary for tokenization
    vocab: Vocabulary,
    /// Whether model is in training mode
    training: bool,
}

impl NeuralErrorEncoder {
    /// Create a new neural encoder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(NeuralEncoderConfig::default())
    }

    /// Create a new neural encoder with custom configuration.
    #[must_use]
    pub fn with_config(config: NeuralEncoderConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.embed_dim);
        let position_embedding = Embedding::new(config.max_seq_len, config.embed_dim);

        let encoder_layers = (0..config.num_layers)
            .map(|_| {
                TransformerLayer::new(
                    config.embed_dim,
                    config.num_heads,
                    config.hidden_dim,
                    config.dropout,
                )
            })
            .collect();

        let output_projection = Linear::new(config.embed_dim, config.output_dim);
        let dropout = Dropout::new(config.dropout);
        let vocab = Vocabulary::for_rust_errors();

        Self {
            config,
            token_embedding,
            position_embedding,
            encoder_layers,
            output_projection,
            dropout,
            vocab,
            training: false,
        }
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode (disables dropout).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode.
    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &NeuralEncoderConfig {
        &self.config
    }

    /// Encode an error message and source context into an embedding.
    ///
    /// # Arguments
    ///
    /// * `error_message` - The compiler error message
    /// * `source_context` - Source code around the error location
    /// * `source_lang` - Source language identifier (e.g., "python", "rust")
    ///
    /// # Returns
    ///
    /// A dense embedding vector of dimension `output_dim`.
    pub fn encode(&self, error_message: &str, source_context: &str, source_lang: &str) -> Vec<f32> {
        // Tokenize input
        let tokens = self.tokenize(error_message, source_context, source_lang);

        // Convert to tensor
        let token_ids: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let seq_len = tokens.len();

        let x = Tensor::new(&token_ids, &[1, seq_len]);

        // Forward pass (no gradients for inference)
        let embedding = no_grad(|| self.forward(&x));

        // Extract embedding vector
        embedding.data().to_vec()
    }

    /// Encode a batch of inputs for training.
    ///
    /// # Arguments
    ///
    /// * `batch` - Vector of (`error_message`, `source_context`, `source_lang`) tuples
    ///
    /// # Returns
    ///
    /// Tensor of shape [`batch_size`, `output_dim`]
    pub fn encode_batch(&self, batch: &[(&str, &str, &str)]) -> Tensor {
        let batch_size = batch.len();
        let max_len = self.config.max_seq_len;

        // Tokenize and pad all inputs
        let mut all_tokens = Vec::with_capacity(batch_size * max_len);
        for (error_msg, source_ctx, lang) in batch {
            let tokens = self.tokenize(error_msg, source_ctx, lang);
            // Pad to max_len
            for i in 0..max_len {
                all_tokens.push(tokens.get(i).copied().unwrap_or(0) as f32);
            }
        }

        let x = Tensor::new(&all_tokens, &[batch_size, max_len]);
        self.forward(&x)
    }

    /// Forward pass through the encoder.
    fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Token embeddings
        let tok_emb = self.token_embedding.forward(x);

        // Position embeddings (broadcast manually)
        let positions: Vec<f32> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|i| i as f32))
            .collect();
        let pos_ids = Tensor::new(&positions, &[batch_size, seq_len]);
        let pos_emb = self.position_embedding.forward(&pos_ids);

        // Combine embeddings
        let mut hidden = tok_emb.add(&pos_emb);

        // Apply dropout
        if self.training {
            hidden = self.dropout.forward(&hidden);
        }

        // Pass through transformer layers
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden, self.training);
        }

        // Mean pooling over sequence dimension
        let pooled = mean_pool(&hidden);

        // Project to output dimension
        let output = self.output_projection.forward(&pooled);

        // L2 normalize for cosine similarity
        l2_normalize(&output)
    }

    /// Tokenize input into token IDs.
    fn tokenize(&self, error_message: &str, source_context: &str, source_lang: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(self.config.max_seq_len);

        // Add special tokens
        tokens.push(self.vocab.cls_token());
        tokens.push(self.vocab.lang_token(source_lang));

        // Tokenize error message
        for token in self.vocab.tokenize(error_message) {
            if tokens.len() >= self.config.max_seq_len - 2 {
                break;
            }
            tokens.push(token);
        }

        // Add separator
        tokens.push(self.vocab.sep_token());

        // Tokenize source context
        for token in self.vocab.tokenize(source_context) {
            if tokens.len() >= self.config.max_seq_len - 1 {
                break;
            }
            tokens.push(token);
        }

        // Add end token
        tokens.push(self.vocab.eos_token());

        tokens
    }

    /// Get total number of parameters.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.embed_dim
            + self.config.max_seq_len * self.config.embed_dim;

        let layer_params = self.config.num_layers
            * (4 * self.config.embed_dim * self.config.embed_dim  // Q, K, V, O projections
               + 2 * self.config.embed_dim * self.config.hidden_dim  // FFN
               + 4 * self.config.embed_dim); // LayerNorm params

        let output_params = self.config.embed_dim * self.config.output_dim + self.config.output_dim;

        embed_params + layer_params + output_params
    }
}

impl Default for NeuralErrorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple embedding layer.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
struct Embedding {
    weight: Tensor,
    #[allow(dead_code)]
    num_embeddings: usize,
    #[allow(dead_code)]
    embedding_dim: usize,
}

impl Embedding {
    fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Initialize with small random values
        let data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|i| (i as f32 * 0.1).sin() * 0.02)
            .collect();

        let weight = Tensor::new(&data, &[num_embeddings, embedding_dim]).requires_grad();

        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, seq_len] of token IDs
        // output: [batch, seq_len, embed_dim]
        embedding_lookup(&self.weight, x)
    }
}

/// Simplified transformer encoder layer.
#[derive(Debug)]
struct TransformerLayer {
    /// Self-attention Q, K, V projections
    qkv_proj: Linear,
    /// Output projection
    out_proj: Linear,
    /// FFN first layer
    ffn1: Linear,
    /// FFN second layer
    ffn2: Linear,
    /// Layer norm 1
    norm1: LayerNorm,
    /// Layer norm 2
    norm2: LayerNorm,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Dropout probability
    dropout_p: f32,
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("mod_part_04.rs");
