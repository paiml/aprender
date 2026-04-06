
impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Hidden dimension (model dimension)
    /// * `num_heads` - Number of attention heads
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `eps` - Layer normalization epsilon
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `hidden_dim` is zero or not divisible by `num_heads`
    /// - `num_heads` is zero
    /// - `intermediate_dim` is zero
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }

        let attn_norm = LayerNorm::new(hidden_dim, eps)?;
        // Use standard MHA with Q/K/V/O projections
        let attention = MultiHeadAttention::mha(hidden_dim, num_heads)?;
        let ffn_norm = LayerNorm::new(hidden_dim, eps)?;
        let ffn = FeedForward::new(hidden_dim, intermediate_dim)?;

        Ok(Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            hidden_dim,
            num_heads,
        })
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape is invalid
    ///
    /// # Note
    ///
    /// This simplified implementation uses the same input for Q, K, V (self-attention).
    /// Production models would compute Q, K, V projections separately.
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Expected last dimension {}, got {}",
                    self.hidden_dim, last_dim
                ),
            });
        }

        // Pre-norm attention block
        let normed = self.attn_norm.forward(input)?;

        // Self-attention with proper Q/K/V/O projections via MultiHeadAttention
        let attn_out = self.attention.forward(&normed)?;

        // Residual connection
        let mut residual1 = Vec::with_capacity(input.data().len());
        for (i, &val) in input.data().iter().enumerate() {
            residual1.push(val + attn_out.data()[i]);
        }
        let after_attn = Tensor::from_vec(shape.to_vec(), residual1)?;

        // Pre-norm FFN block
        let normed2 = self.ffn_norm.forward(&after_attn)?;
        let ffn_out = self.ffn.forward(&normed2)?;

        // Residual connection
        let mut residual2 = Vec::with_capacity(after_attn.data().len());
        for (i, &val) in after_attn.data().iter().enumerate() {
            residual2.push(val + ffn_out.data()[i]);
        }

        Tensor::from_vec(shape.to_vec(), residual2)
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get mutable reference to attention layer normalization
    pub fn attn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.attn_norm
    }

    /// Get mutable reference to multi-head attention
    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention {
        &mut self.attention
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get mutable reference to FFN layer normalization
    pub fn ffn_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.ffn_norm
    }

    /// Get mutable reference to FFN
    pub fn ffn_mut(&mut self) -> &mut FeedForward {
        &mut self.ffn
    }
}

/// Embedding layer for converting token IDs to vectors
///
/// Maps discrete token IDs to continuous vector representations.
/// This is the first layer in a transformer model.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Embedding weights: `[vocab_size, embed_dim]`
    weights: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding layer
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary
    /// * `embed_dim` - Dimension of embedding vectors
    ///
    /// # Errors
    ///
    /// Returns error if `vocab_size` or `embed_dim` is zero
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if embed_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "embed_dim must be > 0".to_string(),
            });
        }

        let weights = vec![0.0; vocab_size * embed_dim];

        Ok(Self {
            vocab_size,
            embed_dim,
            weights,
        })
    }

    /// Look up embeddings for token IDs
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token IDs
    ///
    /// # Returns
    ///
    /// Tensor with shape `[seq_len, embed_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is out of bounds
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let mut output = Vec::with_capacity(seq_len * self.embed_dim);

        for &token_id in token_ids {
            if token_id >= self.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {token_id} out of bounds (vocab_size={})",
                        self.vocab_size
                    ),
                });
            }

            let offset = token_id * self.embed_dim;
            output.extend_from_slice(&self.weights[offset..offset + self.embed_dim]);
        }

        Tensor::from_vec(vec![seq_len, self.embed_dim], output)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get mutable access to weights for loading
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

/// Transformer Language Model
///
/// Complete transformer model for language modeling:
/// - Token embedding
/// - Stack of transformer blocks
/// - Final layer normalization
/// - Output projection (LM head)
///
/// # Architecture
///
/// ```text
/// Token IDs → Embedding → [TransformerBlock × N] → LayerNorm → Linear → Logits
/// ```
#[derive(Debug, Clone)]
pub struct Model {
    /// Token embedding layer
    embedding: Embedding,
    /// Stack of transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Final layer normalization
    final_norm: LayerNorm,
    /// Output projection (LM head)
    lm_head: Linear,
    /// Model configuration
    config: ModelConfig,
}

/// Configuration for the transformer model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Layer normalization epsilon
    pub eps: f32,
}
