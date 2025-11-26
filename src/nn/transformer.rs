//! Transformer architecture components (Vaswani et al., 2017).
//!
//! Implements the attention mechanism and transformer layers for
//! sequence-to-sequence modeling.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::{MultiHeadAttention, TransformerEncoderLayer, Module};
//! use aprender::autograd::Tensor;
//!
//! // Create a transformer encoder layer
//! let encoder = TransformerEncoderLayer::new(512, 8, 2048);
//!
//! // Process a sequence
//! let x = Tensor::randn(&[32, 10, 512]);  // [batch, seq_len, d_model]
//! let y = encoder.forward(&x);            // [batch, seq_len, d_model]
//! ```
//!
//! # References
//!
//! - Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

use super::dropout::Dropout;
use super::linear::Linear;
use super::module::Module;
use super::normalization::LayerNorm;
use crate::autograd::Tensor;

/// Scaled Dot-Product Attention.
///
/// ```text
/// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
/// ```
fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f32,
    training: bool,
) -> (Tensor, Tensor) {
    let d_k = query.shape()[query.ndim() - 1] as f32;
    let scale = 1.0 / d_k.sqrt();

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    let key_t = transpose_last_two(key);
    let scores = matmul_batched(query, &key_t);
    let scores = scale_tensor(&scores, scale);

    // Apply mask (for causal attention or padding)
    let scores = match attn_mask {
        Some(mask) => add_mask(&scores, mask),
        None => scores,
    };

    // Softmax over last dimension
    let attn_weights = softmax_last_dim(&scores);

    // Apply dropout if training
    let attn_weights = if training && dropout_p > 0.0 {
        apply_dropout(&attn_weights, dropout_p)
    } else {
        attn_weights
    };

    // Weighted sum: attn_weights @ V
    let output = matmul_batched(&attn_weights, value);

    (output, attn_weights)
}

/// Multi-Head Attention (Vaswani et al., 2017).
///
/// Allows the model to jointly attend to information from different
/// representation subspaces at different positions.
///
/// # Example
///
/// ```ignore
/// let mha = MultiHeadAttention::new(512, 8);  // d_model=512, num_heads=8
/// let q = Tensor::randn(&[32, 10, 512]);
/// let k = Tensor::randn(&[32, 20, 512]);
/// let v = Tensor::randn(&[32, 20, 512]);
/// let (output, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
/// ```
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    dropout_p: f32,

    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    out_proj: Linear,

    training: bool,
}

impl MultiHeadAttention {
    /// Create a new Multi-Head Attention layer.
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    ///
    /// # Panics
    ///
    /// Panics if `embed_dim` is not divisible by `num_heads`.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            dropout_p: 0.0,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            training: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    /// Forward pass with separate query, key, value inputs.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch, target_len, embed_dim]
    /// * `key` - Key tensor [batch, source_len, embed_dim]
    /// * `value` - Value tensor [batch, source_len, embed_dim]
    /// * `attn_mask` - Optional attention mask [batch, target_len, source_len]
    ///
    /// # Returns
    ///
    /// Tuple of (output, attention_weights)
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let batch_size = query.shape()[0];
        let tgt_len = query.shape()[1];
        let src_len = key.shape()[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape for multi-head: [batch, seq, embed] -> [batch, heads, seq, head_dim]
        let q = reshape_for_attention(&q, batch_size, tgt_len, self.num_heads, self.head_dim);
        let k = reshape_for_attention(&k, batch_size, src_len, self.num_heads, self.head_dim);
        let v = reshape_for_attention(&v, batch_size, src_len, self.num_heads, self.head_dim);

        // Scaled dot-product attention
        let (attn_output, attn_weights) =
            scaled_dot_product_attention(&q, &k, &v, attn_mask, self.dropout_p, self.training);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let attn_output = reshape_from_attention(&attn_output, batch_size, tgt_len, self.embed_dim);

        // Output projection
        let output = self.out_proj.forward(&attn_output);

        (output, attn_weights)
    }

    /// Self-attention: query, key, value are the same.
    pub fn forward_self(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        self.forward_qkv(x, x, x, attn_mask)
    }

    /// Get embed_dim.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get num_heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _) = self.forward_self(input, None);
        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.q_proj.parameters_mut();
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("dropout_p", &self.dropout_p)
            .finish_non_exhaustive()
    }
}

/// Transformer Encoder Layer.
///
/// Consists of self-attention followed by a feed-forward network,
/// with residual connections and layer normalization.
///
/// ```text
/// x = x + Dropout(SelfAttention(LayerNorm(x)))
/// x = x + Dropout(FFN(LayerNorm(x)))
/// ```
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    d_model: usize,
    training: bool,
}

impl TransformerEncoderLayer {
    /// Create a new Transformer Encoder Layer.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of the feedforward network (typically 4 * d_model)
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(0.1),
            dropout1: Dropout::new(0.1),
            dropout2: Dropout::new(0.1),
            d_model,
            training: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Dropout::new(dropout);
        self.dropout1 = Dropout::new(dropout);
        self.dropout2 = Dropout::new(dropout);
        self.self_attn = self.self_attn.with_dropout(dropout);
        self
    }

    /// Forward with optional attention mask.
    pub fn forward_with_mask(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        // Pre-norm architecture (more stable)
        // Self-attention block
        let src_norm = self.norm1.forward(src);
        let (attn_out, _) = self.self_attn.forward_self(&src_norm, src_mask);
        let attn_out = self.dropout1.forward(&attn_out);
        let src = add_tensors(src, &attn_out);

        // Feed-forward block
        let src_norm = self.norm2.forward(&src);
        let ff_out = self.linear1.forward(&src_norm);
        let ff_out = gelu(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.linear2.forward(&ff_out);
        let ff_out = self.dropout2.forward(&ff_out);

        add_tensors(&src, &ff_out)
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.dropout.train();
        self.dropout1.train();
        self.dropout2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.dropout.eval();
        self.dropout1.eval();
        self.dropout2.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for TransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerEncoderLayer")
            .field("d_model", &self.d_model)
            .field("self_attn", &self.self_attn)
            .finish_non_exhaustive()
    }
}

/// Transformer Decoder Layer.
///
/// Like encoder but with an additional cross-attention layer.
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
    dropout1: Dropout,
    dropout2: Dropout,
    dropout3: Dropout,
    d_model: usize,
    training: bool,
}

impl TransformerDecoderLayer {
    /// Create a new Transformer Decoder Layer.
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            cross_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            norm3: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(0.1),
            dropout1: Dropout::new(0.1),
            dropout2: Dropout::new(0.1),
            dropout3: Dropout::new(0.1),
            d_model,
            training: true,
        }
    }

    /// Forward with memory from encoder.
    ///
    /// # Arguments
    ///
    /// * `tgt` - Target sequence [batch, tgt_len, d_model]
    /// * `memory` - Encoder output [batch, src_len, d_model]
    /// * `tgt_mask` - Optional causal mask for target
    /// * `memory_mask` - Optional mask for encoder-decoder attention
    pub fn forward_with_memory(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
    ) -> Tensor {
        // Self-attention (masked)
        let tgt_norm = self.norm1.forward(tgt);
        let (attn_out, _) = self.self_attn.forward_self(&tgt_norm, tgt_mask);
        let attn_out = self.dropout1.forward(&attn_out);
        let tgt = add_tensors(tgt, &attn_out);

        // Cross-attention
        let tgt_norm = self.norm2.forward(&tgt);
        let (cross_out, _) = self
            .cross_attn
            .forward_qkv(&tgt_norm, memory, memory, memory_mask);
        let cross_out = self.dropout2.forward(&cross_out);
        let tgt = add_tensors(&tgt, &cross_out);

        // Feed-forward
        let tgt_norm = self.norm3.forward(&tgt);
        let ff_out = self.linear1.forward(&tgt_norm);
        let ff_out = gelu(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.linear2.forward(&ff_out);
        let ff_out = self.dropout3.forward(&ff_out);

        add_tensors(&tgt, &ff_out)
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // For single input, use it as both target and memory
        self.forward_with_memory(input, input, None, None)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.cross_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        params.extend(self.cross_attn.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.norm3.parameters_mut());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.cross_attn.train();
        self.dropout.train();
        self.dropout1.train();
        self.dropout2.train();
        self.dropout3.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.cross_attn.eval();
        self.dropout.eval();
        self.dropout1.eval();
        self.dropout2.eval();
        self.dropout3.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for TransformerDecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerDecoderLayer")
            .field("d_model", &self.d_model)
            .field("self_attn", &self.self_attn)
            .field("cross_attn", &self.cross_attn)
            .finish_non_exhaustive()
    }
}

/// Sinusoidal Positional Encoding (Vaswani et al., 2017).
///
/// Adds position information to input embeddings using sine and cosine
/// functions of different frequencies.
///
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
#[derive(Debug)]
pub struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    dropout: Dropout,
    /// Pre-computed positional encodings
    pe: Tensor,
    training: bool,
}

impl PositionalEncoding {
    /// Create positional encoding.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `max_len` - Maximum sequence length to pre-compute
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let pe = compute_positional_encoding(d_model, max_len);

        Self {
            d_model,
            max_len,
            dropout: Dropout::new(0.1),
            pe,
            training: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Dropout::new(dropout);
        self
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape()[1];
        assert!(
            seq_len <= self.max_len,
            "Sequence length {seq_len} exceeds max_len {}",
            self.max_len
        );

        // Get positional encodings for this sequence length
        let pe_slice = slice_pe(&self.pe, seq_len, self.d_model);

        // Add to input and apply dropout
        let output = add_positional_encoding(input, &pe_slice);
        self.dropout.forward(&output)
    }

    fn train(&mut self) {
        self.training = true;
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Transpose the last two dimensions of a tensor.
fn transpose_last_two(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return x.clone();
    }

    let last = shape[ndim - 1];
    let second_last = shape[ndim - 2];

    // Compute new shape
    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = last;
    new_shape[ndim - 1] = second_last;

    // Compute batch dimensions
    let batch_size: usize = shape[..ndim - 2].iter().product();
    let matrix_size = last * second_last;

    let mut output = vec![0.0; x.data().len()];

    for b in 0..batch_size {
        let offset = b * matrix_size;
        for i in 0..second_last {
            for j in 0..last {
                // Original: [b, i, j] -> New: [b, j, i]
                output[offset + j * second_last + i] = x.data()[offset + i * last + j];
            }
        }
    }

    Tensor::new(&output, &new_shape)
}

/// Batched matrix multiplication.
fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 4D tensors: [batch, heads, seq, dim]
    if a_shape.len() == 4 && b_shape.len() == 4 {
        let (batch, heads, m, k1) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
        let k2 = b_shape[2];
        let n = b_shape[3];

        assert_eq!(k1, k2, "Inner dimensions must match for matmul");

        let mut output = vec![0.0; batch * heads * m * n];

        for ba in 0..batch {
            for h in 0..heads {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k1 {
                            let a_idx = ba * heads * m * k1 + h * m * k1 + i * k1 + k;
                            let b_idx = ba * heads * k2 * n + h * k2 * n + k * n + j;
                            sum += a.data()[a_idx] * b.data()[b_idx];
                        }
                        let out_idx = ba * heads * m * n + h * m * n + i * n + j;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Tensor::new(&output, &[batch, heads, m, n])
    } else {
        // Fallback for 2D/3D
        a.matmul(b)
    }
}

/// Scale tensor by scalar.
fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| v * scale).collect();
    Tensor::new(&data, x.shape())
}

/// Add attention mask to scores.
fn add_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
    // Mask contains 0 for valid positions and -inf for masked positions
    let data: Vec<f32> = scores
        .data()
        .iter()
        .zip(mask.data().iter())
        .map(|(&s, &m)| s + m)
        .collect();
    Tensor::new(&data, scores.shape())
}

/// Softmax over last dimension.
fn softmax_last_dim(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let mut output = vec![0.0; x.data().len()];

    for b in 0..batch_size {
        let offset = b * last_dim;
        let slice = &x.data()[offset..offset + last_dim];

        // Max for numerical stability
        let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max)
        let exp_vals: Vec<f32> = slice.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        // Normalize
        for (i, exp_v) in exp_vals.iter().enumerate() {
            output[offset + i] = exp_v / sum;
        }
    }

    Tensor::new(&output, shape)
}

/// Apply dropout (simplified).
fn apply_dropout(x: &Tensor, p: f32) -> Tensor {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);

    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if rng.gen::<f32>() < p { 0.0 } else { v * scale })
        .collect();

    Tensor::new(&data, x.shape())
}

/// Reshape for multi-head attention: [batch, seq, embed] -> [batch, heads, seq, head_dim]
fn reshape_for_attention(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let mut output = vec![0.0; batch * num_heads * seq_len * head_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, s, h * head_dim + d]
                    // Output: [b, h, s, d]
                    let in_idx = b * seq_len * (num_heads * head_dim)
                        + s * (num_heads * head_dim)
                        + h * head_dim
                        + d;
                    let out_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    output[out_idx] = x.data()[in_idx];
                }
            }
        }
    }

    Tensor::new(&output, &[batch, num_heads, seq_len, head_dim])
}

/// Reshape from multi-head attention: [batch, heads, seq, head_dim] -> [batch, seq, embed]
fn reshape_from_attention(x: &Tensor, batch: usize, seq_len: usize, embed_dim: usize) -> Tensor {
    let num_heads = x.shape()[1];
    let head_dim = x.shape()[3];

    let mut output = vec![0.0; batch * seq_len * embed_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, h, s, d]
                    // Output: [b, s, h * head_dim + d]
                    let in_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    let out_idx = b * seq_len * embed_dim + s * embed_dim + h * head_dim + d;
                    output[out_idx] = x.data()[in_idx];
                }
            }
        }
    }

    Tensor::new(&output, &[batch, seq_len, embed_dim])
}

/// Element-wise tensor addition.
fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::new(&data, a.shape())
}

/// GELU activation.
fn gelu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| {
            0.5 * v
                * (1.0
                    + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))
                        / std::f32::consts::SQRT_2)
                        .tanh())
        })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Compute sinusoidal positional encoding.
fn compute_positional_encoding(d_model: usize, max_len: usize) -> Tensor {
    let mut pe = vec![0.0; max_len * d_model];

    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
            pe[pos * d_model + 2 * i] = angle.sin();
            pe[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    Tensor::new(&pe, &[max_len, d_model])
}

/// Slice positional encoding for current sequence length.
fn slice_pe(pe: &Tensor, seq_len: usize, d_model: usize) -> Tensor {
    let data: Vec<f32> = pe.data()[..seq_len * d_model].to_vec();
    Tensor::new(&data, &[seq_len, d_model])
}

/// Add positional encoding to input (broadcasting over batch).
fn add_positional_encoding(x: &Tensor, pe: &Tensor) -> Tensor {
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let d_model = x.shape()[2];

    let mut output = vec![0.0; x.data().len()];

    for b in 0..batch_size {
        for s in 0..seq_len {
            for d in 0..d_model {
                let x_idx = b * seq_len * d_model + s * d_model + d;
                let pe_idx = s * d_model + d;
                output[x_idx] = x.data()[x_idx] + pe.data()[pe_idx];
            }
        }
    }

    Tensor::new(&output, x.shape())
}

/// Generate causal (triangular) attention mask.
///
/// Returns a mask where positions can only attend to earlier positions.
pub fn generate_causal_mask(size: usize) -> Tensor {
    let mut data = vec![0.0; size * size];

    for i in 0..size {
        for j in 0..size {
            if j > i {
                data[i * size + j] = f32::NEG_INFINITY;
            }
        }
    }

    Tensor::new(&data, &[size, size])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_shape() {
        let mha = MultiHeadAttention::new(64, 8);

        let q = Tensor::ones(&[2, 10, 64]);
        let k = Tensor::ones(&[2, 20, 64]);
        let v = Tensor::ones(&[2, 20, 64]);

        let (output, attn_weights) = mha.forward_qkv(&q, &k, &v, None);

        assert_eq!(output.shape(), &[2, 10, 64]);
        assert_eq!(attn_weights.shape(), &[2, 8, 10, 20]);
    }

    #[test]
    fn test_multi_head_attention_self() {
        let mha = MultiHeadAttention::new(64, 8);

        let x = Tensor::ones(&[2, 10, 64]);
        let (output, _) = mha.forward_self(&x, None);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_multi_head_attention_parameters() {
        let mha = MultiHeadAttention::new(64, 8);
        let params = mha.parameters();

        // 4 linear layers * 2 params each (weight + bias) = 8
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_transformer_encoder_layer_shape() {
        let layer = TransformerEncoderLayer::new(64, 8, 256);

        let x = Tensor::ones(&[2, 10, 64]);
        let y = layer.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder_layer_parameters() {
        let layer = TransformerEncoderLayer::new(64, 8, 256);
        let params = layer.parameters();

        // Self-attn: 8 params
        // Linear1: 2 params
        // Linear2: 2 params
        // Norm1: 2 params
        // Norm2: 2 params
        // Total: 16
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_transformer_decoder_layer_shape() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);

        let tgt = Tensor::ones(&[2, 10, 64]);
        let memory = Tensor::ones(&[2, 20, 64]);

        let output = layer.forward_with_memory(&tgt, &memory, None, None);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_positional_encoding_shape() {
        let pe = PositionalEncoding::new(64, 100);

        let x = Tensor::ones(&[2, 10, 64]);
        let y = pe.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = generate_causal_mask(4);

        assert_eq!(mask.shape(), &[4, 4]);

        // Check upper triangle is -inf
        assert!(mask.data()[1].is_infinite()); // [0, 1]
        assert!(mask.data()[2].is_infinite()); // [0, 2]
        assert!(mask.data()[3].is_infinite()); // [0, 3]
        assert!(mask.data()[6].is_infinite()); // [1, 2]

        // Check diagonal and below is 0
        assert_eq!(mask.data()[0], 0.0); // [0, 0]
        assert_eq!(mask.data()[4], 0.0); // [1, 0]
        assert_eq!(mask.data()[5], 0.0); // [1, 1]
    }

    #[test]
    fn test_softmax_last_dim() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let y = softmax_last_dim(&x);

        // Check that each row sums to 1
        let row1_sum: f32 = y.data()[0..3].iter().sum();
        let row2_sum: f32 = y.data()[3..6].iter().sum();

        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_transpose_last_two() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let y = transpose_last_two(&x);

        assert_eq!(y.shape(), &[1, 3, 2]);
    }
}
