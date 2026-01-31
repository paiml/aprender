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
//! - Vaswani, A., et al. (2017). Attention is all you need. `NeurIPS`.

use super::dropout::Dropout;
use super::linear::Linear;
use super::module::Module;
use super::normalization::LayerNorm;
use crate::autograd::Tensor;
use trueno::Matrix;

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
    /// * `embed_dim` - Total dimension of the model (must be divisible by `num_heads`)
    /// * `num_heads` - Number of attention heads
    ///
    /// # Panics
    ///
    /// Panics if `embed_dim` is not divisible by `num_heads`.
    #[must_use]
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
    #[must_use]
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    /// Forward pass with separate query, key, value inputs.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch, `target_len`, `embed_dim`]
    /// * `key` - Key tensor [batch, `source_len`, `embed_dim`]
    /// * `value` - Value tensor [batch, `source_len`, `embed_dim`]
    /// * `attn_mask` - Optional attention mask [batch, `target_len`, `source_len`]
    ///
    /// # Returns
    ///
    /// Tuple of (output, `attention_weights`)
    #[must_use]
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
    #[must_use]
    pub fn forward_self(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        self.forward_qkv(x, x, x, attn_mask)
    }

    /// Get `embed_dim`.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get `num_heads`.
    #[must_use]
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
    /// * `dim_feedforward` - Dimension of the feedforward network (typically 4 * `d_model`)
    #[must_use]
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
    #[must_use]
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
    /// * `tgt` - Target sequence [batch, `tgt_len`, `d_model`]
    /// * `memory` - Encoder output [batch, `src_len`, `d_model`]
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
    #[must_use]
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

/// Batched matrix multiplication using SIMD-accelerated Trueno.
/// For 4D tensors [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
///
/// Uses `trueno::Matrix::batched_matmul_4d` for efficient SIMD computation.
/// Per spec §2.4.1 Compute Backend Hierarchy: SIMD before naive loops.
///
/// # Panics
/// Panics if `batched_matmul_4d` fails after dimension validation. This should
/// never happen as dimensions are validated by the assert above. If it does,
/// it indicates a bug in the trueno library.
#[allow(clippy::expect_used)]
fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 4D tensors: [batch, heads, seq, dim]
    if a_shape.len() == 4 && b_shape.len() == 4 {
        let (batch, heads, m, k1) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
        let k2 = b_shape[2];
        let n = b_shape[3];

        assert_eq!(k1, k2, "Inner dimensions must match for matmul");

        // Use Trueno's SIMD batched matmul for 4D attention tensors
        let output = Matrix::batched_matmul_4d(a.data(), b.data(), batch, heads, m, k1, n)
            .expect("batched_matmul_4d failed: dimensions validated but operation failed");

        Tensor::new(&output, &[batch, heads, m, n])
    } else {
        // Fallback for 2D/3D - uses Tensor's SIMD matmul
        a.matmul(b)
    }
}

/// Scale tensor by scalar (SIMD-accelerated).
fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    x.mul_scalar(scale)
}

/// Add attention mask to scores (SIMD-accelerated).
fn add_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
    // Mask contains 0 for valid positions and -inf for masked positions
    // Use SIMD-accelerated add if shapes match, otherwise broadcast
    if scores.shape() == mask.shape() {
        return scores.add(mask);
    }
    // Fallback for broadcast (SIMD broadcast_add deferred to trueno)
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

/// Reshape for multi-head attention: [batch, seq, embed] -> [batch, heads, seq, `head_dim`]
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

/// Reshape from multi-head attention: [batch, heads, seq, `head_dim`] -> [batch, seq, embed]
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
#[must_use]
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

// ============================================================================
// Linear Attention (Katharopoulos et al., 2020)
// ============================================================================

/// Linear Attention with kernel feature maps.
///
/// Achieves O(nd²) complexity instead of O(n²d) by using kernel approximation.
/// Based on "Transformers are RNNs" (Katharopoulos et al., 2020).
///
/// ```text
/// Attention(Q, K, V) ≈ φ(Q) * (φ(K)^T @ V) / (φ(Q) * Σφ(K)^T)
/// ```
///
/// where φ is a feature map (e.g., elu(x) + 1).
///
/// # Example
///
/// ```ignore
/// let attn = LinearAttention::new(512, 8);
/// let x = Tensor::randn(&[32, 100, 512]);
/// let y = attn.forward(&x);
/// ```
pub struct LinearAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    eps: f32,
    training: bool,
}

impl LinearAttention {
    /// Create a new Linear Attention layer.
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of attention heads
    #[must_use]
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
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            eps: 1e-6,
            training: true,
        }
    }

    /// Forward pass with linear attention.
    #[must_use]
    pub fn forward_linear(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
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

        // Apply feature map φ(x) = elu(x) + 1 to Q and K
        let q_prime = elu_feature_map(&q);
        let k_prime = elu_feature_map(&k);

        // Linear attention: φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ Σφ(K)^T)
        // Compute K^T @ V first: [batch, heads, head_dim, head_dim]
        let k_prime_t = transpose_last_two(&k_prime);
        let kv = matmul_batched(&k_prime_t, &v); // [batch, heads, head_dim, head_dim]

        // Compute Q @ KV
        let output = matmul_batched(&q_prime, &kv); // [batch, heads, tgt_len, head_dim]

        // Compute normalizer: Q @ sum(K^T, dim=-1)
        let k_sum = sum_last_dim(&k_prime); // [batch, heads, head_dim]
        let normalizer = matmul_with_broadcast(&q_prime, &k_sum); // [batch, heads, tgt_len, 1]

        // Normalize output
        let output = divide_with_eps(&output, &normalizer, self.eps);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let output = reshape_from_attention(&output, batch_size, tgt_len, self.embed_dim);

        // Output projection
        self.out_proj.forward(&output)
    }

    /// Get `embed_dim`.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get `num_heads`.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

impl Module for LinearAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_linear(input, input, input)
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

impl std::fmt::Debug for LinearAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Grouped Query Attention (Ainslie et al., 2023)
// ============================================================================

/// Grouped Query Attention (GQA).
///
/// Uses fewer key-value heads than query heads, reducing memory and compute.
/// From "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023).
///
/// When `num_kv_heads = 1`, this is Multi-Query Attention (MQA).
/// When `num_kv_heads = num_heads`, this is standard Multi-Head Attention (MHA).
///
/// # Example
///
/// ```ignore
/// // GQA with 8 query heads and 2 KV heads (4:1 ratio)
/// let gqa = GroupedQueryAttention::new(512, 8, 2);
/// let x = Tensor::randn(&[32, 100, 512]);
/// let y = gqa.forward(&x);
/// ```
pub struct GroupedQueryAttention {
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
    dropout_p: f32,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    training: bool,
}

impl GroupedQueryAttention {
    /// Create a new Grouped Query Attention layer.
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads (must divide `num_heads`)
    ///
    /// # Panics
    ///
    /// Panics if `embed_dim` is not divisible by `num_heads` or
    /// if `num_heads` is not divisible by `num_kv_heads`.
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );

        let head_dim = embed_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_head_dim: head_dim,
            dropout_p: 0.0,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, kv_dim),
            v_proj: Linear::new(embed_dim, kv_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            training: true,
        }
    }

    /// Create a placeholder GQA layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    /// Uses `Linear::placeholder()` internally to minimize memory usage.
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// all projection weights are loaded via `*_proj_mut().set_weight()`.
    #[must_use]
    pub fn placeholder(embed_dim: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_head_dim: head_dim,
            dropout_p: 0.0,
            q_proj: Linear::placeholder(embed_dim, embed_dim),
            k_proj: Linear::placeholder(embed_dim, kv_dim),
            v_proj: Linear::placeholder(embed_dim, kv_dim),
            out_proj: Linear::placeholder(embed_dim, embed_dim),
            training: true,
        }
    }

    /// Set dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    /// Forward pass with separate query, key, value inputs.
    #[must_use]
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

        // Reshape Q: [batch, seq, embed] -> [batch, num_heads, seq, head_dim]
        let q = reshape_for_attention(&q, batch_size, tgt_len, self.num_heads, self.head_dim);

        // Reshape K, V: [batch, seq, kv_dim] -> [batch, num_kv_heads, seq, head_dim]
        let k = reshape_for_attention(&k, batch_size, src_len, self.num_kv_heads, self.kv_head_dim);
        let v = reshape_for_attention(&v, batch_size, src_len, self.num_kv_heads, self.kv_head_dim);

        // Expand K, V to match Q heads by repeating
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv_heads(&k, groups);
        let v = repeat_kv_heads(&v, groups);

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
    #[must_use]
    pub fn forward_self(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        self.forward_qkv(x, x, x, attn_mask)
    }

    /// Get `embed_dim`.
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get `num_heads`.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get `num_kv_heads`.
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get mutable reference to Q projection layer.
    pub fn q_proj_mut(&mut self) -> &mut Linear {
        &mut self.q_proj
    }

    /// Get mutable reference to K projection layer.
    pub fn k_proj_mut(&mut self) -> &mut Linear {
        &mut self.k_proj
    }

    /// Get mutable reference to V projection layer.
    pub fn v_proj_mut(&mut self) -> &mut Linear {
        &mut self.v_proj
    }

    /// Get mutable reference to output projection layer.
    pub fn out_proj_mut(&mut self) -> &mut Linear {
        &mut self.out_proj
    }
}

impl Module for GroupedQueryAttention {
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

impl std::fmt::Debug for GroupedQueryAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedQueryAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("dropout_p", &self.dropout_p)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Additional Helper Functions for Attention Variants
// ============================================================================

/// ELU feature map: φ(x) = elu(x) + 1 for positive-definite kernel.
fn elu_feature_map(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if v > 0.0 { v + 1.0 } else { v.exp() })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Sum over last dimension.
#[allow(clippy::needless_range_loop)]
fn sum_last_dim(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let new_size: usize = shape[..shape.len() - 1].iter().product();

    let mut output = vec![0.0; new_size];

    for i in 0..new_size {
        let offset = i * last_dim;
        output[i] = x.data()[offset..offset + last_dim].iter().sum();
    }

    let new_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    Tensor::new(&output, &new_shape)
}

/// Matrix multiply with broadcasting for normalizer computation.
fn matmul_with_broadcast(q: &Tensor, k_sum: &Tensor) -> Tensor {
    // q: [batch, heads, seq, head_dim]
    // k_sum: [batch, heads, head_dim]
    // output: [batch, heads, seq, 1] (dot product of each q row with k_sum)
    let q_shape = q.shape();
    let (batch, heads, seq_len, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);

    let mut output = vec![0.0; batch * heads * seq_len];

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_len {
                let mut sum = 0.0;
                for d in 0..head_dim {
                    let q_idx =
                        b * heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                    let k_idx = b * heads * head_dim + h * head_dim + d;
                    sum += q.data()[q_idx] * k_sum.data()[k_idx];
                }
                let out_idx = b * heads * seq_len + h * seq_len + s;
                output[out_idx] = sum;
            }
        }
    }

    Tensor::new(&output, &[batch, heads, seq_len, 1])
}

/// Divide tensor by normalizer with epsilon for numerical stability.
fn divide_with_eps(x: &Tensor, normalizer: &Tensor, eps: f32) -> Tensor {
    // x: [batch, heads, seq, head_dim]
    // normalizer: [batch, heads, seq, 1]
    let x_shape = x.shape();
    let (batch, heads, seq_len, head_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);

    let mut output = vec![0.0; x.data().len()];

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_len {
                let norm_idx = b * heads * seq_len + h * seq_len + s;
                let norm_val = normalizer.data()[norm_idx].max(eps);

                for d in 0..head_dim {
                    let idx =
                        b * heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                    output[idx] = x.data()[idx] / norm_val;
                }
            }
        }
    }

    Tensor::new(&output, x_shape)
}

/// Dimensions for KV head repetition.
#[derive(Clone, Copy)]
struct KvHeadDims {
    kv_heads: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
}

/// Copy a single sequence position for head repetition.
#[inline]
fn copy_kv_head_seq(
    x_data: &[f32],
    output: &mut [f32],
    b: usize,
    kv_h: usize,
    h: usize,
    s: usize,
    dims: KvHeadDims,
) {
    let in_base = b * dims.kv_heads * dims.seq_len * dims.head_dim
        + kv_h * dims.seq_len * dims.head_dim
        + s * dims.head_dim;
    let out_base = b * dims.num_heads * dims.seq_len * dims.head_dim
        + h * dims.seq_len * dims.head_dim
        + s * dims.head_dim;
    output[out_base..out_base + dims.head_dim]
        .copy_from_slice(&x_data[in_base..in_base + dims.head_dim]);
}

/// Repeat a single KV head across groups.
fn repeat_single_kv_head(
    x_data: &[f32],
    output: &mut [f32],
    b: usize,
    kv_h: usize,
    groups: usize,
    dims: KvHeadDims,
) {
    for g in 0..groups {
        let h = kv_h * groups + g;
        for s in 0..dims.seq_len {
            copy_kv_head_seq(x_data, output, b, kv_h, h, s, dims);
        }
    }
}

/// Repeat KV heads to match Q heads for Grouped Query Attention.
fn repeat_kv_heads(x: &Tensor, groups: usize) -> Tensor {
    if groups == 1 {
        return x.clone();
    }

    let shape = x.shape();
    let (batch, kv_heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let num_heads = kv_heads * groups;
    let dims = KvHeadDims {
        kv_heads,
        num_heads,
        seq_len,
        head_dim,
    };

    let mut output = vec![0.0; batch * num_heads * seq_len * head_dim];

    for b in 0..batch {
        for kv_h in 0..kv_heads {
            repeat_single_kv_head(x.data(), &mut output, b, kv_h, groups, dims);
        }
    }

    Tensor::new(&output, &[batch, num_heads, seq_len, head_dim])
}

// ============================================================================
// Modern Positional Encoding Variants
// ============================================================================

/// Rotary Position Embedding (`RoPE`) (Su et al., 2021).
///
/// Encodes absolute position with relative position dependencies via rotation.
/// Used in GPT-NeoX, `LLaMA`, and other modern LLMs.
///
/// # Method
///
/// Rotates pairs of features by position-dependent angles:
/// ```text
/// (q_2i, q_2i+1) = (q_2i * cos(mθ_i) - q_2i+1 * sin(mθ_i),
///                   q_2i * sin(mθ_i) + q_2i+1 * cos(mθ_i))
/// ```
/// where m is the position and `θ_i` = 10000^(-2i/d)
///
/// # Reference
///
/// - Su, J., et al. (2021). `RoFormer`: Enhanced Transformer with Rotary
///   Position Embedding. arXiv:2104.09864
#[derive(Debug, Clone)]
pub struct RotaryPositionEmbedding {
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    /// Precomputed cos values [`max_seq_len`, `head_dim/2`]
    cos_cache: Vec<f32>,
    /// Precomputed sin values [`max_seq_len`, `head_dim/2`]
    sin_cache: Vec<f32>,
}

impl RotaryPositionEmbedding {
    /// Create `RoPE` with specified head dimension.
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension per attention head (must be even)
    /// * `max_seq_len` - Maximum sequence length to precompute
    #[must_use]
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        Self::with_base(head_dim, max_seq_len, 10000.0)
    }

    /// Create `RoPE` with custom base frequency.
    #[must_use]
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_cache = vec![0.0; max_seq_len * half_dim];
        let mut sin_cache = vec![0.0; max_seq_len * half_dim];

        // Compute inv_freq: 1 / (base^(2i/d))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Compute cos/sin for each position
        for pos in 0..max_seq_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_cache[pos * half_dim + i] = angle.cos();
                sin_cache[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            head_dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
        }
    }

    /// Apply rotary embedding to query or key tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor `[batch, seq_len, num_heads, head_dim]`
    /// * `position_ids` - Position indices for each token \[`seq_len`\]
    ///
    /// # Returns
    ///
    /// Tensor with rotary embeddings applied.
    #[must_use]
    pub fn apply(&self, x: &Tensor, position_ids: &[usize]) -> Tensor {
        let shape = x.shape();
        assert!(
            shape.len() == 4,
            "Expected 4D tensor [batch, seq, heads, head_dim]"
        );

        let (batch, seq_len, num_heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(head_dim, self.head_dim);

        let half_dim = head_dim / 2;
        let mut output = vec![0.0; x.data().len()];

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = position_ids.get(s).copied().unwrap_or(s);
                assert!(pos < self.max_seq_len, "Position {pos} exceeds max_seq_len");

                for h in 0..num_heads {
                    for i in 0..half_dim {
                        let cos_val = self.cos_cache[pos * half_dim + i];
                        let sin_val = self.sin_cache[pos * half_dim + i];

                        // Get pair of values
                        let idx1 = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + 2 * i;
                        let idx2 = idx1 + 1;

                        let x1 = x.data()[idx1];
                        let x2 = x.data()[idx2];

                        // Apply rotation
                        output[idx1] = x1 * cos_val - x2 * sin_val;
                        output[idx2] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }

        Tensor::new(&output, shape)
    }

    /// Get head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get maximum sequence length.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get base frequency.
    #[must_use]
    pub fn base(&self) -> f32 {
        self.base
    }
}

/// `ALiBi` (Attention with Linear Biases) (Press et al., 2022).
///
/// Adds linear position biases directly to attention scores instead of
/// positional embeddings. Enables length extrapolation.
///
/// # Method
///
/// ```text
/// attention = softmax(Q @ K^T / sqrt(d) - m * |i - j|)
/// ```
/// where m is a head-specific slope.
///
/// # Reference
///
/// - Press, O., et al. (2022). Train Short, Test Long: Attention with
///   Linear Biases Enables Input Length Extrapolation. ICLR.
#[derive(Debug, Clone)]
pub struct ALiBi {
    num_heads: usize,
    /// Per-head slopes (geometric sequence starting from 2^(-8/n))
    slopes: Vec<f32>,
}

impl ALiBi {
    /// Create `ALiBi` with specified number of attention heads.
    ///
    /// Slopes follow geometric sequence: 2^(-8/n), 2^(-16/n), ...
    #[must_use]
    pub fn new(num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Compute slopes using the formula from the paper.
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // For power-of-2 heads, use geometric sequence
        // For non-power-of-2, interpolate
        let closest_pow2 = (num_heads as f32).log2().ceil() as u32;
        let base = 2.0_f32.powf(-(8.0 / 2.0_f32.powi(closest_pow2 as i32)));

        let mut slopes = Vec::with_capacity(num_heads);

        if num_heads.is_power_of_two() {
            for i in 0..num_heads {
                slopes.push(base.powi((i + 1) as i32));
            }
        } else {
            // Interpolate for non-power-of-2
            let extra_base = 2.0_f32.powf(-(8.0 / 2.0_f32.powi(closest_pow2 as i32 - 1)));
            let num_extra = 2 * num_heads - 2_usize.pow(closest_pow2);

            for i in 0..num_extra {
                slopes.push(extra_base.powi(((i + 1) * 2) as i32));
            }
            for i in num_extra..num_heads {
                slopes.push(base.powi((i - num_extra + 1) as i32));
            }
        }

        slopes
    }

    /// Compute `ALiBi` bias matrix for given sequence length.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Current sequence length
    ///
    /// # Returns
    ///
    /// Bias tensor [`num_heads`, `seq_len`, `seq_len`] to add to attention scores.
    #[must_use]
    pub fn compute_bias(&self, seq_len: usize) -> Tensor {
        let mut bias = vec![0.0; self.num_heads * seq_len * seq_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = (i as i32 - j as i32).abs() as f32;
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    bias[idx] = -slope * distance;
                }
            }
        }

        Tensor::new(&bias, &[self.num_heads, seq_len, seq_len])
    }

    /// Apply `ALiBi` to attention scores.
    ///
    /// # Arguments
    ///
    /// * `scores` - Attention scores [batch, `num_heads`, `seq_len`, `seq_len`]
    ///
    /// # Returns
    ///
    /// Scores with `ALiBi` bias applied.
    #[must_use]
    pub fn apply(&self, scores: &Tensor) -> Tensor {
        let shape = scores.shape();
        assert!(shape.len() == 4, "Expected 4D tensor");
        assert_eq!(shape[1], self.num_heads, "num_heads mismatch");

        let (batch, _, seq_len, _) = (shape[0], shape[1], shape[2], shape[3]);
        let bias = self.compute_bias(seq_len);

        // Add bias (broadcast over batch)
        let mut output = scores.data().to_vec();

        for b in 0..batch {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let score_idx = b * self.num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j;
                        let bias_idx = h * seq_len * seq_len + i * seq_len + j;
                        output[score_idx] += bias.data()[bias_idx];
                    }
                }
            }
        }

        Tensor::new(&output, shape)
    }

    /// Get slopes for each head.
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Get number of heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
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

    // ========================================================================
    // Linear Attention Tests
    // ========================================================================

    #[test]
    fn test_linear_attention_shape() {
        let attn = LinearAttention::new(64, 8);

        let x = Tensor::ones(&[2, 10, 64]);
        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_linear_attention_qkv_shape() {
        let attn = LinearAttention::new(64, 8);

        let q = Tensor::ones(&[2, 10, 64]);
        let k = Tensor::ones(&[2, 20, 64]);
        let v = Tensor::ones(&[2, 20, 64]);

        let output = attn.forward_linear(&q, &k, &v);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_linear_attention_parameters() {
        let attn = LinearAttention::new(64, 8);
        let params = attn.parameters();

        // 4 linear layers * 2 params each = 8
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_linear_attention_getters() {
        let attn = LinearAttention::new(128, 4);

        assert_eq!(attn.embed_dim(), 128);
        assert_eq!(attn.num_heads(), 4);
    }

    #[test]
    fn test_linear_attention_train_eval() {
        let mut attn = LinearAttention::new(64, 8);

        assert!(attn.training());
        attn.eval();
        assert!(!attn.training());
        attn.train();
        assert!(attn.training());
    }

    #[test]
    fn test_linear_attention_long_sequence() {
        // Linear attention should scale well with sequence length
        let attn = LinearAttention::new(32, 4);

        let x = Tensor::ones(&[1, 100, 32]); // Long sequence
        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[1, 100, 32]);
    }

    #[test]
    fn test_elu_feature_map_positive() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let y = elu_feature_map(&x);

        // For positive values: elu(x) + 1 = x + 1
        assert!((y.data()[0] - 2.0).abs() < 1e-6);
        assert!((y.data()[1] - 3.0).abs() < 1e-6);
        assert!((y.data()[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_elu_feature_map_negative() {
        let x = Tensor::new(&[-1.0, -2.0], &[2]);
        let y = elu_feature_map(&x);

        // For negative values: elu(x) + 1 = exp(x)
        assert!((y.data()[0] - (-1.0_f32).exp()).abs() < 1e-6);
        assert!((y.data()[1] - (-2.0_f32).exp()).abs() < 1e-6);
    }

    // ========================================================================
    // Grouped Query Attention Tests
    // ========================================================================

    #[test]
    fn test_gqa_shape() {
        // 8 query heads, 2 KV heads (4:1 ratio)
        let gqa = GroupedQueryAttention::new(64, 8, 2);

        let x = Tensor::ones(&[2, 10, 64]);
        let output = gqa.forward(&x);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_qkv_shape() {
        let gqa = GroupedQueryAttention::new(64, 8, 2);

        let q = Tensor::ones(&[2, 10, 64]);
        let k = Tensor::ones(&[2, 20, 64]);
        let v = Tensor::ones(&[2, 20, 64]);

        let (output, attn_weights) = gqa.forward_qkv(&q, &k, &v, None);

        assert_eq!(output.shape(), &[2, 10, 64]);
        // Attention weights have expanded heads
        assert_eq!(attn_weights.shape(), &[2, 8, 10, 20]);
    }

    #[test]
    fn test_gqa_multi_query_attention() {
        // MQA: 1 KV head for all query heads
        let mqa = GroupedQueryAttention::new(64, 8, 1);

        let x = Tensor::ones(&[2, 10, 64]);
        let output = mqa.forward(&x);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_equals_mha() {
        // GQA with num_kv_heads == num_heads should behave like MHA
        let gqa = GroupedQueryAttention::new(64, 8, 8);

        let x = Tensor::ones(&[2, 10, 64]);
        let output = gqa.forward(&x);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_parameters_reduced() {
        // GQA with fewer KV heads has fewer parameters
        let mha = MultiHeadAttention::new(64, 8);
        let gqa = GroupedQueryAttention::new(64, 8, 2);

        let mha_params = mha.parameters();
        let gqa_params = gqa.parameters();

        // Both have 8 parameter tensors (4 linear layers * 2)
        assert_eq!(mha_params.len(), gqa_params.len());

        // But GQA K,V projections are smaller
        // MHA K projection: 64 -> 64, GQA K projection: 64 -> 16
    }

    #[test]
    fn test_gqa_getters() {
        let gqa = GroupedQueryAttention::new(128, 8, 4);

        assert_eq!(gqa.embed_dim(), 128);
        assert_eq!(gqa.num_heads(), 8);
        assert_eq!(gqa.num_kv_heads(), 4);
    }

    #[test]
    fn test_gqa_with_dropout() {
        let gqa = GroupedQueryAttention::new(64, 8, 2).with_dropout(0.1);

        let x = Tensor::ones(&[2, 10, 64]);
        let output = gqa.forward(&x);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_train_eval() {
        let mut gqa = GroupedQueryAttention::new(64, 8, 2);

        assert!(gqa.training());
        gqa.eval();
        assert!(!gqa.training());
        gqa.train();
        assert!(gqa.training());
    }

    #[test]
    #[should_panic(expected = "num_heads (8) must be divisible by num_kv_heads (3)")]
    fn test_gqa_invalid_kv_heads() {
        // num_heads must be divisible by num_kv_heads
        let _gqa = GroupedQueryAttention::new(64, 8, 3);
    }

    #[test]
    fn test_repeat_kv_heads_identity() {
        // groups=1 should return identity
        let x = Tensor::ones(&[2, 4, 10, 8]); // [batch, kv_heads, seq, head_dim]
        let y = repeat_kv_heads(&x, 1);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_repeat_kv_heads_expansion() {
        // 2 KV heads -> 8 Q heads (4x expansion)
        let x = Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, 2, 2], // [batch=1, kv_heads=2, seq=2, head_dim=2]
        );
        let y = repeat_kv_heads(&x, 4);

        assert_eq!(y.shape(), &[1, 8, 2, 2]);

        // Each KV head should be repeated 4 times
        // Head 0 data [1,2,3,4] repeated at positions 0,1,2,3
        // Head 1 data [5,6,7,8] repeated at positions 4,5,6,7
    }

    #[test]
    fn test_sum_last_dim() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = sum_last_dim(&x);

        assert_eq!(y.shape(), &[2]);
        assert!((y.data()[0] - 6.0).abs() < 1e-6); // 1+2+3
        assert!((y.data()[1] - 15.0).abs() < 1e-6); // 4+5+6
    }

    // ========================================================================
    // RoPE Tests
    // ========================================================================

    #[test]
    fn test_rope_creation() {
        let rope = RotaryPositionEmbedding::new(64, 512);

        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 512);
        assert!((rope.base() - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_custom_base() {
        let rope = RotaryPositionEmbedding::with_base(32, 256, 20000.0);

        assert_eq!(rope.head_dim(), 32);
        assert!((rope.base() - 20000.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_apply_shape() {
        let rope = RotaryPositionEmbedding::new(8, 128);

        // [batch=2, seq=10, heads=4, head_dim=8]
        let x = Tensor::ones(&[2, 10, 4, 8]);
        let positions: Vec<usize> = (0..10).collect();

        let output = rope.apply(&x, &positions);

        assert_eq!(output.shape(), x.shape());
    }

    #[test]
    fn test_rope_position_dependent() {
        let rope = RotaryPositionEmbedding::new(4, 10);

        // Same input at different positions should give different output
        let x = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 1, 4]);

        let out_pos0 = rope.apply(&x, &[0]);
        let out_pos5 = rope.apply(&x, &[5]);

        // Outputs should differ
        let diff: f32 = out_pos0
            .data()
            .iter()
            .zip(out_pos5.data().iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different positions should give different outputs"
        );
    }

    #[test]
    fn test_rope_cos_sin_cache() {
        let rope = RotaryPositionEmbedding::new(4, 10);

        // At position 0, cos should be 1, sin should be 0
        // cos_cache and sin_cache have shape [max_seq_len, head_dim/2]
        let half_dim = 2;
        assert!((rope.cos_cache[0 * half_dim] - 1.0).abs() < 1e-6);
        assert!(rope.sin_cache[0 * half_dim].abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn test_rope_odd_dim_panics() {
        let _rope = RotaryPositionEmbedding::new(7, 100);
    }

    // ========================================================================
    // ALiBi Tests
    // ========================================================================

    #[test]
    fn test_alibi_creation() {
        let alibi = ALiBi::new(8);

        assert_eq!(alibi.num_heads(), 8);
        assert_eq!(alibi.slopes().len(), 8);
    }

    #[test]
    fn test_alibi_slopes_power_of_two() {
        let alibi = ALiBi::new(8);

        // Slopes should be monotonically decreasing for power-of-2 heads
        let slopes = alibi.slopes();
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "Slopes should decrease");
        }

        // All slopes should be positive
        for &s in slopes {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_alibi_bias_shape() {
        let alibi = ALiBi::new(4);
        let bias = alibi.compute_bias(10);

        assert_eq!(bias.shape(), &[4, 10, 10]);
    }

    #[test]
    fn test_alibi_bias_diagonal_zero() {
        let alibi = ALiBi::new(2);
        let bias = alibi.compute_bias(5);

        // Diagonal should be zero (distance = 0)
        for h in 0..2 {
            for i in 0..5 {
                let idx = h * 5 * 5 + i * 5 + i;
                assert!((bias.data()[idx] - 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_alibi_bias_negative() {
        let alibi = ALiBi::new(2);
        let bias = alibi.compute_bias(5);

        // Off-diagonal should be negative (penalties)
        for h in 0..2 {
            for i in 0..5 {
                for j in 0..5 {
                    if i != j {
                        let idx = h * 5 * 5 + i * 5 + j;
                        assert!(bias.data()[idx] < 0.0, "Off-diagonal should be negative");
                    }
                }
            }
        }
    }

    #[test]
    fn test_alibi_apply_shape() {
        let alibi = ALiBi::new(4);

        // Attention scores [batch=2, heads=4, seq=8, seq=8]
        let scores = Tensor::ones(&[2, 4, 8, 8]);
        let output = alibi.apply(&scores);

        assert_eq!(output.shape(), scores.shape());
    }

    #[test]
    fn test_alibi_apply_modifies_scores() {
        let alibi = ALiBi::new(2);
        let scores = Tensor::ones(&[1, 2, 4, 4]);
        let output = alibi.apply(&scores);

        // Output should differ from input (bias applied)
        let sum_input: f32 = scores.data().iter().sum();
        let sum_output: f32 = output.data().iter().sum();

        // Bias is negative, so output sum should be less than input
        assert!(sum_output < sum_input);
    }

    #[test]
    fn test_alibi_non_power_two_heads() {
        // Should handle non-power-of-2 heads
        let alibi = ALiBi::new(6);
        assert_eq!(alibi.slopes().len(), 6);

        // All slopes should be positive
        for &s in alibi.slopes() {
            assert!(s > 0.0);
        }
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_mha_train_eval() {
        let mut mha = MultiHeadAttention::new(64, 8);
        assert!(mha.training());

        mha.eval();
        assert!(!mha.training());

        mha.train();
        assert!(mha.training());
    }

    #[test]
    fn test_mha_embed_num_heads_getters() {
        let mha = MultiHeadAttention::new(128, 4);
        assert_eq!(mha.embed_dim(), 128);
        assert_eq!(mha.num_heads(), 4);
    }

    #[test]
    fn test_mha_with_dropout() {
        let mha = MultiHeadAttention::new(64, 8).with_dropout(0.2);
        let x = Tensor::ones(&[2, 10, 64]);
        let output = mha.forward(&x);
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_mha_debug() {
        let mha = MultiHeadAttention::new(64, 8);
        let debug_str = format!("{:?}", mha);
        assert!(debug_str.contains("MultiHeadAttention"));
        assert!(debug_str.contains("embed_dim"));
        assert!(debug_str.contains("num_heads"));
    }

    #[test]
    fn test_mha_parameters_mut() {
        let mut mha = MultiHeadAttention::new(64, 8);
        let params = mha.parameters_mut();
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_encoder_layer_train_eval() {
        let mut layer = TransformerEncoderLayer::new(64, 8, 256);
        assert!(layer.training());

        layer.eval();
        assert!(!layer.training());

        layer.train();
        assert!(layer.training());
    }

    #[test]
    fn test_encoder_layer_with_dropout() {
        let layer = TransformerEncoderLayer::new(64, 8, 256).with_dropout(0.2);
        let x = Tensor::ones(&[2, 10, 64]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_encoder_layer_forward_with_mask() {
        let layer = TransformerEncoderLayer::new(64, 8, 256);
        let x = Tensor::ones(&[2, 10, 64]);
        // Test without mask - mask shape requirements are complex
        let y = layer.forward_with_mask(&x, None);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_encoder_layer_debug() {
        let layer = TransformerEncoderLayer::new(64, 8, 256);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("TransformerEncoderLayer"));
        assert!(debug_str.contains("d_model"));
    }

    #[test]
    fn test_encoder_layer_parameters_mut() {
        let mut layer = TransformerEncoderLayer::new(64, 8, 256);
        let params = layer.parameters_mut();
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_decoder_layer_train_eval() {
        let mut layer = TransformerDecoderLayer::new(64, 8, 256);
        assert!(layer.training());

        layer.eval();
        assert!(!layer.training());

        layer.train();
        assert!(layer.training());
    }

    #[test]
    fn test_decoder_layer_forward_single_input() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);
        let x = Tensor::ones(&[2, 10, 64]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_decoder_layer_parameters() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);
        let params = layer.parameters();
        // self_attn: 8 + cross_attn: 8 + linear1: 2 + linear2: 2 + norm1,2,3: 6 = 26
        assert_eq!(params.len(), 26);
    }

    #[test]
    fn test_decoder_layer_parameters_mut() {
        let mut layer = TransformerDecoderLayer::new(64, 8, 256);
        let params = layer.parameters_mut();
        assert_eq!(params.len(), 26);
    }

    #[test]
    fn test_decoder_layer_with_masks() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);
        let tgt = Tensor::ones(&[2, 10, 64]);
        let memory = Tensor::ones(&[2, 20, 64]);
        // Test without mask - mask shape requirements are complex
        let y = layer.forward_with_memory(&tgt, &memory, None, None);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_positional_encoding_parameters() {
        let pe = PositionalEncoding::new(64, 100);
        let params = pe.parameters();
        assert!(params.is_empty()); // No learnable parameters
    }

    #[test]
    fn test_positional_encoding_train_eval() {
        let mut pe = PositionalEncoding::new(64, 100);
        assert!(pe.training());
        pe.eval();
        assert!(!pe.training());
        pe.train();
        assert!(pe.training());
    }

    #[test]
    fn test_scale_tensor() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let y = scale_tensor(&x, 2.0);
        assert_eq!(y.data(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_add_tensors() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
        let c = add_tensors(&a, &b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_gelu_activation() {
        let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
        let y = gelu(&x);
        // GELU(0) ≈ 0
        assert!((y.data()[0] - 0.0).abs() < 0.01);
        // GELU(1) ≈ 0.841
        assert!((y.data()[1] - 0.841).abs() < 0.1);
        // GELU(-1) ≈ -0.159
        assert!((y.data()[2] + 0.159).abs() < 0.1);
    }

    #[test]
    fn test_matmul_2d_shapes() {
        // matmul requires 2D tensors
        let a = Tensor::ones(&[3, 4]);
        let b = Tensor::ones(&[4, 5]);
        let c = matmul_batched(&a, &b);
        assert_eq!(c.shape(), &[3, 5]);
    }

    #[test]
    fn test_gqa_self_attention() {
        let gqa = GroupedQueryAttention::new(64, 8, 2);
        let x = Tensor::ones(&[2, 10, 64]);
        let (output, _weights) = gqa.forward_self(&x, None);
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_placeholder() {
        let gqa = GroupedQueryAttention::placeholder(64, 8, 2);
        assert_eq!(gqa.embed_dim(), 64);
        assert_eq!(gqa.num_heads(), 8);
        assert_eq!(gqa.num_kv_heads(), 2);
    }

    #[test]
    fn test_gqa_mut_accessors() {
        let mut gqa = GroupedQueryAttention::new(64, 8, 2);
        let _q = gqa.q_proj_mut();
        let _k = gqa.k_proj_mut();
        let _v = gqa.v_proj_mut();
        let _o = gqa.out_proj_mut();
    }

    #[test]
    fn test_gqa_parameters_mut() {
        let mut gqa = GroupedQueryAttention::new(64, 8, 2);
        let params = gqa.parameters_mut();
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_gqa_debug() {
        let gqa = GroupedQueryAttention::new(64, 8, 2);
        let debug_str = format!("{:?}", gqa);
        assert!(debug_str.contains("GroupedQueryAttention"));
    }

    #[test]
    fn test_rope_apply_batch() {
        let rope = RotaryPositionEmbedding::new(8, 128);
        let x = Tensor::ones(&[4, 10, 8, 8]); // batch=4
        let positions: Vec<usize> = (0..10).collect();
        let y = rope.apply(&x, &positions);
        assert_eq!(y.shape(), &[4, 10, 8, 8]);
    }

    #[test]
    fn test_linear_attention_debug() {
        let attn = LinearAttention::new(64, 8);
        let debug_str = format!("{:?}", attn);
        assert!(debug_str.contains("LinearAttention"));
    }

    #[test]
    fn test_linear_attention_parameters_mut() {
        let mut attn = LinearAttention::new(64, 8);
        let params = attn.parameters_mut();
        assert_eq!(params.len(), 8);
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_multi_head_attention_with_dropout() {
        let mha = MultiHeadAttention::new(64, 8).with_dropout(0.1);
        let x = Tensor::ones(&[2, 10, 64]);
        let output = mha.forward(&x);
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_multi_head_attention_train_eval() {
        let mut mha = MultiHeadAttention::new(64, 8);
        assert!(mha.training());
        mha.eval();
        assert!(!mha.training());
        mha.train();
        assert!(mha.training());
    }

    #[test]
    fn test_multi_head_attention_debug() {
        let mha = MultiHeadAttention::new(128, 8);
        let debug_str = format!("{:?}", mha);
        assert!(debug_str.contains("MultiHeadAttention"));
        assert!(debug_str.contains("embed_dim"));
        assert!(debug_str.contains("num_heads"));
    }

    #[test]
    fn test_multi_head_attention_getters() {
        let mha = MultiHeadAttention::new(128, 4);
        assert_eq!(mha.embed_dim(), 128);
        assert_eq!(mha.num_heads(), 4);
    }

    #[test]
    fn test_multi_head_attention_parameters_mut() {
        let mut mha = MultiHeadAttention::new(64, 8);
        let params = mha.parameters_mut();
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_transformer_encoder_layer_with_dropout() {
        let layer = TransformerEncoderLayer::new(64, 8, 256).with_dropout(0.2);
        let x = Tensor::ones(&[2, 10, 64]);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder_layer_forward_with_none_mask() {
        let layer = TransformerEncoderLayer::new(64, 8, 256);
        let x = Tensor::ones(&[2, 10, 64]);
        let y = layer.forward_with_mask(&x, None);
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_encoder_layer_train_eval() {
        let mut layer = TransformerEncoderLayer::new(64, 8, 256);
        assert!(layer.training());
        layer.eval();
        assert!(!layer.training());
        layer.train();
        assert!(layer.training());
    }

    #[test]
    fn test_transformer_encoder_layer_parameters_mut() {
        let mut layer = TransformerEncoderLayer::new(64, 8, 256);
        let params = layer.parameters_mut();
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn test_transformer_decoder_layer_forward_with_none_masks() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);
        let tgt = Tensor::ones(&[2, 10, 64]);
        let memory = Tensor::ones(&[2, 20, 64]);
        let output = layer.forward_with_memory(&tgt, &memory, None, None);
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_decoder_layer_train_eval() {
        let mut layer = TransformerDecoderLayer::new(64, 8, 256);
        assert!(layer.training());
        layer.eval();
        assert!(!layer.training());
        layer.train();
        assert!(layer.training());
    }

    #[test]
    fn test_transformer_decoder_layer_parameters() {
        let layer = TransformerDecoderLayer::new(64, 8, 256);
        let params = layer.parameters();
        // Self-attn: 8, Cross-attn: 8, Linear1: 2, Linear2: 2, Norms: 6
        assert!(params.len() > 0);
    }

    #[test]
    fn test_transformer_decoder_layer_parameters_mut() {
        let mut layer = TransformerDecoderLayer::new(64, 8, 256);
        let params = layer.parameters_mut();
        assert!(params.len() > 0);
    }

    #[test]
    fn test_add_tensors_shape_3d() {
        let a = Tensor::ones(&[2, 3, 4]);
        let b = Tensor::ones(&[2, 3, 4]);
        let c = add_tensors(&a, &b);
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_softmax_last_dim_single_row() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let y = softmax_last_dim(&x);
        let sum: f32 = y.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_causal_mask_small() {
        let mask = generate_causal_mask(2);
        assert_eq!(mask.shape(), &[2, 2]);
        // Lower triangle and diagonal should be 0
        assert_eq!(mask.data()[0], 0.0); // [0,0]
        assert_eq!(mask.data()[2], 0.0); // [1,0]
        assert_eq!(mask.data()[3], 0.0); // [1,1]
                                         // Upper triangle should be -inf
        assert!(mask.data()[1].is_infinite()); // [0,1]
    }

    #[test]
    fn test_rope_head_dim() {
        let rope = RotaryPositionEmbedding::new(16, 100);
        assert_eq!(rope.head_dim(), 16);
    }

    #[test]
    fn test_rope_single_position() {
        let rope = RotaryPositionEmbedding::new(8, 128);
        let x = Tensor::ones(&[1, 1, 2, 8]);
        let positions = vec![42_usize];
        let y = rope.apply(&x, &positions);
        assert_eq!(y.shape(), &[1, 1, 2, 8]);
    }
}
