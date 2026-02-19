use super::mod_part_03::{add_positional_encoding, slice_pe};
#[allow(clippy::wildcard_imports)]
use super::*;

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
pub(super) fn transpose_last_two(x: &Tensor) -> Tensor {
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
pub(super) fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
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
pub(super) fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    x.mul_scalar(scale)
}

/// Add attention mask to scores (SIMD-accelerated).
pub(super) fn add_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
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
///
/// ONE PATH: Delegates to `nn::functional::softmax` (UCBD §4).
pub(super) fn softmax_last_dim(x: &Tensor) -> Tensor {
    crate::nn::functional::softmax(x, -1)
}

/// ONE PATH: Delegates to `nn::functional::dropout` (UCBD §4).
pub(super) fn apply_dropout(x: &Tensor, p: f32) -> Tensor {
    crate::nn::functional::dropout(x, p, true)
}

/// Reshape for multi-head attention: [batch, seq, embed] -> [batch, heads, seq, `head_dim`]
pub(super) fn reshape_for_attention(
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
pub(super) fn reshape_from_attention(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    embed_dim: usize,
) -> Tensor {
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
pub(super) fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
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
pub(super) fn gelu(x: &Tensor) -> Tensor {
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
