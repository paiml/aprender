use super::*;
use super::mod_part_03::GroupedQueryAttention;

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
pub(super) fn elu_feature_map(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if v > 0.0 { v + 1.0 } else { v.exp() })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Sum over last dimension.
#[allow(clippy::needless_range_loop)]
pub(super) fn sum_last_dim(x: &Tensor) -> Tensor {
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
pub(super) fn matmul_with_broadcast(q: &Tensor, k_sum: &Tensor) -> Tensor {
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
pub(super) fn divide_with_eps(x: &Tensor, normalizer: &Tensor, eps: f32) -> Tensor {
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
pub(super) fn repeat_kv_heads(x: &Tensor, groups: usize) -> Tensor {
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
    pub(crate) head_dim: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) base: f32,
    /// Precomputed cos values [`max_seq_len`, `head_dim/2`]
    pub(crate) cos_cache: Vec<f32>,
    /// Precomputed sin values [`max_seq_len`, `head_dim/2`]
    pub(crate) sin_cache: Vec<f32>,
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
    pub(crate) num_heads: usize,
    /// Per-head slopes (geometric sequence starting from 2^(-8/n))
    pub(crate) slopes: Vec<f32>,
}
