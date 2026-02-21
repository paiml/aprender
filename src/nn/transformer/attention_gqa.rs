use super::attention_helpers::{
    divide_with_eps, elu_feature_map, matmul_with_broadcast, repeat_kv_heads, sum_last_dim,
};
use super::positional_encoding::{
    matmul_batched, reshape_for_attention, reshape_from_attention, transpose_last_two,
};
#[allow(clippy::wildcard_imports)]
use super::*;

/// Slice positional encoding for current sequence length.
pub(super) fn slice_pe(pe: &Tensor, seq_len: usize, d_model: usize) -> Tensor {
    let data: Vec<f32> = pe.data()[..seq_len * d_model].to_vec();
    Tensor::new(&data, &[seq_len, d_model])
}

/// Add positional encoding to input (broadcasting over batch).
pub(super) fn add_positional_encoding(x: &Tensor, pe: &Tensor) -> Tensor {
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
    pub(crate) embed_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_head_dim: usize,
    pub(crate) dropout_p: f32,
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) out_proj: Linear,
    pub(crate) training: bool,
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
