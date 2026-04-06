
impl FusedQKVAttention {
    /// Create a new fused QKV attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension per attention head
    /// * `hidden_dim` - Total hidden dimension (must be divisible by head_dim)
    ///
    /// # Errors
    ///
    /// Returns error if head_dim is 0, hidden_dim is 0, or hidden_dim % head_dim != 0
    pub fn new(head_dim: usize, hidden_dim: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if !hidden_dim.is_multiple_of(head_dim) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim ({}) must be divisible by head_dim ({})",
                    hidden_dim, head_dim
                ),
            });
        }

        let num_heads = hidden_dim / head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let proj_size = hidden_dim * hidden_dim;

        // Initialize with small random-like values for non-degenerate behavior
        let init_weight = |size: usize| -> Vec<f32> {
            (0..size).map(|i| (i as f32 * 0.001).sin() * 0.02).collect()
        };

        Ok(Self {
            head_dim,
            hidden_dim,
            num_heads,
            scale,
            w_q: init_weight(proj_size),
            w_k: init_weight(proj_size),
            w_v: init_weight(proj_size),
            w_o: init_weight(proj_size),
        })
    }

    /// Forward pass with fused QKV projection and attention
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [seq_len, hidden_dim]
    ///
    /// # Returns
    ///
    /// Output tensor [seq_len, hidden_dim]
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match hidden_dim
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(RealizarError::InvalidShape {
                reason: "Input must have at least 2 dimensions [seq_len, hidden_dim]".to_string(),
            });
        }

        let seq_len = shape[0];
        let input_dim = shape[shape.len() - 1];

        if input_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input hidden_dim ({}) doesn't match layer hidden_dim ({})",
                    input_dim, self.hidden_dim
                ),
            });
        }

        let data = input.data();

        // Compute Q, K, V projections
        let mut q = vec![0.0f32; seq_len * self.hidden_dim];
        let mut k = vec![0.0f32; seq_len * self.hidden_dim];
        let mut v = vec![0.0f32; seq_len * self.hidden_dim];

        // Matrix multiply: [seq_len, hidden_dim] @ [hidden_dim, hidden_dim]
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                let mut sum_q = 0.0f32;
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for l in 0..self.hidden_dim {
                    let inp = data[i * self.hidden_dim + l];
                    sum_q += inp * self.w_q[l * self.hidden_dim + j];
                    sum_k += inp * self.w_k[l * self.hidden_dim + j];
                    sum_v += inp * self.w_v[l * self.hidden_dim + j];
                }
                q[i * self.hidden_dim + j] = sum_q;
                k[i * self.hidden_dim + j] = sum_k;
                v[i * self.hidden_dim + j] = sum_v;
            }
        }

        // Compute attention per head
        let mut output = vec![0.0f32; seq_len * self.hidden_dim];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;

            // Compute attention scores for this head
            for i in 0..seq_len {
                // Find max for numerical stability (causal: only j <= i)
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=i {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_idx = i * self.hidden_dim + head_offset + d;
                        let k_idx = j * self.hidden_dim + head_offset + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    let score = dot * self.scale;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Compute softmax with log-sum-exp trick
                // Using enumerate() pattern for causal attention where j <= i
                let mut sum_exp = 0.0f32;
                let mut scores = vec![0.0f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q_idx = i * self.hidden_dim + head_offset + d;
                        let k_idx = j * self.hidden_dim + head_offset + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    *score = (dot * self.scale - max_score).exp();
                    sum_exp += *score;
                }

                // Normalize and compute weighted sum of values
                if sum_exp > 0.0 {
                    for d in 0..self.head_dim {
                        let mut weighted_sum = 0.0f32;
                        for (j, &score) in scores.iter().enumerate() {
                            let v_idx = j * self.hidden_dim + head_offset + d;
                            weighted_sum += (score / sum_exp) * v[v_idx];
                        }
                        output[i * self.hidden_dim + head_offset + d] = weighted_sum;
                    }
                }
            }
        }

        // Output projection
        let mut final_output = vec![0.0f32; seq_len * self.hidden_dim];
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                let mut sum = 0.0f32;
                for l in 0..self.hidden_dim {
                    sum += output[i * self.hidden_dim + l] * self.w_o[l * self.hidden_dim + j];
                }
                final_output[i * self.hidden_dim + j] = sum;
            }
        }

        Tensor::from_vec(vec![seq_len, self.hidden_dim], final_output)
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get mutable access to Q projection weights for loading
    pub fn w_q_mut(&mut self) -> &mut [f32] {
        &mut self.w_q
    }

    /// Get mutable access to K projection weights for loading
    pub fn w_k_mut(&mut self) -> &mut [f32] {
        &mut self.w_k
    }

    /// Get mutable access to V projection weights for loading
    pub fn w_v_mut(&mut self) -> &mut [f32] {
        &mut self.w_v
    }

    /// Get mutable access to output projection weights for loading
    pub fn w_o_mut(&mut self) -> &mut [f32] {
        &mut self.w_o
    }
}

/// Multi-Head Attention with support for MHA, MQA, and GQA
///
/// Implements three attention variants through configurable `KV` head count:
///
/// **Multi-Head Attention (MHA):** `num_kv_heads = num_heads`
/// - Each head has separate Q, K, V projections
/// - `KV` cache: `O(num_heads * seq_len * head_dim)`
/// - Standard attention mechanism
///
/// **Multi-Query Attention (MQA):** `num_kv_heads = 1`
/// - Each head has separate Q projection
/// - All heads share single K, V projection
/// - `KV` cache: `O(seq_len * head_dim)` - reduces by `num_heads` factor
/// - Used in `PaLM`, Falcon, `StarCoder`
///
/// **Grouped-Query Attention (GQA):** `1 < num_kv_heads < num_heads`
/// - Heads grouped into `num_kv_heads` groups
/// - Each group shares K, V projections
/// - `KV` cache: `O(num_kv_heads * seq_len * head_dim)`
/// - Used in `Llama-2`, Mistral, `CodeLlama`
///
/// # Architecture
///
/// ```text
/// Input [hidden_dim]
///   |
///   ├─> Q_proj [hidden_dim -> hidden_dim] -> split into num_heads
///   ├─> K_proj [hidden_dim -> num_kv_heads * head_dim]
///   └─> V_proj [hidden_dim -> num_kv_heads * head_dim]
///   |
///   ├─> Attention (grouped by num_kv_heads)
///   |
///   └─> O_proj [hidden_dim -> hidden_dim]
///       |
///     Output [hidden_dim]
/// ```
///
/// # References
///
/// - "Attention is All You Need" - Vaswani et al., 2017 (MHA)
/// - "Fast Transformer Decoding: One Write-Head is All You Need" - Shazeer, 2019 (MQA)
/// - "`PaLM`: Scaling Language Modeling with Pathways" - Chowdhery et al., 2022 (MQA)
/// - "`GQA`: Training Generalized Multi-Query Transformer" - Ainslie et al., 2023 (GQA)
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads (Q heads)
    num_heads: usize,
    /// Number of key/value heads (for GQA/MQA)
    num_kv_heads: usize,
    /// Dimension per attention head
    head_dim: usize,
    /// Total hidden dimension (`num_heads * head_dim`)
    hidden_dim: usize,
    /// Query projection: `hidden_dim -> hidden_dim`
    q_proj: Linear,
    /// Key projection: `hidden_dim -> num_kv_heads * head_dim`
    k_proj: Linear,
    /// Value projection: `hidden_dim -> num_kv_heads * head_dim`
    v_proj: Linear,
    /// Output projection: `hidden_dim -> hidden_dim`
    o_proj: Linear,
    /// Per-head attention mechanism
    attention: Attention,
}
