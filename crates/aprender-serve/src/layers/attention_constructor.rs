
impl MultiHeadAttention {
    /// Create a new Multi-Head Attention layer with configurable `KV` heads
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Total hidden dimension (must be divisible by `num_heads`)
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (must divide `num_heads`)
    ///
    /// # Modes
    ///
    /// - MHA: `num_kv_heads = num_heads` (standard multi-head)
    /// - MQA: `num_kv_heads = 1` (all heads share K/V)
    /// - GQA: `1 < num_kv_heads < num_heads` (grouped heads)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `hidden_dim` is zero or not divisible by `num_heads`
    /// - `num_heads` is zero or not divisible by `num_kv_heads`
    /// - `num_kv_heads` is zero or greater than `num_heads`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Standard Multi-Head Attention (MHA)
    /// let mha = MultiHeadAttention::new(512, 8, 8)?;
    ///
    /// // Multi-Query Attention (MQA)
    /// let mqa = MultiHeadAttention::new(512, 8, 1)?;
    ///
    /// // Grouped-Query Attention (GQA) - 4 heads per group
    /// let gqa = MultiHeadAttention::new(512, 8, 2)?;
    /// ```
    pub fn new(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> Result<Self> {
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
        if num_kv_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_kv_heads must be > 0".to_string(),
            });
        }
        if num_kv_heads > num_heads {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_kv_heads {num_kv_heads} cannot be greater than num_heads {num_heads}"
                ),
            });
        }
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
                ),
            });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
                ),
            });
        }

        let head_dim = hidden_dim / num_heads;

        // Q projection: always hidden_dim -> hidden_dim (all query heads)
        let q_proj = Linear::new(hidden_dim, hidden_dim)?;

        // K/V projections: hidden_dim -> num_kv_heads * head_dim
        let kv_dim = num_kv_heads * head_dim;
        let k_proj = Linear::new(hidden_dim, kv_dim)?;
        let v_proj = Linear::new(hidden_dim, kv_dim)?;

        // Output projection: hidden_dim -> hidden_dim
        let o_proj = Linear::new(hidden_dim, hidden_dim)?;

        // Per-head attention mechanism
        let attention = Attention::new(head_dim)?;

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attention,
        })
    }

    /// Create standard Multi-Head Attention (MHA) - each head has separate K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `hidden_dim` is not divisible by `num_heads`
    pub fn mha(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Self::new(hidden_dim, num_heads, num_heads)
    }

    /// Create Multi-Query Attention (MQA) - all heads share K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `hidden_dim` is not divisible by `num_heads`
    pub fn mqa(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Self::new(hidden_dim, num_heads, 1)
    }

    /// Create Grouped-Query Attention (GQA) - heads grouped to share K/V
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if:
    /// - `hidden_dim` is 0
    /// - `num_heads` is 0
    /// - `num_kv_heads` is 0
    /// - `num_kv_heads` is greater than `num_heads`
    /// - `hidden_dim` is not divisible by `num_heads`
    /// - `num_heads` is not divisible by `num_kv_heads`
    pub fn gqa(hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> Result<Self> {
        contract_pre_gqa!();
        let result = Self::new(hidden_dim, num_heads, num_kv_heads);
        if result.is_ok() {
            contract_post_gqa!(&());
        }
        result
    }

    /// Forward pass through multi-head attention
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
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected 2D tensor [seq_len, hidden_dim], got shape {shape:?}"),
            });
        }

        let seq_len = shape[0];
        let input_dim = shape[1];

        if input_dim != self.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected hidden_dim={}, got {}", self.hidden_dim, input_dim),
            });
        }

        // Project Q, K, V
        let q = self.q_proj.forward(input)?; // [seq_len, hidden_dim]
        let k = self.k_proj.forward(input)?; // [seq_len, kv_dim]
        let v = self.v_proj.forward(input)?; // [seq_len, kv_dim]

        // Reshape Q into heads: [seq_len, num_heads, head_dim]
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // Calculate heads per group for GQA
        let heads_per_group = self.num_heads / self.num_kv_heads;

        // Process each query head
        let mut head_outputs = Vec::with_capacity(self.num_heads);

        for head_idx in 0..self.num_heads {
            // Extract Q for this head
            let mut q_head_data = Vec::with_capacity(seq_len * self.head_dim);
            for seq_idx in 0..seq_len {
                let q_row_start = seq_idx * self.hidden_dim;
                let head_start = q_row_start + head_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    q_head_data.push(q_data[head_start + offset]);
                }
            }
            let q_head = Tensor::from_vec(vec![seq_len, self.head_dim], q_head_data)?;

            // Determine which KV head this Q head uses (for GQA/MQA/MHA)
            let kv_head_idx = head_idx / heads_per_group;
            let kv_dim = self.num_kv_heads * self.head_dim;

            // Extract K, V for the corresponding KV head
            let mut k_head_data = Vec::with_capacity(seq_len * self.head_dim);
            let mut v_head_data = Vec::with_capacity(seq_len * self.head_dim);
            for seq_idx in 0..seq_len {
                let kv_row_start = seq_idx * kv_dim;
                let kv_head_start = kv_row_start + kv_head_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    k_head_data.push(k_data[kv_head_start + offset]);
                    v_head_data.push(v_data[kv_head_start + offset]);
                }
            }
            let k_head = Tensor::from_vec(vec![seq_len, self.head_dim], k_head_data)?;
            let v_head = Tensor::from_vec(vec![seq_len, self.head_dim], v_head_data)?;

            // Compute attention for this head
            let head_output = self.attention.forward(&q_head, &k_head, &v_head)?;
            head_outputs.push(head_output);
        }

        // Concatenate all head outputs: [seq_len, hidden_dim]
        let mut concat_data = Vec::with_capacity(seq_len * self.hidden_dim);
        for seq_idx in 0..seq_len {
            for head_output in &head_outputs {
                let head_output_data = head_output.data();
                let head_row_start = seq_idx * self.head_dim;
                for offset in 0..self.head_dim {
                    concat_data.push(head_output_data[head_row_start + offset]);
                }
            }
        }

        let concat = Tensor::from_vec(vec![seq_len, self.hidden_dim], concat_data)?;

        // Output projection
        self.o_proj.forward(&concat)
    }

    /// Get number of query heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get number of key/value heads
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
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

    /// Check if using Multi-Query Attention (MQA)
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Check if using Grouped-Query Attention (GQA)
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads > 1 && self.num_kv_heads < self.num_heads
    }

    /// Check if using standard Multi-Head Attention (MHA)
    #[must_use]
    pub fn is_mha(&self) -> bool {
        self.num_kv_heads == self.num_heads
    }
}
