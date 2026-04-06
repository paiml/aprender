
impl FlashAttentionBrick {
    /// Create new flash attention brick with default tile size.
    #[must_use]
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            tile_size: 128, // Optimal for L2 cache on most GPUs
            budget: TokenBudget::from_latency(5.0), // 5µs target (2x vs 10µs naive)
            use_online_softmax: true,
        }
    }

    /// Create with custom tile size (for tuning).
    #[must_use]
    pub fn with_tile_size(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        tile_size: usize,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            tile_size,
            budget: TokenBudget::from_latency(5.0),
            use_online_softmax: true,
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// GQA group size (query heads per KV head).
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }

    /// Compute FLOPs per token for attention at given sequence length.
    ///
    /// FLOPs = 2 * H * D * S (Q @ K^T) + 2 * H * S * D (attn @ V)
    ///       = 4 * H * D * S
    #[must_use]
    pub fn flops(&self, seq_len: usize) -> u64 {
        4 * self.num_heads as u64 * self.head_dim as u64 * seq_len as u64
    }

    /// Compute memory bandwidth for naive vs flash attention.
    ///
    /// Naive: Reads full KV cache for each head
    /// Flash: Reads KV cache in tiles, better cache reuse
    #[must_use]
    pub fn memory_bytes(&self, seq_len: usize) -> (u64, u64) {
        let kv_bytes = 2 * self.num_kv_heads as u64 * self.head_dim as u64 * seq_len as u64 * 4; // K + V, f32
        let naive = kv_bytes + self.num_heads as u64 * seq_len as u64 * 4; // + attention matrix
        let flash = kv_bytes; // No attention matrix materialized
        (naive, flash)
    }

    /// Compute arithmetic intensity (FLOPs / bytes).
    #[must_use]
    pub fn arithmetic_intensity(&self, seq_len: usize) -> f64 {
        let (_, flash_bytes) = self.memory_bytes(seq_len);
        self.flops(seq_len) as f64 / flash_bytes as f64
    }

    /// Number of tiles needed for given sequence length.
    #[must_use]
    pub fn num_tiles(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.tile_size)
    }

    /// Compute flash attention with online softmax algorithm.
    ///
    /// **REAL IMPLEMENTATION** - FlashAttention-2 (Dao et al. 2023)
    ///
    /// # Arguments
    /// * `query` - Query tensor [num_heads, head_dim]
    /// * `keys` - Key cache [seq_len, num_kv_heads, head_dim]
    /// * `values` - Value cache [seq_len, num_kv_heads, head_dim]
    ///
    /// # Returns
    /// * Output tensor [num_heads, head_dim]
    pub fn forward(
        &self,
        query: &[f32],  // [num_heads * head_dim]
        keys: &[f32],   // [seq_len * num_kv_heads * head_dim]
        values: &[f32], // [seq_len * num_kv_heads * head_dim]
        seq_len: usize,
    ) -> Result<Vec<f32>, BrickError> {
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(BrickError::InvalidInput("Zero dimension".to_string()));
        }
        if query.len() != self.num_heads * self.head_dim {
            return Err(BrickError::InvalidInput(format!(
                "Query length {} != num_heads * head_dim = {}",
                query.len(),
                self.num_heads * self.head_dim
            )));
        }
        let expected_kv_len = seq_len * self.num_kv_heads * self.head_dim;
        if keys.len() != expected_kv_len || values.len() != expected_kv_len {
            return Err(BrickError::InvalidInput(format!(
                "KV length {} != seq_len * num_kv_heads * head_dim = {}",
                keys.len(),
                expected_kv_len
            )));
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let group_size = self.group_size();
        let mut output = vec![0.0f32; self.num_heads * self.head_dim];

        // Process each query head
        for h in 0..self.num_heads {
            let kv_head = h / group_size; // GQA: map query head to KV head
            let q_start = h * self.head_dim;

            // Online softmax variables (FlashAttention-2)
            let mut m = f32::NEG_INFINITY; // Running max
            let mut l = 0.0f32; // Running sum of exp
            let mut o = vec![0.0f32; self.head_dim]; // Running output

            // Process in tiles for cache efficiency
            for tile_start in (0..seq_len).step_by(self.tile_size) {
                let tile_end = (tile_start + self.tile_size).min(seq_len);

                for s in tile_start..tile_end {
                    // Compute Q @ K^T for this position
                    let k_start = (s * self.num_kv_heads + kv_head) * self.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..self.head_dim {
                        score += query[q_start + d] * keys[k_start + d];
                    }
                    score *= scale;

                    // Online softmax update (Milakov & Gimelshein, 2018)
                    let m_new = m.max(score);
                    let exp_old = (m - m_new).exp();
                    let exp_score = (score - m_new).exp();

                    // Update running sum
                    l = l * exp_old + exp_score;

                    // Update running output: O = O * exp(m_old - m_new) + exp(score - m_new) * V
                    let v_start = (s * self.num_kv_heads + kv_head) * self.head_dim;
                    for d in 0..self.head_dim {
                        o[d] = o[d] * exp_old + exp_score * values[v_start + d];
                    }

                    m = m_new;
                }
            }

            // Normalize output: O = O / l
            if l > 0.0 {
                for d in 0..self.head_dim {
                    output[h * self.head_dim + d] = o[d] / l;
                }
            }
        }

        Ok(output)
    }

    /// Execute flash attention with timing (for benchmarking).
    pub fn forward_timed(
        &self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<TokenResult<Vec<f32>>, BrickError> {
        let start = Instant::now();
        let output = self.forward(query, keys, values, seq_len)?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(TokenResult {
            output,
            tokens_processed: 1,
            us_per_token: elapsed_us,
            tokens_per_sec: 1_000_000.0 / elapsed_us,
            budget_met: elapsed_us <= self.budget.us_per_token,
        })
    }

    /// Legacy stub for backward compatibility (prefer `forward()`)
    #[deprecated(note = "Use forward() for real implementation")]
    pub fn execute(&self, _seq_len: usize) -> Result<Vec<f32>, BrickError> {
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(BrickError::InvalidInput("Zero dimension".to_string()));
        }
        Ok(vec![0.0; self.num_heads * self.head_dim])
    }
}

impl ComputeBrick for FlashAttentionBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
            // Attention outputs should be bounded
            BrickAssertion::bounds(-100.0, 100.0),
            // Custom assertions for flash attention
            BrickAssertion {
                name: "online_softmax".to_string(),
                description: "Uses online softmax (no full attention matrix)".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "online_softmax".to_string(),
                },
            },
            BrickAssertion {
                name: "tiled_kv_access".to_string(),
                description: "KV cache accessed in tiles for cache locality".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "tiled_kv_access".to_string(),
                },
            },
        ]
    }

    fn can_run(&self) -> bool {
        self.num_heads > 0 && self.head_dim > 0 && self.tile_size > 0
    }
}

/// FFN brick (SwiGLU).
#[derive(Debug)]
pub struct FfnBrick {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Intermediate dimension
    pub intermediate_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl FfnBrick {
    /// Create a new FFN brick.
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
            budget: TokenBudget::from_latency(12.2), // 12.2µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for FfnBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "ffn"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// Output projection brick.
#[derive(Debug)]
pub struct OProjBrick {
    /// Input dimension (num_heads * head_dim)
    pub in_dim: usize,
    /// Output dimension (hidden_dim)
    pub out_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl OProjBrick {
    /// Create a new O projection brick.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            budget: TokenBudget::from_latency(3.5), // 3.5µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for OProjBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "o_proj"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::no_nan(),
            BrickAssertion::no_inf(),
            BrickAssertion::budget_met(),
        ]
    }
}

/// ActivationQuantBrick - Q8 activation quantization for memory bandwidth reduction (P2).
///
/// **Purpose**: Quantize intermediate activations from f32 to int8 to reduce
/// memory bandwidth by 4x during inter-layer communication.
///
/// **Pipeline**:
/// ```text
/// Layer N output (f32) → Q8 quantize → transfer → Q8 dequantize → Layer N+1 input (f32)
///
/// Memory bandwidth reduction:
///   f32: 4 bytes/element
///   int8: 1 byte/element + 2 floats (scale, zero_point)
///   Effective: ~4x reduction for large activations
/// ```
///
/// **Algorithm** (per-tensor affine quantization):
/// ```text
/// Quantize:
///   scale = (max - min) / 255
///   zero_point = round(-min / scale)
///   q[i] = clamp(round(x[i] / scale + zero_point), 0, 255)
///
/// Dequantize:
///   x[i] = (q[i] - zero_point) * scale
/// ```
///
/// **Performance**: 2x memory BW improvement with ~0.1% accuracy loss
///
/// **Reference**: Jacob, B., et al. (2018). "Quantization and Training of
/// Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR '18.
#[derive(Debug, Clone)]
pub struct ActivationQuantBrick {
    /// Activation dimension (e.g., hidden_dim or intermediate_dim)
    pub dim: usize,
    /// Budget (target: 0.5µs for quant+dequant overhead)
    budget: TokenBudget,
    /// Use per-channel quantization (more accurate but slower)
    pub per_channel: bool,
}
