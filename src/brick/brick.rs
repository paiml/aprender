
/// CB-BUDGET: Validate that a ComputeBrick implementation has assertions and budget.
///
/// This is the module-level budget validation gate required by CBTOP-SPEC-001.
/// Call this at brick registration time to ensure contract compliance.
pub fn validate_brick_contract(brick: &dyn ComputeBrick<Output = Vec<f32>>) -> Result<(), BrickError> {
    let assertions = brick.assertions();
    if assertions.is_empty() {
        return Err(BrickError::AssertionFailed {
            name: format!("{}/CB-BUDGET", brick.name()),
            expected: "at least 1 assertion".to_string(),
            actual: "0 assertions".to_string(),
        });
    }
    let budget = brick.budget();
    if budget.us_per_token <= 0.0 || budget.tokens_per_sec <= 0.0 {
        return Err(BrickError::BudgetExceeded {
            limit_us: 0.0,
            actual_us: budget.us_per_token,
        });
    }
    Ok(())
}

// ============================================================================
// Transformer Brick Implementations
// ============================================================================

/// RMSNorm brick - layer normalization.
#[derive(Debug)]
pub struct RmsNormBrick {
    /// Weight vector
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Budget
    budget: TokenBudget,
}

impl RmsNormBrick {
    /// Create a new RMSNorm brick.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self {
            weight,
            eps,
            budget: TokenBudget::from_latency(1.5), // 1.5µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Run RMSNorm on input.
    pub fn run(&self, input: &[f32]) -> Result<TokenResult<Vec<f32>>, BrickError> {
        if input.len() != self.weight.len() {
            return Err(BrickError::InvalidInput(format!(
                "Input len {} != weight len {}",
                input.len(),
                self.weight.len()
            )));
        }

        let start = Instant::now();

        // Compute RMS
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + self.eps).sqrt();

        // Normalize and scale
        let output: Vec<f32> = input
            .iter()
            .zip(self.weight.iter())
            .map(|(x, w)| (x / rms) * w)
            .collect();

        let elapsed_us = start.elapsed().as_micros() as f64;
        let result = TokenResult::new(output, 1, elapsed_us, &self.budget);

        // Check assertions
        for assertion in self.assertions() {
            assertion.check_f32(&result.output, result.budget_met)?;
        }

        Ok(result)
    }
}

impl ComputeBrick for RmsNormBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "rms_norm"
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

/// QKV projection brick.
#[derive(Debug)]
pub struct QkvBrick {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Q output dimension
    pub q_dim: usize,
    /// K output dimension
    pub k_dim: usize,
    /// V output dimension
    pub v_dim: usize,
    /// Budget
    budget: TokenBudget,
    /// Has bias (Qwen2 has large biases)
    pub has_bias: bool,
}

impl QkvBrick {
    /// Create a new QKV brick.
    pub fn new(hidden_dim: usize, q_dim: usize, k_dim: usize, v_dim: usize) -> Self {
        Self {
            hidden_dim,
            q_dim,
            k_dim,
            v_dim,
            budget: TokenBudget::from_latency(6.0), // 6µs target
            has_bias: false,
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Mark as having bias.
    #[must_use]
    pub fn with_bias(mut self) -> Self {
        self.has_bias = true;
        self
    }

    /// Total output dimension.
    pub fn total_out_dim(&self) -> usize {
        self.q_dim + self.k_dim + self.v_dim
    }
}

impl ComputeBrick for QkvBrick {
    type Output = (Vec<f32>, Vec<f32>, Vec<f32>);

    fn name(&self) -> &'static str {
        "qkv_proj"
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

/// RoPE brick - rotary position embedding.
#[derive(Debug)]
pub struct RopeBrick {
    /// Head dimension
    pub head_dim: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Base theta
    pub theta: f32,
    /// RoPE type (0=NORM, 2=NEOX)
    pub rope_type: u32,
    /// Budget
    budget: TokenBudget,
}

impl RopeBrick {
    /// Create a new RoPE brick.
    pub fn new(head_dim: usize, num_heads: usize, theta: f32, rope_type: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            theta,
            rope_type,
            budget: TokenBudget::from_latency(1.0), // 1µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for RopeBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "rope"
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

/// Attention brick.
#[derive(Debug)]
pub struct AttentionBrick {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Budget
    budget: TokenBudget,
}

impl AttentionBrick {
    /// Create a new attention brick.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            budget: TokenBudget::from_latency(10.0), // 10µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// GQA group size.
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }
}

impl ComputeBrick for AttentionBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "attention"
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
        ]
    }
}

/// FlashAttentionBrick - incremental flash attention for decode (P1 optimization).
///
/// **Algorithm** (FlashAttention-2, Dao et al. 2023):
/// ```text
/// For decode (single query token):
///   Q: [1, H, D]     (single query)
///   K: [S, H_kv, D]  (full KV cache)
///   V: [S, H_kv, D]  (full KV cache)
///
///   Online softmax (no full attention matrix materialization):
///   for tile in KV_tiles(TILE_SIZE=128):
///       S_tile = Q @ K_tile^T / sqrt(D)    # [1, H, TILE_SIZE]
///       m_new = max(m_old, max(S_tile))    # Running max
///       P_tile = exp(S_tile - m_new)       # Stable softmax numerator
///       O = O * exp(m_old - m_new) + P_tile @ V_tile  # Accumulate
///       l = l * exp(m_old - m_new) + sum(P_tile)      # Running denominator
///   O = O / l  # Final output
/// ```
///
/// **Performance vs naive**:
/// - Naive: O(S) memory for attention matrix
/// - Flash: O(TILE_SIZE) memory, 2x speedup from better cache locality
///
/// **Reference**: Dao, T., et al. (2023). "FlashAttention-2: Faster Attention
/// with Better Parallelism and Work Partitioning." arXiv:2307.08691.
#[derive(Debug, Clone)]
pub struct FlashAttentionBrick {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Tile size for KV cache (default: 128 for L2 cache fit)
    pub tile_size: usize,
    /// Budget (target: 5.0µs for 2x improvement over naive)
    budget: TokenBudget,
    /// Use online softmax (FlashAttention algorithm)
    pub use_online_softmax: bool,
}
