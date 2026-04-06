
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

/// Tokenize brick - BPE encoding performance (GH-378).
///
/// Measures tokenizer encode latency as a ComputeBrick. Unlike transformer
/// bricks (which operate on f32 tensors), this brick operates on text→token_ids.
/// The budget is based on measured GH-378 results: 70µs for a 636-char payload
/// on the priority-queue BPE encoder (1.49x faster than HF tokenizers v0.22).
#[derive(Debug)]
pub struct TokenizeBrick {
    /// Input text length (chars) for budget calibration
    pub input_chars: usize,
    /// Budget
    budget: TokenBudget,
}

impl TokenizeBrick {
    /// Create a new tokenize brick with default budget (80µs for ~636 chars).
    pub fn new() -> Self {
        Self {
            input_chars: 636,
            budget: TokenBudget::from_latency(80.0), // 80µs target (GH-378: measured 70µs)
        }
    }

    /// Create with specific input size and proportional budget.
    pub fn for_input_chars(chars: usize) -> Self {
        // Linear budget scaling: 80µs per 636 chars
        let budget_us = (chars as f64 / 636.0) * 80.0;
        Self {
            input_chars: chars,
            budget: TokenBudget::from_latency(budget_us.max(1.0)),
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl Default for TokenizeBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBrick for TokenizeBrick {
    type Output = Vec<u32>;

    fn name(&self) -> &'static str {
        "tokenize"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::budget_met()]
    }
}

// ============================================================================
// Training Bricks
// ============================================================================

/// LoRA forward pass brick — measures rank-decomposed projection latency.
///
/// LoRA projects x through two small matrices: A [rank × d_in] and B [d_out × rank].
/// Budget calibrated for rank-16 projections on typical hidden dimensions.
#[derive(Debug)]
pub struct LoraForwardBrick {
    /// Input dimension (e.g., 3584 for Qwen2.5-7B hidden_dim)
    pub d_in: usize,
    /// Output dimension (same as d_in for Q/V projections)
    pub d_out: usize,
    /// LoRA rank (16 default from InstructConfig)
    pub rank: usize,
    /// LoRA alpha scaling factor
    pub alpha: f32,
    /// Budget
    budget: TokenBudget,
}

impl LoraForwardBrick {
    /// Create a LoRA forward brick with model dimensions.
    pub fn new(d_in: usize, d_out: usize, rank: usize, alpha: f32) -> Self {
        Self {
            d_in,
            d_out,
            rank,
            alpha,
            budget: TokenBudget::from_latency(5.0), // 5µs target for rank-16
        }
    }

    /// Create with default rank-16 config for a given hidden dimension.
    pub fn for_hidden_dim(hidden_dim: usize) -> Self {
        Self::new(hidden_dim, hidden_dim, 16, 32.0)
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for LoraForwardBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "lora_forward"
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

/// Optimizer step brick — measures SIMD AdamW update latency over LoRA parameters.
#[derive(Debug)]
pub struct OptimizerStepBrick {
    /// Total trainable parameters
    pub num_params: usize,
    /// Budget
    budget: TokenBudget,
}

impl OptimizerStepBrick {
    /// Create an optimizer step brick for the given parameter count.
    pub fn new(num_params: usize) -> Self {
        Self {
            num_params,
            budget: TokenBudget::from_latency(50.0), // 50µs target for SIMD AdamW
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for OptimizerStepBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "optimizer_step"
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

/// Loss computation brick — measures cross-entropy loss over logits.
#[derive(Debug)]
pub struct LossComputeBrick {
    /// Vocabulary size (e.g., 152064 for Qwen2.5)
    pub vocab_size: usize,
    /// Response token count
    pub seq_len: usize,
    /// Budget
    budget: TokenBudget,
}

impl LossComputeBrick {
    /// Create a loss compute brick for the given vocabulary and sequence length.
    pub fn new(vocab_size: usize, seq_len: usize) -> Self {
        Self {
            vocab_size,
            seq_len,
            budget: TokenBudget::from_latency(20.0), // 20µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for LossComputeBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "loss_compute"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::budget_met(),
            BrickAssertion::bounds(0.0, 30.0),
        ]
    }
}

/// Full training step brick — composite: forward + backward + optimizer.
///
/// Budget is the sum of sub-brick budgets scaled by layer count.
#[derive(Debug)]
pub struct TrainingStepBrick {
    /// Model hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// LoRA rank
    pub lora_rank: usize,
    /// Budget
    budget: TokenBudget,
}

impl TrainingStepBrick {
    /// Create a training step brick from model config dimensions.
    pub fn from_model_config(hidden_dim: usize, num_layers: usize, lora_rank: usize) -> Self {
        Self {
            hidden_dim,
            num_layers,
            lora_rank,
            budget: TokenBudget::from_latency(5000.0), // 5ms target for full step
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for TrainingStepBrick {
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "train_step"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::budget_met()]
    }
}

// ============================================================================
// Serving Bricks
// ============================================================================

/// TTFT (Time to First Token) brick — measures prefill + first decode latency.
#[derive(Debug)]
pub struct ServeTtftBrick {
    /// Input prompt length in tokens
    pub prompt_tokens: usize,
    /// Budget
    budget: TokenBudget,
}

impl ServeTtftBrick {
    /// Create a TTFT brick for the given prompt length.
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            budget: TokenBudget::from_latency(500.0), // 500µs target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for ServeTtftBrick {
    type Output = Vec<u32>;

    fn name(&self) -> &'static str {
        "ttft"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::budget_met()]
    }
}

/// Decode throughput brick — measures sustained token generation rate.
#[derive(Debug)]
pub struct ServeThroughputBrick {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Budget
    budget: TokenBudget,
}

impl ServeThroughputBrick {
    /// Create a throughput brick with 50 tok/s decode target.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: TokenBudget::from_throughput(50.0), // 50 tok/s target
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for ServeThroughputBrick {
    type Output = Vec<u32>;

    fn name(&self) -> &'static str {
        "throughput"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::budget_met()]
    }
}

/// Batch generation brick — measures concurrent request throughput.
///
/// Budget scales from single-request baseline with 70% efficiency factor.
#[derive(Debug)]
pub struct ServeBatchBrick {
    /// Number of concurrent requests
    pub batch_size: usize,
    /// Per-request generation length
    pub max_tokens: usize,
    /// Budget
    budget: TokenBudget,
}

impl ServeBatchBrick {
    /// Create a batch brick with scaled throughput target (70% efficiency).
    pub fn new(batch_size: usize, max_tokens: usize) -> Self {
        let throughput = 50.0 * batch_size as f64 * 0.7;
        Self {
            batch_size,
            max_tokens,
            budget: TokenBudget::from_throughput(throughput)
                .with_batch_size(batch_size),
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl ComputeBrick for ServeBatchBrick {
    type Output = Vec<u32>;

    fn name(&self) -> &'static str {
        "batch_generate"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::budget_met()]
    }
}
