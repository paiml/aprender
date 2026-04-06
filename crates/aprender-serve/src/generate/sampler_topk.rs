
impl Sampler for DynTempSampler {
    fn name(&self) -> &'static str {
        "dyn_temp"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        *logits = apply_dynamic_temperature(logits, &self.config);
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Top-K sampler
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// Number of top tokens to consider
    pub k: usize,
}

impl TopKSampler {
    /// Create a new top-k sampler
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Sampler for TopKSampler {
    fn name(&self) -> &'static str {
        "top_k"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        let data = logits.data_mut();
        if self.k == 0 || self.k >= data.len() {
            return; // k=0: no-op; k >= vocab: nothing to mask
        }

        // Find the k-th largest value using partial sort — O(n) average vs O(n log n) full sort.
        // We sort a values-only Vec (4 bytes/element) instead of (usize, f32) pairs (16 bytes).
        let mut values: Vec<f32> = data.to_vec();
        values.select_nth_unstable_by(self.k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = values[self.k - 1];

        // Mask everything below threshold in-place — no second allocation
        for logit in data.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Top-P (nucleus) sampler
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// Cumulative probability threshold (0.0 to 1.0)
    pub p: f32,
}

impl TopPSampler {
    /// Create a new top-p sampler
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Sampler for TopPSampler {
    fn name(&self) -> &'static str {
        "top_p"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        let data = logits.data();

        // Calculate softmax
        let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
        let mut indexed: Vec<(usize, f32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit, (logit - max_logit).exp() / exp_sum))
            .collect();

        // Sort by probability descending
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, _, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below cutoff
        let mut new_data = vec![f32::NEG_INFINITY; data.len()];
        for (idx, logit, _) in indexed.iter().take(cutoff_idx) {
            new_data[*idx] = *logit;
        }

        if let Ok(result) = Tensor::from_vec(logits.shape().to_vec(), new_data) {
            *logits = result;
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Repetition penalty sampler
#[derive(Debug, Clone)]
pub struct RepetitionPenaltySampler {
    /// Repetition penalty configuration
    pub config: RepetitionPenaltyConfig,
}

impl RepetitionPenaltySampler {
    /// Create a new repetition penalty sampler
    pub fn new(config: RepetitionPenaltyConfig) -> Self {
        Self { config }
    }
}

impl Sampler for RepetitionPenaltySampler {
    fn name(&self) -> &'static str {
        "repetition_penalty"
    }

    fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext) {
        *logits = apply_repetition_penalty(logits, &context.tokens, &self.config);
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Infill sampler
#[derive(Debug, Clone)]
pub struct InfillSampler {
    /// Infill/FIM configuration
    pub config: InfillConfig,
}

impl InfillSampler {
    /// Create a new infill sampler
    pub fn new(config: InfillConfig) -> Self {
        Self { config }
    }
}

impl Sampler for InfillSampler {
    fn name(&self) -> &'static str {
        "infill"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        let result = apply_infill_sampling(logits, &self.config);
        *logits = result.logits;
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// =============================================================================
// LogitProcessor Trait (RLZR-GEN-001)
// =============================================================================
//
// Composable logit processing for text generation pipelines.
// Based on HuggingFace Transformers LogitsProcessor pattern.
//
// References:
// - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration"
// - Wolf et al. (2020) "Transformers: State-of-the-Art NLP"
// =============================================================================

/// Context available during logit processing
///
/// Provides information about the current generation state to processors.
#[derive(Debug, Clone)]
pub struct LogitProcessorContext<'a> {
    /// Previously generated tokens (including initial prompt)
    pub tokens: &'a [u32],
    /// Current generation step (0-indexed, after initial tokens)
    pub step: usize,
    /// Vocabulary size
    pub n_vocab: usize,
}

impl<'a> LogitProcessorContext<'a> {
    /// Create a new context
    #[must_use]
    pub fn new(tokens: &'a [u32], step: usize, n_vocab: usize) -> Self {
        Self {
            tokens,
            step,
            n_vocab,
        }
    }
}

/// Logit processor trait for composable pre-sampling transforms
///
/// Processors are applied in order before sampling. They can:
/// - Set logits to -inf to suppress tokens
/// - Add penalties (repetition, length)
/// - Scale logits (temperature)
///
/// # Example
///
/// ```rust,ignore
/// use realizar::generate::{LogitProcessor, LogitProcessorContext};
///
/// struct MyProcessor;
///
/// impl LogitProcessor for MyProcessor {
///     fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
///         // Suppress token 0
///         logits[0] = f32::NEG_INFINITY;
///     }
/// }
/// ```
pub trait LogitProcessor: Send + Sync {
    /// Process logits in-place before sampling
    ///
    /// # Arguments
    ///
    /// * `logits` - Mutable slice of logits to modify
    /// * `ctx` - Context with token history and generation state
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext);

    /// Human-readable name for debugging and tracing
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

/// Suppress specific tokens by setting their logits to -inf
///
/// Use this to prevent certain tokens from being generated, such as:
/// - Special tokens (SOT, PREV, SOLM in Whisper)
/// - Profanity or sensitive content
/// - Invalid tokens for the current context
#[derive(Debug, Clone)]
pub struct TokenSuppressor {
    /// Token IDs to suppress
    suppress_ids: Vec<u32>,
}

impl TokenSuppressor {
    /// Create a new token suppressor
    ///
    /// # Arguments
    ///
    /// * `suppress_ids` - Token IDs to suppress (set to -inf)
    #[must_use]
    pub fn new(suppress_ids: Vec<u32>) -> Self {
        Self { suppress_ids }
    }

    /// Create from a slice of token IDs
    #[must_use]
    pub fn from_slice(suppress_ids: &[u32]) -> Self {
        Self {
            suppress_ids: suppress_ids.to_vec(),
        }
    }
}

impl LogitProcessor for TokenSuppressor {
    fn process(&self, logits: &mut [f32], _ctx: &LogitProcessorContext) {
        for &token_id in &self.suppress_ids {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "token_suppressor"
    }
}

/// Penalize repeated tokens to reduce repetitive generation
///
/// Applies a penalty to tokens that have appeared in the recent context.
/// Penalty > 1.0 reduces probability, < 1.0 increases it.
///
/// Based on: Keskar et al. (2019) "CTRL: A Conditional Transformer Language Model"
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    /// Penalty multiplier (> 1.0 to penalize, < 1.0 to encourage)
    penalty: f32,
    /// Look-back window size (0 = entire history)
    window: usize,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty processor
    ///
    /// # Arguments
    ///
    /// * `penalty` - Penalty multiplier (typical: 1.0-2.0)
    /// * `window` - Look-back window (0 = use all tokens)
    #[must_use]
    pub fn new(penalty: f32, window: usize) -> Self {
        Self { penalty, window }
    }

    /// Create with default window (entire history)
    #[must_use]
    pub fn with_penalty(penalty: f32) -> Self {
        Self { penalty, window: 0 }
    }
}

impl LogitProcessor for RepetitionPenalty {
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        // Determine which tokens to consider
        let tokens = if self.window > 0 && ctx.tokens.len() > self.window {
            &ctx.tokens[ctx.tokens.len() - self.window..]
        } else {
            ctx.tokens
        };

        // Apply penalty to tokens that have appeared
        for &token_id in tokens {
            if (token_id as usize) < logits.len() {
                let logit = logits[token_id as usize];
                // Apply penalty: divide positive logits, multiply negative logits
                logits[token_id as usize] = if logit > 0.0 {
                    logit / self.penalty
                } else {
                    logit * self.penalty
                };
            }
        }
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Scale logits by temperature
///
/// Temperature > 1.0 increases randomness (flatter distribution)
/// Temperature < 1.0 decreases randomness (sharper distribution)
/// Temperature = 1.0 has no effect
#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    /// Temperature value (must be > 0)
    temperature: f32,
}

impl TemperatureScaler {
    /// Create a new temperature scaler
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature value (> 0)
    ///
    /// # Panics
    ///
    /// Panics if temperature <= 0
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        Self { temperature }
    }
}

impl LogitProcessor for TemperatureScaler {
    fn process(&self, logits: &mut [f32], _ctx: &LogitProcessorContext) {
        if (self.temperature - 1.0).abs() > 1e-6 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &'static str {
        "temperature_scaler"
    }
}

/// Chain of logit processors applied in order
///
/// Allows composing multiple processors into a single processing step.
#[derive(Default)]
pub struct LogitProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}
