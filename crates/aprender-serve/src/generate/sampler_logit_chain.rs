
impl LogitProcessorChain {
    /// Create an empty processor chain
    #[must_use]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain (builder pattern)
    #[must_use]
    pub fn with_processor<P: LogitProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors.push(Box::new(processor));
        self
    }

    /// Add a boxed processor to the chain (builder pattern)
    #[must_use]
    pub fn with_boxed_processor(mut self, processor: Box<dyn LogitProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Process logits through all processors in order
    pub fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        for processor in &self.processors {
            processor.process(logits, ctx);
        }
    }

    /// Get the number of processors in the chain
    #[must_use]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the chain is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get processor names for debugging
    #[must_use]
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

impl LogitProcessor for LogitProcessorChain {
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        LogitProcessorChain::process(self, logits, ctx);
    }

    fn name(&self) -> &'static str {
        "processor_chain"
    }
}

/// Model trait for generation pipeline
///
/// Implement this trait to use your model with GenerationPipeline.
pub trait GenerativeModel {
    /// Forward pass producing logits for next token
    ///
    /// # Arguments
    ///
    /// * `tokens` - Current token sequence
    ///
    /// # Returns
    ///
    /// Logits for vocabulary (shape: [vocab_size])
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Reset any cached state (e.g., KV cache)
    fn reset(&mut self) {}
}

/// Generation pipeline with processor chain
///
/// Orchestrates the generation loop with:
/// 1. Model forward pass
/// 2. Logit processing
/// 3. Token sampling
/// 4. EOS detection
///
/// # Example
///
/// ```rust,ignore
/// use realizar::generate::{GenerationPipeline, TokenSuppressor, GenerationConfig};
///
/// let pipeline = GenerationPipeline::new(model)
///     .add_processor(TokenSuppressor::new(vec![0, 1, 2]))
///     .with_config(GenerationConfig::greedy().with_eos_token_id(50256));
///
/// let tokens = pipeline.generate(&[1, 2, 3])?;
/// ```
pub struct GenerationPipeline<M: GenerativeModel> {
    model: M,
    processors: LogitProcessorChain,
    config: GenerationConfig,
}

impl<M: GenerativeModel> GenerationPipeline<M> {
    /// Create a new generation pipeline
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            processors: LogitProcessorChain::new(),
            config: GenerationConfig::default(),
        }
    }

    /// Add a logit processor to the pipeline
    #[must_use]
    pub fn add_processor<P: LogitProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors = self.processors.with_processor(processor);
        self
    }

    /// Set generation configuration
    #[must_use]
    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate tokens starting from initial sequence
    ///
    /// # Arguments
    ///
    /// * `initial_tokens` - Starting token sequence (prompt)
    ///
    /// # Returns
    ///
    /// Generated token sequence (including initial tokens)
    pub fn generate(&mut self, initial_tokens: &[u32]) -> Result<Vec<u32>> {
        let mut tokens = initial_tokens.to_vec();
        let n_vocab = self.model.vocab_size();
        let eos_token = self.config.eos_token_id;

        // Simple PRNG for sampling (deterministic with seed)
        let mut rng_state = self.config.seed.unwrap_or(42);

        for step in 0..self.config.max_tokens {
            // Forward pass
            let mut logits = self.model.forward(&tokens)?;

            // Apply logit processors
            let ctx = LogitProcessorContext::new(&tokens, step, n_vocab);
            self.processors.process(&mut logits, &ctx);

            // Sample next token
            let logits_tensor = Tensor::from_vec(vec![logits.len()], logits)?;

            // Simple LCG for RNG
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let rng_value = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

            let next_token = sample_token(&logits_tensor, &self.config, rng_value)? as u32;

            tokens.push(next_token);

            // Check for EOS
            if let Some(eos) = eos_token {
                if next_token == eos as u32 {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Get reference to the model
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get reference to the processor chain
    #[must_use]
    pub fn processors(&self) -> &LogitProcessorChain {
        &self.processors
    }

    /// Get reference to the config
    #[must_use]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }
}

// Tests extracted to tests.rs (PMAT-802)

#[cfg(test)]
#[path = "sampler_tests.rs"]
mod sampler_tests;
