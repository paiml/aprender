//! Text generation and sampling strategies
//!
//! This module provides the generation loop for autoregressive text generation
//! and various sampling strategies for token selection.
//!
//! # Sampling Strategies
//!
//! - **Greedy**: Always select the most probable token
//! - **Top-k**: Sample from the k most probable tokens
//! - **Top-p (nucleus)**: Sample from tokens with cumulative probability ≤ p
//! - **Temperature**: Scale logits before softmax to control randomness

use crate::{
    error::{RealizarError, Result},
    layers::softmax,
    tensor::Tensor,
};

// Submodules
mod algorithms;
pub mod cancel;
mod sampler;

// aprender#2376(3): cooperative cancellation for the decode loops.
pub use cancel::{CancelOnDrop, CancelToken};

// Re-exports from algorithms (unique sampling algorithms)
pub use algorithms::{
    analyze_token_healing, apply_cfg, apply_dry_penalty, apply_xtc, sample_eta, sample_min_p,
    sample_mirostat, sample_tfs, sample_typical, CfgConfig, DryConfig, EtaConfig, MirostatState,
    TokenHealingConfig, TokenHealingResult, XtcConfig,
};

// Re-exports from sampler (advanced sampling infrastructure)
pub use sampler::{
    apply_all_penalties, apply_dynamic_temperature, apply_infill_sampling, apply_logit_bias,
    apply_presence_frequency_penalty, apply_repetition_penalty, AdvancedGenerationConfig,
    BeamHypothesis, BeamSearchConfig, BeamSearchState, DynTempConfig, DynTempSampler,
    GenerationPipeline, GenerativeModel, InfillConfig, InfillResult, InfillSampler, LogitBias,
    LogitProcessor, LogitProcessorChain, LogitProcessorContext, PresenceFrequencyPenalty,
    PromptCache, PromptCacheEntry, PromptCacheStats, RepetitionPenalty, RepetitionPenaltyConfig,
    RepetitionPenaltySampler, Sampler, SamplerChain, SamplerContext, StopSequenceDetector,
    StreamingGenerator, TemperatureSampler, TemperatureScaler, TokenSuppressor, TopKSampler,
    TopPSampler,
};

// Shared [0,1) RNG-state mapper (defined in sampler_logit_chain.rs, include!d into sampler).
// Re-exported pub(crate) so every sampling loop (here + layers::model_model) uses the one
// f32-safe construction instead of the buggy `(state >> 33)/(1<<31)` idiom.
pub(crate) use sampler::lcg_state_to_unit_f32;

/// Sample from a probability distribution using a random value
///
/// # Arguments
///
/// * `probs` - Probabilities (must sum to 1)
/// * `indices` - Corresponding indices for each probability
/// * `rng_value` - Random value in [0, 1)
///
/// # Returns
///
/// Selected index
pub(crate) fn sample_from_distribution(probs: &[f32], indices: &[usize], rng_value: f32) -> usize {
    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if rng_value < cumsum {
            return indices[i];
        }
    }
    // Fallback to last token
    indices[indices.len() - 1]
}

/// Convert logits to softmax probabilities for a subset
///
/// # Arguments
///
/// * `indexed` - Pairs of (index, logit) sorted by logit descending
///
/// # Returns
///
/// Probabilities for the subset
pub(crate) fn logits_to_probs(indexed: &[(usize, f32)]) -> Vec<f32> {
    let max_logit = indexed[0].1;
    let exp_vals: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|e| e / sum_exp).collect()
}

/// Build nucleus for top-p sampling
///
/// # Arguments
///
/// * `indexed` - Pairs of (index, prob) sorted by prob descending
/// * `p` - Cumulative probability threshold
///
/// # Returns
///
/// Nucleus of (index, prob) pairs with cumulative probability >= p
pub(crate) fn build_nucleus(indexed: &[(usize, f32)], p: f32) -> Vec<(usize, f32)> {
    let mut cumsum = 0.0;
    let mut nucleus = Vec::new();
    for &(idx, prob) in indexed {
        nucleus.push((idx, prob));
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }
    nucleus
}

/// Sampling strategy for token selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Always select the most probable token
    Greedy,
    /// Sample from the k most probable tokens
    TopK {
        /// Number of top tokens to consider
        k: usize,
    },
    /// Sample from tokens with cumulative probability ≤ p
    TopP {
        /// Cumulative probability threshold
        p: f32,
    },
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Temperature for scaling logits (1.0 = no scaling)
    pub temperature: f32,
    /// Token ID for end-of-sequence
    pub eos_token_id: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Cooperative cancellation signal, polled once per decode step.
    ///
    /// aprender#2376(3). Defaults to [`CancelToken::never`] — zero-cost, never
    /// cancels — so every existing caller is unaffected.
    pub cancel: CancelToken,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            eos_token_id: None,
            seed: None,
            cancel: CancelToken::never(),
        }
    }
}

impl GenerationConfig {
    /// Create a new generation config with greedy sampling
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            ..Default::default()
        }
    }

    /// Create a new generation config with top-k sampling
    #[must_use]
    pub fn top_k(k: usize) -> Self {
        Self {
            strategy: SamplingStrategy::TopK { k },
            ..Default::default()
        }
    }

    /// Create a new generation config with top-p (nucleus) sampling
    #[must_use]
    pub fn top_p(p: f32) -> Self {
        Self {
            strategy: SamplingStrategy::TopP { p },
            ..Default::default()
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set maximum tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set end-of-sequence token ID
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Attach a cooperative cancellation signal (aprender#2376(3)).
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }
}

/// Apply temperature scaling to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `temperature` - Temperature value (> 0)
///
/// # Returns
///
/// Scaled logits
///
/// # Errors
///
/// Returns error if temperature is not positive
pub fn apply_temperature(logits: &Tensor<f32>, temperature: f32) -> Result<Tensor<f32>> {
    contract_pre_temperature!();
    // A non-finite temperature (NaN / ±inf) MUST be rejected. `NaN <= 0.0` is FALSE
    // (IEEE-754 comparisons with NaN are unordered), so the old `temperature <= 0.0`
    // guard let NaN through; then `x / NaN = NaN` poisons every logit -> NaN softmax ->
    // the cumulative draw (`rng_value < cumsum`) never fires -> a silent, biased fallback
    // to the last token, with no error surfaced. Require a positive, FINITE temperature.
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(RealizarError::InvalidShape {
            reason: "Temperature must be a positive finite number".to_string(),
        });
    }

    if (temperature - 1.0).abs() < 1e-6 {
        // No scaling needed
        return Ok(logits.clone());
    }

    let data = logits.data();
    let scaled: Vec<f32> = data.iter().map(|&x| x / temperature).collect();
    Tensor::from_vec(logits.shape().to_vec(), scaled)
}

/// Greedy sampling: select the token with highest probability
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_greedy(logits: &Tensor<f32>) -> Result<usize> {
    contract_pre_greedy!();
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(max_idx)
}

/// Top-k sampling: sample from the k most probable tokens
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `k` - Number of top tokens to consider
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if k is 0 or logits are empty
pub fn sample_top_k(logits: &Tensor<f32>, k: usize, rng_value: f32) -> Result<usize> {
    contract_pre_top_k!();
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if k == 0 {
        return Err(RealizarError::InvalidShape {
            reason: "k must be > 0".to_string(),
        });
    }

    // PERF-034: was a full O(V log V) `sort_by` over the entire vocabulary to keep
    // `k` entries, plus a fresh V-element `Vec<(usize, f32)>` (2.4 MB on Qwen2.5) and
    // the stable sort's n/2 scratch, once per token. `retain_top_k_sorted` selects in
    // O(V) into a reusable per-thread buffer; it reproduces the stable sort's tie
    // order exactly (value descending, then index ascending). See `sampling_select`.
    crate::sampling_select::with_candidate_scratch(|indexed| {
        indexed.extend(data.iter().copied().enumerate());
        crate::sampling_select::retain_top_k_sorted(indexed, k.min(data.len()));

        // Convert to probabilities and sample
        let probs = logits_to_probs(indexed);
        let indices: Vec<usize> = indexed.iter().map(|(idx, _)| *idx).collect();
        Ok(sample_from_distribution(&probs, &indices, rng_value))
    })
}

/// Top-p (nucleus) sampling: sample from tokens with cumulative probability ≤ p
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `p` - Cumulative probability threshold
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if p is not in (0, 1] or logits are empty
pub fn sample_top_p(logits: &Tensor<f32>, p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if p <= 0.0 || p > 1.0 {
        return Err(RealizarError::InvalidShape {
            reason: "p must be in (0, 1]".to_string(),
        });
    }

    // Convert logits to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Create (index, prob) pairs and sort by prob descending.
    //
    // PERF-034: top-p needs the whole ranking (the nucleus is unbounded), so this
    // stays a full sort — but into the reusable per-thread buffer and with
    // `sort_unstable_by`, which allocates nothing where the stable `sort_by`
    // allocates an n/2 scratch (~1.2 MB on a 152k vocabulary, per token). Identical
    // result: the explicit index tiebreak is the order the stable sort already gave.
    let nucleus = crate::sampling_select::with_candidate_scratch(|indexed| {
        indexed.extend(probs.iter().copied().enumerate());
        crate::sampling_select::sort_desc_by_index(indexed);

        // Build nucleus (cumulative probability <= p)
        build_nucleus(indexed, p)
    });

    // Renormalize and sample
    let nucleus_sum: f32 = nucleus.iter().map(|(_, prob)| prob).sum();
    let normalized_probs: Vec<f32> = nucleus.iter().map(|(_, prob)| prob / nucleus_sum).collect();
    let indices: Vec<usize> = nucleus.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(
        &normalized_probs,
        &indices,
        rng_value,
    ))
}

/// Sample a token based on the sampling strategy
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `config` - Generation configuration
/// * `rng_value` - Random value in [0, 1) for sampling (ignored for greedy)
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if temperature is invalid or sampling fails
pub fn sample_token(
    logits: &Tensor<f32>,
    config: &GenerationConfig,
    rng_value: f32,
) -> Result<usize> {
    // Apply temperature
    let scaled_logits = apply_temperature(logits, config.temperature)?;

    match config.strategy {
        SamplingStrategy::Greedy => sample_greedy(&scaled_logits),
        SamplingStrategy::TopK { k } => sample_top_k(&scaled_logits, k, rng_value),
        SamplingStrategy::TopP { p } => sample_top_p(&scaled_logits, p, rng_value),
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod generate_tests;

// Additional tests for coverage (Part 2)
#[cfg(test)]
#[path = "tests_sample_greedy.rs"]
mod generate_tests_part_02;

// Algorithm-specific tests
#[cfg(test)]
mod algorithms_tests;

// FALSIFY-SA: Sampling contract tests (sampling-algorithms-v1.yaml)
#[cfg(test)]
#[path = "tests_sampling_contract.rs"]
mod tests_sampling_contract;
