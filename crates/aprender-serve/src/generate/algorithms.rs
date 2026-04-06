//! Advanced Sampling Algorithms (PMAT-802)
//!
//! Unique sampling algorithms not in sampler.rs:
//! - Min-p sampling
//! - Mirostat (v1/v2) adaptive sampling
//! - Tail-Free Sampling (TFS)
//! - Typical sampling
//! - DRY (Don't Repeat Yourself) penalty
//! - XTC (Exclude Top Choices) sampling
//! - Eta sampling
//! - Token healing
//! - Classifier-Free Guidance (CFG)

use super::{sample_from_distribution, sample_greedy};
use crate::error::{RealizarError, Result};
use crate::layers::softmax;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Sample using min-p (minimum probability) sampling.
///
/// Filters tokens with probability below `min_p * max_prob` threshold.
pub fn sample_min_p(logits: &Tensor<f32>, min_p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&min_p) {
        return Err(RealizarError::InvalidShape {
            reason: "min_p must be in [0, 1]".to_string(),
        });
    }

    // Convert to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Find max probability
    let max_prob = probs.iter().copied().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;

    // Keep tokens above threshold
    let mut candidates: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p >= threshold)
        .collect();

    // Sort by probability descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if candidates.is_empty() {
        // Fallback to argmax
        return sample_greedy(logits);
    }

    // Renormalize and sample
    let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = candidates.iter().map(|(_, p)| p / sum).collect();
    let indices: Vec<usize> = candidates.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

/// Mirostat sampling state for adaptive perplexity targeting
///
/// Implements Mirostat 2.0 algorithm from the paper:
/// "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity"
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprise value (tau)
    pub tau: f32,
    /// Learning rate (eta)
    pub eta: f32,
    /// Current surprise estimate (mu)
    pub mu: f32,
}

impl Default for MirostatState {
    fn default() -> Self {
        Self {
            tau: 5.0, // Default target surprise
            eta: 0.1, // Learning rate
            mu: 10.0, // Initial mu = 2 * tau
        }
    }
}

impl MirostatState {
    /// Create new Mirostat state with specified tau
    pub fn new(tau: f32) -> Self {
        Self {
            tau,
            eta: 0.1,
            mu: 2.0 * tau,
        }
    }

    /// Set learning rate
    #[must_use]
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Update mu based on observed surprise
    pub fn update(&mut self, observed_surprise: f32) {
        self.mu -= self.eta * (observed_surprise - self.tau);
    }
}

/// Mirostat 2.0 sampling: adaptive sampling to target perplexity
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `state` - Mirostat state (will be updated)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_mirostat(
    logits: &Tensor<f32>,
    state: &mut MirostatState,
    rng_value: f32,
) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Save top candidate for fallback
    let top_candidate = indexed[0];

    // Calculate surprise values and find cutoff
    let mut candidates = Vec::new();
    for (idx, prob) in indexed {
        let surprise = -prob.ln();
        if surprise > state.mu {
            break;
        }
        candidates.push((idx, prob));
    }

    // Ensure at least one candidate
    if candidates.is_empty() {
        candidates.push(top_candidate);
    }

    // Renormalize and sample
    let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = candidates.iter().map(|(_, p)| p / sum).collect();
    let indices: Vec<usize> = candidates.iter().map(|(idx, _)| *idx).collect();

    let selected = sample_from_distribution(&normalized, &indices, rng_value);
    let selected_idx = indices.iter().position(|&i| i == selected).unwrap_or(0);
    let selected_prob = candidates[selected_idx].1;

    // Update mu based on observed surprise
    let observed_surprise = -selected_prob.ln();
    state.update(observed_surprise);

    Ok(selected)
}

/// Tail-Free Sampling (TFS): Filter tokens based on probability second derivatives
///
/// TFS analyzes the "tail" of the probability distribution and removes tokens
/// in the low-probability tail. It computes second derivatives to find where
/// the distribution starts to flatten out.
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `z` - TFS parameter (0.0 to 1.0, higher = more tokens kept)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_tfs(logits: &Tensor<f32>, z: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if indexed.len() < 3 {
        // Not enough tokens for second derivative, use greedy
        return Ok(indexed[0].0);
    }

    // Compute first derivatives (differences between consecutive probabilities)
    let first_derivatives: Vec<f32> = indexed
        .windows(2)
        .map(|w| (w[0].1 - w[1].1).abs())
        .collect();

    // Compute second derivatives
    let second_derivatives: Vec<f32> = first_derivatives
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .collect();

    // Normalize second derivatives
    let sum_second: f32 = second_derivatives.iter().sum();
    let normalized: Vec<f32> = if sum_second > 1e-9 {
        second_derivatives.iter().map(|&x| x / sum_second).collect()
    } else {
        vec![1.0 / second_derivatives.len() as f32; second_derivatives.len()]
    };

    // Find cumulative sum and cutoff point
    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &val) in normalized.iter().enumerate() {
        cumsum += val;
        if cumsum > z {
            cutoff_idx = i + 2; // +2 because second derivative is 2 steps behind
            break;
        }
    }

    // Keep tokens up to cutoff
    let kept: Vec<(usize, f32)> = indexed.into_iter().take(cutoff_idx.max(1)).collect();

    // Renormalize and sample
    let sum_kept: f32 = kept.iter().map(|(_, p)| p).sum();
    let normalized_kept: Vec<f32> = kept.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = kept.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(
        &normalized_kept,
        &indices,
        rng_value,
    ))
}

/// Locally Typical Sampling: Sample based on local typicality
///
/// Typical sampling selects tokens whose information content is close to
/// the expected information content (entropy) of the distribution.
/// This tends to produce more "typical" text.
///
/// Reference: Meister et al. (2022) "Locally Typical Sampling"
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `p` - Cumulative probability mass to keep (0.0 to 1.0)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_typical(logits: &Tensor<f32>, p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Compute entropy (expected information content)
    let entropy: f32 = -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Compute information content for each token: -log(p)
    // Then compute deviation from entropy: |info - entropy|
    let mut indexed: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &prob)| prob > 1e-10)
        .map(|(i, &prob)| {
            let info = -prob.ln();
            let deviation = (info - entropy).abs();
            (i, prob, deviation)
        })
        .collect();

    // Sort by deviation (most typical first)
    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Keep tokens until cumulative probability exceeds p
    let mut cumsum = 0.0;
    let mut kept: Vec<(usize, f32)> = Vec::new();
    for (idx, prob, _) in indexed {
        kept.push((idx, prob));
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    // Ensure at least one token
    if kept.is_empty() {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        return Ok(max_idx);
    }

    // Renormalize and sample
    let sum_kept: f32 = kept.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = kept.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = kept.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

/// DRY (Don't Repeat Yourself) sampling configuration
///
/// DRY sampling penalizes n-gram repetitions to prevent the model from
/// generating repetitive sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryConfig {
    /// Multiplier for the penalty (higher = stronger penalty)
    pub multiplier: f32,
    /// Base value for exponential penalty growth
    pub base: f32,
    /// Minimum n-gram length to consider
    pub allowed_length: usize,
    /// Maximum sequence length to check for repetitions
    pub penalty_last_n: usize,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 256,
        }
    }
}

impl DryConfig {
    /// Create new DRY config with specified multiplier
    pub fn new(multiplier: f32) -> Self {
        Self {
            multiplier,
            ..Default::default()
        }
    }

    /// Set the base for exponential penalty
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set minimum n-gram length
    #[must_use]
    pub fn with_allowed_length(mut self, len: usize) -> Self {
        self.allowed_length = len;
        self
    }

    /// Set penalty window size
    #[must_use]
    pub fn with_penalty_last_n(mut self, n: usize) -> Self {
        self.penalty_last_n = n;
        self
    }

    /// Check if DRY is enabled
    pub fn is_enabled(&self) -> bool {
        self.multiplier > 0.0
    }
}

include!("dry_penalty.rs");
include!("classifier_free_guidance.rs");
include!("sampling_tests.rs");
