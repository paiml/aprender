//! Advanced Sampling Strategies (PMAT-802)
//!
//! Extracted from generate/mod.rs - Advanced sampling algorithms.
//!
//! ## Contents
//! - Stop sequence detection
//! - Repetition, presence/frequency penalties
//! - Min-p, Mirostat, TFS, typical sampling
//! - DRY, XTC, Eta sampling
//! - Token healing

use super::{apply_temperature, sample_greedy, sample_token, GenerationConfig};
use crate::error::Result;
use crate::tensor::Tensor;
use std::collections::HashMap;

// ==================== Advanced Sampling Features ====================

use serde::{Deserialize, Serialize};

/// Stop sequence detector for generation termination
///
/// Detects when generated text matches stop sequences and signals termination.
/// Supports both token ID sequences and string patterns.
#[derive(Debug, Clone, Default)]
pub struct StopSequenceDetector {
    /// Token ID sequences to stop on
    token_sequences: Vec<Vec<usize>>,
    /// String patterns to stop on
    string_patterns: Vec<String>,
    /// Buffer for partial matches (token-based)
    token_buffer: Vec<usize>,
    /// Maximum sequence length to track
    max_seq_len: usize,
}

impl StopSequenceDetector {
    /// Create new stop sequence detector
    pub fn new() -> Self {
        Self {
            token_sequences: Vec::new(),
            string_patterns: Vec::new(),
            token_buffer: Vec::new(),
            max_seq_len: 0,
        }
    }

    /// Add a token ID sequence as stop condition
    #[must_use]
    pub fn with_token_sequence(mut self, sequence: Vec<usize>) -> Self {
        if !sequence.is_empty() {
            self.max_seq_len = self.max_seq_len.max(sequence.len());
            self.token_sequences.push(sequence);
        }
        self
    }

    /// Add a string pattern as stop condition
    #[must_use]
    pub fn with_string_pattern(mut self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.string_patterns.push(pattern);
        }
        self
    }

    /// Add multiple stop sequences from strings
    #[must_use]
    pub fn with_stop_strings(mut self, stops: Vec<String>) -> Self {
        for stop in stops {
            if !stop.is_empty() {
                self.string_patterns.push(stop);
            }
        }
        self
    }

    /// Check if a new token triggers a stop condition
    ///
    /// Returns true if generation should stop.
    pub fn check_token(&mut self, token_id: usize) -> bool {
        // Add to buffer
        self.token_buffer.push(token_id);

        // Trim buffer to max sequence length
        if self.token_buffer.len() > self.max_seq_len && self.max_seq_len > 0 {
            self.token_buffer.remove(0);
        }

        // Check token sequences
        for seq in &self.token_sequences {
            if self.token_buffer.ends_with(seq) {
                return true;
            }
        }

        false
    }

    /// Check if generated text contains a stop string
    ///
    /// Returns Some(position) if stop found, None otherwise.
    pub fn check_text(&self, text: &str) -> Option<usize> {
        for pattern in &self.string_patterns {
            if let Some(pos) = text.find(pattern) {
                return Some(pos);
            }
        }
        None
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.token_buffer.clear();
    }

    /// Check if detector has any stop conditions configured
    pub fn has_conditions(&self) -> bool {
        !self.token_sequences.is_empty() || !self.string_patterns.is_empty()
    }
}

/// Repetition penalty configuration
///
/// Penalizes tokens that have appeared in the context to reduce repetition.
/// Higher values = stronger penalty (1.0 = no penalty).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionPenaltyConfig {
    /// Penalty multiplier for repeated tokens (1.0 = no penalty, >1.0 = penalty)
    pub penalty: f32,
    /// Number of recent tokens to consider (0 = all)
    pub window_size: usize,
}

impl Default for RepetitionPenaltyConfig {
    fn default() -> Self {
        Self {
            penalty: 1.0, // No penalty by default
            window_size: 64,
        }
    }
}

impl RepetitionPenaltyConfig {
    /// Create with specified penalty
    pub fn new(penalty: f32) -> Self {
        Self {
            penalty,
            window_size: 64,
        }
    }

    /// Set window size for context
    #[must_use]
    pub fn with_window(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Check if penalty is enabled
    pub fn is_enabled(&self) -> bool {
        (self.penalty - 1.0).abs() > 1e-6
    }
}

/// Apply repetition penalty to logits
///
/// Divides logits of tokens that appear in context by the penalty factor.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - Repetition penalty configuration
///
/// # Returns
///
/// Logits with repetition penalty applied
pub fn apply_repetition_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &RepetitionPenaltyConfig,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();
    let vocab_size = data.len();

    // Get relevant context window
    let window_start = if config.window_size > 0 && context_tokens.len() > config.window_size {
        context_tokens.len() - config.window_size
    } else {
        0
    };
    let relevant_tokens = &context_tokens[window_start..];

    // Apply penalty to each token in context
    for &token_id in relevant_tokens {
        if token_id < vocab_size {
            let logit = penalized[token_id];
            // For positive logits, divide by penalty
            // For negative logits, multiply by penalty
            penalized[token_id] = if logit > 0.0 {
                logit / config.penalty
            } else {
                logit * config.penalty
            };
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Presence and frequency penalty configuration (OpenAI-style)
///
/// - Presence penalty: Constant penalty for tokens that appear at least once
/// - Frequency penalty: Penalty proportional to token frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceFrequencyPenalty {
    /// Presence penalty (penalty if token appeared at all)
    pub presence_penalty: f32,
    /// Frequency penalty (penalty per occurrence)
    pub frequency_penalty: f32,
}

impl Default for PresenceFrequencyPenalty {
    fn default() -> Self {
        Self {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl PresenceFrequencyPenalty {
    /// Create new penalty config
    pub fn new(presence: f32, frequency: f32) -> Self {
        Self {
            presence_penalty: presence,
            frequency_penalty: frequency,
        }
    }

    /// Check if any penalty is enabled
    pub fn is_enabled(&self) -> bool {
        self.presence_penalty.abs() > 1e-6 || self.frequency_penalty.abs() > 1e-6
    }
}

/// Apply presence and frequency penalties to logits
///
/// Formula: logit -= presence_penalty * (1 if token in context else 0)
/// Formula: logit -= frequency_penalty * count(token in context)
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - Presence/frequency penalty configuration
///
/// # Returns
///
/// Logits with penalties applied
pub fn apply_presence_frequency_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &PresenceFrequencyPenalty,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();
    let vocab_size = data.len();

    // Count token frequencies
    let mut token_counts: HashMap<usize, usize> = HashMap::new();
    for &token_id in context_tokens {
        if token_id < vocab_size {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }
    }

    // Apply penalties
    for (token_id, count) in token_counts {
        let presence = if count > 0 { 1.0 } else { 0.0 };
        penalized[token_id] -= config.presence_penalty * presence;
        penalized[token_id] -= config.frequency_penalty * (count as f32);
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Logit bias configuration
///
/// Allows adjusting specific token probabilities before sampling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogitBias {
    /// Map of token ID to bias value (added to logit)
    biases: HashMap<usize, f32>,
}

impl LogitBias {
    /// Create empty logit bias
    pub fn new() -> Self {
        Self {
            biases: HashMap::new(),
        }
    }

    /// Add bias for a specific token
    #[must_use]
    pub fn with_bias(mut self, token_id: usize, bias: f32) -> Self {
        self.biases.insert(token_id, bias);
        self
    }

    /// Add multiple biases from a map
    #[must_use]
    pub fn with_biases(mut self, biases: HashMap<usize, f32>) -> Self {
        self.biases.extend(biases);
        self
    }

    /// Check if any biases are configured
    pub fn is_empty(&self) -> bool {
        self.biases.is_empty()
    }

    /// Get bias for a token (0.0 if not set)
    pub fn get(&self, token_id: usize) -> f32 {
        self.biases.get(&token_id).copied().unwrap_or(0.0)
    }
}

/// Apply logit bias to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `bias` - Logit bias configuration
///
/// # Returns
///
/// Logits with biases applied
pub fn apply_logit_bias(logits: &Tensor<f32>, bias: &LogitBias) -> Tensor<f32> {
    if bias.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut biased = data.to_vec();
    let vocab_size = data.len();

    for (&token_id, &bias_value) in &bias.biases {
        if token_id < vocab_size {
            biased[token_id] += bias_value;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), biased).expect("Shape should match original logits")
}

// ===== Prompt Caching =====

/// Prompt cache entry
#[derive(Debug, Clone)]
pub struct PromptCacheEntry {
    /// Token sequence
    pub tokens: Vec<usize>,
    /// Cached KV state (simplified - in practice would be actual KV tensors)
    pub kv_hash: u64,
    /// Number of times this entry has been hit
    pub hit_count: usize,
    /// Last access timestamp
    pub last_access: std::time::Instant,
}

/// Prompt cache for efficient prefix reuse
///
/// Caches prompt prefixes to avoid recomputation when generating multiple
/// completions with the same prefix.
#[derive(Debug)]
pub struct PromptCache {
    /// Cache entries keyed by token sequence hash
    entries: std::collections::HashMap<u64, PromptCacheEntry>,
    /// Maximum cache size
    max_entries: usize,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new(100)
    }
}

include!("prompt_cache.rs");
include!("dynamic_temperature.rs");
include!("sampler_topk.rs");
include!("sampler_logit_chain.rs");
