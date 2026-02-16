//! Sequence generation and decoding algorithms.
//!
//! This module provides decoding strategies for sequence-to-sequence models:
//!
//! - **Greedy decoding**: Select the most likely token at each step
//! - **Beam search**: Maintain top-k candidates at each step
//! - **Nucleus (top-p) sampling**: Sample from the smallest set of tokens
//!   with cumulative probability >= p
//! - **Top-k sampling**: Sample from the k most likely tokens
//! - **Temperature sampling**: Scale logits before sampling
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::generation::{BeamSearch, NucleusSampler, GenerationConfig};
//!
//! // Beam search decoding
//! let beam = BeamSearch::new(5);  // beam_size=5
//! let output = beam.generate(&model, &input, 50);
//!
//! // Nucleus sampling
//! let sampler = NucleusSampler::new(0.95);  // top_p=0.95
//! let output = sampler.sample(&logits);
//! ```
//!
//! # References
//!
//! - Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration.
//!   ICLR. (Nucleus sampling)
//! - Freitag, M., & Al-Onaizan, Y. (2017). Beam Search Strategies for Neural
//!   Machine Translation. ACL Workshop.

use crate::autograd::Tensor;

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum length of generated sequence
    pub max_length: usize,
    /// Minimum length of generated sequence
    pub min_length: usize,
    /// Temperature for sampling (higher = more random)
    pub temperature: f32,
    /// Top-k sampling: keep only top k tokens
    pub top_k: Option<usize>,
    /// Nucleus (top-p) sampling: keep tokens with cumulative prob >= p
    pub top_p: Option<f32>,
    /// Beam search width
    pub num_beams: usize,
    /// Length penalty (>1 favors longer sequences)
    pub length_penalty: f32,
    /// Repetition penalty (>1 penalizes repetition)
    pub repetition_penalty: f32,
    /// Early stopping for beam search
    pub early_stopping: bool,
    /// End-of-sequence token ID
    pub eos_token_id: Option<usize>,
    /// Pad token ID
    pub pad_token_id: Option<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 50,
            min_length: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            num_beams: 1,
            length_penalty: 1.0,
            repetition_penalty: 1.0,
            early_stopping: false,
            eos_token_id: None,
            pad_token_id: None,
        }
    }
}

impl GenerationConfig {
    /// Create a new generation config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum generation length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set temperature for sampling.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set nucleus (top-p) sampling.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set beam search width.
    #[must_use]
    pub fn with_num_beams(mut self, num_beams: usize) -> Self {
        self.num_beams = num_beams;
        self
    }

    /// Set EOS token ID.
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }
}

/// A single beam hypothesis in beam search.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token IDs in this hypothesis
    pub tokens: Vec<usize>,
    /// Log probability score
    pub score: f32,
    /// Whether this hypothesis is complete (reached EOS)
    pub is_done: bool,
}

impl BeamHypothesis {
    /// Create a new beam hypothesis.
    #[must_use]
    pub fn new(tokens: Vec<usize>, score: f32) -> Self {
        Self {
            tokens,
            score,
            is_done: false,
        }
    }

    /// Get the length-normalized score.
    #[must_use]
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        let len = self.tokens.len() as f32;
        self.score / len.powf(length_penalty)
    }
}

/// Beam search decoder for sequence generation.
///
/// Maintains the top-k (`beam_size`) hypotheses at each decoding step.
///
/// # Example
///
/// ```ignore
/// let beam_search = BeamSearch::new(5);  // beam_size=5
/// let hypotheses = beam_search.search(&logits_fn, start_token, max_len);
/// ```
pub struct BeamSearch {
    beam_size: usize,
    length_penalty: f32,
    early_stopping: bool,
    eos_token_id: Option<usize>,
}

impl BeamSearch {
    /// Create a new beam search decoder.
    ///
    /// # Arguments
    ///
    /// * `beam_size` - Number of beams to maintain
    #[must_use]
    pub fn new(beam_size: usize) -> Self {
        Self {
            beam_size,
            length_penalty: 1.0,
            early_stopping: false,
            eos_token_id: None,
        }
    }

    /// Set length penalty (>1 favors longer sequences).
    #[must_use]
    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty;
        self
    }

    /// Enable early stopping when all beams reach EOS.
    #[must_use]
    pub fn with_early_stopping(mut self) -> Self {
        self.early_stopping = true;
        self
    }

    /// Set EOS token ID.
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Perform beam search given log probabilities.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log probabilities for each token `[vocab_size]`
    /// * `current_beams` - Current beam hypotheses
    ///
    /// # Returns
    ///
    /// Updated beam hypotheses after one step.
    #[must_use]
    pub fn step(
        &self,
        log_probs: &Tensor,
        current_beams: &[BeamHypothesis],
    ) -> Vec<BeamHypothesis> {
        let vocab_size = log_probs.shape()[0];
        let mut candidates: Vec<BeamHypothesis> = Vec::new();

        for beam in current_beams {
            if beam.is_done {
                candidates.push(beam.clone());
                continue;
            }

            // Expand each beam with all vocabulary items
            for token_id in 0..vocab_size {
                let token_score = log_probs.data()[token_id];
                let new_score = beam.score + token_score;

                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(token_id);

                let mut new_beam = BeamHypothesis::new(new_tokens, new_score);

                // Check if reached EOS
                if Some(token_id) == self.eos_token_id {
                    new_beam.is_done = true;
                }

                candidates.push(new_beam);
            }
        }

        // Sort by normalized score and keep top beam_size
        candidates.sort_by(|a, b| {
            b.normalized_score(self.length_penalty)
                .partial_cmp(&a.normalized_score(self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(self.beam_size);
        candidates
    }

    /// Initialize beam search with a start token.
    #[must_use]
    pub fn init(&self, start_token: usize) -> Vec<BeamHypothesis> {
        vec![BeamHypothesis::new(vec![start_token], 0.0)]
    }

    /// Check if all beams are done.
    #[must_use]
    pub fn all_done(&self, beams: &[BeamHypothesis]) -> bool {
        beams.iter().all(|b| b.is_done)
    }

    /// Get the best hypothesis.
    #[must_use]
    pub fn best(&self, beams: &[BeamHypothesis]) -> Option<BeamHypothesis> {
        beams
            .iter()
            .max_by(|a, b| {
                a.normalized_score(self.length_penalty)
                    .partial_cmp(&b.normalized_score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Get `beam_size`.
    #[must_use]
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// Get `length_penalty`.
    #[must_use]
    pub fn length_penalty(&self) -> f32 {
        self.length_penalty
    }
}

impl std::fmt::Debug for BeamSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeamSearch")
            .field("beam_size", &self.beam_size)
            .field("length_penalty", &self.length_penalty)
            .field("early_stopping", &self.early_stopping)
            .field("eos_token_id", &self.eos_token_id)
            .finish()
    }
}

/// Nucleus (top-p) sampler for diverse text generation.
///
/// Samples from the smallest set of tokens whose cumulative probability
/// exceeds the threshold p. This provides a balance between diversity
/// and quality.
///
/// # Reference
///
/// Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration.
///
/// # Example
///
/// ```ignore
/// let sampler = NucleusSampler::new(0.95);
/// let token = sampler.sample(&logits);
/// ```
pub struct NucleusSampler {
    pub(crate) top_p: f32,
    pub(crate) temperature: f32,
    pub(crate) min_tokens_to_keep: usize,
}

#[path = "mod_part_02.rs"]
mod mod_part_02;
pub use mod_part_02::*;

#[path = "mod_part_03.rs"]
mod mod_part_03;
