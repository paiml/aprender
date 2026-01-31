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
    top_p: f32,
    temperature: f32,
    min_tokens_to_keep: usize,
}

impl NucleusSampler {
    /// Create a new nucleus sampler.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Cumulative probability threshold (0.0, 1.0]
    #[must_use]
    pub fn new(top_p: f32) -> Self {
        assert!(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0.0, 1.0]");
        Self {
            top_p,
            temperature: 1.0,
            min_tokens_to_keep: 1,
        }
    }

    /// Set temperature for sampling.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set minimum tokens to keep (even if below `top_p` threshold).
    #[must_use]
    pub fn with_min_tokens_to_keep(mut self, min_tokens: usize) -> Self {
        self.min_tokens_to_keep = min_tokens;
        self
    }

    /// Filter logits using nucleus (top-p) sampling.
    ///
    /// Sets logits outside the nucleus to -infinity.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw logits from the model `[vocab_size]`
    ///
    /// # Returns
    ///
    /// Filtered logits tensor.
    #[must_use]
    pub fn filter(&self, logits: &Tensor) -> Tensor {
        let vocab_size = logits.data().len();

        // Apply temperature
        let scaled_logits: Vec<f32> = logits
            .data()
            .iter()
            .map(|&x| x / self.temperature)
            .collect();

        // Convert to probabilities (softmax)
        let max_logit = scaled_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // Sort by probability (descending)
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cumulative probability cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = vocab_size;

        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= self.top_p && i >= self.min_tokens_to_keep - 1 {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Create mask: keep only top_p tokens
        let mut filtered_logits = vec![f32::NEG_INFINITY; vocab_size];
        for &idx in &indices[..cutoff_idx] {
            filtered_logits[idx] = scaled_logits[idx];
        }

        Tensor::new(&filtered_logits, &[vocab_size])
    }

    /// Sample a token from the filtered distribution.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw logits from the model `[vocab_size]`
    ///
    /// # Returns
    ///
    /// Sampled token ID.
    #[must_use]
    pub fn sample(&self, logits: &Tensor) -> usize {
        let filtered = self.filter(logits);
        sample_from_logits(&filtered)
    }

    /// Get `top_p` value.
    #[must_use]
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Get temperature value.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl std::fmt::Debug for NucleusSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NucleusSampler")
            .field("top_p", &self.top_p)
            .field("temperature", &self.temperature)
            .field("min_tokens_to_keep", &self.min_tokens_to_keep)
            .finish()
    }
}

/// Top-k sampler for text generation.
///
/// Samples from the k most likely tokens.
///
/// # Example
///
/// ```ignore
/// let sampler = TopKSampler::new(50);
/// let token = sampler.sample(&logits);
/// ```
pub struct TopKSampler {
    top_k: usize,
    temperature: f32,
}

impl TopKSampler {
    /// Create a new top-k sampler.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Number of top tokens to keep
    #[must_use]
    pub fn new(top_k: usize) -> Self {
        assert!(top_k > 0, "top_k must be > 0");
        Self {
            top_k,
            temperature: 1.0,
        }
    }

    /// Set temperature for sampling.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Filter logits to keep only top-k tokens.
    #[must_use]
    pub fn filter(&self, logits: &Tensor) -> Tensor {
        let vocab_size = logits.data().len();
        let k = self.top_k.min(vocab_size);

        // Apply temperature
        let scaled_logits: Vec<f32> = logits
            .data()
            .iter()
            .map(|&x| x / self.temperature)
            .collect();

        // Find top-k indices
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_by(|&a, &b| {
            scaled_logits[b]
                .partial_cmp(&scaled_logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create filtered logits
        let mut filtered_logits = vec![f32::NEG_INFINITY; vocab_size];
        for &idx in &indices[..k] {
            filtered_logits[idx] = scaled_logits[idx];
        }

        Tensor::new(&filtered_logits, &[vocab_size])
    }

    /// Sample a token from the filtered distribution.
    #[must_use]
    pub fn sample(&self, logits: &Tensor) -> usize {
        let filtered = self.filter(logits);
        sample_from_logits(&filtered)
    }

    /// Get `top_k` value.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

impl std::fmt::Debug for TopKSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopKSampler")
            .field("top_k", &self.top_k)
            .field("temperature", &self.temperature)
            .finish()
    }
}

/// Greedy decoder - always selects the most likely token.
#[derive(Debug, Default)]
pub struct GreedyDecoder;

impl GreedyDecoder {
    /// Create a new greedy decoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decode the most likely token.
    #[must_use]
    pub fn decode(&self, logits: &Tensor) -> usize {
        argmax(logits.data())
    }
}

/// Apply repetition penalty to logits.
///
/// # Arguments
///
/// * `logits` - Raw logits `[vocab_size]`
/// * `generated_tokens` - Previously generated token IDs
/// * `penalty` - Repetition penalty (>1 reduces probability of repeated tokens)
///
/// # Returns
///
/// Penalized logits tensor.
#[must_use]
pub fn apply_repetition_penalty(
    logits: &Tensor,
    generated_tokens: &[usize],
    penalty: f32,
) -> Tensor {
    let mut data = logits.data().to_vec();

    for &token_id in generated_tokens {
        if token_id < data.len() {
            // Apply penalty: divide if positive, multiply if negative
            if data[token_id] > 0.0 {
                data[token_id] /= penalty;
            } else {
                data[token_id] *= penalty;
            }
        }
    }

    Tensor::new(&data, logits.shape())
}

/// Apply temperature scaling to logits.
#[must_use]
pub fn apply_temperature(logits: &Tensor, temperature: f32) -> Tensor {
    assert!(temperature > 0.0, "Temperature must be positive");
    let data: Vec<f32> = logits.data().iter().map(|&x| x / temperature).collect();
    Tensor::new(&data, logits.shape())
}

// Helper: Sample from logits using the Gumbel-softmax trick.
fn sample_from_logits(logits: &Tensor) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Softmax
    let max_logit = logits
        .data()
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits
        .data()
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum: f32 = exp_logits.iter().sum();

    if sum <= 0.0 {
        // Fallback: return first valid token
        return 0;
    }

    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Sample using cumulative distribution
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }

    // Fallback: return last token
    probs.len() - 1
}

// Helper: argmax
fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Teacher Forcing scheduler for seq2seq training.
///
/// During training, controls the probability of using ground truth tokens
/// vs model predictions as input to the decoder.
///
/// # Schedules
///
/// - **Linear**: Decreases teacher forcing ratio linearly
/// - **Exponential**: Decreases exponentially
/// - **`InverseSquareRoot`**: Slow initial decay, faster later
///
/// # Example
///
/// ```ignore
/// use aprender::nn::generation::TeacherForcing;
///
/// let scheduler = TeacherForcing::linear(1.0, 0.1, 1000);
///
/// for step in 0..1000 {
///     let tf_ratio = scheduler.get_ratio(step);
///     // Use ground truth with probability tf_ratio
///     // Use model prediction with probability (1 - tf_ratio)
/// }
/// ```
///
/// # Reference
///
/// Williams, R. J., & Zipser, D. (1989). A Learning Algorithm for Continually
/// Running Fully Recurrent Neural Networks.
#[derive(Debug, Clone)]
pub struct TeacherForcing {
    schedule: TeacherForcingSchedule,
    initial_ratio: f32,
    final_ratio: f32,
    num_steps: usize,
}

/// Teacher forcing schedule type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TeacherForcingSchedule {
    /// Constant ratio throughout training.
    Constant,
    /// Linear decay from initial to final ratio.
    Linear,
    /// Exponential decay.
    Exponential,
    /// Inverse square root decay.
    InverseSquareRoot,
}

impl TeacherForcing {
    /// Create constant teacher forcing (ratio doesn't change).
    #[must_use]
    pub fn constant(ratio: f32) -> Self {
        assert!((0.0..=1.0).contains(&ratio), "Ratio must be in [0, 1]");
        Self {
            schedule: TeacherForcingSchedule::Constant,
            initial_ratio: ratio,
            final_ratio: ratio,
            num_steps: 1,
        }
    }

    /// Create linear decay schedule.
    #[must_use]
    pub fn linear(initial: f32, final_ratio: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&final_ratio),
            "Final ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::Linear,
            initial_ratio: initial,
            final_ratio,
            num_steps,
        }
    }

    /// Create exponential decay schedule.
    #[must_use]
    pub fn exponential(initial: f32, final_ratio: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&final_ratio),
            "Final ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::Exponential,
            initial_ratio: initial,
            final_ratio,
            num_steps,
        }
    }

    /// Create inverse square root decay schedule.
    #[must_use]
    pub fn inverse_sqrt(initial: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::InverseSquareRoot,
            initial_ratio: initial,
            final_ratio: 0.0,
            num_steps,
        }
    }

    /// Get teacher forcing ratio for given step.
    #[must_use]
    pub fn get_ratio(&self, step: usize) -> f32 {
        match self.schedule {
            TeacherForcingSchedule::Constant => self.initial_ratio,

            TeacherForcingSchedule::Linear => {
                if step >= self.num_steps {
                    self.final_ratio
                } else {
                    let progress = step as f32 / self.num_steps as f32;
                    self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
                }
            }

            TeacherForcingSchedule::Exponential => {
                if step >= self.num_steps {
                    self.final_ratio
                } else {
                    let decay = (self.final_ratio / self.initial_ratio.max(1e-10))
                        .powf(step as f32 / self.num_steps as f32);
                    self.initial_ratio * decay
                }
            }

            TeacherForcingSchedule::InverseSquareRoot => {
                self.initial_ratio / (1.0 + step as f32).sqrt()
            }
        }
    }

    /// Decide whether to use teacher forcing at this step.
    ///
    /// Returns true with probability equal to the current ratio.
    #[must_use]
    pub fn should_use_teacher(&self, step: usize) -> bool {
        let ratio = self.get_ratio(step);
        rand::random::<f32>() < ratio
    }

    #[must_use]
    pub fn schedule(&self) -> TeacherForcingSchedule {
        self.schedule
    }

    #[must_use]
    pub fn initial_ratio(&self) -> f32 {
        self.initial_ratio
    }

    #[must_use]
    pub fn final_ratio(&self) -> f32 {
        self.final_ratio
    }
}


#[cfg(test)]
mod tests;
