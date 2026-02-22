#[allow(clippy::wildcard_imports)]
use super::*;

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
pub(super) fn sample_from_logits(logits: &Tensor) -> usize {
    use rand::Rng;
    let mut rng = rand::rng();

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
    let r: f32 = rng.random();
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
pub(super) fn argmax(data: &[f32]) -> usize {
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
    pub(crate) schedule: TeacherForcingSchedule,
    pub(crate) initial_ratio: f32,
    pub(crate) final_ratio: f32,
    pub(crate) num_steps: usize,
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
