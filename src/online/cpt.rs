//! Continual Pre-Training (CPT) Pipeline (GH-448)
//!
//! Adapts a pre-trained language model to a new domain by continuing
//! causal language modeling on domain-specific text corpora. Unlike
//! fine-tuning (which adds task-specific heads), CPT preserves the
//! autoregressive objective while shifting the distribution.
//!
//! # Key Design Decisions
//!
//! - **Data mixing**: Blend domain data with general data to prevent
//!   catastrophic forgetting (configurable ratio)
//! - **Learning rate**: Use small LR (1e-5 to 5e-5) with linear warmup
//!   to avoid destroying pre-trained representations
//! - **Replay buffer**: Optional experience replay from original training
//!   distribution to maintain general capabilities
//!
//! # References
//!
//! - Gururangan et al. 2020: "Don't Stop Pretraining"
//! - Ke et al. 2023: "Continual Pre-training of Language Models"

use crate::error::{AprenderError, Result};

/// Configuration for continual pre-training.
#[derive(Debug, Clone)]
pub struct CptConfig {
    /// Learning rate for domain adaptation
    pub learning_rate: f64,
    /// Warmup steps (linear warmup from 0 to learning_rate)
    pub warmup_steps: usize,
    /// Total training steps
    pub total_steps: usize,
    /// Sequence length for causal LM
    pub seq_length: usize,
    /// Domain data mixing ratio (0.0 = all general, 1.0 = all domain)
    pub domain_mix_ratio: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Random seed
    pub seed: u64,
    /// Maximum gradient norm (gradient clipping)
    pub max_grad_norm: f64,
    /// Replay buffer size (0 = disabled)
    pub replay_buffer_size: usize,
}

impl Default for CptConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            warmup_steps: 100,
            total_steps: 1000,
            seq_length: 512,
            domain_mix_ratio: 0.7,
            weight_decay: 0.01,
            seed: 42,
            max_grad_norm: 1.0,
            replay_buffer_size: 0,
        }
    }
}

impl CptConfig {
    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "learning_rate must be positive finite, got {}",
                    self.learning_rate
                ),
            });
        }
        if self.total_steps == 0 {
            return Err(AprenderError::FormatError {
                message: "total_steps must be > 0".to_string(),
            });
        }
        if self.domain_mix_ratio < 0.0 || self.domain_mix_ratio > 1.0 {
            return Err(AprenderError::FormatError {
                message: format!(
                    "domain_mix_ratio must be in [0.0, 1.0], got {}",
                    self.domain_mix_ratio
                ),
            });
        }
        if self.seq_length == 0 {
            return Err(AprenderError::FormatError {
                message: "seq_length must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Learning rate schedule for CPT.
///
/// Linear warmup followed by cosine decay, which is the standard schedule
/// for continual pre-training (Gururangan et al. 2020).
#[derive(Debug, Clone)]
pub struct CptSchedule {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl CptSchedule {
    /// Create a new learning rate schedule.
    #[must_use]
    pub fn new(config: &CptConfig) -> Self {
        Self {
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.total_steps,
        }
    }

    /// Get learning rate at a given step.
    #[must_use]
    pub fn lr_at_step(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f64 / self.warmup_steps.max(1) as f64)
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps).max(1) as f64;
            let progress = progress.min(1.0);
            self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}

/// Data mixer for blending domain and general corpora.
///
/// Implements the mixing strategy from "Don't Stop Pretraining":
/// each batch contains `domain_mix_ratio` fraction of domain data
/// and `1 - domain_mix_ratio` fraction of general data.
#[derive(Debug, Clone)]
pub struct DataMixer {
    domain_mix_ratio: f64,
    seed: u64,
    step: usize,
}

impl DataMixer {
    /// Create a new data mixer.
    #[must_use]
    pub fn new(domain_mix_ratio: f64, seed: u64) -> Self {
        Self {
            domain_mix_ratio,
            seed,
            step: 0,
        }
    }

    /// Determine whether the next sample should be from domain or general data.
    ///
    /// Returns `true` for domain data, `false` for general data.
    pub fn next_is_domain(&mut self) -> bool {
        // Deterministic mixing using LCG
        self.step += 1;
        let state = self
            .seed
            .wrapping_add(self.step as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (state >> 33) as f64 / (1u64 << 31) as f64;
        r < self.domain_mix_ratio
    }

    /// Mix two token sequences according to the mixing ratio.
    ///
    /// Returns a batch where each element is drawn from either domain
    /// or general data according to the configured ratio.
    pub fn mix_batches(
        &mut self,
        domain_tokens: &[Vec<u32>],
        general_tokens: &[Vec<u32>],
        batch_size: usize,
    ) -> Vec<Vec<u32>> {
        let mut batch = Vec::with_capacity(batch_size);
        let mut domain_idx = 0;
        let mut general_idx = 0;

        for _ in 0..batch_size {
            if self.next_is_domain() && domain_idx < domain_tokens.len() {
                batch.push(domain_tokens[domain_idx].clone());
                domain_idx += 1;
            } else if general_idx < general_tokens.len() {
                batch.push(general_tokens[general_idx].clone());
                general_idx += 1;
            } else if domain_idx < domain_tokens.len() {
                batch.push(domain_tokens[domain_idx].clone());
                domain_idx += 1;
            }
        }
        batch
    }

    /// Get the mixing ratio.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.domain_mix_ratio
    }
}

/// Experience replay buffer for catastrophic forgetting prevention.
///
/// Stores a fixed-size buffer of examples from the original training
/// distribution. During CPT, a fraction of each batch is replaced with
/// replay examples to maintain general capabilities.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    buffer: Vec<Vec<u32>>,
    capacity: usize,
    insert_idx: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            insert_idx: 0,
        }
    }

    /// Add an example to the buffer (ring buffer semantics).
    pub fn add(&mut self, tokens: Vec<u32>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(tokens);
        } else {
            self.buffer[self.insert_idx % self.capacity] = tokens;
        }
        self.insert_idx += 1;
    }

    /// Sample `n` examples from the buffer (deterministic, cyclic).
    #[must_use]
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<u32>> {
        if self.buffer.is_empty() {
            return vec![];
        }

        let mut samples = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (state >> 33) as usize % self.buffer.len();
            samples.push(self.buffer[idx].clone());
        }
        samples
    }

    /// Get current buffer size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// CPT training progress tracker.
#[derive(Debug, Clone)]
pub struct CptProgress {
    /// Current step
    pub step: usize,
    /// Total steps
    pub total_steps: usize,
    /// Current learning rate
    pub current_lr: f64,
    /// Running average loss
    pub avg_loss: f64,
    /// Domain samples seen
    pub domain_samples: usize,
    /// General samples seen
    pub general_samples: usize,
    /// Replay samples used
    pub replay_samples: usize,
}

impl CptProgress {
    /// Create initial progress.
    #[must_use]
    pub fn new(total_steps: usize) -> Self {
        Self {
            step: 0,
            total_steps,
            current_lr: 0.0,
            avg_loss: 0.0,
            domain_samples: 0,
            general_samples: 0,
            replay_samples: 0,
        }
    }

    /// Update progress with a new training step.
    pub fn update(&mut self, lr: f64, loss: f64, domain: bool) {
        self.step += 1;
        self.current_lr = lr;
        // Exponential moving average of loss
        let alpha = 0.99_f64.min(1.0 - 1.0 / (self.step as f64 + 1.0));
        self.avg_loss = alpha * self.avg_loss + (1.0 - alpha) * loss;
        if domain {
            self.domain_samples += 1;
        } else {
            self.general_samples += 1;
        }
    }

    /// Get progress as a fraction [0.0, 1.0].
    #[must_use]
    pub fn fraction(&self) -> f64 {
        self.step as f64 / self.total_steps.max(1) as f64
    }

    /// Check if training is complete.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.step >= self.total_steps
    }
}

#[cfg(test)]
#[path = "cpt_tests.rs"]
mod tests;
