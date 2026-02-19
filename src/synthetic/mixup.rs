//! `MixUp` Data Augmentation.
//!
//! Implements `MixUp` (Zhang et al., 2018) for creating synthetic samples
//! via convex combinations in embedding space.
//!
//! # References
//!
//! Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
//! mixup: Beyond Empirical Risk Minimization. ICLR.

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;

// ============================================================================
// Embedding Trait
// ============================================================================

/// Trait for types that can be embedded into a vector space.
///
/// Implement this trait to enable `MixUp` interpolation for your data type.
///
/// # Example
///
/// ```
/// use aprender::synthetic::mixup::Embeddable;
///
/// #[derive(Clone)]
/// struct TextSample {
///     text: String,
///     embedding: Vec<f32>,
/// }
///
/// impl Embeddable for TextSample {
///     fn embedding(&self) -> &[f32] {
///         &self.embedding
///     }
///
///     fn from_embedding(embedding: Vec<f32>, reference: &Self) -> Self {
///         TextSample {
///             text: format!("[mixup from: {}]", reference.text),
///             embedding,
///         }
///     }
/// }
/// ```
pub trait Embeddable: Clone {
    /// Get the embedding vector for this sample.
    fn embedding(&self) -> &[f32];

    /// Create a new sample from an embedding, using reference for metadata.
    fn from_embedding(embedding: Vec<f32>, reference: &Self) -> Self;

    /// Get the embedding dimension.
    fn embedding_dim(&self) -> usize {
        self.embedding().len()
    }
}

// ============================================================================
// MixUp Configuration
// ============================================================================

/// Configuration for `MixUp` augmentation.
#[derive(Debug, Clone)]
pub struct MixUpConfig {
    /// Alpha parameter for Beta distribution (higher = more uniform mixing).
    pub alpha: f32,
    /// Whether to mix samples from different classes only.
    pub cross_class_only: bool,
    /// Minimum lambda value (avoids near-copies).
    pub lambda_min: f32,
    /// Maximum lambda value.
    pub lambda_max: f32,
}

impl Default for MixUpConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            cross_class_only: false,
            lambda_min: 0.2,
            lambda_max: 0.8,
        }
    }
}

impl MixUpConfig {
    /// Create a new `MixUp` configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the alpha parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.max(0.01);
        self
    }

    /// Enable cross-class mixing only.
    #[must_use]
    pub fn with_cross_class_only(mut self, enabled: bool) -> Self {
        self.cross_class_only = enabled;
        self
    }

    /// Set the lambda range.
    #[must_use]
    pub fn with_lambda_range(mut self, min: f32, max: f32) -> Self {
        self.lambda_min = min.clamp(0.0, 1.0);
        self.lambda_max = max.clamp(0.0, 1.0);
        if self.lambda_min > self.lambda_max {
            std::mem::swap(&mut self.lambda_min, &mut self.lambda_max);
        }
        self
    }
}

// ============================================================================
// Simple RNG for deterministic mixing
// ============================================================================

/// Linear Congruential Generator for deterministic randomness.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005);
        self.state = self.state.wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    /// Sample from Beta(alpha, alpha) using inverse transform.
    /// Simplified approximation for alpha in [0.1, 2.0].
    fn beta(&mut self, alpha: f32) -> f32 {
        // Use Kumaraswamy distribution as approximation
        // For alpha near 1, this gives uniform-like behavior
        // For alpha < 1, gives U-shaped; for alpha > 1, gives bell-shaped
        let u = self.next_f32().max(0.001);
        let a = alpha;
        let b = alpha;

        // Kumaraswamy CDF inverse approximation
        let x = (1.0 - (1.0 - u).powf(1.0 / b)).powf(1.0 / a);
        x.clamp(0.0, 1.0)
    }
}

// ============================================================================
// MixUp Generator
// ============================================================================

/// `MixUp` synthetic data generator.
///
/// Creates synthetic samples by interpolating between pairs of samples
/// in embedding space: x' = λ*x1 + (1-λ)*x2
///
/// # Example
///
/// ```
/// use aprender::synthetic::mixup::{MixUpGenerator, MixUpConfig, Embeddable};
/// use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
///
/// #[derive(Clone, Debug)]
/// struct Sample {
///     data: Vec<f32>,
/// }
///
/// impl Embeddable for Sample {
///     fn embedding(&self) -> &[f32] { &self.data }
///     fn from_embedding(embedding: Vec<f32>, _: &Self) -> Self {
///         Sample { data: embedding }
///     }
/// }
///
/// let gen = MixUpGenerator::<Sample>::new();
/// let seeds = vec![
///     Sample { data: vec![1.0, 0.0] },
///     Sample { data: vec![0.0, 1.0] },
/// ];
/// let config = SyntheticConfig::default().with_augmentation_ratio(1.0);
/// let mixed = gen.generate(&seeds, &config).expect("generation failed");
/// ```
#[derive(Debug, Clone)]
pub struct MixUpGenerator<T: Embeddable> {
    config: MixUpConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Embeddable> MixUpGenerator<T> {
    /// Create a new `MixUp` generator with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MixUpConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(mut self, config: MixUpConfig) -> Self {
        self.config = config;
        self
    }

    /// Interpolate two embeddings.
    fn interpolate(e1: &[f32], e2: &[f32], lambda: f32) -> Vec<f32> {
        e1.iter()
            .zip(e2.iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect()
    }

    /// ONE PATH: Delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
    fn cosine_similarity(e1: &[f32], e2: &[f32]) -> f32 {
        crate::nn::functional::cosine_similarity_slice(e1, e2)
    }

    /// Compute embedding variance as diversity measure.
    fn embedding_variance(embeddings: &[Vec<f32>]) -> f32 {
        if embeddings.is_empty() || embeddings[0].is_empty() {
            return 0.0;
        }

        let dim = embeddings[0].len();
        let n = embeddings.len() as f32;

        // Compute mean embedding
        let mut mean = vec![0.0; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v / n;
            }
        }

        // Compute variance
        let mut var = 0.0;
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                var += (v - mean[i]).powi(2);
            }
        }

        var / (n * dim as f32)
    }
}

impl<T: Embeddable> Default for MixUpGenerator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Embeddable + std::fmt::Debug> SyntheticGenerator for MixUpGenerator<T> {
    type Input = T;
    type Output = T;

    fn generate(&self, seeds: &[T], config: &SyntheticConfig) -> Result<Vec<T>> {
        if seeds.len() < 2 {
            return Ok(Vec::new());
        }

        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target);
        let mut rng = SimpleRng::new(config.seed);

        let mut attempts = 0;
        let max_attempts = target * config.max_attempts;

        while results.len() < target && attempts < max_attempts {
            attempts += 1;

            // Select two different samples
            let i = rng.next_usize(seeds.len());
            let mut j = rng.next_usize(seeds.len());
            if j == i {
                j = (j + 1) % seeds.len();
            }

            // Sample lambda from Beta distribution
            let raw_lambda = rng.beta(self.config.alpha);
            let lambda = self.config.lambda_min
                + raw_lambda * (self.config.lambda_max - self.config.lambda_min);

            // Interpolate embeddings
            let e1 = seeds[i].embedding();
            let e2 = seeds[j].embedding();

            if e1.len() != e2.len() || e1.is_empty() {
                continue;
            }

            let mixed_embedding = Self::interpolate(e1, e2, lambda);

            // Create mixed sample using first seed as reference
            let mixed = T::from_embedding(mixed_embedding, &seeds[i]);

            // Quality check
            let quality = self.quality_score(&mixed, &seeds[i]);
            if config.meets_quality(quality) {
                results.push(mixed);
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &T, seed: &T) -> f32 {
        // Quality is based on embedding similarity to seed
        let sim = Self::cosine_similarity(generated.embedding(), seed.embedding());

        // Transform to quality score: similarity should be moderate (not too high = copy)
        // Ideal range: 0.3 to 0.9
        let quality = if sim < 0.3 {
            sim / 0.3 * 0.5 // Low similarity = lower quality
        } else if sim > 0.9 {
            1.0 - (sim - 0.9) / 0.1 * 0.5 // Too similar = lower quality
        } else {
            0.5 + (sim - 0.3) / 0.6 * 0.5 // Sweet spot
        };

        quality.clamp(0.0, 1.0)
    }

    fn diversity_score(&self, batch: &[T]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        let embeddings: Vec<Vec<f32>> = batch.iter().map(|s| s.embedding().to_vec()).collect();

        // Use embedding variance as diversity measure
        let variance = Self::embedding_variance(&embeddings);

        // Normalize to [0, 1] assuming typical variance range
        (variance * 10.0).min(1.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "mixup_tests.rs"]
mod tests;
