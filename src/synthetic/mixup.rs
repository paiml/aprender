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

    /// Compute cosine similarity between two embeddings.
    fn cosine_similarity(e1: &[f32], e2: &[f32]) -> f32 {
        if e1.len() != e2.len() || e1.is_empty() {
            return 0.0;
        }

        let dot: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = e2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 < f32::EPSILON || norm2 < f32::EPSILON {
            return 0.0;
        }

        (dot / (norm1 * norm2)).clamp(-1.0, 1.0)
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
mod tests {
    use super::*;

    // Test sample type
    #[derive(Clone, Debug, PartialEq)]
    struct TestSample {
        embedding: Vec<f32>,
        label: i32,
    }

    impl TestSample {
        fn new(embedding: Vec<f32>, label: i32) -> Self {
            Self { embedding, label }
        }
    }

    impl Embeddable for TestSample {
        fn embedding(&self) -> &[f32] {
            &self.embedding
        }

        fn from_embedding(embedding: Vec<f32>, reference: &Self) -> Self {
            Self {
                embedding,
                label: reference.label, // Keep reference label
            }
        }
    }

    // ========================================================================
    // MixUpConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = MixUpConfig::default();
        assert!((config.alpha - 0.4).abs() < f32::EPSILON);
        assert!(!config.cross_class_only);
        assert!((config.lambda_min - 0.2).abs() < f32::EPSILON);
        assert!((config.lambda_max - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_alpha() {
        let config = MixUpConfig::new().with_alpha(1.0);
        assert!((config.alpha - 1.0).abs() < f32::EPSILON);

        // Alpha should be at least 0.01
        let config = MixUpConfig::new().with_alpha(-1.0);
        assert!((config.alpha - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_cross_class() {
        let config = MixUpConfig::new().with_cross_class_only(true);
        assert!(config.cross_class_only);
    }

    #[test]
    fn test_config_with_lambda_range() {
        let config = MixUpConfig::new().with_lambda_range(0.3, 0.7);
        assert!((config.lambda_min - 0.3).abs() < f32::EPSILON);
        assert!((config.lambda_max - 0.7).abs() < f32::EPSILON);

        // Should swap if min > max
        let config = MixUpConfig::new().with_lambda_range(0.8, 0.2);
        assert!((config.lambda_min - 0.2).abs() < f32::EPSILON);
        assert!((config.lambda_max - 0.8).abs() < f32::EPSILON);

        // Should clamp to [0, 1]
        let config = MixUpConfig::new().with_lambda_range(-0.5, 1.5);
        assert!((config.lambda_min - 0.0).abs() < f32::EPSILON);
        assert!((config.lambda_max - 1.0).abs() < f32::EPSILON);
    }

    // ========================================================================
    // SimpleRng Tests
    // ========================================================================

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(43);

        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_rng_f32_range() {
        let mut rng = SimpleRng::new(12345);

        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_usize_range() {
        let mut rng = SimpleRng::new(12345);

        for _ in 0..100 {
            let v = rng.next_usize(10);
            assert!(v < 10);
        }

        // Edge case: max = 0
        assert_eq!(rng.next_usize(0), 0);
    }

    #[test]
    fn test_rng_beta_range() {
        let mut rng = SimpleRng::new(12345);

        for alpha in [0.1, 0.5, 1.0, 2.0] {
            for _ in 0..50 {
                let v = rng.beta(alpha);
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    // ========================================================================
    // MixUpGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = MixUpGenerator::<TestSample>::new();
        assert!((gen.config.alpha - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_with_config() {
        let config = MixUpConfig::new().with_alpha(1.0);
        let gen = MixUpGenerator::<TestSample>::new().with_config(config);
        assert!((gen.config.alpha - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_default() {
        let gen = MixUpGenerator::<TestSample>::default();
        assert!((gen.config.alpha - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolate() {
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];

        // lambda = 0.5 should give midpoint
        let result = MixUpGenerator::<TestSample>::interpolate(&e1, &e2, 0.5);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert!((result[1] - 0.5).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);

        // lambda = 1.0 should give e1
        let result = MixUpGenerator::<TestSample>::interpolate(&e1, &e2, 1.0);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 0.0).abs() < f32::EPSILON);

        // lambda = 0.0 should give e2
        let result = MixUpGenerator::<TestSample>::interpolate(&e1, &e2, 0.0);
        assert!((result[0] - 0.0).abs() < f32::EPSILON);
        assert!((result[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let e1 = vec![1.0, 0.0, 0.0];
        let sim = MixUpGenerator::<TestSample>::cosine_similarity(&e1, &e1);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let e2 = vec![0.0, 1.0, 0.0];
        let sim = MixUpGenerator::<TestSample>::cosine_similarity(&e1, &e2);
        assert!(sim.abs() < 0.001);

        // Opposite vectors
        let e3 = vec![-1.0, 0.0, 0.0];
        let sim = MixUpGenerator::<TestSample>::cosine_similarity(&e1, &e3);
        assert!((sim - (-1.0)).abs() < 0.001);

        // Empty vectors
        let empty: Vec<f32> = vec![];
        let sim = MixUpGenerator::<TestSample>::cosine_similarity(&empty, &empty);
        assert!((sim - 0.0).abs() < f32::EPSILON);

        // Different lengths
        let e4 = vec![1.0, 0.0];
        let sim = MixUpGenerator::<TestSample>::cosine_similarity(&e1, &e4);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_embedding_variance() {
        // Single embedding
        let embeddings = vec![vec![1.0, 0.0]];
        let var = MixUpGenerator::<TestSample>::embedding_variance(&embeddings);
        assert!((var - 0.0).abs() < f32::EPSILON);

        // Identical embeddings
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let var = MixUpGenerator::<TestSample>::embedding_variance(&embeddings);
        assert!((var - 0.0).abs() < f32::EPSILON);

        // Different embeddings
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let var = MixUpGenerator::<TestSample>::embedding_variance(&embeddings);
        assert!(var > 0.0);

        // Empty
        let var = MixUpGenerator::<TestSample>::embedding_variance(&[]);
        assert!((var - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generate_basic() {
        let gen = MixUpGenerator::<TestSample>::new();
        let seeds = vec![
            TestSample::new(vec![1.0, 0.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0, 0.0], 1),
            TestSample::new(vec![0.0, 0.0, 1.0], 2),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.1);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should generate some samples
        assert!(!result.is_empty());

        // All samples should have valid embeddings
        for sample in &result {
            assert_eq!(sample.embedding().len(), 3);
        }
    }

    #[test]
    fn test_generate_insufficient_seeds() {
        let gen = MixUpGenerator::<TestSample>::new();

        // 0 seeds
        let result = gen
            .generate(&[], &SyntheticConfig::default())
            .expect("should succeed");
        assert!(result.is_empty());

        // 1 seed - need at least 2 for mixing
        let seeds = vec![TestSample::new(vec![1.0, 0.0], 0)];
        let result = gen
            .generate(&seeds, &SyntheticConfig::default())
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_respects_target() {
        let gen = MixUpGenerator::<TestSample>::new();
        let seeds = vec![
            TestSample::new(vec![1.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0], 1),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0) // Target: 4 samples
            .with_quality_threshold(0.0); // Accept all

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should generate up to target (may be fewer due to quality)
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_generate_deterministic() {
        let gen = MixUpGenerator::<TestSample>::new();
        let seeds = vec![
            TestSample::new(vec![1.0, 0.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0, 0.0], 1),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.1)
            .with_seed(12345);

        let result1 = gen.generate(&seeds, &config).expect("generation failed");
        let result2 = gen.generate(&seeds, &config).expect("generation failed");

        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.embedding(), r2.embedding());
        }
    }

    #[test]
    fn test_quality_score() {
        let gen = MixUpGenerator::<TestSample>::new();

        let seed = TestSample::new(vec![1.0, 0.0, 0.0], 0);

        // Identical sample - should have moderate quality (not too high)
        let identical = TestSample::new(vec![1.0, 0.0, 0.0], 0);
        let score = gen.quality_score(&identical, &seed);
        assert!(score < 1.0); // Not perfect since it's too similar

        // Somewhat similar sample - should have good quality
        let similar = TestSample::new(vec![0.8, 0.2, 0.0], 0);
        let score = gen.quality_score(&similar, &seed);
        assert!(score > 0.3);

        // Very different sample - lower quality
        let different = TestSample::new(vec![0.0, 0.0, 1.0], 0);
        let score = gen.quality_score(&different, &seed);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_diversity_score() {
        let gen = MixUpGenerator::<TestSample>::new();

        // Empty batch
        let score = gen.diversity_score(&[]);
        assert!((score - 0.0).abs() < f32::EPSILON);

        // Single sample
        let single = vec![TestSample::new(vec![1.0, 0.0], 0)];
        let score = gen.diversity_score(&single);
        assert!((score - 0.0).abs() < f32::EPSILON);

        // Diverse batch
        let diverse = vec![
            TestSample::new(vec![1.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0], 1),
            TestSample::new(vec![-1.0, 0.0], 2),
        ];
        let score = gen.diversity_score(&diverse);
        assert!(score > 0.0);

        // Homogeneous batch
        let homogeneous = vec![
            TestSample::new(vec![1.0, 0.0], 0),
            TestSample::new(vec![1.0, 0.0], 0),
            TestSample::new(vec![1.0, 0.0], 0),
        ];
        let homo_score = gen.diversity_score(&homogeneous);
        assert!(homo_score < score);
    }

    #[test]
    fn test_generate_mixed_embeddings() {
        let gen = MixUpGenerator::<TestSample>::new();
        let seeds = vec![
            TestSample::new(vec![1.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0], 1),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_quality_threshold(0.0)
            .with_seed(42);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Mixed samples should have embeddings between the two seeds
        for sample in &result {
            let e = sample.embedding();
            // Each component should be in [0, 1] (interpolation of unit vectors)
            assert!((0.0..=1.0).contains(&e[0]));
            assert!((0.0..=1.0).contains(&e[1]));
        }
    }

    #[test]
    fn test_embeddable_trait() {
        let sample = TestSample::new(vec![1.0, 2.0, 3.0], 5);

        assert_eq!(sample.embedding(), &[1.0, 2.0, 3.0]);
        assert_eq!(sample.embedding_dim(), 3);

        let new_emb = vec![4.0, 5.0, 6.0];
        let new_sample = TestSample::from_embedding(new_emb.clone(), &sample);
        assert_eq!(new_sample.embedding(), &[4.0, 5.0, 6.0]);
        assert_eq!(new_sample.label, 5); // Kept from reference
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_mixup_pipeline() {
        let gen = MixUpGenerator::new().with_config(
            MixUpConfig::new()
                .with_alpha(0.5)
                .with_lambda_range(0.3, 0.7),
        );

        // Create samples with distinct embeddings
        let seeds = vec![
            TestSample::new(vec![1.0, 0.0, 0.0, 0.0], 0),
            TestSample::new(vec![0.0, 1.0, 0.0, 0.0], 1),
            TestSample::new(vec![0.0, 0.0, 1.0, 0.0], 2),
            TestSample::new(vec![0.0, 0.0, 0.0, 1.0], 3),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_quality_threshold(0.2)
            .with_seed(9999);

        let synthetic = gen.generate(&seeds, &config).expect("generation failed");

        // Verify generated samples
        for sample in &synthetic {
            // Should have correct dimension
            assert_eq!(sample.embedding_dim(), 4);

            // Quality should meet threshold
            let quality = gen.quality_score(sample, &seeds[0]);
            assert!(quality >= 0.0);
        }

        // Diversity should be reasonable
        if !synthetic.is_empty() {
            let diversity = gen.diversity_score(&synthetic);
            assert!((0.0..=1.0).contains(&diversity));
        }
    }
}
