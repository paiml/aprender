pub(crate) use super::*;

// Test sample type
#[derive(Clone, Debug, PartialEq)]
pub(super) struct TestSample {
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
