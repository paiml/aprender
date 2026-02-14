use super::*;

/// Simple test generator for unit tests
#[derive(Debug)]
struct DoubleGenerator;

impl SyntheticGenerator for DoubleGenerator {
    type Input = i32;
    type Output = i32;

    fn generate(&self, seeds: &[i32], _config: &SyntheticConfig) -> Result<Vec<i32>> {
        Ok(seeds.iter().map(|x| x * 2).collect())
    }

    fn quality_score(&self, generated: &i32, seed: &i32) -> f32 {
        if *generated == seed * 2 {
            1.0
        } else {
            0.0
        }
    }

    fn diversity_score(&self, batch: &[i32]) -> f32 {
        use std::collections::HashSet;
        let unique: HashSet<_> = batch.iter().collect();
        if batch.is_empty() {
            0.0
        } else {
            unique.len() as f32 / batch.len() as f32
        }
    }
}

#[test]
fn test_synthetic_generator_trait() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2, 3];
    let config = SyntheticConfig::default();

    let result = gen.generate(&seeds, &config).expect("generation failed");
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_quality_score() {
    let gen = DoubleGenerator;
    assert!((gen.quality_score(&4, &2) - 1.0).abs() < f32::EPSILON);
    assert!((gen.quality_score(&5, &2) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_diversity_score() {
    let gen = DoubleGenerator;
    assert!((gen.diversity_score(&[1, 2, 3]) - 1.0).abs() < f32::EPSILON);
    assert!((gen.diversity_score(&[1, 1, 1]) - (1.0 / 3.0)).abs() < f32::EPSILON);
    assert!((gen.diversity_score(&[]) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_generate_batched() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2, 3, 4, 5];
    let config = SyntheticConfig::default();

    let result = generate_batched(&gen, &seeds, &config, 2).expect("batched generation failed");
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[test]
fn test_generate_batched_single_batch() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2, 3];
    let config = SyntheticConfig::default();

    let result =
        generate_batched(&gen, &seeds, &config, 100).expect("batched generation failed");
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_generate_batched_empty() {
    let gen = DoubleGenerator;
    let seeds: Vec<i32> = vec![];
    let config = SyntheticConfig::default();

    let result = generate_batched(&gen, &seeds, &config, 2).expect("batched generation failed");
    assert!(result.is_empty());
}

#[test]
fn test_synthetic_stream_basic() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2, 3, 4, 5];
    let config = SyntheticConfig::default();

    let stream = SyntheticStream::new(&gen, &seeds, &config, 2);
    let results: Vec<_> = stream.map(|r| r.expect("generation failed")).collect();

    assert_eq!(results.len(), 3); // [1,2], [3,4], [5]
    assert_eq!(results[0], vec![2, 4]);
    assert_eq!(results[1], vec![6, 8]);
    assert_eq!(results[2], vec![10]);
}

#[test]
fn test_synthetic_stream_has_next() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2];
    let config = SyntheticConfig::default();

    let mut stream = SyntheticStream::new(&gen, &seeds, &config, 1);
    assert!(stream.has_next());
    assert_eq!(stream.remaining(), 2);

    stream.next();
    assert!(stream.has_next());
    assert_eq!(stream.remaining(), 1);

    stream.next();
    assert!(!stream.has_next());
    assert_eq!(stream.remaining(), 0);
}

#[test]
fn test_synthetic_stream_empty() {
    let gen = DoubleGenerator;
    let seeds: Vec<i32> = vec![];
    let config = SyntheticConfig::default();

    let mut stream = SyntheticStream::new(&gen, &seeds, &config, 2);
    assert!(!stream.has_next());
    assert!(stream.next().is_none());
}

#[test]
fn test_batch_size_zero_becomes_one() {
    let gen = DoubleGenerator;
    let seeds = vec![1, 2, 3];
    let config = SyntheticConfig::default();

    // batch_size of 0 should be treated as 1
    let result = generate_batched(&gen, &seeds, &config, 0).expect("generation failed");
    assert_eq!(result, vec![2, 4, 6]);
}

// ============================================================================
// EXTREME TDD: Andon Integration Tests
// ============================================================================

#[test]
fn test_check_andon_disabled() {
    let config = SyntheticConfig::default().with_andon_enabled(false);
    let andon = TestAndon::new();

    // Should not trigger even with 100% rejection
    let result = check_andon::<TestAndon>(0, 100, 0.5, &config, Some(&andon));
    assert!(result.is_ok());
    assert!(andon.events().is_empty());
}

#[test]
fn test_check_andon_empty_total() {
    let config = SyntheticConfig::default();
    let andon = TestAndon::new();

    // Zero total should not trigger
    let result = check_andon::<TestAndon>(0, 0, 0.5, &config, Some(&andon));
    assert!(result.is_ok());
}

#[test]
fn test_check_andon_high_rejection_halts() {
    let config = SyntheticConfig::default().with_andon_rejection_threshold(0.90);
    let andon = TestAndon::new();

    // 95% rejection rate (5 accepted out of 100)
    let result = check_andon::<TestAndon>(5, 100, 0.5, &config, Some(&andon));
    assert!(result.is_err());
    assert!(andon.was_halted());
    assert_eq!(andon.count_high_rejection(), 1);
}

#[test]
fn test_check_andon_acceptable_rejection() {
    let config = SyntheticConfig::default().with_andon_rejection_threshold(0.90);
    let andon = TestAndon::new();

    // 80% rejection rate (20 accepted out of 100) - below threshold
    let result = check_andon::<TestAndon>(20, 100, 0.5, &config, Some(&andon));
    assert!(result.is_ok());
    assert!(!andon.was_halted());
}

#[test]
fn test_check_andon_diversity_collapse_warns() {
    let config =
        SyntheticConfig::default().with_andon(AndonConfig::new().with_diversity_minimum(0.2));
    let andon = TestAndon::new();

    // Low diversity (0.1 < 0.2 minimum) but good acceptance
    let result = check_andon::<TestAndon>(80, 100, 0.1, &config, Some(&andon));
    assert!(result.is_ok()); // Diversity collapse is warning, not halt
    assert!(!andon.was_halted());
    assert_eq!(andon.events().len(), 1);
}

#[test]
fn test_check_andon_no_handler() {
    let config = SyntheticConfig::default().with_andon_rejection_threshold(0.90);

    // High rejection but no handler - should not error
    let result = check_andon::<DefaultAndon>(5, 100, 0.5, &config, None);
    assert!(result.is_ok());
}

#[test]
fn test_check_andon_multiple_conditions() {
    let config = SyntheticConfig::default()
        .with_andon_rejection_threshold(0.90)
        .with_andon(AndonConfig::new().with_diversity_minimum(0.2));
    let andon = TestAndon::new();

    // Both high rejection AND low diversity
    let result = check_andon::<TestAndon>(3, 100, 0.05, &config, Some(&andon));
    assert!(result.is_err()); // Rejection halts first
    assert!(andon.was_halted());
}
