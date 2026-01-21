//! Corpus Management for Online Learning
//!
//! Provides efficient corpus storage with deduplication, importance sampling,
//! and configurable eviction policies.
//!
//! # References
//!
//! - [Vitter 1985] "Random Sampling with a Reservoir"
//! - [Settles 2009] "Active Learning Literature Survey"
//!
//! # Toyota Way Principles
//!
//! - **Muda Elimination**: Deduplication avoids redundant training data
//! - **Heijunka**: Eviction policies level data quality over time

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::error::{AprenderError, Result};

/// Eviction policy for corpus buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// First-in-first-out - remove oldest samples
    FIFO,
    /// Remove lowest-weight samples
    ImportanceWeighted,
    /// Reservoir sampling for uniform distribution
    /// Reference: [Vitter 1985] "Random Sampling with a Reservoir"
    #[default]
    Reservoir,
    /// Keep diverse samples (maximize coverage)
    DiversitySampling,
}

/// Sample source for provenance tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum SampleSource {
    /// Synthetic generated data
    Synthetic,
    /// Hand-crafted training samples
    HandCrafted,
    /// Examples from documentation
    Examples,
    /// Production data from CITL
    #[default]
    Production,
    /// External dataset
    External(String),
}

/// A single sample in the corpus
#[derive(Debug, Clone)]
pub struct Sample {
    /// Feature vector
    pub features: Vec<f64>,
    /// Target value(s)
    pub target: Vec<f64>,
    /// Importance weight
    pub weight: f64,
    /// Source for provenance
    pub source: SampleSource,
    /// Optional timestamp
    pub timestamp: Option<u64>,
}

impl Sample {
    /// Create a new sample
    #[must_use]
    pub fn new(features: Vec<f64>, target: Vec<f64>) -> Self {
        Self {
            features,
            target,
            weight: 1.0,
            source: SampleSource::Production,
            timestamp: None,
        }
    }

    /// Create with weight
    #[must_use]
    pub fn with_weight(features: Vec<f64>, target: Vec<f64>, weight: f64) -> Self {
        Self {
            features,
            target,
            weight,
            source: SampleSource::Production,
            timestamp: None,
        }
    }

    /// Create with source
    #[must_use]
    pub fn with_source(features: Vec<f64>, target: Vec<f64>, source: SampleSource) -> Self {
        Self {
            features,
            target,
            weight: 1.0,
            source,
            timestamp: None,
        }
    }

    /// Compute a hash for deduplication
    fn compute_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash features (quantized to avoid float precision issues)
        for &f in &self.features {
            let quantized = (f * 1000.0).round() as i64;
            quantized.hash(&mut hasher);
        }

        // Hash target
        for &t in &self.target {
            let quantized = (t * 1000.0).round() as i64;
            quantized.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Configuration for corpus buffer
#[derive(Debug, Clone)]
pub struct CorpusBufferConfig {
    /// Maximum buffer size
    pub max_size: usize,
    /// Eviction policy
    pub policy: EvictionPolicy,
    /// Enable deduplication
    pub deduplicate: bool,
    /// Random seed for reservoir sampling
    pub seed: Option<u64>,
}

impl Default for CorpusBufferConfig {
    fn default() -> Self {
        Self {
            max_size: 10_000,
            policy: EvictionPolicy::Reservoir,
            deduplicate: true,
            seed: None,
        }
    }
}

/// Efficient corpus storage with deduplication
///
/// Reference: [Settles 2009] "Active Learning Literature Survey"
/// - Importance sampling for corpus construction
/// - Hash-based deduplication to avoid redundancy
#[derive(Debug)]
pub struct CorpusBuffer {
    /// Stored samples
    samples: Vec<Sample>,
    /// Hash set for deduplication
    seen_hashes: HashSet<u64>,
    /// Configuration
    config: CorpusBufferConfig,
    /// Reservoir sampling state
    n_seen: u64,
    /// Simple PRNG state for reservoir sampling
    rng_state: u64,
}

impl CorpusBuffer {
    /// Create a new corpus buffer with default config
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self::with_config(CorpusBufferConfig {
            max_size,
            ..Default::default()
        })
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: CorpusBufferConfig) -> Self {
        let rng_state = config.seed.unwrap_or(12345);
        Self {
            samples: Vec::with_capacity(config.max_size.min(1024)),
            seen_hashes: HashSet::new(),
            config,
            n_seen: 0,
            rng_state,
        }
    }

    /// Simple xorshift PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Get number of samples in buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Check if buffer is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.config.max_size
    }

    /// Add sample with deduplication
    ///
    /// Returns true if sample was added, false if duplicate
    pub fn add(&mut self, sample: Sample) -> bool {
        // Check for duplicate
        if self.config.deduplicate {
            let hash = sample.compute_hash();
            if self.seen_hashes.contains(&hash) {
                return false;
            }
            self.seen_hashes.insert(hash);
        }

        self.n_seen += 1;

        // Handle full buffer
        if self.samples.len() >= self.config.max_size {
            match self.config.policy {
                EvictionPolicy::FIFO => {
                    // Remove oldest (first) element
                    if self.config.deduplicate && !self.samples.is_empty() {
                        let old_hash = self.samples[0].compute_hash();
                        self.seen_hashes.remove(&old_hash);
                    }
                    self.samples.remove(0);
                    self.samples.push(sample);
                }
                EvictionPolicy::ImportanceWeighted => {
                    // Remove lowest weight sample
                    if let Some((idx, _)) =
                        self.samples.iter().enumerate().min_by(|(_, a), (_, b)| {
                            a.weight
                                .partial_cmp(&b.weight)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        if self.samples[idx].weight < sample.weight {
                            if self.config.deduplicate {
                                let old_hash = self.samples[idx].compute_hash();
                                self.seen_hashes.remove(&old_hash);
                            }
                            self.samples.remove(idx);
                            self.samples.push(sample);
                        }
                    }
                }
                EvictionPolicy::Reservoir => {
                    // Reservoir sampling: replace random element with probability max_size/n_seen
                    let prob = self.config.max_size as f64 / self.n_seen as f64;
                    let rand = (self.next_random() as f64) / (u64::MAX as f64);

                    if rand < prob {
                        let idx = (self.next_random() as usize) % self.samples.len();
                        if self.config.deduplicate {
                            let old_hash = self.samples[idx].compute_hash();
                            self.seen_hashes.remove(&old_hash);
                        }
                        self.samples[idx] = sample;
                    }
                }
                EvictionPolicy::DiversitySampling => {
                    // Find most similar sample and replace if new is more diverse
                    let new_hash = sample.compute_hash();
                    if let Some((idx, _)) =
                        self.samples.iter().enumerate().min_by(|(_, a), (_, b)| {
                            let dist_a = self.distance(&sample.features, &a.features);
                            let dist_b = self.distance(&sample.features, &b.features);
                            dist_a
                                .partial_cmp(&dist_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        // Replace if the new sample is different enough
                        let dist = self.distance(&sample.features, &self.samples[idx].features);
                        if dist > 0.1 {
                            // threshold for "different enough"
                            if self.config.deduplicate {
                                let old_hash = self.samples[idx].compute_hash();
                                self.seen_hashes.remove(&old_hash);
                                self.seen_hashes.insert(new_hash);
                            }
                            self.samples[idx] = sample;
                        }
                    }
                }
            }
        } else {
            self.samples.push(sample);
        }

        true
    }

    /// Add raw features and target
    pub fn add_raw(&mut self, features: Vec<f64>, target: Vec<f64>) -> bool {
        self.add(Sample::new(features, target))
    }

    /// Euclidean distance between two vectors
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let _ = self; // suppress unused self warning - method for consistency
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get all samples
    #[must_use]
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Export corpus as (features, targets)
    #[must_use]
    pub fn to_dataset(&self) -> (Vec<f64>, Vec<f64>, usize, usize) {
        if self.samples.is_empty() {
            return (vec![], vec![], 0, 0);
        }

        let n_features = self.samples[0].features.len();
        let n_targets = self.samples[0].target.len();
        let n_samples = self.samples.len();

        let mut features = Vec::with_capacity(n_samples * n_features);
        let mut targets = Vec::with_capacity(n_samples * n_targets);

        for sample in &self.samples {
            features.extend(&sample.features);
            targets.extend(&sample.target);
        }

        (features, targets, n_samples, n_features)
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
        self.seen_hashes.clear();
        self.n_seen = 0;
    }

    /// Get sample weights
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.weight).collect()
    }

    /// Update weight of sample at index
    pub fn update_weight(&mut self, idx: usize, weight: f64) -> Result<()> {
        if idx >= self.samples.len() {
            return Err(AprenderError::index_out_of_bounds(idx, self.samples.len()));
        }
        self.samples[idx].weight = weight;
        Ok(())
    }

    /// Get samples by source
    #[must_use]
    pub fn samples_by_source(&self, source: &SampleSource) -> Vec<&Sample> {
        self.samples
            .iter()
            .filter(|s| &s.source == source)
            .collect()
    }
}

/// Source for corpus merger
#[derive(Debug, Clone)]
pub struct CorpusSource {
    /// Source name for provenance
    pub name: String,
    /// Samples
    pub samples: Vec<Sample>,
    /// Weight multiplier (1.0 = normal)
    pub weight: f64,
    /// Priority (higher = prefer in dedup)
    pub priority: u8,
}

impl CorpusSource {
    /// Create a new corpus source
    pub fn new(name: impl Into<String>, samples: Vec<Sample>) -> Self {
        Self {
            name: name.into(),
            samples,
            weight: 1.0,
            priority: 0,
        }
    }

    /// Set weight multiplier
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Provenance tracking for merged corpus
#[derive(Debug, Clone)]
pub struct CorpusProvenance {
    /// Sources and their contributions
    pub sources: HashMap<String, (usize, usize)>, // (original, effective)
    /// Final merged size
    pub final_size: usize,
    /// Duplicates removed
    pub duplicates_removed: usize,
}

impl CorpusProvenance {
    /// Create new provenance tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            final_size: 0,
            duplicates_removed: 0,
        }
    }

    /// Add source contribution
    pub fn add_source(&mut self, name: &str, original: usize, effective: usize) {
        self.sources.insert(name.to_string(), (original, effective));
    }

    /// Set final merged size
    pub fn set_final_size(&mut self, size: usize) {
        self.final_size = size;
    }
}

impl Default for CorpusProvenance {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge multiple data sources with configurable weighting
///
/// Used by ruchy Oracle to combine:
/// - Synthetic data
/// - Hand-crafted corpus
/// - Examples corpus
/// - Production corpus
#[derive(Debug)]
pub struct CorpusMerger {
    /// Sources to merge
    sources: Vec<CorpusSource>,
    /// Enable deduplication
    deduplicate: bool,
    /// Random seed for shuffling
    shuffle_seed: Option<u64>,
}

impl CorpusMerger {
    /// Create a new corpus merger
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            deduplicate: true,
            shuffle_seed: None,
        }
    }

    /// Add a source
    pub fn add_source(&mut self, source: CorpusSource) -> &mut Self {
        self.sources.push(source);
        self
    }

    /// Set deduplication flag
    #[must_use]
    pub fn deduplicate(mut self, enable: bool) -> Self {
        self.deduplicate = enable;
        self
    }

    /// Set shuffle seed
    #[must_use]
    pub fn shuffle_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    /// Merge all sources into unified dataset
    pub fn merge(&self) -> Result<(CorpusBuffer, CorpusProvenance)> {
        let mut provenance = CorpusProvenance::new();
        let mut all_samples: Vec<(Sample, u8)> = Vec::new(); // (sample, priority)

        // Collect all samples with weights applied
        for source in &self.sources {
            let original_count = source.samples.len();
            let effective_count = (original_count as f64 * source.weight).round() as usize;

            // Sample with replacement if weight > 1, otherwise take all
            if source.weight >= 1.0 {
                // Take all and potentially repeat
                let repeats = source.weight.floor() as usize;
                let remainder = source.weight.fract();

                for sample in &source.samples {
                    for _ in 0..repeats {
                        let mut s = sample.clone();
                        s.weight *= source.weight;
                        all_samples.push((s, source.priority));
                    }
                }

                // Partial sampling for remainder
                let extra = (source.samples.len() as f64 * remainder).round() as usize;
                for sample in source.samples.iter().take(extra) {
                    let mut s = sample.clone();
                    s.weight *= source.weight;
                    all_samples.push((s, source.priority));
                }
            } else {
                // Subsample
                let take = (source.samples.len() as f64 * source.weight).round() as usize;
                for sample in source.samples.iter().take(take) {
                    all_samples.push((sample.clone(), source.priority));
                }
            }

            provenance.add_source(&source.name, original_count, effective_count);
        }

        // Sort by priority (higher first) for deduplication
        all_samples.sort_by(|a, b| b.1.cmp(&a.1));

        // Create buffer and deduplicate
        let config = CorpusBufferConfig {
            max_size: all_samples.len(),
            deduplicate: self.deduplicate,
            policy: EvictionPolicy::FIFO,
            seed: self.shuffle_seed,
        };

        let mut buffer = CorpusBuffer::with_config(config);
        let mut duplicates = 0;

        for (sample, _) in all_samples {
            if !buffer.add(sample) {
                duplicates += 1;
            }
        }

        provenance.duplicates_removed = duplicates;
        provenance.set_final_size(buffer.len());

        // Shuffle if seed provided
        if let Some(seed) = self.shuffle_seed {
            buffer.rng_state = seed;
            // Simple Fisher-Yates shuffle
            let n = buffer.samples.len();
            for i in (1..n).rev() {
                let j = (buffer.next_random() as usize) % (i + 1);
                buffer.samples.swap(i, j);
            }
        }

        Ok((buffer, provenance))
    }
}

impl Default for CorpusMerger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_buffer_basic() {
        let mut buffer = CorpusBuffer::new(100);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_corpus_buffer_deduplication() {
        let mut buffer = CorpusBuffer::new(100);

        // Add same sample twice
        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        let added = buffer.add_raw(vec![1.0, 2.0], vec![3.0]);

        assert!(!added, "Duplicate should not be added");
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_corpus_buffer_no_deduplication() {
        let config = CorpusBufferConfig {
            max_size: 100,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        // Add same sample twice
        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        let added = buffer.add_raw(vec![1.0, 2.0], vec![3.0]);

        assert!(added, "Duplicate should be added when dedup disabled");
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_corpus_buffer_fifo_eviction() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::FIFO,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        buffer.add_raw(vec![3.0], vec![3.0]);
        buffer.add_raw(vec![4.0], vec![4.0]);

        assert_eq!(buffer.len(), 3);
        // First sample should be evicted
        assert_eq!(buffer.samples()[0].features[0], 2.0);
    }

    #[test]
    fn test_corpus_buffer_importance_weighted_eviction() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::ImportanceWeighted,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add(Sample::with_weight(vec![1.0], vec![1.0], 0.5));
        buffer.add(Sample::with_weight(vec![2.0], vec![2.0], 0.8));
        buffer.add(Sample::with_weight(vec![3.0], vec![3.0], 0.3));
        buffer.add(Sample::with_weight(vec![4.0], vec![4.0], 1.0));

        assert_eq!(buffer.len(), 3);

        // Check that lowest weight (0.3) was evicted
        let weights: Vec<f64> = buffer.samples().iter().map(|s| s.weight).collect();
        assert!(!weights.contains(&0.3));
    }

    #[test]
    fn test_corpus_buffer_reservoir_sampling() {
        let config = CorpusBufferConfig {
            max_size: 10,
            policy: EvictionPolicy::Reservoir,
            deduplicate: false,
            seed: Some(42),
        };
        let mut buffer = CorpusBuffer::with_config(config);

        // Add many samples
        for i in 0..100 {
            buffer.add_raw(vec![i as f64], vec![(i * 2) as f64]);
        }

        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn test_corpus_buffer_to_dataset() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        buffer.add_raw(vec![4.0, 5.0], vec![6.0]);

        let (features, targets, n_samples, n_features) = buffer.to_dataset();

        assert_eq!(n_samples, 2);
        assert_eq!(n_features, 2);
        assert_eq!(features.len(), 4);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_corpus_buffer_empty_dataset() {
        let buffer = CorpusBuffer::new(100);
        let (features, targets, n_samples, n_features) = buffer.to_dataset();

        assert!(features.is_empty());
        assert!(targets.is_empty());
        assert_eq!(n_samples, 0);
        assert_eq!(n_features, 0);
    }

    #[test]
    fn test_corpus_buffer_clear() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_corpus_buffer_weights() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add(Sample::with_weight(vec![1.0], vec![1.0], 0.5));
        buffer.add(Sample::with_weight(vec![2.0], vec![2.0], 1.5));

        let weights = buffer.weights();
        assert_eq!(weights, vec![0.5, 1.5]);
    }

    #[test]
    fn test_corpus_buffer_update_weight() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0], vec![1.0]);

        buffer.update_weight(0, 2.0).unwrap();
        assert_eq!(buffer.samples()[0].weight, 2.0);

        // Invalid index
        assert!(buffer.update_weight(10, 1.0).is_err());
    }

    #[test]
    fn test_corpus_buffer_samples_by_source() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add(Sample::with_source(
            vec![1.0],
            vec![1.0],
            SampleSource::Synthetic,
        ));
        buffer.add(Sample::with_source(
            vec![2.0],
            vec![2.0],
            SampleSource::Production,
        ));
        buffer.add(Sample::with_source(
            vec![3.0],
            vec![3.0],
            SampleSource::Synthetic,
        ));

        let synthetic = buffer.samples_by_source(&SampleSource::Synthetic);
        assert_eq!(synthetic.len(), 2);

        let production = buffer.samples_by_source(&SampleSource::Production);
        assert_eq!(production.len(), 1);
    }

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(sample.features, vec![1.0, 2.0]);
        assert_eq!(sample.target, vec![3.0]);
        assert_eq!(sample.weight, 1.0);
        assert!(sample.timestamp.is_none());
    }

    #[test]
    fn test_corpus_source() {
        let samples = vec![Sample::new(vec![1.0], vec![1.0])];
        let source = CorpusSource::new("test", samples)
            .with_weight(2.0)
            .with_priority(5);

        assert_eq!(source.name, "test");
        assert_eq!(source.weight, 2.0);
        assert_eq!(source.priority, 5);
    }

    #[test]
    fn test_corpus_merger_basic() {
        let samples1 = vec![
            Sample::new(vec![1.0], vec![1.0]),
            Sample::new(vec![2.0], vec![2.0]),
        ];
        let samples2 = vec![
            Sample::new(vec![3.0], vec![3.0]),
            Sample::new(vec![4.0], vec![4.0]),
        ];

        let mut merger = CorpusMerger::new();
        merger.add_source(CorpusSource::new("source1", samples1));
        merger.add_source(CorpusSource::new("source2", samples2));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 4);
        assert_eq!(provenance.sources.len(), 2);
    }

    #[test]
    fn test_corpus_merger_with_weights() {
        let samples = vec![
            Sample::new(vec![1.0], vec![1.0]),
            Sample::new(vec![2.0], vec![2.0]),
        ];

        let mut merger = CorpusMerger::new().deduplicate(false); // Disable dedup for this test
        merger.add_source(CorpusSource::new("weighted", samples).with_weight(2.0));

        let (buffer, _) = merger.merge().unwrap();

        // Weight 2.0 should double the samples (4 with repeats)
        assert!(
            buffer.len() >= 4,
            "Expected at least 4 samples, got {}",
            buffer.len()
        );
    }

    #[test]
    fn test_corpus_merger_deduplication() {
        let samples1 = vec![Sample::new(vec![1.0], vec![1.0])];
        let samples2 = vec![Sample::new(vec![1.0], vec![1.0])]; // Duplicate

        let mut merger = CorpusMerger::new();
        merger.add_source(CorpusSource::new("source1", samples1).with_priority(1));
        merger.add_source(CorpusSource::new("source2", samples2).with_priority(0));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 1);
        assert_eq!(provenance.duplicates_removed, 1);
    }

    #[test]
    fn test_corpus_merger_no_deduplication() {
        let samples1 = vec![Sample::new(vec![1.0], vec![1.0])];
        let samples2 = vec![Sample::new(vec![1.0], vec![1.0])];

        let mut merger = CorpusMerger::new().deduplicate(false);
        merger.add_source(CorpusSource::new("source1", samples1));
        merger.add_source(CorpusSource::new("source2", samples2));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 2);
        assert_eq!(provenance.duplicates_removed, 0);
    }

    #[test]
    fn test_corpus_merger_shuffle() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample::new(vec![i as f64], vec![i as f64]))
            .collect();

        let mut merger = CorpusMerger::new().shuffle_seed(42);
        merger.add_source(CorpusSource::new("ordered", samples));

        let (buffer, _) = merger.merge().unwrap();

        // Check that order is different (with high probability)
        let features: Vec<f64> = buffer.samples().iter().map(|s| s.features[0]).collect();
        let ordered: Vec<f64> = (0..10).map(|i| i as f64).collect();

        assert_ne!(features, ordered);
    }

    #[test]
    fn test_corpus_provenance() {
        let mut provenance = CorpusProvenance::new();

        provenance.add_source("test1", 100, 200);
        provenance.add_source("test2", 50, 50);
        provenance.set_final_size(250);

        assert_eq!(provenance.sources.len(), 2);
        assert_eq!(provenance.final_size, 250);
    }

    #[test]
    fn test_eviction_policy_default() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Reservoir);
    }

    #[test]
    fn test_sample_source_default() {
        assert_eq!(SampleSource::default(), SampleSource::Production);
    }

    #[test]
    fn test_corpus_buffer_config_default() {
        let config = CorpusBufferConfig::default();
        assert_eq!(config.max_size, 10_000);
        assert!(config.deduplicate);
        assert_eq!(config.policy, EvictionPolicy::Reservoir);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_eviction_policy_eq() {
        assert_eq!(EvictionPolicy::FIFO, EvictionPolicy::FIFO);
        assert_ne!(EvictionPolicy::FIFO, EvictionPolicy::Reservoir);
    }

    #[test]
    fn test_eviction_policy_debug() {
        let policy = EvictionPolicy::DiversitySampling;
        let debug = format!("{:?}", policy);
        assert!(debug.contains("DiversitySampling"));
    }

    #[test]
    fn test_eviction_policy_clone() {
        let policy = EvictionPolicy::ImportanceWeighted;
        let cloned = policy;
        assert_eq!(policy, cloned);
    }

    #[test]
    fn test_sample_source_external() {
        let source = SampleSource::External("dataset.csv".to_string());
        let debug = format!("{:?}", source);
        assert!(debug.contains("External"));
        assert!(debug.contains("dataset.csv"));
    }

    #[test]
    fn test_sample_source_eq() {
        assert_eq!(SampleSource::Synthetic, SampleSource::Synthetic);
        assert_ne!(SampleSource::Synthetic, SampleSource::HandCrafted);
        assert_eq!(
            SampleSource::External("a".to_string()),
            SampleSource::External("a".to_string())
        );
        assert_ne!(
            SampleSource::External("a".to_string()),
            SampleSource::External("b".to_string())
        );
    }

    #[test]
    fn test_sample_source_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SampleSource::Synthetic);
        set.insert(SampleSource::Production);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_sample_debug() {
        let sample = Sample::new(vec![1.0], vec![2.0]);
        let debug = format!("{:?}", sample);
        assert!(debug.contains("Sample"));
    }

    #[test]
    fn test_sample_clone() {
        let original = Sample::with_weight(vec![1.0, 2.0], vec![3.0], 0.5);
        let cloned = original.clone();
        assert_eq!(original.features, cloned.features);
        assert_eq!(original.target, cloned.target);
        assert_eq!(original.weight, cloned.weight);
    }

    #[test]
    fn test_corpus_buffer_config_debug() {
        let config = CorpusBufferConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CorpusBufferConfig"));
    }

    #[test]
    fn test_corpus_buffer_config_clone() {
        let original = CorpusBufferConfig::default();
        let cloned = original.clone();
        assert_eq!(original.max_size, cloned.max_size);
    }

    #[test]
    fn test_corpus_buffer_debug() {
        let buffer = CorpusBuffer::new(10);
        let debug = format!("{:?}", buffer);
        assert!(debug.contains("CorpusBuffer"));
    }

    #[test]
    fn test_corpus_source_debug() {
        let source = CorpusSource::new("test", vec![]);
        let debug = format!("{:?}", source);
        assert!(debug.contains("CorpusSource"));
    }

    #[test]
    fn test_corpus_source_clone() {
        let original = CorpusSource::new("test", vec![])
            .with_weight(2.0)
            .with_priority(3);
        let cloned = original.clone();
        assert_eq!(original.name, cloned.name);
        assert_eq!(original.weight, cloned.weight);
        assert_eq!(original.priority, cloned.priority);
    }

    #[test]
    fn test_corpus_provenance_debug() {
        let prov = CorpusProvenance::new();
        let debug = format!("{:?}", prov);
        assert!(debug.contains("CorpusProvenance"));
    }

    #[test]
    fn test_corpus_provenance_clone() {
        let mut original = CorpusProvenance::new();
        original.add_source("test", 10, 20);
        original.set_final_size(20);
        let cloned = original.clone();
        assert_eq!(original.final_size, cloned.final_size);
    }

    #[test]
    fn test_corpus_merger_debug() {
        let merger = CorpusMerger::new();
        let debug = format!("{:?}", merger);
        assert!(debug.contains("CorpusMerger"));
    }

    #[test]
    fn test_corpus_merger_default() {
        let merger = CorpusMerger::default();
        assert!(merger.deduplicate);
    }

    #[test]
    fn test_corpus_provenance_default() {
        let prov = CorpusProvenance::default();
        assert_eq!(prov.final_size, 0);
        assert!(prov.sources.is_empty());
    }

    #[test]
    fn test_corpus_buffer_diversity_sampling() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::DiversitySampling,
            deduplicate: false,
            seed: Some(42),
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![0.0], vec![0.0]);
        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        // Add a sample that's diverse from existing
        buffer.add_raw(vec![100.0], vec![100.0]);

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_corpus_buffer_is_full() {
        let mut buffer = CorpusBuffer::new(2);
        assert!(!buffer.is_full());

        buffer.add_raw(vec![1.0], vec![1.0]);
        assert!(!buffer.is_full());

        buffer.add_raw(vec![2.0], vec![2.0]);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_sample_with_timestamp() {
        let mut sample = Sample::new(vec![1.0], vec![2.0]);
        sample.timestamp = Some(12345);
        assert_eq!(sample.timestamp, Some(12345));
    }

    #[test]
    fn test_corpus_buffer_fifo_with_dedup() {
        let config = CorpusBufferConfig {
            max_size: 2,
            policy: EvictionPolicy::FIFO,
            deduplicate: true,
            seed: None,
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        buffer.add_raw(vec![3.0], vec![3.0]); // Should evict first

        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_corpus_merger_subsample() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample::new(vec![i as f64], vec![i as f64]))
            .collect();

        let mut merger = CorpusMerger::new().deduplicate(false);
        merger.add_source(CorpusSource::new("subsampled", samples).with_weight(0.5));

        let (buffer, _) = merger.merge().unwrap();

        // Weight 0.5 should halve the samples
        assert!(buffer.len() <= 5);
    }

    #[test]
    fn test_sample_sources_all_variants() {
        let sources = vec![
            SampleSource::Synthetic,
            SampleSource::HandCrafted,
            SampleSource::Examples,
            SampleSource::Production,
            SampleSource::External("test".to_string()),
        ];

        for source in sources {
            let debug = format!("{:?}", source);
            assert!(!debug.is_empty());
        }
    }
}
