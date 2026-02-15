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

include!("corpus_part_02.rs");
include!("corpus_part_03.rs");
