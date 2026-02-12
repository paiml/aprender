//! Weak Supervision for Synthetic Data Generation.
//!
//! Implements programmatic labeling functions following Snorkel (Ratner et al., 2017)
//! for creating training data from heuristic rules.
//!
//! # References
//!
//! Ratner, A., Bach, S. H., et al. (2017). Snorkel: Rapid Training Data Creation
//! with Weak Supervision. VLDB Endowment, 11(3), 269-282.

use std::collections::HashMap;

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;

// ============================================================================
// Label Types
// ============================================================================

/// Result of a labeling function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelVote {
    /// Positive class.
    Positive,
    /// Negative class.
    Negative,
    /// Specific class label.
    Class(i32),
    /// Abstain - labeling function doesn't apply.
    Abstain,
}

impl LabelVote {
    /// Check if this is an abstain vote.
    #[must_use]
    pub fn is_abstain(&self) -> bool {
        matches!(self, Self::Abstain)
    }

    /// Convert to numeric label (Abstain = None).
    #[must_use]
    pub fn to_label(&self) -> Option<i32> {
        match self {
            Self::Positive => Some(1),
            Self::Negative => Some(0),
            Self::Class(c) => Some(*c),
            Self::Abstain => None,
        }
    }
}

// ============================================================================
// Labeling Function Trait
// ============================================================================

/// A labeling function that assigns labels to samples.
///
/// Labeling functions encapsulate heuristic rules for generating noisy labels.
/// They can abstain if the rule doesn't apply to a sample.
///
/// # Example
///
/// ```
/// use aprender::synthetic::weak_supervision::{LabelingFunction, LabelVote};
///
/// struct ContainsGit;
///
/// impl LabelingFunction<String> for ContainsGit {
///     fn name(&self) -> &str { "contains_git" }
///
///     fn apply(&self, sample: &String) -> LabelVote {
///         if sample.contains("git") {
///             LabelVote::Positive
///         } else {
///             LabelVote::Abstain
///         }
///     }
/// }
/// ```
pub trait LabelingFunction<T>: Send + Sync {
    /// Name of this labeling function (for debugging/metrics).
    fn name(&self) -> &str;

    /// Apply the labeling function to a sample.
    fn apply(&self, sample: &T) -> LabelVote;

    /// Optional weight for this LF in aggregation (default 1.0).
    fn weight(&self) -> f32 {
        1.0
    }
}

// ============================================================================
// Aggregation Strategies
// ============================================================================

/// Strategy for aggregating multiple labeling function votes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationStrategy {
    /// Simple majority vote (ties = abstain).
    #[default]
    MajorityVote,
    /// Weighted majority vote using LF weights.
    WeightedVote,
    /// Require all non-abstaining LFs to agree.
    Unanimous,
    /// Use any non-abstaining vote (first wins).
    Any,
}

// ============================================================================
// Labeled Sample
// ============================================================================

/// A sample with a weak label and confidence.
#[derive(Debug, Clone)]
pub struct LabeledSample<T> {
    /// The original sample.
    pub sample: T,
    /// Aggregated label.
    pub label: i32,
    /// Confidence in the label [0.0, 1.0].
    pub confidence: f32,
    /// Number of LFs that voted (non-abstaining).
    pub num_votes: usize,
    /// Individual LF votes for analysis.
    pub votes: Vec<(String, LabelVote)>,
}

impl<T> LabeledSample<T> {
    /// Create a new labeled sample.
    pub fn new(sample: T, label: i32, confidence: f32) -> Self {
        Self {
            sample,
            label,
            confidence,
            num_votes: 0,
            votes: Vec::new(),
        }
    }

    /// Add voting details.
    pub fn with_votes(mut self, num_votes: usize, votes: Vec<(String, LabelVote)>) -> Self {
        self.num_votes = num_votes;
        self.votes = votes;
        self
    }
}

// ============================================================================
// Weak Supervision Configuration
// ============================================================================

/// Configuration for weak supervision.
#[derive(Debug, Clone)]
pub struct WeakSupervisionConfig {
    /// Aggregation strategy.
    pub aggregation: AggregationStrategy,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Minimum number of non-abstaining votes.
    pub min_votes: usize,
    /// Whether to include abstained samples with default label.
    pub include_abstained: bool,
    /// Default label for fully-abstained samples.
    pub default_label: i32,
}

impl Default for WeakSupervisionConfig {
    fn default() -> Self {
        Self {
            aggregation: AggregationStrategy::MajorityVote,
            min_confidence: 0.5,
            min_votes: 1,
            include_abstained: false,
            default_label: 0,
        }
    }
}

impl WeakSupervisionConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set aggregation strategy.
    #[must_use]
    pub fn with_aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation = strategy;
        self
    }

    /// Set minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set minimum votes required.
    #[must_use]
    pub fn with_min_votes(mut self, votes: usize) -> Self {
        self.min_votes = votes.max(1);
        self
    }

    /// Enable including abstained samples.
    #[must_use]
    pub fn with_include_abstained(mut self, include: bool, default_label: i32) -> Self {
        self.include_abstained = include;
        self.default_label = default_label;
        self
    }
}

// ============================================================================
// Weak Supervision Generator
// ============================================================================

/// Weak supervision generator using labeling functions.
///
/// Applies multiple labeling functions to unlabeled samples and aggregates
/// their votes to create weakly-labeled training data.
///
/// # Example
///
/// ```
/// use aprender::synthetic::weak_supervision::{
///     WeakSupervisionGenerator, LabelingFunction, LabelVote,
///     WeakSupervisionConfig, AggregationStrategy,
/// };
/// use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
///
/// // Define labeling functions
/// struct LengthLF;
/// impl LabelingFunction<String> for LengthLF {
///     fn name(&self) -> &str { "length" }
///     fn apply(&self, s: &String) -> LabelVote {
///         if s.len() > 10 { LabelVote::Positive }
///         else if s.len() < 5 { LabelVote::Negative }
///         else { LabelVote::Abstain }
///     }
/// }
///
/// let mut gen = WeakSupervisionGenerator::<String>::new();
/// gen.add_lf(Box::new(LengthLF));
///
/// let samples = vec!["short".to_string(), "this is a longer string".to_string()];
/// let config = SyntheticConfig::default();
/// let labeled = gen.generate(&samples, &config).expect("weak supervision generation should succeed");
/// ```
pub struct WeakSupervisionGenerator<T> {
    /// Labeling functions.
    labeling_functions: Vec<Box<dyn LabelingFunction<T>>>,
    /// Configuration.
    config: WeakSupervisionConfig,
}

impl<T> std::fmt::Debug for WeakSupervisionGenerator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakSupervisionGenerator")
            .field("num_lfs", &self.labeling_functions.len())
            .field("config", &self.config)
            .finish()
    }
}

impl<T> WeakSupervisionGenerator<T> {
    /// Create a new weak supervision generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            labeling_functions: Vec::new(),
            config: WeakSupervisionConfig::default(),
        }
    }

    /// Set configuration.
    #[must_use]
    pub fn with_config(mut self, config: WeakSupervisionConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a labeling function.
    pub fn add_lf(&mut self, lf: Box<dyn LabelingFunction<T>>) {
        self.labeling_functions.push(lf);
    }

    /// Get number of labeling functions.
    #[must_use]
    pub fn num_lfs(&self) -> usize {
        self.labeling_functions.len()
    }

    /// Apply all LFs to a sample and collect votes.
    fn collect_votes(&self, sample: &T) -> Vec<(String, LabelVote, f32)> {
        self.labeling_functions
            .iter()
            .map(|lf| (lf.name().to_string(), lf.apply(sample), lf.weight()))
            .collect()
    }

    /// Aggregate votes using the configured strategy.
    fn aggregate_votes(&self, votes: &[(String, LabelVote, f32)]) -> Option<(i32, f32)> {
        let non_abstain: Vec<_> = votes.iter().filter(|(_, v, _)| !v.is_abstain()).collect();

        if non_abstain.len() < self.config.min_votes {
            return if self.config.include_abstained {
                Some((self.config.default_label, 0.0))
            } else {
                None
            };
        }

        match self.config.aggregation {
            AggregationStrategy::MajorityVote => self.majority_vote(&non_abstain),
            AggregationStrategy::WeightedVote => self.weighted_vote(&non_abstain),
            AggregationStrategy::Unanimous => Self::unanimous_vote(&non_abstain),
            AggregationStrategy::Any => Self::any_vote(&non_abstain),
        }
    }

    fn majority_vote(&self, votes: &[&(String, LabelVote, f32)]) -> Option<(i32, f32)> {
        if votes.is_empty() {
            return None;
        }

        // Count votes per label
        let mut counts: HashMap<i32, usize> = HashMap::new();
        for (_, vote, _) in votes {
            if let Some(label) = vote.to_label() {
                *counts.entry(label).or_insert(0) += 1;
            }
        }

        // Find majority
        let total = votes.len();
        let (label, count) = counts.into_iter().max_by_key(|(_, c)| *c)?;

        let confidence = count as f32 / total as f32;

        // Check for tie (if another label has same count)
        // Simplified: just use the one we found
        if confidence >= self.config.min_confidence {
            Some((label, confidence))
        } else {
            None
        }
    }

    fn weighted_vote(&self, votes: &[&(String, LabelVote, f32)]) -> Option<(i32, f32)> {
        if votes.is_empty() {
            return None;
        }

        // Sum weights per label
        let mut weights: HashMap<i32, f32> = HashMap::new();
        let mut total_weight = 0.0;

        for (_, vote, weight) in votes {
            if let Some(label) = vote.to_label() {
                *weights.entry(label).or_insert(0.0) += weight;
                total_weight += weight;
            }
        }

        if total_weight < f32::EPSILON {
            return None;
        }

        // Find highest weight
        let (label, weight) = weights
            .into_iter()
            .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal))?;

        let confidence = weight / total_weight;

        if confidence >= self.config.min_confidence {
            Some((label, confidence))
        } else {
            None
        }
    }

    fn unanimous_vote(votes: &[&(String, LabelVote, f32)]) -> Option<(i32, f32)> {
        if votes.is_empty() {
            return None;
        }

        let first_label = votes[0].1.to_label()?;

        // Check all votes match
        let unanimous = votes
            .iter()
            .all(|(_, v, _)| v.to_label() == Some(first_label));

        if unanimous {
            Some((first_label, 1.0))
        } else {
            None
        }
    }

    fn any_vote(votes: &[&(String, LabelVote, f32)]) -> Option<(i32, f32)> {
        // Take first non-abstaining vote
        for (_, vote, _) in votes {
            if let Some(label) = vote.to_label() {
                return Some((label, 1.0 / votes.len() as f32));
            }
        }
        None
    }
}

impl<T> Default for WeakSupervisionGenerator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + std::fmt::Debug> SyntheticGenerator for WeakSupervisionGenerator<T> {
    type Input = T;
    type Output = LabeledSample<T>;

    fn generate(&self, seeds: &[T], config: &SyntheticConfig) -> Result<Vec<LabeledSample<T>>> {
        if self.labeling_functions.is_empty() {
            return Ok(Vec::new());
        }

        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target.min(seeds.len()));

        for sample in seeds {
            if results.len() >= target {
                break;
            }

            // Collect votes from all LFs
            let votes = self.collect_votes(sample);

            // Aggregate votes
            if let Some((label, confidence)) = self.aggregate_votes(&votes) {
                // Check quality threshold
                if confidence >= config.quality_threshold {
                    let num_votes = votes.iter().filter(|(_, v, _)| !v.is_abstain()).count();
                    let vote_details: Vec<_> = votes
                        .iter()
                        .map(|(name, vote, _)| (name.clone(), *vote))
                        .collect();

                    results.push(
                        LabeledSample::new(sample.clone(), label, confidence)
                            .with_votes(num_votes, vote_details),
                    );
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &LabeledSample<T>, _seed: &T) -> f32 {
        // Quality is the confidence in the label
        generated.confidence
    }

    fn diversity_score(&self, batch: &[LabeledSample<T>]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Diversity is based on label distribution entropy
        let mut label_counts: HashMap<i32, usize> = HashMap::new();
        for sample in batch {
            *label_counts.entry(sample.label).or_insert(0) += 1;
        }

        let n = batch.len() as f32;
        let mut entropy = 0.0;

        for count in label_counts.values() {
            let p = *count as f32 / n;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize by max entropy (uniform distribution over unique labels)
        let max_entropy = (label_counts.len() as f32).ln();
        if max_entropy > f32::EPSILON {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Built-in Labeling Functions
// ============================================================================

/// Keyword-based labeling function for text.
#[derive(Debug)]
pub struct KeywordLF {
    name: String,
    keywords: Vec<String>,
    label: LabelVote,
    weight: f32,
}

impl KeywordLF {
    /// Create a new keyword labeling function.
    #[must_use]
    pub fn new(name: impl Into<String>, keywords: &[&str], label: LabelVote) -> Self {
        Self {
            name: name.into(),
            keywords: keywords.iter().map(|s| (*s).to_string()).collect(),
            label,
            weight: 1.0,
        }
    }

    /// Set the weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }
}

impl LabelingFunction<String> for KeywordLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        let lower = sample.to_lowercase();
        if self.keywords.iter().any(|kw| lower.contains(kw)) {
            self.label
        } else {
            LabelVote::Abstain
        }
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

/// Length-based labeling function for text.
#[derive(Debug)]
pub struct LengthLF {
    name: String,
    min_len: usize,
    max_len: usize,
    label: LabelVote,
}

impl LengthLF {
    /// Create a new length labeling function.
    #[must_use]
    pub fn new(name: impl Into<String>, min_len: usize, max_len: usize, label: LabelVote) -> Self {
        Self {
            name: name.into(),
            min_len,
            max_len,
            label,
        }
    }
}

impl LabelingFunction<String> for LengthLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        let len = sample.len();
        if len >= self.min_len && len <= self.max_len {
            self.label
        } else {
            LabelVote::Abstain
        }
    }
}

/// Regex-based labeling function for text.
#[derive(Debug)]
pub struct PatternLF {
    name: String,
    pattern: String,
    label: LabelVote,
}

impl PatternLF {
    /// Create a new pattern labeling function (simple substring match).
    #[must_use]
    pub fn new(name: impl Into<String>, pattern: impl Into<String>, label: LabelVote) -> Self {
        Self {
            name: name.into(),
            pattern: pattern.into(),
            label,
        }
    }
}

impl LabelingFunction<String> for PatternLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        if sample.contains(&self.pattern) {
            self.label
        } else {
            LabelVote::Abstain
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LabelVote Tests
    // ========================================================================

    #[test]
    fn test_label_vote_is_abstain() {
        assert!(!LabelVote::Positive.is_abstain());
        assert!(!LabelVote::Negative.is_abstain());
        assert!(!LabelVote::Class(5).is_abstain());
        assert!(LabelVote::Abstain.is_abstain());
    }

    #[test]
    fn test_label_vote_to_label() {
        assert_eq!(LabelVote::Positive.to_label(), Some(1));
        assert_eq!(LabelVote::Negative.to_label(), Some(0));
        assert_eq!(LabelVote::Class(42).to_label(), Some(42));
        assert_eq!(LabelVote::Abstain.to_label(), None);
    }

    // ========================================================================
    // WeakSupervisionConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = WeakSupervisionConfig::default();
        assert_eq!(config.aggregation, AggregationStrategy::MajorityVote);
        assert!((config.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_votes, 1);
        assert!(!config.include_abstained);
    }

    #[test]
    fn test_config_with_aggregation() {
        let config = WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous);
        assert_eq!(config.aggregation, AggregationStrategy::Unanimous);
    }

    #[test]
    fn test_config_with_min_confidence() {
        let config = WeakSupervisionConfig::new().with_min_confidence(0.8);
        assert!((config.min_confidence - 0.8).abs() < f32::EPSILON);

        // Should clamp
        let config = WeakSupervisionConfig::new().with_min_confidence(1.5);
        assert!((config.min_confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_min_votes() {
        let config = WeakSupervisionConfig::new().with_min_votes(3);
        assert_eq!(config.min_votes, 3);

        // Should be at least 1
        let config = WeakSupervisionConfig::new().with_min_votes(0);
        assert_eq!(config.min_votes, 1);
    }

    #[test]
    fn test_config_with_include_abstained() {
        let config = WeakSupervisionConfig::new().with_include_abstained(true, 99);
        assert!(config.include_abstained);
        assert_eq!(config.default_label, 99);
    }

    // ========================================================================
    // LabeledSample Tests
    // ========================================================================

    #[test]
    fn test_labeled_sample_new() {
        let sample = LabeledSample::new("test".to_string(), 1, 0.9);
        assert_eq!(sample.sample, "test");
        assert_eq!(sample.label, 1);
        assert!((sample.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_labeled_sample_with_votes() {
        let sample = LabeledSample::new("test".to_string(), 1, 0.9).with_votes(
            2,
            vec![
                ("lf1".to_string(), LabelVote::Positive),
                ("lf2".to_string(), LabelVote::Positive),
            ],
        );
        assert_eq!(sample.num_votes, 2);
        assert_eq!(sample.votes.len(), 2);
    }

    // ========================================================================
    // Built-in LF Tests
    // ========================================================================

    #[test]
    fn test_keyword_lf() {
        let lf = KeywordLF::new("git_cmd", &["git", "commit"], LabelVote::Positive);

        assert_eq!(lf.name(), "git_cmd");
        assert_eq!(lf.apply(&"git status".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"commit message".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"cargo build".to_string()), LabelVote::Abstain);
    }

    #[test]
    fn test_keyword_lf_case_insensitive() {
        let lf = KeywordLF::new("test", &["git"], LabelVote::Positive);
        assert_eq!(lf.apply(&"GIT status".to_string()), LabelVote::Positive);
    }

    #[test]
    fn test_keyword_lf_with_weight() {
        let lf = KeywordLF::new("test", &["x"], LabelVote::Positive).with_weight(2.5);
        assert!((lf.weight() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_length_lf() {
        let lf = LengthLF::new("short", 0, 5, LabelVote::Negative);

        assert_eq!(lf.name(), "short");
        assert_eq!(lf.apply(&"hi".to_string()), LabelVote::Negative);
        assert_eq!(lf.apply(&"hello".to_string()), LabelVote::Negative);
        assert_eq!(lf.apply(&"hello world".to_string()), LabelVote::Abstain);
    }

    #[test]
    fn test_pattern_lf() {
        let lf = PatternLF::new("has_equals", "=", LabelVote::Positive);

        assert_eq!(lf.name(), "has_equals");
        assert_eq!(lf.apply(&"x=1".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"no equals".to_string()), LabelVote::Abstain);
    }

    // ========================================================================
    // WeakSupervisionGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = WeakSupervisionGenerator::<String>::new();
        assert_eq!(gen.num_lfs(), 0);
    }

    #[test]
    fn test_generator_add_lf() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "test",
            &["x"],
            LabelVote::Positive,
        )));
        assert_eq!(gen.num_lfs(), 1);
    }

    #[test]
    fn test_generator_no_lfs() {
        let gen = WeakSupervisionGenerator::<String>::new();
        let samples = vec!["test".to_string()];
        let result = gen
            .generate(&samples, &SyntheticConfig::default())
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_generator_majority_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["git"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["status"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf3",
            &["cargo"],
            LabelVote::Negative,
        )));

        let samples = vec!["git status".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 1); // Positive (2 votes vs 0)
    }

    #[test]
    fn test_generator_weighted_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_aggregation(AggregationStrategy::WeightedVote)
                .with_min_confidence(0.0),
        );

        gen.add_lf(Box::new(
            KeywordLF::new("lf1", &["test"], LabelVote::Positive).with_weight(1.0),
        ));
        gen.add_lf(Box::new(
            KeywordLF::new("lf2", &["test"], LabelVote::Negative).with_weight(3.0),
        ));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 0); // Negative wins (weight 3 vs 1)
    }

    #[test]
    fn test_generator_unanimous() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 1);
        assert!((result[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_unanimous_disagree() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["test"],
            LabelVote::Negative,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // No unanimous agreement
    }

    #[test]
    fn test_generator_any_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new()
            .with_config(WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Any));

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_generator_min_votes() {
        let mut gen = WeakSupervisionGenerator::<String>::new()
            .with_config(WeakSupervisionConfig::new().with_min_votes(2));

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // Only 1 vote, need 2
    }

    #[test]
    fn test_generator_all_abstain() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["nomatch"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // All abstained
    }

    #[test]
    fn test_generator_include_abstained() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_include_abstained(true, 99)
                .with_min_confidence(0.0),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["nomatch"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 99); // Default label
    }

    #[test]
    fn test_quality_score() {
        let gen = WeakSupervisionGenerator::<String>::new();

        let high_conf = LabeledSample::new("test".to_string(), 1, 0.9);
        let low_conf = LabeledSample::new("test".to_string(), 1, 0.3);

        assert!(gen.quality_score(&high_conf, &String::new()) > 0.5);
        assert!(gen.quality_score(&low_conf, &String::new()) < 0.5);
    }

    #[test]
    fn test_diversity_score() {
        let gen = WeakSupervisionGenerator::<String>::new();

        // Empty batch
        let score = gen.diversity_score(&[]);
        assert!((score - 0.0).abs() < f32::EPSILON);

        // Single label
        let single_label = vec![
            LabeledSample::new("a".to_string(), 1, 1.0),
            LabeledSample::new("b".to_string(), 1, 1.0),
        ];
        let score = gen.diversity_score(&single_label);
        assert!((score - 0.0).abs() < f32::EPSILON); // No diversity

        // Two labels, equal distribution
        let diverse = vec![
            LabeledSample::new("a".to_string(), 0, 1.0),
            LabeledSample::new("b".to_string(), 1, 1.0),
        ];
        let score = gen.diversity_score(&diverse);
        assert!((score - 1.0).abs() < f32::EPSILON); // Max diversity
    }

    #[test]
    fn test_generator_respects_target() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new("lf1", &["a"], LabelVote::Positive)));

        let samples = vec![
            "a1".to_string(),
            "a2".to_string(),
            "a3".to_string(),
            "a4".to_string(),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(0.5) // Target: 2
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.len() <= 2);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_weak_supervision_pipeline() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_aggregation(AggregationStrategy::MajorityVote)
                .with_min_confidence(0.5),
        );

        // Add multiple LFs
        gen.add_lf(Box::new(KeywordLF::new(
            "git_positive",
            &["git"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "cargo_positive",
            &["cargo"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(LengthLF::new(
            "short_negative",
            0,
            5,
            LabelVote::Negative,
        )));
        gen.add_lf(Box::new(PatternLF::new(
            "equals_positive",
            "=",
            LabelVote::Positive,
        )));

        let samples = vec![
            "git status".to_string(),        // git_positive votes
            "cargo build".to_string(),       // cargo_positive votes
            "hi".to_string(),                // short_negative votes
            "git log --oneline".to_string(), // git_positive votes
            "x=1".to_string(),               // equals_positive + short_negative
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");

        // Verify labels make sense
        for labeled in &result {
            assert!(labeled.num_votes > 0);
            assert!(!labeled.votes.is_empty());
        }

        // Check diversity
        let diversity = gen.diversity_score(&result);
        assert!((0.0..=1.0).contains(&diversity));
    }
}
