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

include!("labeling.rs");
include!("weak_supervision_part_03.rs");
