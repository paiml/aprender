//! Active Learning strategies for label-efficient training.
//!
//! # Strategies
//! - Uncertainty Sampling: Query most uncertain samples
//! - Query-by-Committee: Query samples with highest disagreement
//! - Entropy Sampling: Query highest entropy predictions

use crate::primitives::Vector;

/// Active learning query strategy trait.
pub trait QueryStrategy {
    /// Score samples for querying (higher = more informative).
    fn score(&self, predictions: &[Vector<f32>]) -> Vec<f32>;

    /// Select top-k samples to query.
    fn select(&self, predictions: &[Vector<f32>], k: usize) -> Vec<usize> {
        let scores = self.score(predictions);
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    }
}

/// Uncertainty sampling: query samples with lowest confidence.
#[derive(Debug, Clone, Default)]
pub struct UncertaintySampling;

impl UncertaintySampling {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl QueryStrategy for UncertaintySampling {
    // Contract: active-learning-v1, equation = "uncertainty_score"
    fn score(&self, predictions: &[Vector<f32>]) -> Vec<f32> {
        predictions
            .iter()
            .map(|p| {
                let max_prob = p.as_slice().iter().fold(0.0_f32, |a, &b| a.max(b));
                1.0 - max_prob // Lower confidence = higher score
            })
            .collect()
    }
}

/// Margin sampling: query samples with smallest margin between top 2.
#[derive(Debug, Clone, Default)]
pub struct MarginSampling;

impl MarginSampling {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl QueryStrategy for MarginSampling {
    // Contract: active-learning-v1, equation = "margin_score"
    fn score(&self, predictions: &[Vector<f32>]) -> Vec<f32> {
        predictions
            .iter()
            .map(|p| {
                let mut sorted: Vec<f32> = p.as_slice().to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                if sorted.len() >= 2 {
                    1.0 - (sorted[0] - sorted[1]) // Smaller margin = higher score
                } else {
                    1.0
                }
            })
            .collect()
    }
}

/// Entropy sampling: query samples with highest entropy.
#[derive(Debug, Clone, Default)]
pub struct EntropySampling;

impl EntropySampling {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl QueryStrategy for EntropySampling {
    // Contract: active-learning-v1, equation = "entropy_score"
    fn score(&self, predictions: &[Vector<f32>]) -> Vec<f32> {
        predictions
            .iter()
            .map(|p| {
                let mut entropy = 0.0;
                for &prob in p.as_slice() {
                    if prob > 1e-10 {
                        entropy -= prob * prob.ln();
                    }
                }
                entropy
            })
            .collect()
    }
}

/// Query-by-Committee: query samples with highest disagreement.
#[derive(Debug, Clone)]
pub struct QueryByCommittee {
    n_members: usize,
}

impl QueryByCommittee {
    #[must_use]
    pub fn new(n_members: usize) -> Self {
        Self { n_members }
    }

    /// Score based on vote entropy across committee members.
    // Contract: active-learning-v1, equation = "qbc_score"
    #[must_use]
    pub fn score_committee(&self, committee_preds: &[Vec<Vector<f32>>]) -> Vec<f32> {
        if committee_preds.is_empty() || committee_preds[0].is_empty() {
            return vec![];
        }

        let n_samples = committee_preds[0].len();
        let n_classes = committee_preds[0][0].len();

        (0..n_samples)
            .map(|i| {
                let mut votes = vec![0.0; n_classes];
                for member in committee_preds {
                    let pred = &member[i];
                    let class = pred
                        .as_slice()
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or(0, |(idx, _)| idx);
                    votes[class] += 1.0;
                }
                // Vote entropy
                let n = committee_preds.len() as f32;
                let mut entropy = 0.0;
                for &v in &votes {
                    if v > 0.0 {
                        let p = v / n;
                        entropy -= p * p.ln();
                    }
                }
                entropy
            })
            .collect()
    }

    #[must_use]
    pub fn n_members(&self) -> usize {
        self.n_members
    }
}

/// Random sampling baseline.
#[derive(Debug, Clone, Default)]
pub struct RandomSampling;

impl RandomSampling {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn select(&self, n_samples: usize, k: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);
        indices
    }
}

/// Core-Set selection for diversity-based active learning.
///
/// Selects samples that maximize coverage of the feature space,
/// ensuring diverse representation in the labeled set.
///
/// # Algorithm
///
/// Uses greedy furthest-first traversal to select points that
/// are maximally distant from already selected points.
///
/// # Reference
///
/// - Sener, O., & Savarese, S. (2018). Active Learning for Convolutional
///   Neural Networks: A Core-Set Approach. ICLR.
#[derive(Debug, Clone)]
pub struct CoreSetSelection {
    /// Already labeled indices (to avoid re-selecting)
    labeled_indices: Vec<usize>,
}

impl CoreSetSelection {
    /// Create a new core-set selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            labeled_indices: Vec::new(),
        }
    }

    /// Create with initial labeled indices.
    #[must_use]
    pub fn with_labeled(labeled_indices: Vec<usize>) -> Self {
        Self { labeled_indices }
    }

    /// Select k samples using furthest-first traversal.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Feature embeddings for all samples
    /// * `k` - Number of samples to select
    ///
    /// # Returns
    ///
    /// Indices of selected samples.
    pub fn select(&self, embeddings: &[Vec<f32>], k: usize) -> Vec<usize> {
        let n = embeddings.len();
        if n == 0 || k == 0 {
            return vec![];
        }

        let mut selected = self.labeled_indices.clone();
        let mut available: Vec<usize> = (0..n).filter(|i| !selected.contains(i)).collect();

        if available.is_empty() {
            return vec![];
        }

        // If no labeled points, start with first available
        if selected.is_empty() {
            let first = available.remove(0);
            selected.push(first);
        }

        let k_to_select = k.min(available.len());

        // Furthest-first traversal
        for _ in 0..k_to_select {
            if available.is_empty() {
                break;
            }

            // Find point with maximum minimum distance to selected set
            let mut best_idx = 0;
            let mut best_dist = f32::NEG_INFINITY;

            for (i, &candidate) in available.iter().enumerate() {
                let min_dist = selected
                    .iter()
                    .map(|&s| Self::euclidean_distance(&embeddings[candidate], &embeddings[s]))
                    .fold(f32::INFINITY, f32::min);

                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = i;
                }
            }

            let chosen = available.remove(best_idx);
            selected.push(chosen);
        }

        // Return only newly selected (not the initial labeled ones)
        selected
            .into_iter()
            .filter(|i| !self.labeled_indices.contains(i))
            .collect()
    }

    /// Compute diversity score for a set of indices.
    ///
    /// Higher score indicates more diverse selection.
    #[must_use]
    pub fn diversity_score(&self, embeddings: &[Vec<f32>], indices: &[usize]) -> f32 {
        if indices.len() < 2 {
            return 0.0;
        }

        // Average pairwise distance
        let mut total_dist = 0.0;
        let mut count = 0;

        for (i, &idx1) in indices.iter().enumerate() {
            for &idx2 in indices.iter().skip(i + 1) {
                total_dist += Self::euclidean_distance(&embeddings[idx1], &embeddings[idx2]);
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f32
        } else {
            0.0
        }
    }

    /// ONE PATH: Delegates to `nn::functional::euclidean_distance` (UCBD §4).
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        crate::nn::functional::euclidean_distance(a, b)
    }
}

impl Default for CoreSetSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// Expected Model Change strategy.
///
/// Selects samples that would cause the largest gradient update if labeled.
///
/// # Reference
///
/// - Settles, B., et al. (2008). An Analysis of Active Learning Strategies
///   for Sequence Labeling Tasks. EMNLP.
#[derive(Debug, Clone)]
pub struct ExpectedModelChange {
    /// Gradient norm threshold for filtering
    min_grad_norm: f32,
}

impl ExpectedModelChange {
    /// Create a new expected model change selector.
    #[must_use]
    pub fn new() -> Self {
        Self { min_grad_norm: 0.0 }
    }

    /// Create with minimum gradient norm threshold.
    #[must_use]
    pub fn with_min_grad(min_grad_norm: f32) -> Self {
        Self { min_grad_norm }
    }

    /// Score samples by expected gradient magnitude.
    ///
    /// Uses prediction uncertainty as proxy for gradient magnitude.
    #[must_use]
    pub fn score(&self, predictions: &[Vector<f32>], gradient_norms: Option<&[f32]>) -> Vec<f32> {
        if let Some(norms) = gradient_norms {
            // Use actual gradient norms if provided
            norms
                .iter()
                .map(|&n| if n >= self.min_grad_norm { n } else { 0.0 })
                .collect()
        } else {
            // Approximate using prediction entropy
            predictions
                .iter()
                .map(|p| {
                    let mut entropy = 0.0;
                    for &prob in p.as_slice() {
                        if prob > 1e-10 {
                            entropy -= prob * prob.ln();
                        }
                    }
                    entropy
                })
                .collect()
        }
    }

    /// Select top-k samples by expected model change.
    #[must_use]
    pub fn select(&self, predictions: &[Vector<f32>], k: usize) -> Vec<usize> {
        let scores = self.score(predictions, None);
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    }
}

impl Default for ExpectedModelChange {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "active_learning_tests.rs"]
mod tests;
