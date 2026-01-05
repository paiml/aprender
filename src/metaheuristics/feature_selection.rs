//! Feature Selection Utilities
//!
//! High-level utilities for using metaheuristics (especially Binary GA)
//! to select optimal feature subsets for machine learning models.
//!
//! # Example
//!
//! ```
//! use aprender::metaheuristics::{FeatureSelector, SelectionCriterion, Budget};
//!
//! // Feature matrix: 100 samples, 20 features
//! let n_features = 20;
//!
//! // Custom evaluator: returns (accuracy, n_selected_features)
//! let evaluator = |selected: &[bool]| -> (f64, usize) {
//!     let n_selected = selected.iter().filter(|&&s| s).count();
//!     if n_selected == 0 {
//!         return (0.0, 0);
//!     }
//!     // Simulate: more features = diminishing returns
//!     let accuracy = 0.5 + 0.4 * (1.0 - (-0.2 * n_selected as f64).exp());
//!     (accuracy, n_selected)
//! };
//!
//! let mut selector = FeatureSelector::new(n_features)
//!     .with_criterion(SelectionCriterion::MaxAccuracyMinFeatures { alpha: 0.01 })
//!     .with_seed(42);
//!
//! let result = selector.select(evaluator, Budget::Evaluations(1000));
//!
//! println!("Selected {} features with score {:.3}", result.n_selected, result.score);
//! ```

use super::{BinaryGA, Budget, PerturbativeMetaheuristic, SearchSpace};

/// Criterion for evaluating feature subsets.
#[derive(Debug, Clone, Copy)]
pub enum SelectionCriterion {
    /// Maximize accuracy only (no penalty for feature count)
    MaxAccuracy,

    /// Minimize features only (for compression/interpretability)
    MinFeatures,

    /// Balance accuracy and feature count: score = accuracy - alpha * `n_features`
    MaxAccuracyMinFeatures {
        /// Penalty per selected feature
        alpha: f64,
    },

    /// Maximize accuracy with hard constraint on max features
    MaxAccuracyWithLimit {
        /// Maximum allowed features
        max_features: usize,
    },

    /// AIC-like criterion: score = accuracy - 2 * `n_features` / `n_samples`
    AIC {
        /// Number of training samples
        n_samples: usize,
    },

    /// BIC-like criterion: score = accuracy - log(n) * `n_features` / `n_samples`
    BIC {
        /// Number of training samples
        n_samples: usize,
    },
}

/// Result of feature selection.
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Indices of selected features
    pub selected_indices: Vec<usize>,
    /// Number of selected features
    pub n_selected: usize,
    /// Final optimization score (depends on criterion)
    pub score: f64,
    /// Accuracy achieved (if provided by evaluator)
    pub accuracy: Option<f64>,
    /// Boolean mask: true for selected features
    pub mask: Vec<bool>,
    /// Number of objective evaluations used
    pub evaluations: usize,
}

/// High-level feature selector using Binary GA.
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    /// Number of features in the dataset
    n_features: usize,
    /// Selection criterion
    criterion: SelectionCriterion,
    /// Population size for Binary GA
    population_size: usize,
    /// Mutation probability
    mutation_prob: f64,
    /// Random seed
    seed: Option<u64>,
}

impl FeatureSelector {
    /// Create new feature selector for given number of features.
    #[must_use] 
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            criterion: SelectionCriterion::MaxAccuracyMinFeatures { alpha: 0.01 },
            population_size: 50,
            mutation_prob: 1.0 / n_features as f64, // Adaptive mutation
            seed: None,
        }
    }

    /// Set selection criterion.
    #[must_use] 
    pub fn with_criterion(mut self, criterion: SelectionCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set population size for Binary GA.
    #[must_use] 
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size.max(10);
        self
    }

    /// Set mutation probability.
    #[must_use] 
    pub fn with_mutation_prob(mut self, prob: f64) -> Self {
        self.mutation_prob = prob.clamp(0.001, 0.5);
        self
    }

    /// Set random seed for reproducibility.
    #[must_use] 
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run feature selection.
    ///
    /// The evaluator function takes a boolean mask and returns (accuracy, `n_selected`).
    pub fn select<F>(&mut self, evaluator: F, budget: Budget) -> FeatureSelectionResult
    where
        F: Fn(&[bool]) -> (f64, usize),
    {
        let space = SearchSpace::binary(self.n_features);

        // Build objective function based on criterion
        let objective = |bits: &[f64]| -> f64 {
            let mask: Vec<bool> = bits.iter().map(|&b| b > 0.5).collect();
            let (accuracy, n_selected) = evaluator(&mask);

            // Convert to minimization problem (lower is better)
            match self.criterion {
                SelectionCriterion::MaxAccuracy => -accuracy,

                SelectionCriterion::MinFeatures => n_selected as f64,

                SelectionCriterion::MaxAccuracyMinFeatures { alpha } => {
                    -accuracy + alpha * n_selected as f64
                }

                SelectionCriterion::MaxAccuracyWithLimit { max_features } => {
                    if n_selected > max_features {
                        // Heavy penalty for exceeding limit
                        1000.0 + n_selected as f64
                    } else {
                        -accuracy
                    }
                }

                SelectionCriterion::AIC { n_samples } => {
                    -accuracy + 2.0 * n_selected as f64 / n_samples as f64
                }

                SelectionCriterion::BIC { n_samples } => {
                    let log_n = (n_samples as f64).ln();
                    -accuracy + log_n * n_selected as f64 / n_samples as f64
                }
            }
        };

        // Configure Binary GA
        let mut ga = BinaryGA::default()
            .with_population_size(self.population_size)
            .with_mutation_prob(self.mutation_prob);

        if let Some(seed) = self.seed {
            ga = ga.with_seed(seed);
        }

        // Run optimization
        let result = ga.optimize(&objective, &space, budget);

        // Convert solution to mask
        let mask: Vec<bool> = result.solution.iter().map(|&b| b > 0.5).collect();
        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();
        let n_selected = selected_indices.len();

        // Re-evaluate to get accuracy
        let (accuracy, _) = evaluator(&mask);

        FeatureSelectionResult {
            selected_indices,
            n_selected,
            score: -result.objective_value, // Convert back from minimization
            accuracy: Some(accuracy),
            mask,
            evaluations: result.evaluations,
        }
    }
}

/// Convenience function for quick feature selection.
///
/// Uses default settings with `MaxAccuracyMinFeatures` criterion.
pub fn select_features<F>(n_features: usize, evaluator: F, budget: Budget) -> FeatureSelectionResult
where
    F: Fn(&[bool]) -> (f64, usize),
{
    let mut selector = FeatureSelector::new(n_features);
    selector.select(evaluator, budget)
}

/// Feature importance ranking using perturbation.
///
/// Evaluates the impact of removing each feature individually.
pub fn rank_features<F>(n_features: usize, evaluator: F) -> Vec<(usize, f64)>
where
    F: Fn(&[bool]) -> (f64, usize),
{
    // Baseline: all features selected
    let all_selected = vec![true; n_features];
    let (baseline_acc, _) = evaluator(&all_selected);

    // Measure drop when each feature is removed
    let mut importance: Vec<(usize, f64)> = (0..n_features)
        .map(|i| {
            let mut mask = all_selected.clone();
            mask[i] = false;
            let (acc, _) = evaluator(&mask);
            let drop = baseline_acc - acc;
            (i, drop)
        })
        .collect();

    // Sort by importance (descending)
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    importance
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================
    // EXTREME TDD: Tests written first
    // ==========================================================

    #[test]
    fn test_feature_selector_basic() {
        let n_features = 10;

        // Evaluator: accuracy increases with first 3 features, then plateaus
        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let selected: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|(_, &s)| s)
                .map(|(i, _)| i)
                .collect();
            let n = selected.len();

            if n == 0 {
                return (0.0, 0);
            }

            // First 3 features are important
            let important_count = selected.iter().filter(|&&i| i < 3).count();
            let accuracy = 0.5 + 0.15 * important_count as f64;
            (accuracy, n)
        };

        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::MaxAccuracyMinFeatures { alpha: 0.02 })
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(500));

        assert!(result.n_selected > 0, "Should select at least one feature");
        assert!(
            result.n_selected <= n_features,
            "Can't select more than exist"
        );
        assert!(
            result.accuracy.unwrap() > 0.5,
            "Should achieve decent accuracy"
        );
    }

    #[test]
    fn test_feature_selector_min_features() {
        let n_features = 5;

        // Constant accuracy regardless of features
        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            (0.8, n) // Always 80% accuracy
        };

        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::MinFeatures)
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(200));

        // Should select 0 or very few features since accuracy is constant
        assert!(result.n_selected <= 2, "Should minimize features");
    }

    #[test]
    fn test_feature_selector_max_limit() {
        let n_features = 10;

        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            let accuracy = 0.5 + 0.05 * n as f64; // More features = better
            (accuracy, n)
        };

        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::MaxAccuracyWithLimit { max_features: 3 })
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(500));

        assert!(
            result.n_selected <= 3,
            "Should respect max_features limit, got {}",
            result.n_selected
        );
    }

    #[test]
    fn test_feature_selector_aic() {
        let n_features = 8;
        let n_samples = 100;

        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            let accuracy = if n == 0 {
                0.0
            } else {
                0.7 + 0.02 * n.min(4) as f64
            };
            (accuracy, n)
        };

        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::AIC { n_samples })
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(500));

        assert!(result.n_selected > 0);
        assert!(result.n_selected < n_features);
    }

    #[test]
    fn test_convenience_function() {
        let result = select_features(
            5,
            |mask| {
                let n = mask.iter().filter(|&&s| s).count();
                (0.6 + 0.1 * n.min(2) as f64, n)
            },
            Budget::Evaluations(200),
        );

        assert!(!result.mask.is_empty());
        assert_eq!(result.mask.len(), 5);
    }

    #[test]
    fn test_rank_features() {
        let n_features = 4;

        // Feature 0 is most important, feature 3 is useless
        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            let mut acc = 0.5;
            if mask[0] {
                acc += 0.2;
            }
            if mask[1] {
                acc += 0.1;
            }
            if mask[2] {
                acc += 0.05;
            }
            // mask[3] contributes nothing
            (acc, n)
        };

        let ranking = rank_features(n_features, evaluator);

        // Feature 0 should rank highest (biggest drop when removed)
        assert_eq!(ranking[0].0, 0, "Feature 0 should be most important");
        assert!(
            ranking[0].1 > ranking[3].1,
            "Feature 0 more important than 3"
        );
    }

    #[test]
    fn test_feature_selection_result_structure() {
        let result = FeatureSelectionResult {
            selected_indices: vec![0, 2, 5],
            n_selected: 3,
            score: 0.85,
            accuracy: Some(0.9),
            mask: vec![true, false, true, false, false, true],
            evaluations: 500,
        };

        assert_eq!(result.selected_indices.len(), result.n_selected);
        assert_eq!(
            result.mask.iter().filter(|&&s| s).count(),
            result.n_selected
        );
    }

    #[test]
    fn test_selector_builder_pattern() {
        let selector = FeatureSelector::new(20)
            .with_criterion(SelectionCriterion::MaxAccuracy)
            .with_population_size(100)
            .with_mutation_prob(0.05)
            .with_seed(123);

        assert_eq!(selector.n_features, 20);
        assert_eq!(selector.population_size, 100);
        assert!((selector.mutation_prob - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_bic_criterion() {
        let n_features = 6;
        let n_samples = 50;

        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            (0.75, n)
        };

        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::BIC { n_samples })
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(300));

        // BIC penalizes more than AIC for large n, should select fewer
        assert!(result.n_selected < n_features);
    }

    #[test]
    fn test_empty_selection_handling() {
        let n_features = 3;

        // Evaluator that handles empty selection
        let evaluator = |mask: &[bool]| -> (f64, usize) {
            let n = mask.iter().filter(|&&s| s).count();
            if n == 0 {
                (0.0, 0)
            } else {
                (0.8, n)
            }
        };

        // MinFeatures should still work (might select 0)
        let mut selector = FeatureSelector::new(n_features)
            .with_criterion(SelectionCriterion::MinFeatures)
            .with_seed(42);

        let result = selector.select(evaluator, Budget::Evaluations(100));

        // Should handle gracefully regardless of result
        assert!(result.mask.len() == n_features);
    }
}
