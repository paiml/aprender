//! ML-based Anomaly Detection and Forecasting
//!
//! This module provides machine learning components for stack diagnostics:
//! - Isolation Forest for anomaly detection
//! - Error Forecaster for time series prediction
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: ML anomaly detection surfaces issues automatically
//! - **Genchi Genbutsu**: Evidence-based diagnosis from actual data

use super::diagnostics::{
    Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode, StackDiagnostics,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Feature extraction
// ============================================================================

/// Feature indices for the 6-element vector
const FEAT_DEMO_SCORE: usize = 0;
const FEAT_COVERAGE: usize = 1;
const FEAT_COMPLEXITY: usize = 3;
const FEAT_DEAD_CODE: usize = 5;

/// Extract a 6-element feature vector from component metrics.
///
/// Layout: `[demo_score, coverage, mutation_score, complexity_avg, satd_count, dead_code_pct]`
fn extract_features(metrics: &ComponentMetrics) -> Vec<f64> {
    vec![
        metrics.demo_score,
        metrics.coverage,
        metrics.mutation_score,
        metrics.complexity_avg,
        metrics.satd_count as f64,
        metrics.dead_code_pct,
    ]
}

// ============================================================================
// Anomaly category dispatch table
// ============================================================================

/// Rule for categorizing an anomaly based on a single feature threshold.
struct CategoryRule {
    feature_index: usize,
    threshold: f64,
    /// true = trigger when feature < threshold; false = trigger when feature > threshold
    below: bool,
    category: AnomalyCategory,
    description_template: &'static str,
    recommendation: &'static str,
}

/// Ordered table of anomaly rules. First matching rule wins.
const CATEGORY_RULES: &[CategoryRule] = &[
    CategoryRule {
        feature_index: FEAT_DEMO_SCORE,
        threshold: 70.0,
        below: true,
        category: AnomalyCategory::QualityRegression,
        description_template: "Quality score {val:.1} is significantly below healthy threshold",
        recommendation: "Review recent changes for quality regressions",
    },
    CategoryRule {
        feature_index: FEAT_COVERAGE,
        threshold: 50.0,
        below: true,
        category: AnomalyCategory::CoverageDrop,
        description_template: "Test coverage {val:.1}% is dangerously low",
        recommendation: "Run `cargo tarpaulin` and add tests for uncovered paths",
    },
    CategoryRule {
        feature_index: FEAT_COMPLEXITY,
        threshold: 15.0,
        below: false,
        category: AnomalyCategory::ComplexityIncrease,
        description_template: "Average complexity {val:.1} indicates maintainability risk",
        recommendation: "Consider refactoring complex functions (>10 cyclomatic complexity)",
    },
    CategoryRule {
        feature_index: FEAT_DEAD_CODE,
        threshold: 10.0,
        below: false,
        category: AnomalyCategory::DependencyRisk,
        description_template: "Dead code {val:.1}% suggests technical debt accumulation",
        recommendation: "Run `cargo udeps` to identify and remove dead code",
    },
];

/// Check whether a rule matches a given feature vector.
fn rule_matches(rule: &CategoryRule, features: &[f64]) -> bool {
    let val = features[rule.feature_index];
    if rule.below {
        val < rule.threshold
    } else {
        val > rule.threshold
    }
}

/// Find the first matching rule for a feature vector.
fn find_matching_rule(features: &[f64]) -> Option<&'static CategoryRule> {
    CATEGORY_RULES.iter().find(|r| rule_matches(r, features))
}

/// Render a description template, replacing `{val:.1}` with the feature value.
fn render_description(template: &str, features: &[f64], feature_index: usize) -> String {
    let val = features[feature_index];
    template.replace("{val:.1}", &format!("{val:.1}"))
}

// ============================================================================
// Simple PRNG (for reproducible isolation forest without external deps)
// ============================================================================

/// Simple Linear Congruential Generator for reproducible randomness
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn seed_from_u64(seed: u64) -> Self {
        Self { state: seed ^ 0x5DEECE66D }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        if range.is_empty() {
            return range.start;
        }
        let len = range.end - range.start;
        range.start + (self.next_u64() as usize % len)
    }

    fn gen_range_f64(&mut self, range: std::ops::Range<f64>) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        range.start + t * (range.end - range.start)
    }
}

// ============================================================================
// Isolation Forest (ML Anomaly Detection)
// ============================================================================

/// Isolation Forest for anomaly detection
/// Implements a simplified version of the algorithm from Liu et al. (2008)
#[derive(Debug)]
pub struct IsolationForest {
    /// Number of trees in the forest
    n_trees: usize,
    /// Subsample size for each tree
    sample_size: usize,
    /// Random seed for reproducibility
    seed: u64,
    /// Trained isolation trees
    trees: Vec<IsolationTree>,
    /// Feature names for interpretation
    feature_names: Vec<String>,
}

impl IsolationForest {
    /// Create a new Isolation Forest
    pub fn new(n_trees: usize, sample_size: usize, seed: u64) -> Self {
        Self { n_trees, sample_size, seed, trees: Vec::new(), feature_names: Vec::new() }
    }

    /// Default forest configuration
    pub fn default_forest() -> Self {
        Self::new(100, 256, 42)
    }

    /// Set feature names for interpretability
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit the forest on data points
    /// Each row is a data point, each column is a feature
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        let mut rng = SimpleRng::seed_from_u64(self.seed);
        let n_samples = data.len();
        let max_depth = (self.sample_size as f64).log2().ceil() as usize;

        self.trees.clear();

        for _ in 0..self.n_trees {
            // Sample data points
            let sample: Vec<Vec<f64>> = (0..self.sample_size.min(n_samples))
                .map(|_| {
                    let idx = rng.gen_range(0..n_samples);
                    data[idx].clone()
                })
                .collect();

            // Build tree
            let tree = IsolationTree::build(&sample, max_depth, &mut rng);
            self.trees.push(tree);
        }
    }

    /// Compute anomaly scores for data points
    /// Returns scores in [0, 1] where higher = more anomalous
    pub fn score(&self, data: &[Vec<f64>]) -> Vec<f64> {
        if self.trees.is_empty() || data.is_empty() {
            return vec![0.0; data.len()];
        }

        let n = self.sample_size as f64;
        let c_n = average_path_length(n);

        data.iter()
            .map(|point| {
                let avg_path_length: f64 =
                    self.trees.iter().map(|tree| tree.path_length(point, 0) as f64).sum::<f64>()
                        / self.trees.len() as f64;

                // Anomaly score: 2^(-avg_path_length / c(n))
                // Higher score = more anomalous
                2.0_f64.powf(-avg_path_length / c_n)
            })
            .collect()
    }

    /// Predict anomalies with threshold
    pub fn predict(&self, data: &[Vec<f64>], threshold: f64) -> Vec<bool> {
        self.score(data).into_iter().map(|s| s > threshold).collect()
    }

    /// Detect anomalies in component metrics and return Anomaly objects
    pub fn detect_anomalies(&self, diagnostics: &StackDiagnostics, threshold: f64) -> Vec<Anomaly> {
        let components: Vec<_> = diagnostics.components().collect();
        if components.is_empty() {
            return Vec::new();
        }

        // Extract feature vectors
        let data: Vec<Vec<f64>> = components.iter().map(|c| extract_features(&c.metrics)).collect();

        let scores = self.score(&data);
        let mut anomalies = Vec::new();

        for (i, (component, score)) in components.iter().zip(scores.iter()).enumerate() {
            if *score > threshold {
                let category = self.categorize_anomaly(&data[i]);
                let description = self.describe_anomaly(&data[i], &category);

                let mut anomaly =
                    Anomaly::new(component.name.clone(), *score, category, description);

                // Add evidence
                anomaly = anomaly
                    .with_evidence(format!("Isolation score: {:.3}", score))
                    .with_evidence(format!("Demo score: {:.1}", component.metrics.demo_score))
                    .with_evidence(format!("Coverage: {:.1}%", component.metrics.coverage));

                // Add recommendation
                let rec = self.recommend_action(&category, &data[i]);
                anomaly = anomaly.with_recommendation(rec);

                anomalies.push(anomaly);
            }
        }

        // Sort by score descending
        anomalies
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        anomalies
    }

    /// Categorize the anomaly based on which features are most deviant
    fn categorize_anomaly(&self, features: &[f64]) -> AnomalyCategory {
        if features.len() < 6 {
            return AnomalyCategory::Other;
        }
        find_matching_rule(features).map(|r| r.category).unwrap_or(AnomalyCategory::Other)
    }

    /// Generate human-readable description
    fn describe_anomaly(&self, features: &[f64], category: &AnomalyCategory) -> String {
        if features.len() < 6 {
            return "Unusual metric combination detected".to_string();
        }
        find_matching_rule(features)
            .filter(|r| r.category == *category)
            .map(|r| render_description(r.description_template, features, r.feature_index))
            .unwrap_or_else(|| "Unusual metric combination detected".to_string())
    }

    /// Generate actionable recommendation
    fn recommend_action(&self, category: &AnomalyCategory, features: &[f64]) -> String {
        // Special sub-rule: QualityRegression with low coverage gets a coverage-specific tip
        if *category == AnomalyCategory::QualityRegression
            && features.len() >= 6
            && features[FEAT_COVERAGE] < 80.0
        {
            return "Add tests to improve coverage above 80%".to_string();
        }

        find_matching_rule(features)
            .filter(|r| r.category == *category)
            .map(|r| r.recommendation.to_string())
            .unwrap_or_else(|| "Review component metrics for unusual patterns".to_string())
    }
}

/// A single isolation tree node
#[derive(Debug)]
enum IsolationTree {
    /// Internal node with split
    Internal {
        split_feature: usize,
        split_value: f64,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
    /// External (leaf) node
    External { size: usize },
}

impl IsolationTree {
    /// Build an isolation tree from data
    fn build(data: &[Vec<f64>], max_depth: usize, rng: &mut SimpleRng) -> Self {
        if data.is_empty() {
            return IsolationTree::External { size: 0 };
        }

        if max_depth == 0 || data.len() <= 1 {
            return IsolationTree::External { size: data.len() };
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return IsolationTree::External { size: data.len() };
        }

        // Random feature
        let feature = rng.gen_range(0..n_features);

        // Find min/max for this feature
        let values: Vec<f64> = data.iter().filter_map(|row| row.get(feature).copied()).collect();
        if values.is_empty() {
            return IsolationTree::External { size: data.len() };
        }

        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationTree::External { size: data.len() };
        }

        // Random split value
        let split_value = rng.gen_range_f64(min_val..max_val);

        // Partition data
        let (left_data, right_data): (Vec<_>, Vec<_>) = data
            .iter()
            .cloned()
            .partition(|row| row.get(feature).is_some_and(|&v| v < split_value));

        // Handle edge case where all data goes to one side
        if left_data.is_empty() || right_data.is_empty() {
            return IsolationTree::External { size: data.len() };
        }

        IsolationTree::Internal {
            split_feature: feature,
            split_value,
            left: Box::new(IsolationTree::build(&left_data, max_depth - 1, rng)),
            right: Box::new(IsolationTree::build(&right_data, max_depth - 1, rng)),
        }
    }

    /// Compute path length for a point
    fn path_length(&self, point: &[f64], current_depth: usize) -> usize {
        match self {
            IsolationTree::External { size } => {
                current_depth + average_path_length(*size as f64) as usize
            }
            IsolationTree::Internal { split_feature, split_value, left, right } => {
                let value = point.get(*split_feature).copied().unwrap_or(0.0);
                if value < *split_value {
                    left.path_length(point, current_depth + 1)
                } else {
                    right.path_length(point, current_depth + 1)
                }
            }
        }
    }
}

/// Average path length of unsuccessful search in BST
fn average_path_length(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    2.0 * (n.ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
}

// ============================================================================
// Time Series Forecasting (Error Prediction)
// ============================================================================

/// Simple exponential smoothing for time series forecasting
#[derive(Debug, Clone)]
pub struct ErrorForecaster {
    /// Smoothing parameter alpha (0-1)
    alpha: f64,
    /// Historical observations
    history: Vec<f64>,
    /// Current smoothed value
    level: f64,
}

impl ErrorForecaster {
    /// Create a new error forecaster
    pub fn new(alpha: f64) -> Self {
        Self { alpha: alpha.clamp(0.0, 1.0), history: Vec::new(), level: 0.0 }
    }

    /// Default forecaster with alpha=0.3
    pub fn default_forecaster() -> Self {
        Self::new(0.3)
    }

    /// Add an observation
    pub fn observe(&mut self, value: f64) {
        if self.history.is_empty() {
            self.level = value;
        } else {
            // Exponential smoothing: L_t = alpha * Y_t + (1 - alpha) * L_{t-1}
            self.level = self.alpha * value + (1.0 - self.alpha) * self.level;
        }
        self.history.push(value);
    }

    /// Forecast next n values
    pub fn forecast(&self, n: usize) -> Vec<f64> {
        // Simple exponential smoothing forecasts are constant
        vec![self.level; n]
    }

    /// Compute forecast error metrics
    pub fn error_metrics(&self) -> ForecastMetrics {
        if self.history.len() < 2 {
            return ForecastMetrics::default();
        }

        // Compute in-sample errors
        let mut errors = Vec::new();
        let mut level = self.history[0];

        for &actual in self.history.iter().skip(1) {
            let forecast = level;
            errors.push(actual - forecast);
            level = self.alpha * actual + (1.0 - self.alpha) * level;
        }

        let n = errors.len() as f64;
        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / n;
        let mse = errors.iter().map(|e| e * e).sum::<f64>() / n;
        let rmse = mse.sqrt();

        // MAPE (avoid division by zero)
        let mape = if self.history.iter().skip(1).all(|&v| v.abs() > f64::EPSILON) {
            let sum: f64 =
                errors.iter().zip(self.history.iter().skip(1)).map(|(e, a)| (e / a).abs()).sum();
            sum / n * 100.0
        } else {
            f64::NAN
        };

        ForecastMetrics { mae, mse, rmse, mape }
    }

    /// Get historical observations
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Get current level
    pub fn current_level(&self) -> f64 {
        self.level
    }
}

/// Forecast error metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "diagnostics_ml_tests.rs"]
mod tests;
