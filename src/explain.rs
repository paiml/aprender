//! Model Explainability (SHAP, LIME, Attention)
//!
//! Per spec §13: Model explainability for APR classical ML models.
//! Implements SHAP TreeExplainer for tree ensembles and KernelSHAP for any model.
//!
//! ## Methods
//!
//! | Method | Type | Models | Output |
//! |--------|------|--------|--------|
//! | **TreeSHAP** | Model-specific | Tree ensembles | Feature contributions |
//! | **KernelSHAP** | Model-agnostic | Any | Feature contributions |
//!
//! ## References
//!
//! - [16] Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
//! - [17] Ribeiro et al. (2016) "Why Should I Trust You? Explaining Predictions"

// Module-level clippy allows for explainability module
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unused_self)] // Methods designed for future expansion
#![allow(clippy::unnecessary_wraps)] // Result wrapping for API consistency
#![allow(clippy::option_if_let_else)] // map_or is more readable for our use case

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Error type for explainability operations
#[derive(Debug, Error)]
pub enum ExplainError {
    /// Model does not support explainability
    #[error("Model does not support explainability: {reason}")]
    UnsupportedModel {
        /// Why the model is unsupported
        reason: String,
    },

    /// Invalid input dimensions
    #[error("Invalid input: expected {expected} features, got {actual}")]
    InvalidInput {
        /// Expected number of features
        expected: usize,
        /// Actual number of features provided
        actual: usize,
    },

    /// Background dataset required but not provided
    #[error("Background dataset required for KernelSHAP")]
    NoBackground,

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Trait for models that can be explained
pub trait Explainable {
    /// Predict for a single instance
    fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError>;

    /// Predict for multiple instances (batch)
    fn predict_batch(&self, instances: &[Vec<f32>]) -> Result<Vec<f32>, ExplainError> {
        instances.iter().map(|x| self.predict(x)).collect()
    }

    /// Number of features expected
    fn n_features(&self) -> usize;

    /// Check if this is a tree-based model
    fn is_tree_model(&self) -> bool {
        false
    }

    /// Get tree structure for TreeSHAP (if applicable)
    fn get_tree_structure(&self) -> Option<&TreeStructure> {
        None
    }
}

/// Tree structure for TreeSHAP algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStructure {
    /// Number of trees in the ensemble
    pub n_trees: usize,
    /// Number of features
    pub n_features: usize,
    /// Trees in the ensemble
    pub trees: Vec<DecisionTree>,
}

/// A single decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Feature index used at each node (-1 for leaf)
    pub feature: Vec<i32>,
    /// Threshold at each node
    pub threshold: Vec<f32>,
    /// Left child index
    pub left: Vec<usize>,
    /// Right child index
    pub right: Vec<usize>,
    /// Value at leaf nodes
    pub value: Vec<f32>,
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(
        feature: Vec<i32>,
        threshold: Vec<f32>,
        left: Vec<usize>,
        right: Vec<usize>,
        value: Vec<f32>,
    ) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            value,
        }
    }

    /// Get the number of nodes in the tree
    pub fn n_nodes(&self) -> usize {
        self.feature.len()
    }

    /// Check if a node is a leaf
    pub fn is_leaf(&self, node: usize) -> bool {
        self.feature.get(node).is_none_or(|&f| f < 0)
    }

    /// Predict for a single instance
    pub fn predict(&self, instance: &[f32]) -> f32 {
        contract_pre_predict!();
        let mut node = 0;
        while !self.is_leaf(node) {
            let feature_idx = self.feature[node] as usize;
            if instance
                .get(feature_idx)
                .is_some_and(|&v| v <= self.threshold[node])
            {
                node = self.left[node];
            } else {
                node = self.right[node];
            }
        }
        let result = self.value.get(node).copied().unwrap_or(0.0);
        contract_post_predict!(&result);
        result
    }
}

/// SHAP explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapExplanation {
    /// Expected model output E[f(X)]
    pub base_value: f32,
    /// SHAP values for each feature (φᵢ)
    /// sum(shap_values) + base_value ≈ prediction
    pub shap_values: Vec<f32>,
    /// Feature names for display
    pub feature_names: Vec<String>,
    /// The actual prediction for this instance
    pub prediction: f32,
}

impl ShapExplanation {
    /// Create a new SHAP explanation
    pub fn new(base_value: f32, shap_values: Vec<f32>, prediction: f32) -> Self {
        let n = shap_values.len();
        Self {
            base_value,
            shap_values,
            feature_names: (0..n).map(|i| format!("feature_{i}")).collect(),
            prediction,
        }
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Get the most important features (sorted by absolute SHAP value)
    pub fn top_features(&self, n: usize) -> Vec<(String, f32)> {
        let mut indexed: Vec<_> = self
            .shap_values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed
            .into_iter()
            .take(n)
            .map(|(i, v)| {
                let name = self
                    .feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{i}"));
                (name, v)
            })
            .collect()
    }

    /// Verify SHAP consistency: sum(shap_values) + base_value ≈ prediction
    pub fn verify_consistency(&self, tolerance: f32) -> bool {
        let sum: f32 = self.shap_values.iter().sum();
        (self.base_value + sum - self.prediction).abs() < tolerance
    }
}

impl fmt::Display for ShapExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SHAP Explanation:")?;
        writeln!(f, "  Base value: {:.4}", self.base_value)?;
        writeln!(f, "  Prediction: {:.4}", self.prediction)?;
        writeln!(f, "  Top features:")?;
        for (name, value) in self.top_features(5) {
            let sign = if value >= 0.0 { "+" } else { "" };
            writeln!(f, "    {name}: {sign}{value:.4}")?;
        }
        Ok(())
    }
}

/// SHAP explainer for APR classical ML models
/// Reference: [16] Lundberg & Lee (2017) SHAP
pub struct ShapExplainer {
    /// Background dataset for computing expected values
    background: Vec<Vec<f32>>,
    /// Number of samples for KernelSHAP
    nsamples: usize,
    /// Feature names
    feature_names: Vec<String>,
}

include!("explain_nsamples_shap_explainer.rs");
include!("simple.rs");
