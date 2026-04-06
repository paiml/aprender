//! Feature change tracking for counterfactual explanations.

use serde::{Deserialize, Serialize};

/// A single feature change in a counterfactual
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureChange {
    /// Feature index
    pub feature_idx: usize,
    /// Optional human-readable feature name
    pub feature_name: Option<String>,
    /// Original value
    pub original_value: f32,
    /// Counterfactual value (value that would flip the decision)
    pub counterfactual_value: f32,
    /// Change amount (counterfactual - original)
    pub delta: f32,
}

impl FeatureChange {
    /// Create a new feature change
    pub fn new(feature_idx: usize, original: f32, counterfactual: f32) -> Self {
        Self {
            feature_idx,
            feature_name: None,
            original_value: original,
            counterfactual_value: counterfactual,
            delta: counterfactual - original,
        }
    }

    /// Set the feature name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.feature_name = Some(name.into());
        self
    }

    /// Get the absolute change
    pub fn abs_delta(&self) -> f32 {
        self.delta.abs()
    }
}
