//! ENT-032: Ensemble configuration struct

use super::strategy::EnsembleStrategy;
use super::Model;

/// Configuration for ensemble merging
#[derive(Clone, Debug, Default)]
pub struct EnsembleConfig {
    /// Base model for delta-based methods (TIES, DARE)
    /// If None, uses first model as base for delta methods
    pub base: Option<Model>,

    /// Merging strategy
    pub strategy: EnsembleStrategy,
}

impl EnsembleConfig {
    /// Create config for weighted averaging
    pub fn weighted_average(weights: Vec<f32>) -> Self {
        Self { base: None, strategy: EnsembleStrategy::WeightedAverage { weights } }
    }

    /// Create config for uniform averaging
    pub fn uniform_average() -> Self {
        Self { base: None, strategy: EnsembleStrategy::WeightedAverage { weights: Vec::new() } }
    }

    /// Create config for TIES merging
    pub fn ties(base: Model, density: f32) -> Self {
        Self { base: Some(base), strategy: EnsembleStrategy::Ties { density } }
    }

    /// Create config for DARE merging
    pub fn dare(base: Model, drop_prob: f32, seed: Option<u64>) -> Self {
        Self { base: Some(base), strategy: EnsembleStrategy::Dare { drop_prob, seed } }
    }

    /// Create config for iterative SLERP
    pub fn iterative_slerp(t: f32) -> Self {
        Self { base: None, strategy: EnsembleStrategy::IterativeSlerp { t } }
    }

    /// Create config for hierarchical merging
    pub fn hierarchical(leaf_strategy: EnsembleStrategy) -> Self {
        Self {
            base: None,
            strategy: EnsembleStrategy::Hierarchical { leaf_strategy: Box::new(leaf_strategy) },
        }
    }

    /// Set base model for delta-based methods
    pub fn with_base(mut self, base: Model) -> Self {
        self.base = Some(base);
        self
    }
}
