//! ENT-032: Ensemble merging strategy enum

/// Strategy for ensemble merging
#[derive(Clone, Debug)]
pub enum EnsembleStrategy {
    /// Simple weighted average (equivalent to DARE with drop_prob=0)
    WeightedAverage { weights: Vec<f32> },

    /// TIES merge with configurable density
    Ties { density: f32 },

    /// DARE merge with dropout
    Dare { drop_prob: f32, seed: Option<u64> },

    /// Iterative SLERP: merge models pairwise until one remains
    IterativeSlerp { t: f32 },

    /// Hierarchical: merge in tree structure for balanced combination
    Hierarchical { leaf_strategy: Box<EnsembleStrategy> },
}

impl Default for EnsembleStrategy {
    fn default() -> Self {
        Self::WeightedAverage {
            weights: Vec::new(), // Will use uniform weights
        }
    }
}
