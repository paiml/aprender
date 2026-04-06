//! ENT-032: Hierarchical merging

use super::strategy::EnsembleStrategy;
use super::weighted::weighted_average_merge;
use super::{
    dare_merge, slerp_merge as slerp_merge_impl, ties_merge, DareConfig, MergeError, Model,
    SlerpConfig, TiesConfig,
};

/// Hierarchical merge: tree-based for balanced combination
///
/// For 4 models: (m1 + m2) + (m3 + m4)
/// More balanced than iterative for large N
pub fn hierarchical_merge(
    models: &[Model],
    leaf_strategy: &EnsembleStrategy,
    base: Option<&Model>,
) -> Result<Model, MergeError> {
    if models.len() == 1 {
        return Ok(models[0].clone());
    }

    if models.len() == 2 {
        return merge_pair(&models[0], &models[1], leaf_strategy, base);
    }

    // Split and recurse
    let mid = models.len() / 2;
    let left = hierarchical_merge(&models[..mid], leaf_strategy, base)?;
    let right = hierarchical_merge(&models[mid..], leaf_strategy, base)?;

    merge_pair(&left, &right, leaf_strategy, base)
}

/// Merge two models using specified strategy
pub fn merge_pair(
    m1: &Model,
    m2: &Model,
    strategy: &EnsembleStrategy,
    base: Option<&Model>,
) -> Result<Model, MergeError> {
    match strategy {
        EnsembleStrategy::WeightedAverage { weights } => {
            let w = if weights.len() == 2 { weights.clone() } else { vec![0.5, 0.5] };
            weighted_average_merge(&[m1.clone(), m2.clone()], &w)
        }
        EnsembleStrategy::IterativeSlerp { t } => {
            let config = SlerpConfig::new(*t)?;
            slerp_merge_impl(m1, m2, &config)
        }
        EnsembleStrategy::Ties { density } => {
            let base =
                base.ok_or_else(|| MergeError::InvalidConfig("TIES requires base".to_string()))?;
            let config = TiesConfig::new(*density)?;
            ties_merge(&[m1.clone(), m2.clone()], base, &config)
        }
        EnsembleStrategy::Dare { drop_prob, seed } => {
            let base =
                base.ok_or_else(|| MergeError::InvalidConfig("DARE requires base".to_string()))?;
            let mut config = DareConfig::new(*drop_prob)?;
            if let Some(s) = seed {
                config = config.with_seed(*s);
            }
            dare_merge(&[m1.clone(), m2.clone()], base, &config)
        }
        EnsembleStrategy::Hierarchical { .. } => {
            // For hierarchical, default to weighted average at leaves
            weighted_average_merge(&[m1.clone(), m2.clone()], &[0.5, 0.5])
        }
    }
}
