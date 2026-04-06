//! ENT-032: Main ensemble merge function

use super::config::EnsembleConfig;
use super::hierarchical::hierarchical_merge;
use super::slerp::iterative_slerp_merge;
use super::strategy::EnsembleStrategy;
use super::weighted::weighted_average_merge;
use super::{dare_merge, ties_merge, DareConfig, MergeError, Model, TiesConfig};

/// Merge multiple models using the specified strategy
///
/// # Arguments
/// * `models` - Models to merge (must have at least 2)
/// * `config` - Ensemble configuration
///
/// # Returns
/// Merged model combining all inputs
pub fn ensemble_merge(models: &[Model], config: &EnsembleConfig) -> Result<Model, MergeError> {
    if models.len() < 2 {
        return Err(MergeError::InsufficientModels { min: 2, got: models.len() });
    }

    match &config.strategy {
        EnsembleStrategy::WeightedAverage { weights } => weighted_average_merge(models, weights),
        EnsembleStrategy::Ties { density } => {
            let base = config
                .base
                .as_ref()
                .ok_or_else(|| MergeError::InvalidConfig("TIES requires base model".to_string()))?;
            let ties_config = TiesConfig::new(*density)?;
            ties_merge(models, base, &ties_config)
        }
        EnsembleStrategy::Dare { drop_prob, seed } => {
            let base = config
                .base
                .as_ref()
                .ok_or_else(|| MergeError::InvalidConfig("DARE requires base model".to_string()))?;
            let mut dare_config = DareConfig::new(*drop_prob)?;
            if let Some(s) = seed {
                dare_config = dare_config.with_seed(*s);
            }
            dare_merge(models, base, &dare_config)
        }
        EnsembleStrategy::IterativeSlerp { t } => iterative_slerp_merge(models, *t),
        EnsembleStrategy::Hierarchical { leaf_strategy } => {
            hierarchical_merge(models, leaf_strategy, config.base.as_ref())
        }
    }
}
