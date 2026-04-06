//! Tests for EnsembleConfig and EnsembleStrategy defaults and utilities

use super::common::make_model;
use crate::merge::ensemble::{EnsembleConfig, EnsembleStrategy};

#[test]
fn test_ensemble_config_with_base() {
    let base = make_model(vec![0.0, 0.0]);
    let config = EnsembleConfig::uniform_average().with_base(base);

    assert!(config.base.is_some());
}

#[test]
fn test_ensemble_strategy_default() {
    let strategy = EnsembleStrategy::default();
    matches!(strategy, EnsembleStrategy::WeightedAverage { weights } if weights.is_empty());
}

#[test]
fn test_ensemble_config_default() {
    let config = EnsembleConfig::default();
    assert!(config.base.is_none());
    matches!(config.strategy, EnsembleStrategy::WeightedAverage { .. });
}
