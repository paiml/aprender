//! Tests for TIES and DARE ensemble merging

use super::common::make_model;
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig, EnsembleStrategy, MergeError};

#[test]
fn test_ensemble_ties() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, -3.0, 4.0]);
    let m2 = make_model(vec![1.0, -2.0, 3.0, 4.0]);

    let config = EnsembleConfig::ties(base, 0.5);
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    // Should produce valid output
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_ensemble_dare() {
    let base = make_model(vec![0.0, 0.0]);
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::dare(base, 0.0, Some(42));
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    // With drop_prob=0, should be average: (2+4)/2=3, (4+6)/2=5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_ties_without_base() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig { base: None, strategy: EnsembleStrategy::Ties { density: 0.5 } };

    let result = ensemble_merge(&[m1, m2], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_dare_without_base() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig {
        base: None,
        strategy: EnsembleStrategy::Dare { drop_prob: 0.5, seed: None },
    };

    let result = ensemble_merge(&[m1, m2], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_ensemble_dare_without_seed() {
    let base = make_model(vec![0.0, 0.0]);
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::dare(base, 0.3, None);
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}
