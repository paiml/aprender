//! Tests for hierarchical ensemble merging

use super::common::make_model;
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig, EnsembleStrategy, MergeError};

#[test]
fn test_hierarchical_four_models() {
    let models: Vec<_> = (0..4).map(|i| make_model(vec![i as f32 * 2.0, i as f32 * 3.0])).collect();

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage { weights: vec![0.5, 0.5] });
    let result = ensemble_merge(&models, &config).expect("config should be valid");

    // ((0+2)/2 + (4+6)/2) / 2 = (1 + 5) / 2 = 3
    // ((0+3)/2 + (6+9)/2) / 2 = (1.5 + 7.5) / 2 = 4.5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 4.5).abs() < 1e-5);
}

#[test]
fn test_hierarchical_with_slerp() {
    let m1 = make_model(vec![1.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0]);
    let m3 = make_model(vec![-1.0, 0.0]);
    let m4 = make_model(vec![0.0, -1.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::IterativeSlerp { t: 0.5 });
    let result = ensemble_merge(&[m1, m2, m3, m4], &config).expect("config should be valid");

    // Result should be finite
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_with_ties() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = make_model(vec![5.0, 6.0, 7.0, 8.0]);
    let m3 = make_model(vec![9.0, 10.0, 11.0, 12.0]);
    let m4 = make_model(vec![13.0, 14.0, 15.0, 16.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::Ties { density: 0.5 }).with_base(base);

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).expect("config should be valid");
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_with_dare() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = make_model(vec![5.0, 6.0, 7.0, 8.0]);
    let m3 = make_model(vec![9.0, 10.0, 11.0, 12.0]);
    let m4 = make_model(vec![13.0, 14.0, 15.0, 16.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::Dare { drop_prob: 0.3, seed: Some(42) })
            .with_base(base);

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).expect("config should be valid");
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_three_models() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage { weights: vec![] });

    let result = ensemble_merge(&[m1, m2, m3], &config).expect("config should be valid");
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_nested_hierarchical_fallback() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    // Nested hierarchical - the inner hierarchical falls back to weighted average
    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Hierarchical {
        leaf_strategy: Box::new(EnsembleStrategy::WeightedAverage { weights: vec![] }),
    });

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).expect("config should be valid");
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_ties_without_base_error() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Ties { density: 0.5 });
    // No base provided

    let result = ensemble_merge(&[m1, m2, m3, m4], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_hierarchical_dare_without_base_error() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::Dare { drop_prob: 0.3, seed: None });
    // No base provided

    let result = ensemble_merge(&[m1, m2, m3, m4], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_merge_pair_with_weighted_avg_specific_weights() {
    let m1 = make_model(vec![0.0, 0.0]);
    let m2 = make_model(vec![10.0, 10.0]);

    // Using hierarchical will call merge_pair with specific weights
    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage { weights: vec![0.3, 0.7] });

    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");
    // 0*0.3 + 10*0.7 = 7
    assert!((result["w"].data()[0] - 7.0).abs() < 1e-5);
}
