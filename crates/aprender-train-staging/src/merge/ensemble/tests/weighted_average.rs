//! Tests for weighted average ensemble merging

use super::common::make_model;
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig, MergeError};

#[test]
fn test_uniform_average_two_models() {
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    // (2+4)/2 = 3, (4+6)/2 = 5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_uniform_average_three_models() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![2.0, 4.0]);
    let m3 = make_model(vec![3.0, 6.0]);

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2, m3], &config).expect("config should be valid");

    // (1+2+3)/3 = 2, (2+4+6)/3 = 4
    assert!((result["w"].data()[0] - 2.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average() {
    let m1 = make_model(vec![0.0]);
    let m2 = make_model(vec![10.0]);

    // 70% m1, 30% m2
    let config = EnsembleConfig::weighted_average(vec![0.7, 0.3]);
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    // 0*0.7 + 10*0.3 = 3.0
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_unnormalized() {
    let m1 = make_model(vec![0.0]);
    let m2 = make_model(vec![10.0]);

    // Unnormalized weights: 7, 3 -> normalized to 0.7, 0.3
    let config = EnsembleConfig::weighted_average(vec![7.0, 3.0]);
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_wrong_length() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig::weighted_average(vec![1.0]); // Wrong length!
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}
