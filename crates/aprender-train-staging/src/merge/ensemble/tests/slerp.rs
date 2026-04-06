//! Tests for iterative SLERP ensemble merging

use super::common::{make_model, models_approx_equal};
use crate::merge::ensemble::{ensemble_merge, EnsembleConfig};

#[test]
fn test_iterative_slerp_two_models() {
    let m1 = make_model(vec![1.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0]);

    let config = EnsembleConfig::iterative_slerp(0.5);
    let result = ensemble_merge(&[m1, m2], &config).expect("config should be valid");

    // At t=0.5, perpendicular vectors should blend to (1/sqrt(2), 1/sqrt(2))
    let expected = 1.0 / 2.0f32.sqrt();
    assert!((result["w"].data()[0] - expected).abs() < 1e-4);
    assert!((result["w"].data()[1] - expected).abs() < 1e-4);
}

#[test]
fn test_iterative_slerp_three_models() {
    let m1 = make_model(vec![1.0, 0.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0, 0.0]);
    let m3 = make_model(vec![0.0, 0.0, 1.0]);

    let config = EnsembleConfig::iterative_slerp(0.5);
    let result = ensemble_merge(&[m1, m2, m3], &config).expect("config should be valid");

    // Result should be finite and have reasonable values
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_iterative_slerp_t0_returns_first() {
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![4.0, 5.0, 6.0]);
    let m3 = make_model(vec![7.0, 8.0, 9.0]);

    let config = EnsembleConfig::iterative_slerp(0.0);
    let result = ensemble_merge(&[m1.clone(), m2, m3], &config).expect("config should be valid");

    assert!(models_approx_equal(&result, &m1, 1e-5));
}
