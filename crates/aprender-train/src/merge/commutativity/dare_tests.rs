//! DARE commutativity tests

use super::helpers::{make_model, models_approx_equal};
use crate::merge::{dare_merge, DareConfig, Model};

#[test]
fn dare_zero_drop_is_commutative() {
    // With drop_prob=0, DARE becomes simple averaging which is commutative
    let base = make_model(vec![0.0, 0.0]);
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = DareConfig::new(0.0).expect("config should be valid");

    let r1 = dare_merge(&[m1.clone(), m2.clone()], &base, &config).expect("config should be valid");
    let r2 = dare_merge(&[m2, m1], &base, &config).expect("config should be valid");

    assert!(models_approx_equal(&r1, &r2, 1e-5), "DARE with drop_prob=0 should be commutative");
}

#[test]
fn dare_self_merge_identity() {
    // Merging identical models with drop_prob=0 should preserve values
    let base = make_model(vec![0.0, 0.0, 0.0]);
    let m = make_model(vec![1.0, 2.0, 3.0]);

    let config = DareConfig::new(0.0).expect("config should be valid");
    let result =
        dare_merge(&[m.clone(), m.clone()], &base, &config).expect("config should be valid");

    assert!(
        models_approx_equal(&result, &m, 1e-5),
        "DARE of identical models should preserve values"
    );
}

#[test]
fn dare_permutation_invariance_with_zero_drop() {
    // Order of models shouldn't matter for averaging
    let base = make_model(vec![0.0, 0.0, 0.0]);
    let models = vec![
        make_model(vec![1.0, 2.0, 3.0]),
        make_model(vec![4.0, 5.0, 6.0]),
        make_model(vec![7.0, 8.0, 9.0]),
    ];

    let config = DareConfig::new(0.0).expect("config should be valid");

    let r1 = dare_merge(&models, &base, &config).expect("config should be valid");

    // Permute: [2, 0, 1]
    let permuted: Vec<Model> = vec![models[2].clone(), models[0].clone(), models[1].clone()];
    let r2 = dare_merge(&permuted, &base, &config).expect("config should be valid");

    assert!(
        models_approx_equal(&r1, &r2, 1e-5),
        "DARE with drop_prob=0 should be permutation-invariant"
    );
}
