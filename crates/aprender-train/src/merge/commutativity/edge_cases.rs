//! Edge case tests for merge commutativity

use super::helpers::{make_model, models_approx_equal};
use crate::merge::{dare_merge, slerp_merge, ties_merge, DareConfig, SlerpConfig, TiesConfig};

#[test]
fn slerp_parallel_vectors_commutativity() {
    // Parallel vectors should also be commutative (linear interp fallback)
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![2.0, 4.0, 6.0]); // Parallel to m1

    let t = 0.3;
    let c1 = SlerpConfig::new(t).expect("slerp config creation should succeed");
    let c2 = SlerpConfig::new(1.0 - t).expect("slerp config creation should succeed");

    let r1 = slerp_merge(&m1, &m2, &c1).expect("operation should succeed");
    let r2 = slerp_merge(&m2, &m1, &c2).expect("operation should succeed");

    assert!(
        models_approx_equal(&r1, &r2, 1e-4),
        "SLERP with parallel vectors should be commutative"
    );
}

#[test]
fn slerp_antiparallel_vectors_commutativity() {
    // Anti-parallel vectors
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![-1.0, -2.0, -3.0]);

    let t = 0.4;
    let c1 = SlerpConfig::new(t).expect("slerp config creation should succeed");
    let c2 = SlerpConfig::new(1.0 - t).expect("slerp config creation should succeed");

    let r1 = slerp_merge(&m1, &m2, &c1).expect("operation should succeed");
    let r2 = slerp_merge(&m2, &m1, &c2).expect("operation should succeed");

    assert!(
        models_approx_equal(&r1, &r2, 1e-4),
        "SLERP with anti-parallel vectors should be commutative"
    );
}

#[test]
fn dare_with_base_equals_models() {
    // When base equals models, result should equal base
    let base = make_model(vec![5.0, 10.0, 15.0]);
    let config = DareConfig::new(0.5).expect("config should be valid").with_seed(42);

    let result =
        dare_merge(&[base.clone(), base.clone()], &base, &config).expect("config should be valid");

    assert!(
        models_approx_equal(&result, &base, 1e-5),
        "DARE when models equal base should return base"
    );
}

#[test]
fn ties_with_base_equals_models() {
    // When base equals models, result should equal base
    let base = make_model(vec![5.0, 10.0, 15.0]);
    let config = TiesConfig::new(0.5).expect("config should be valid");

    let result =
        ties_merge(&[base.clone(), base.clone()], &base, &config).expect("config should be valid");

    assert!(
        models_approx_equal(&result, &base, 1e-5),
        "TIES when models equal base should return base"
    );
}

#[test]
fn all_methods_handle_single_element() {
    // Single element vectors
    let base = make_model(vec![0.0]);
    let m1 = make_model(vec![5.0]);
    let m2 = make_model(vec![10.0]);

    // SLERP
    let slerp_r = slerp_merge(
        &m1,
        &m2,
        &SlerpConfig::new(0.5).expect("slerp config creation should succeed"),
    )
    .expect("slerp config creation should succeed");
    assert!(slerp_r["w"].data()[0].is_finite());

    // DARE
    let dare_r = dare_merge(
        &[m1.clone(), m2.clone()],
        &base,
        &DareConfig::new(0.0).expect("config should be valid"),
    )
    .expect("config should be valid");
    assert!((dare_r["w"].data()[0] - 7.5).abs() < 1e-5); // (5+10)/2

    // TIES
    let ties_r =
        ties_merge(&[m1, m2], &base, &TiesConfig::new(0.5).expect("config should be valid"))
            .expect("config should be valid");
    assert!(ties_r["w"].data()[0].is_finite());
}
