//! TIES commutativity tests

use super::helpers::{make_model, models_approx_equal};
use crate::merge::{ties_merge, Model, TiesConfig};

#[test]
fn ties_permutation_invariance() {
    // TIES result should be independent of model ordering
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let models = vec![
        make_model(vec![1.0, 2.0, -3.0, 4.0]),
        make_model(vec![1.0, -2.0, 3.0, 4.0]),
        make_model(vec![-1.0, 2.0, 3.0, 4.0]),
    ];

    let config = TiesConfig::new(0.5).expect("config should be valid");

    let r1 = ties_merge(&models, &base, &config).expect("config should be valid");

    // All permutations should yield same result
    let perms =
        [vec![0, 1, 2], vec![0, 2, 1], vec![1, 0, 2], vec![1, 2, 0], vec![2, 0, 1], vec![2, 1, 0]];

    for perm in perms {
        let permuted: Vec<Model> = perm.iter().map(|&i| models[i].clone()).collect();
        let r = ties_merge(&permuted, &base, &config).expect("config should be valid");
        assert!(models_approx_equal(&r1, &r, 1e-5), "TIES should be permutation-invariant");
    }
}

#[test]
fn ties_self_merge_preserves_direction() {
    // Merging identical models should preserve sign and direction
    let base = make_model(vec![0.0, 0.0, 0.0]);
    let m = make_model(vec![1.0, -2.0, 3.0]);

    let config = TiesConfig::new(0.8).expect("config should be valid"); // High density to keep most values

    let result =
        ties_merge(&[m.clone(), m.clone()], &base, &config).expect("config should be valid");

    // Signs should match original
    let orig = m["w"].data();
    let merged = result["w"].data();
    for (o, r) in orig.iter().zip(merged.iter()) {
        if r.abs() > 1e-6 {
            assert!(
                o.signum() == r.signum(),
                "TIES of identical models should preserve sign: {o} vs {r}"
            );
        }
    }
}
