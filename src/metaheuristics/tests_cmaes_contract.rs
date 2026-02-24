// =========================================================================
// FALSIFY-CMA: cma-es-kernel-v1.yaml contract (aprender CMA-ES)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 13+ CMA-ES tests but zero FALSIFY-CMA-* tests
//   Why 2: unit tests verify convergence, not optimizer invariants
//   Why 3: no mapping from cma-es-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: CMA-ES was "obviously correct" (Hansen's tutorial)
//
// References:
//   - provable-contracts/contracts/cma-es-kernel-v1.yaml
//   - Hansen (2016) "The CMA Evolution Strategy: A Tutorial"
// =========================================================================

pub(crate) use super::*;

/// FALSIFY-CMA-001: Step size positivity — sigma > 0 after many generations
///
/// Contract: sigma > 0 at every generation. CSA step-size adaptation
/// should never drive sigma to zero or negative.
#[test]
fn falsify_cma_001_step_size_positivity() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut cma = CmaEs::new(5).with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    cma.optimize(&sphere, &space, Budget::Evaluations(5000));

    assert!(
        cma.sigma > 0.0,
        "FALSIFIED CMA-001: sigma = {} <= 0 after optimization",
        cma.sigma
    );
    assert!(
        cma.sigma.is_finite(),
        "FALSIFIED CMA-001: sigma = {} (not finite)",
        cma.sigma
    );
}

/// FALSIFY-CMA-002: Covariance positive definiteness — c_diag[i] > 0
///
/// For diagonal CMA-ES, positive definiteness = all diagonal entries > 0.
#[test]
fn falsify_cma_002_covariance_positive_definite() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut cma = CmaEs::new(5).with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    cma.optimize(&sphere, &space, Budget::Evaluations(5000));

    for (i, &c) in cma.c_diag.iter().enumerate() {
        assert!(
            c > 0.0,
            "FALSIFIED CMA-002: c_diag[{i}] = {c} <= 0 (not positive definite)"
        );
        assert!(
            c.is_finite(),
            "FALSIFIED CMA-002: c_diag[{i}] = {c} (not finite)"
        );
    }
}

/// FALSIFY-CMA-003: Weight normalization — |sum(w_i) - 1.0| < 1e-10
///
/// Recombination weights must form a convex combination (sum to 1).
#[test]
fn falsify_cma_003_weight_normalization() {
    for dim in [2, 5, 10, 20] {
        let cma = CmaEs::new(dim);
        let sum: f64 = cma.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "FALSIFIED CMA-003: sum(weights) = {sum} for dim={dim}, expected 1.0"
        );
        // All weights should be positive
        for (i, &w) in cma.weights.iter().enumerate() {
            assert!(
                w > 0.0,
                "FALSIFIED CMA-003: weight[{i}] = {w} <= 0 for dim={dim}"
            );
        }
    }
}

/// FALSIFY-CMA-004: Covariance symmetry — trivially true for diagonal
///
/// Diagonal covariance is always symmetric (C = diag(c) = C^T).
/// We verify the invariant holds after optimization.
#[test]
fn falsify_cma_004_covariance_symmetry() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut cma = CmaEs::new(5).with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    cma.optimize(&sphere, &space, Budget::Evaluations(3000));

    // For diagonal representation, symmetry is trivial.
    // Verify the diagonal structure is maintained (dim elements).
    assert_eq!(
        cma.c_diag.len(),
        5,
        "FALSIFIED CMA-004: c_diag length {} != dim 5",
        cma.c_diag.len()
    );
    // Verify all finite (a broken update could produce NaN)
    for (i, &c) in cma.c_diag.iter().enumerate() {
        assert!(
            c.is_finite(),
            "FALSIFIED CMA-004: c_diag[{i}] = {c} (not finite, symmetry meaningless)"
        );
    }
}

/// FALSIFY-CMA-006: Boundary — dimension 1 reduces to (1+1)-ES behavior
///
/// In 1D, CMA-ES should still converge on a simple quadratic.
#[test]
fn falsify_cma_006_dimension_one_boundary() {
    let quadratic = |x: &[f64]| x[0] * x[0];
    let mut cma = CmaEs::new(1).with_seed(42);
    let space = SearchSpace::continuous(1, -10.0, 10.0);
    let result = cma.optimize(&quadratic, &space, Budget::Evaluations(2000));

    assert!(
        result.objective_value < 0.1,
        "FALSIFIED CMA-006: dim=1 objective = {} >= 0.1 (should converge on quadratic)",
        result.objective_value
    );
    assert!(
        cma.sigma > 0.0,
        "FALSIFIED CMA-006: sigma = {} <= 0 for dim=1",
        cma.sigma
    );
    assert_eq!(
        cma.c_diag.len(),
        1,
        "FALSIFIED CMA-006: c_diag.len() = {} for dim=1",
        cma.c_diag.len()
    );
    assert!(
        cma.c_diag[0] > 0.0,
        "FALSIFIED CMA-006: c_diag[0] = {} <= 0 for dim=1",
        cma.c_diag[0]
    );
}

mod cma_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CMA-001-prop: Step size positive for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_cma_001_prop_step_size_positive(
            dim in 2..=10usize,
            seed in 0..500u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut cma = CmaEs::new(dim).with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            cma.optimize(&sphere, &space, Budget::Evaluations(1000));

            prop_assert!(
                cma.sigma > 0.0,
                "FALSIFIED CMA-001-prop: sigma={} <= 0 for dim={}, seed={}",
                cma.sigma, dim, seed
            );
        }
    }

    /// FALSIFY-CMA-003-prop: Weight normalization for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn falsify_cma_003_prop_weight_normalization(
            dim in 1..=30usize,
        ) {
            let cma = CmaEs::new(dim);
            let sum: f64 = cma.weights.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-10,
                "FALSIFIED CMA-003-prop: sum(weights)={} for dim={}",
                sum, dim
            );
            for (i, &w) in cma.weights.iter().enumerate() {
                prop_assert!(
                    w > 0.0,
                    "FALSIFIED CMA-003-prop: weight[{}]={} <= 0 for dim={}",
                    i, w, dim
                );
            }
        }
    }

    /// FALSIFY-CMA-006-prop: Dimension 1 convergence for random seeds
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_cma_006_prop_dim1_convergence(
            seed in 0..500u64,
        ) {
            let quadratic = |x: &[f64]| -> f64 { x[0] * x[0] };
            let mut cma = CmaEs::new(1).with_seed(seed);
            let space = SearchSpace::continuous(1, -10.0, 10.0);
            let result = cma.optimize(&quadratic, &space, Budget::Evaluations(2000));

            prop_assert!(
                result.objective_value < 1.0,
                "FALSIFIED CMA-006-prop: dim=1 obj={} for seed={}",
                result.objective_value, seed
            );
            prop_assert!(
                cma.sigma > 0.0,
                "FALSIFIED CMA-006-prop: sigma={} <= 0 for seed={}",
                cma.sigma, seed
            );
        }
    }
}
