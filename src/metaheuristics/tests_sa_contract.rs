// =========================================================================
// FALSIFY-MH (SA): metaheuristics-v1.yaml contract (aprender SA)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MH-* tests for SA
//   Why 2: SA tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metaheuristics-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: SA was "obviously correct" (Metropolis criterion well-studied)
//
// References:
//   - provable-contracts/contracts/metaheuristics-v1.yaml
//   - Kirkpatrick et al. (1983) "Optimization by Simulated Annealing"
// =========================================================================

use super::*;

/// FALSIFY-MH-004: SA finds near-optimal on sphere function
#[test]
fn falsify_mh_004_sa_sphere_convergence() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut sa = SimulatedAnnealing::default().with_seed(42);
    let space = SearchSpace::continuous(2, -5.0, 5.0);
    let result = sa.optimize(&sphere, &space, Budget::Evaluations(10000));

    assert!(
        result.objective_value < 1.0,
        "FALSIFIED MH-004: SA sphere objective {} >= 1.0",
        result.objective_value
    );
}

/// FALSIFY-MH-005: SA solution stays within bounds
#[test]
fn falsify_mh_005_sa_within_bounds() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut sa = SimulatedAnnealing::default().with_seed(42);
    let space = SearchSpace::continuous(3, -3.0, 3.0);
    let result = sa.optimize(&sphere, &space, Budget::Evaluations(5000));

    for (i, &v) in result.solution.iter().enumerate() {
        assert!(
            (-3.0..=3.0).contains(&v),
            "FALSIFIED MH-005: solution[{i}]={v} outside bounds [-3, 3]"
        );
    }
}

/// FALSIFY-MH-006: SA objective value is finite
#[test]
fn falsify_mh_006_sa_finite_objective() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut sa = SimulatedAnnealing::default().with_seed(42);
    let space = SearchSpace::continuous(2, -5.0, 5.0);
    let result = sa.optimize(&sphere, &space, Budget::Evaluations(2000));

    assert!(
        result.objective_value.is_finite(),
        "FALSIFIED MH-006: SA objective is not finite"
    );
    for (i, &v) in result.solution.iter().enumerate() {
        assert!(
            v.is_finite(),
            "FALSIFIED MH-006: solution[{i}] is not finite"
        );
    }
}

mod sa_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MH-005-prop: SA solution within bounds for random dims/seeds
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_005_prop_within_bounds(
            dim in 2..=4usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut sa = SimulatedAnnealing::default().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = sa.optimize(&sphere, &space, Budget::Evaluations(5000));

            for (i, &v) in result.solution.iter().enumerate() {
                prop_assert!(
                    (-5.0..=5.0).contains(&v),
                    "FALSIFIED MH-005-prop: SA solution[{}]={} outside [-5,5]",
                    i, v
                );
            }
        }
    }

    /// FALSIFY-MH-006-prop: SA objective and solution finite for random seeds
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_006_prop_finite(
            dim in 2..=4usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut sa = SimulatedAnnealing::default().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = sa.optimize(&sphere, &space, Budget::Evaluations(2000));

            prop_assert!(
                result.objective_value.is_finite(),
                "FALSIFIED MH-006-prop: SA objective not finite (seed={}, dim={})",
                seed, dim
            );
            for (i, &v) in result.solution.iter().enumerate() {
                prop_assert!(
                    v.is_finite(),
                    "FALSIFIED MH-006-prop: SA solution[{}] not finite",
                    i
                );
            }
        }
    }
}
