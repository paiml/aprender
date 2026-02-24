// =========================================================================
// FALSIFY-MH (DE): metaheuristics-v1.yaml contract (aprender DE)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MH-* tests for DE
//   Why 2: DE tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metaheuristics-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: DE was "obviously correct" (mutation/crossover/selection)
//
// References:
//   - provable-contracts/contracts/metaheuristics-v1.yaml
//   - Storn & Price (1997) "Differential Evolution"
// =========================================================================

use super::*;

/// FALSIFY-MH-010: DE finds near-optimal on sphere function
#[test]
fn falsify_mh_010_de_sphere_convergence() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(2, -5.0, 5.0);
    let result = de.optimize(&sphere, &space, Budget::Evaluations(5000));

    assert!(
        result.objective_value < 1.0,
        "FALSIFIED MH-010: DE sphere objective {} >= 1.0",
        result.objective_value
    );
}

/// FALSIFY-MH-011: DE solution stays within bounds
#[test]
fn falsify_mh_011_de_within_bounds() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(3, -3.0, 3.0);
    let result = de.optimize(&sphere, &space, Budget::Evaluations(3000));

    for (i, &v) in result.solution.iter().enumerate() {
        assert!(
            (-3.0..=3.0).contains(&v),
            "FALSIFIED MH-011: DE solution[{i}]={v} outside bounds [-3, 3]"
        );
    }
}

/// FALSIFY-MH-012: DE solution dimension matches search space
#[test]
fn falsify_mh_012_de_solution_dimension() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    let result = de.optimize(&sphere, &space, Budget::Evaluations(3000));

    assert_eq!(
        result.solution.len(),
        5,
        "FALSIFIED MH-012: DE solution dim {} != 5",
        result.solution.len()
    );
}

mod de_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MH-011-prop: DE solution within bounds for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_011_prop_within_bounds(
            dim in 2..=5usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut de = DifferentialEvolution::new().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = de.optimize(&sphere, &space, Budget::Evaluations(2000));

            for (i, &v) in result.solution.iter().enumerate() {
                prop_assert!(
                    (-5.0..=5.0).contains(&v),
                    "FALSIFIED MH-011-prop: solution[{}]={} outside [-5,5]",
                    i, v
                );
            }
        }
    }

    /// FALSIFY-MH-012-prop: Solution dimension matches for random dims
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_012_prop_solution_dimension(
            dim in 1..=6usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut de = DifferentialEvolution::new().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = de.optimize(&sphere, &space, Budget::Evaluations(1000));

            prop_assert_eq!(
                result.solution.len(),
                dim,
                "FALSIFIED MH-012-prop: solution dim {} != {}",
                result.solution.len(), dim
            );
        }
    }
}
