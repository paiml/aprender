// =========================================================================
// FALSIFY-MH (GA): metaheuristics-v1.yaml contract (aprender GA)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MH-* tests for GA
//   Why 2: GA tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metaheuristics-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: GA was "obviously correct" (selection/crossover/mutation)
//
// References:
//   - provable-contracts/contracts/metaheuristics-v1.yaml
//   - Deb & Agrawal (1995) "Simulated Binary Crossover"
// =========================================================================

use super::*;

/// FALSIFY-MH-007: GA finds near-optimal on sphere function
#[test]
fn falsify_mh_007_ga_sphere_convergence() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut ga = GeneticAlgorithm::default()
        .with_seed(42)
        .with_population_size(50);
    let space = SearchSpace::continuous(2, -5.0, 5.0);
    let result = ga.optimize(&sphere, &space, Budget::Evaluations(5000));

    assert!(
        result.objective_value < 5.0,
        "FALSIFIED MH-007: GA sphere objective {} >= 5.0",
        result.objective_value
    );
}

/// FALSIFY-MH-008: GA solution dimension matches search space
#[test]
fn falsify_mh_008_ga_solution_dimension() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut ga = GeneticAlgorithm::default()
        .with_seed(42)
        .with_population_size(30);
    let space = SearchSpace::continuous(4, -5.0, 5.0);
    let result = ga.optimize(&sphere, &space, Budget::Evaluations(2000));

    assert_eq!(
        result.solution.len(),
        4,
        "FALSIFIED MH-008: GA solution dim {} != 4",
        result.solution.len()
    );
}

/// FALSIFY-MH-009: GA objective is finite
#[test]
fn falsify_mh_009_ga_finite_objective() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut ga = GeneticAlgorithm::default()
        .with_seed(42)
        .with_population_size(30);
    let space = SearchSpace::continuous(3, -5.0, 5.0);
    let result = ga.optimize(&sphere, &space, Budget::Evaluations(2000));

    assert!(
        result.objective_value.is_finite(),
        "FALSIFIED MH-009: GA objective is not finite"
    );
}

mod ga_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MH-008-prop: GA solution dimension matches for random dims
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_008_prop_solution_dimension(
            dim in 1..=5usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut ga = GeneticAlgorithm::default()
                .with_seed(seed)
                .with_population_size(30);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = ga.optimize(&sphere, &space, Budget::Evaluations(1000));

            prop_assert_eq!(
                result.solution.len(),
                dim,
                "FALSIFIED MH-008-prop: GA solution dim {} != {}",
                result.solution.len(), dim
            );
        }
    }

    /// FALSIFY-MH-009-prop: GA objective finite for random seeds
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_009_prop_finite_objective(
            dim in 2..=4usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut ga = GeneticAlgorithm::default()
                .with_seed(seed)
                .with_population_size(30);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = ga.optimize(&sphere, &space, Budget::Evaluations(1000));

            prop_assert!(
                result.objective_value.is_finite(),
                "FALSIFIED MH-009-prop: GA objective not finite (seed={}, dim={})",
                seed, dim
            );
        }
    }
}
