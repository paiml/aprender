// =========================================================================
// FALSIFY-MH (PSO): metaheuristics-v1.yaml contract (aprender PSO)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MH-* tests for metaheuristics
//   Why 2: metaheuristic tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metaheuristics-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: PSO was "obviously correct" (swarm updates well-studied)
//
// References:
//   - provable-contracts/contracts/metaheuristics-v1.yaml
//   - Kennedy & Eberhart (1995) "Particle Swarm Optimization"
// =========================================================================

use super::*;

/// FALSIFY-MH-001: PSO finds near-optimal on sphere function f(x)=Σx²
#[test]
fn falsify_mh_001_pso_sphere_convergence() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut pso = ParticleSwarm::default().with_seed(42);
    let space = SearchSpace::continuous(2, -5.0, 5.0);
    let result = pso.optimize(&sphere, &space, Budget::Evaluations(5000));

    assert!(
        result.objective_value < 1.0,
        "FALSIFIED MH-001: PSO sphere objective {} >= 1.0",
        result.objective_value
    );
}

/// FALSIFY-MH-002: PSO solution dimension matches search space
#[test]
fn falsify_mh_002_pso_solution_dimension() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut pso = ParticleSwarm::default().with_seed(42);
    let space = SearchSpace::continuous(3, -5.0, 5.0);
    let result = pso.optimize(&sphere, &space, Budget::Evaluations(1000));

    assert_eq!(
        result.solution.len(),
        3,
        "FALSIFIED MH-002: solution dim {} != search space dim 3",
        result.solution.len()
    );
}

/// FALSIFY-MH-003: PSO solution stays within bounds
#[test]
fn falsify_mh_003_pso_within_bounds() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut pso = ParticleSwarm::default().with_seed(42);
    let space = SearchSpace::continuous(3, -2.0, 2.0);
    let result = pso.optimize(&sphere, &space, Budget::Evaluations(2000));

    for (i, &v) in result.solution.iter().enumerate() {
        assert!(
            (-2.0..=2.0).contains(&v),
            "FALSIFIED MH-003: solution[{i}]={v} outside bounds [-2, 2]"
        );
    }
}

mod pso_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-MH-002-prop: PSO solution dimension matches for random dims
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_002_prop_solution_dimension(
            dim in 1..=5usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut pso = ParticleSwarm::default().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = pso.optimize(&sphere, &space, Budget::Evaluations(1000));

            prop_assert_eq!(
                result.solution.len(),
                dim,
                "FALSIFIED MH-002-prop: PSO solution dim {} != {}",
                result.solution.len(), dim
            );
        }
    }

    /// FALSIFY-MH-003-prop: PSO solution within bounds for random dims/seeds
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_mh_003_prop_within_bounds(
            dim in 2..=4usize,
            seed in 0..200u64,
        ) {
            let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
            let mut pso = ParticleSwarm::default().with_seed(seed);
            let space = SearchSpace::continuous(dim, -5.0, 5.0);
            let result = pso.optimize(&sphere, &space, Budget::Evaluations(2000));

            for (i, &v) in result.solution.iter().enumerate() {
                prop_assert!(
                    (-5.0..=5.0).contains(&v),
                    "FALSIFIED MH-003-prop: PSO solution[{}]={} outside [-5,5]",
                    i, v
                );
            }
        }
    }
}
