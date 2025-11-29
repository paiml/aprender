//! Property-based tests for aprender-tsp.
//!
//! Uses proptest to verify invariants across many random inputs.

use aprender_tsp::{
    model::{TspModelMetadata, TspParams},
    AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspAlgorithm, TspInstance, TspModel,
    TspSolver,
};
use proptest::prelude::*;
use tempfile::TempDir;

// ============================================================================
// Instance Generation Strategies
// ============================================================================

/// Generate random coordinates for a TSP instance
fn random_coords(n: usize) -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((0.0..100.0f64, 0.0..100.0f64), n)
}

/// Generate a random instance with 3-20 cities
fn random_instance() -> impl Strategy<Value = TspInstance> {
    (3usize..20)
        .prop_flat_map(|n| random_coords(n))
        .prop_map(|coords| TspInstance::from_coords("random", coords).unwrap())
}

// ============================================================================
// Model Persistence Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_model_roundtrip_preserves_aco_params(
        alpha in 0.1..5.0f64,
        beta in 0.1..5.0f64,
        rho in 0.01..0.5f64,
        q0 in 0.5..1.0f64,
        num_ants in 5usize..50
    ) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let params = TspParams::Aco { alpha, beta, rho, q0, num_ants };
        let model = TspModel::new(TspAlgorithm::Aco).with_params(params);

        model.save(&path).unwrap();
        let loaded = TspModel::load(&path).unwrap();

        if let TspParams::Aco {
            alpha: a,
            beta: b,
            rho: r,
            q0: q,
            num_ants: n
        } = loaded.params {
            prop_assert!((a - alpha).abs() < 1e-10);
            prop_assert!((b - beta).abs() < 1e-10);
            prop_assert!((r - rho).abs() < 1e-10);
            prop_assert!((q - q0).abs() < 1e-10);
            prop_assert_eq!(n, num_ants);
        } else {
            prop_assert!(false, "Wrong params type");
        }
    }

    #[test]
    fn prop_model_roundtrip_preserves_tabu_params(
        tenure in 5usize..100,
        max_neighbors in 10usize..500
    ) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let params = TspParams::Tabu { tenure, max_neighbors };
        let model = TspModel::new(TspAlgorithm::Tabu).with_params(params);

        model.save(&path).unwrap();
        let loaded = TspModel::load(&path).unwrap();

        if let TspParams::Tabu { tenure: t, max_neighbors: m } = loaded.params {
            prop_assert_eq!(t, tenure);
            prop_assert_eq!(m, max_neighbors);
        } else {
            prop_assert!(false, "Wrong params type");
        }
    }

    #[test]
    fn prop_model_roundtrip_preserves_ga_params(
        population_size in 10usize..200,
        crossover_rate in 0.5..1.0f64,
        mutation_rate in 0.01..0.5f64
    ) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let params = TspParams::Ga { population_size, crossover_rate, mutation_rate };
        let model = TspModel::new(TspAlgorithm::Ga).with_params(params);

        model.save(&path).unwrap();
        let loaded = TspModel::load(&path).unwrap();

        if let TspParams::Ga { population_size: p, crossover_rate: c, mutation_rate: m } = loaded.params {
            prop_assert_eq!(p, population_size);
            prop_assert!((c - crossover_rate).abs() < 1e-10);
            prop_assert!((m - mutation_rate).abs() < 1e-10);
        } else {
            prop_assert!(false, "Wrong params type");
        }
    }

    #[test]
    fn prop_model_roundtrip_preserves_hybrid_params(
        ga_frac in 0.0..1.0f64,
        tabu_frac in 0.0..1.0f64,
        aco_frac in 0.0..1.0f64
    ) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let params = TspParams::Hybrid {
            ga_fraction: ga_frac,
            tabu_fraction: tabu_frac,
            aco_fraction: aco_frac
        };
        let model = TspModel::new(TspAlgorithm::Hybrid).with_params(params);

        model.save(&path).unwrap();
        let loaded = TspModel::load(&path).unwrap();

        if let TspParams::Hybrid { ga_fraction, tabu_fraction, aco_fraction } = loaded.params {
            prop_assert!((ga_fraction - ga_frac).abs() < 1e-10);
            prop_assert!((tabu_fraction - tabu_frac).abs() < 1e-10);
            prop_assert!((aco_fraction - aco_frac).abs() < 1e-10);
        } else {
            prop_assert!(false, "Wrong params type");
        }
    }

    #[test]
    fn prop_model_roundtrip_preserves_metadata(
        instances in 0u32..1000,
        avg_size in 0u32..1000,
        gap in 0.0..100.0f64,
        time in 0.0..10000.0f64
    ) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let metadata = TspModelMetadata {
            trained_instances: instances,
            avg_instance_size: avg_size,
            best_known_gap: gap,
            training_time_secs: time,
        };

        let model = TspModel::new(TspAlgorithm::Aco).with_metadata(metadata);
        model.save(&path).unwrap();
        let loaded = TspModel::load(&path).unwrap();

        prop_assert_eq!(loaded.metadata.trained_instances, instances);
        prop_assert_eq!(loaded.metadata.avg_instance_size, avg_size);
        prop_assert!((loaded.metadata.best_known_gap - gap).abs() < 1e-10);
        prop_assert!((loaded.metadata.training_time_secs - time).abs() < 1e-10);
    }
}

// ============================================================================
// Instance Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_distance_symmetric(
        instance in random_instance()
    ) {
        for i in 0..instance.num_cities() {
            for j in 0..instance.num_cities() {
                prop_assert!((instance.distance(i, j) - instance.distance(j, i)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_distance_non_negative(
        instance in random_instance()
    ) {
        for i in 0..instance.num_cities() {
            for j in 0..instance.num_cities() {
                prop_assert!(instance.distance(i, j) >= 0.0);
            }
        }
    }

    #[test]
    fn prop_distance_self_zero(
        instance in random_instance()
    ) {
        for i in 0..instance.num_cities() {
            prop_assert!((instance.distance(i, i) - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn prop_tour_length_positive(
        instance in random_instance()
    ) {
        let tour: Vec<usize> = (0..instance.num_cities()).collect();
        let length = instance.tour_length(&tour);
        prop_assert!(length >= 0.0);
    }
}

// ============================================================================
// Solver Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_aco_produces_valid_tour(
        seed in 0u64..10000
    ) {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let instance = TspInstance::from_coords("test", coords).unwrap();

        let mut solver = AcoSolver::new().with_seed(seed);
        let result = solver.solve(&instance, Budget::Iterations(30)).unwrap();

        prop_assert!(instance.validate_tour(&result.tour).is_ok());
    }

    #[test]
    fn prop_tabu_produces_valid_tour(
        seed in 0u64..10000
    ) {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let instance = TspInstance::from_coords("test", coords).unwrap();

        let mut solver = TabuSolver::new().with_seed(seed);
        let result = solver.solve(&instance, Budget::Iterations(30)).unwrap();

        prop_assert!(instance.validate_tour(&result.tour).is_ok());
    }

    #[test]
    fn prop_ga_produces_valid_tour(
        seed in 0u64..10000
    ) {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let instance = TspInstance::from_coords("test", coords).unwrap();

        let mut solver = GaSolver::new().with_seed(seed).with_population_size(20);
        let result = solver.solve(&instance, Budget::Iterations(30)).unwrap();

        prop_assert!(instance.validate_tour(&result.tour).is_ok());
    }

    #[test]
    fn prop_hybrid_produces_valid_tour(
        seed in 0u64..10000
    ) {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let instance = TspInstance::from_coords("test", coords).unwrap();

        let mut solver = HybridSolver::new().with_seed(seed).with_ga_population(10);
        let result = solver.solve(&instance, Budget::Iterations(30)).unwrap();

        prop_assert!(instance.validate_tour(&result.tour).is_ok());
    }

    #[test]
    fn prop_solver_deterministic_with_same_seed(
        seed in 0u64..10000
    ) {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let instance = TspInstance::from_coords("test", coords).unwrap();

        let mut solver1 = AcoSolver::new().with_seed(seed);
        let mut solver2 = AcoSolver::new().with_seed(seed);

        let result1 = solver1.solve(&instance, Budget::Iterations(50)).unwrap();
        let result2 = solver2.solve(&instance, Budget::Iterations(50)).unwrap();

        prop_assert!((result1.length - result2.length).abs() < 1e-10);
    }
}

// ============================================================================
// Distance Matrix Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_from_matrix_roundtrip(
        size in 3usize..10
    ) {
        // Create symmetric distance matrix
        let mut matrix = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in i+1..size {
                let d = ((i + j) as f64) * 1.5;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }

        let instance = TspInstance::from_matrix("test", matrix.clone()).unwrap();

        prop_assert_eq!(instance.dimension, size);
        for i in 0..size {
            for j in 0..size {
                prop_assert!((instance.distance(i, j) - matrix[i][j]).abs() < 1e-10);
            }
        }
    }
}
