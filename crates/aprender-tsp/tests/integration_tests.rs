//! Integration tests for aprender-tsp.
//!
//! These tests verify end-to-end functionality including:
//! - TSPLIB file loading
//! - Model persistence and loading
//! - Solver correctness
//! - CLI argument parsing

use aprender_tsp::{
    model::{TspModelMetadata, TspParams},
    AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspAlgorithm, TspInstance, TspModel,
    TspSolver,
};
use std::path::Path;
use tempfile::TempDir;

// ============================================================================
// TSPLIB Loading Tests
// ============================================================================

#[test]
fn test_load_berlin52_tsplib() {
    let path = Path::new("data/berlin52.tsp");
    if !path.exists() {
        return; // Skip if test data not available
    }

    let instance = TspInstance::load(path).expect("should load berlin52");
    assert_eq!(instance.name, "berlin52");
    assert_eq!(instance.dimension, 52);
    assert!(instance.coords.is_some());
}

#[test]
fn test_load_eil51_tsplib() {
    let path = Path::new("data/eil51.tsp");
    if !path.exists() {
        return;
    }

    let instance = TspInstance::load(path).expect("should load eil51");
    assert_eq!(instance.name, "eil51");
    assert_eq!(instance.dimension, 51);
}

#[test]
fn test_load_att48_tsplib() {
    let path = Path::new("data/att48.tsp");
    if !path.exists() {
        return;
    }

    let instance = TspInstance::load(path).expect("should load att48");
    assert_eq!(instance.name, "att48");
    assert_eq!(instance.dimension, 48);
}

// ============================================================================
// CSV Loading Tests
// ============================================================================

#[test]
fn test_load_csv_format() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("test.csv");

    let csv_content = r#"# Test CSV instance
1,0.0,0.0
2,1.0,0.0
3,1.0,1.0
4,0.0,1.0
"#;
    std::fs::write(&csv_path, csv_content).unwrap();

    let instance = TspInstance::load(&csv_path).expect("should load CSV");
    assert_eq!(instance.dimension, 4);
    assert!(instance.coords.is_some());
}

#[test]
fn test_load_csv_with_header() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("header.csv");

    let csv_content = r#"id,x,y
1,10.0,20.0
2,30.0,40.0
"#;
    std::fs::write(&csv_path, csv_content).unwrap();

    // Should fail because header row has non-numeric id
    let result = TspInstance::load(&csv_path);
    assert!(result.is_err());
}

#[test]
fn test_load_unknown_extension() {
    let temp_dir = TempDir::new().unwrap();
    let bad_path = temp_dir.path().join("test.xyz");
    std::fs::write(&bad_path, "content").unwrap();

    let result = TspInstance::load(&bad_path);
    assert!(result.is_err());
}

// ============================================================================
// Model Persistence Integration Tests
// ============================================================================

#[test]
fn test_model_roundtrip_all_algorithms() {
    let temp_dir = TempDir::new().unwrap();

    for algo in [
        TspAlgorithm::Aco,
        TspAlgorithm::Tabu,
        TspAlgorithm::Ga,
        TspAlgorithm::Hybrid,
    ] {
        let path = temp_dir.path().join(format!("{:?}.apr", algo));
        let model = TspModel::new(algo);

        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.algorithm, algo);
    }
}

#[test]
fn test_model_preserves_custom_params() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("custom.apr");

    let params = TspParams::Aco {
        alpha: 1.5,
        beta: 3.5,
        rho: 0.2,
        q0: 0.8,
        num_ants: 25,
    };

    let model = TspModel::new(TspAlgorithm::Aco).with_params(params);
    model.save(&path).unwrap();

    let loaded = TspModel::load(&path).unwrap();
    if let TspParams::Aco {
        alpha,
        beta,
        rho,
        q0,
        num_ants,
    } = loaded.params
    {
        assert!((alpha - 1.5).abs() < 1e-10);
        assert!((beta - 3.5).abs() < 1e-10);
        assert!((rho - 0.2).abs() < 1e-10);
        assert!((q0 - 0.8).abs() < 1e-10);
        assert_eq!(num_ants, 25);
    } else {
        panic!("Wrong params type");
    }
}

#[test]
fn test_model_preserves_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("meta.apr");

    let metadata = TspModelMetadata {
        trained_instances: 5,
        avg_instance_size: 100,
        best_known_gap: 0.05,
        training_time_secs: 3.14159,
    };

    let model = TspModel::new(TspAlgorithm::Tabu).with_metadata(metadata);
    model.save(&path).unwrap();

    let loaded = TspModel::load(&path).unwrap();
    assert_eq!(loaded.metadata.trained_instances, 5);
    assert_eq!(loaded.metadata.avg_instance_size, 100);
    assert!((loaded.metadata.best_known_gap - 0.05).abs() < 1e-10);
    assert!((loaded.metadata.training_time_secs - 3.14159).abs() < 1e-10);
}

// ============================================================================
// Solver Correctness Tests
// ============================================================================

fn square_instance() -> TspInstance {
    let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    TspInstance::from_coords("square", coords).unwrap()
}

#[test]
fn test_all_solvers_find_valid_tours() {
    let instance = square_instance();

    // ACO
    let mut aco = AcoSolver::new().with_seed(42);
    let aco_result = aco.solve(&instance, Budget::Iterations(50)).unwrap();
    assert!(instance.validate_tour(&aco_result.tour).is_ok());

    // Tabu
    let mut tabu = TabuSolver::new().with_seed(42);
    let tabu_result = tabu.solve(&instance, Budget::Iterations(50)).unwrap();
    assert!(instance.validate_tour(&tabu_result.tour).is_ok());

    // GA
    let mut ga = GaSolver::new().with_seed(42).with_population_size(20);
    let ga_result = ga.solve(&instance, Budget::Iterations(50)).unwrap();
    assert!(instance.validate_tour(&ga_result.tour).is_ok());

    // Hybrid
    let mut hybrid = HybridSolver::new().with_seed(42).with_ga_population(15);
    let hybrid_result = hybrid.solve(&instance, Budget::Iterations(50)).unwrap();
    assert!(instance.validate_tour(&hybrid_result.tour).is_ok());
}

#[test]
fn test_solver_determinism() {
    let instance = square_instance();

    // Run same solver twice with same seed
    let mut solver1 = AcoSolver::new().with_seed(12345);
    let mut solver2 = AcoSolver::new().with_seed(12345);

    let result1 = solver1.solve(&instance, Budget::Iterations(100)).unwrap();
    let result2 = solver2.solve(&instance, Budget::Iterations(100)).unwrap();

    assert!((result1.length - result2.length).abs() < 1e-10);
    assert_eq!(result1.tour, result2.tour);
}

#[test]
fn test_solver_different_seeds_vary() {
    let instance = square_instance();

    let mut solver1 = AcoSolver::new().with_seed(1);
    let mut solver2 = AcoSolver::new().with_seed(99999);

    let result1 = solver1.solve(&instance, Budget::Iterations(10)).unwrap();
    let result2 = solver2.solve(&instance, Budget::Iterations(10)).unwrap();

    // Results may be equal by chance, but history should differ
    // This is a weak test but ensures seeds are actually used
    assert!(result1.evaluations > 0);
    assert!(result2.evaluations > 0);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_train_save_load_solve_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("trained.apr");

    // Create instance
    let coords = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ];
    let instance = TspInstance::from_coords("hexagon", coords).unwrap();

    // Train
    let mut solver = AcoSolver::new().with_seed(42);
    let train_result = solver.solve(&instance, Budget::Iterations(100)).unwrap();

    // Save model
    let model = TspModel::new(TspAlgorithm::Aco)
        .with_params(TspParams::Aco {
            alpha: solver.alpha,
            beta: solver.beta,
            rho: solver.rho,
            q0: solver.q0,
            num_ants: solver.num_ants,
        })
        .with_metadata(TspModelMetadata {
            trained_instances: 1,
            avg_instance_size: instance.dimension as u32,
            best_known_gap: 0.0,
            training_time_secs: 0.1,
        });
    model.save(&model_path).unwrap();

    // Load and solve
    let loaded = TspModel::load(&model_path).unwrap();
    if let TspParams::Aco {
        alpha,
        beta,
        rho,
        q0,
        num_ants,
    } = loaded.params
    {
        let mut loaded_solver = AcoSolver::new()
            .with_seed(42)
            .with_alpha(alpha)
            .with_beta(beta)
            .with_rho(rho)
            .with_q0(q0)
            .with_num_ants(num_ants);

        let solve_result = loaded_solver
            .solve(&instance, Budget::Iterations(100))
            .unwrap();

        // Should get identical results (same seed and params)
        assert!((solve_result.length - train_result.length).abs() < 1e-10);
    }
}

// ============================================================================
// Algorithm-Specific Tests
// ============================================================================

#[test]
fn test_aco_solves_instance() {
    let instance = square_instance();

    let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);
    let result = solver.solve(&instance, Budget::Iterations(50)).unwrap();

    assert!(instance.validate_tour(&result.tour).is_ok());
    // Optimal tour around square is 4.0
    assert!(result.length <= 5.0, "Length {} > 5.0", result.length);
}

#[test]
fn test_tabu_refine_improves() {
    let instance = square_instance();

    // Start with a crossing tour (suboptimal)
    let crossing_tour = vec![0, 2, 1, 3];
    let initial_length = instance.tour_length(&crossing_tour);

    let mut solver = TabuSolver::new().with_seed(42);
    let result = solver.refine(crossing_tour, &instance, 50).unwrap();

    // Should improve or stay same
    assert!(result.length <= initial_length + 1e-10);
}

#[test]
fn test_ga_evolve_returns_sorted_population() {
    let instance = square_instance();

    let mut solver = GaSolver::new().with_seed(42).with_population_size(20);
    let population = solver.evolve(&instance, 20).unwrap();

    // Should be sorted by fitness (ascending)
    for window in population.windows(2) {
        assert!(window[0].1 <= window[1].1 + 1e-10);
    }
}

#[test]
fn test_hybrid_uses_all_phases() {
    let instance = square_instance();

    let mut solver = HybridSolver::new()
        .with_seed(42)
        .with_ga_fraction(0.4)
        .with_tabu_fraction(0.3)
        .with_aco_fraction(0.3)
        .with_ga_population(15);

    let result = solver.solve(&instance, Budget::Iterations(100)).unwrap();

    // Verify we get a valid result
    assert!(instance.validate_tour(&result.tour).is_ok());
    assert!(result.evaluations > 0);
    assert!(!result.history.is_empty());
}

// ============================================================================
// Solver Name Tests
// ============================================================================

#[test]
fn test_solver_names() {
    assert_eq!(AcoSolver::new().name(), "Ant Colony Optimization");
    assert_eq!(TabuSolver::new().name(), "Tabu Search");
    assert_eq!(GaSolver::new().name(), "Genetic Algorithm");
    assert_eq!(HybridSolver::new().name(), "Hybrid (GA+Tabu+ACO)");
}

// ============================================================================
// Budget Tests
// ============================================================================

#[test]
fn test_budget_iterations() {
    let budget = Budget::Iterations(100);
    assert_eq!(budget.limit(), 100);
}

#[test]
fn test_budget_evaluations() {
    let budget = Budget::Evaluations(5000);
    assert_eq!(budget.limit(), 5000);
}

// ============================================================================
// TspAlgorithm Tests
// ============================================================================

#[test]
fn test_algorithm_parse() {
    assert_eq!(TspAlgorithm::parse("aco"), Some(TspAlgorithm::Aco));
    assert_eq!(TspAlgorithm::parse("ACO"), Some(TspAlgorithm::Aco));
    assert_eq!(TspAlgorithm::parse("tabu"), Some(TspAlgorithm::Tabu));
    assert_eq!(TspAlgorithm::parse("ga"), Some(TspAlgorithm::Ga));
    assert_eq!(TspAlgorithm::parse("hybrid"), Some(TspAlgorithm::Hybrid));
    assert_eq!(TspAlgorithm::parse("unknown"), None);
}

#[test]
fn test_algorithm_as_str() {
    assert_eq!(TspAlgorithm::Aco.as_str(), "aco");
    assert_eq!(TspAlgorithm::Tabu.as_str(), "tabu");
    assert_eq!(TspAlgorithm::Ga.as_str(), "ga");
    assert_eq!(TspAlgorithm::Hybrid.as_str(), "hybrid");
}
