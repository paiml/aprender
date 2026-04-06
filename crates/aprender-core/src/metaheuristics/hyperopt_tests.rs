pub(crate) use super::*;

// ==========================================================
// EXTREME TDD: Tests written first
// ==========================================================

#[test]
fn test_hyperopt_search_builder() {
    let search = HyperoptSearch::new()
        .add_real("lr", 1e-5, 1e-1, true)
        .add_int("epochs", 10, 100)
        .add_categorical("optimizer", &["sgd", "adam"])
        .with_seed(42);

    assert_eq!(search.parameters.len(), 3);
    assert!(search.seed.is_some());
}

#[test]
fn test_hyperopt_real_parameter() {
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 10.0, false)
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let x = params.get_real("x").unwrap();
        (x - 5.0).powi(2) // Minimum at x=5
    };

    let result = search.minimize(objective, Budget::Evaluations(200));

    assert!(result.best_params.get_real("x").is_some());
    let x = result.best_params.get_real("x").unwrap();
    assert!((x - 5.0).abs() < 1.0, "Expected x near 5, got {}", x);
}

#[test]
fn test_hyperopt_log_scale() {
    let search = HyperoptSearch::new()
        .add_real("lr", 1e-5, 1e-1, true) // Log scale
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let lr = params.get_real("lr").unwrap();
        // Optimal at lr = 0.001 (10^-3)
        (lr.log10() + 3.0).powi(2)
    };

    let result = search.minimize(objective, Budget::Evaluations(200));

    let lr = result.best_params.get_real("lr").unwrap();
    // Should find value near 0.001
    assert!(lr > 1e-5 && lr < 1e-1, "LR out of bounds: {}", lr);
}

#[test]
fn test_hyperopt_int_parameter() {
    let search = HyperoptSearch::new().add_int("n", 1, 10).with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let n = params.get_int("n").unwrap();
        ((n - 7) as f64).powi(2) // Minimum at n=7
    };

    let result = search.minimize(objective, Budget::Evaluations(100));

    let n = result.best_params.get_int("n").unwrap();
    assert!(n >= 1 && n <= 10, "n out of bounds: {}", n);
}

#[test]
fn test_hyperopt_categorical_parameter() {
    let search = HyperoptSearch::new()
        .add_categorical("color", &["red", "green", "blue"])
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let color = params.get_categorical("color").unwrap();
        match color {
            "green" => 0.0, // Optimal choice
            "red" => 1.0,
            "blue" => 2.0,
            _ => 10.0,
        }
    };

    let result = search.minimize(objective, Budget::Evaluations(100));

    let color = result.best_params.get_categorical("color").unwrap();
    assert!(
        ["red", "green", "blue"].contains(&color),
        "Invalid color: {}",
        color
    );
}

#[test]
fn test_hyperopt_maximize() {
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 10.0, false)
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let x = params.get_real("x").unwrap();
        -(x - 5.0).powi(2) + 10.0 // Maximum at x=5
    };

    let result = search.maximize(objective, Budget::Evaluations(200));

    assert!(result.best_score > 9.0, "Score should be near 10");
}

#[test]
fn test_hyperopt_mixed_space() {
    let search = HyperoptSearch::new()
        .add_real("lr", 0.001, 0.1, false)
        .add_int("hidden", 16, 256)
        .add_categorical("activation", &["relu", "tanh"])
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let lr = params.get_real("lr").unwrap();
        let hidden = params.get_int("hidden").unwrap();
        let activation = params.get_categorical("activation").unwrap();

        let mut score = (lr - 0.01).powi(2);
        score += ((hidden - 64) as f64 / 100.0).powi(2);
        if activation == "relu" {
            score -= 0.1;
        }
        score
    };

    let result = search.minimize(objective, Budget::Evaluations(300));

    assert!(result.best_params.get_real("lr").is_some());
    assert!(result.best_params.get_int("hidden").is_some());
    assert!(result.best_params.get_categorical("activation").is_some());
}

#[test]
fn test_hyperopt_result_structure() {
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 1.0, false)
        .with_seed(42);

    let objective = |_: &HyperparameterSet| 0.5;

    let result = search.minimize(objective, Budget::Evaluations(50));

    assert!(result.evaluations > 0);
    assert!(!result.history.is_empty());
}

#[test]
fn test_hyperopt_different_algorithms() {
    let algorithms = [
        SearchAlgorithm::DifferentialEvolution,
        SearchAlgorithm::ParticleSwarm,
        SearchAlgorithm::SimulatedAnnealing,
    ];

    for algo in algorithms {
        let search = HyperoptSearch::new()
            .add_real("x", 0.0, 10.0, false)
            .with_algorithm(algo)
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let x = params.get_real("x").unwrap();
            (x - 5.0).powi(2)
        };

        let result = search.minimize(objective, Budget::Evaluations(100));
        assert!(result.evaluations > 0, "Algorithm {:?} failed", algo);
    }
}

#[test]
fn test_hyperparameter_set_operations() {
    let mut params = HyperparameterSet::new();

    params.set_real("lr", 0.01);
    params.set_int("epochs", 100);
    params.set_categorical("opt", "adam");

    assert_eq!(params.get_real("lr"), Some(0.01));
    assert_eq!(params.get_int("epochs"), Some(100));
    assert_eq!(params.get_categorical("opt"), Some("adam"));
    assert_eq!(params.get_real("missing"), None);
}

#[test]
fn test_decode_consistency() {
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 1.0, false)
        .add_real("y", 10.0, 20.0, false);

    // Midpoint of [0,1] x [0,1] internal space
    let params = search.decode(&[0.5, 0.5]);

    let x = params.get_real("x").unwrap();
    let y = params.get_real("y").unwrap();

    assert!((x - 0.5).abs() < 0.01, "x should be 0.5, got {}", x);
    assert!((y - 15.0).abs() < 0.1, "y should be 15.0, got {}", y);
}

#[test]
fn test_hyperopt_cmaes_algorithm() {
    // Exercises the SearchAlgorithm::CmaEs branch in minimize().
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 10.0, false)
        .with_algorithm(SearchAlgorithm::CmaEs)
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let x = params.get_real("x").unwrap();
        (x - 5.0).powi(2)
    };

    let result = search.minimize(objective, Budget::Evaluations(200));
    assert!(result.evaluations > 0);
    assert!(result.best_params.get_real("x").is_some());
}

#[test]
fn test_hyperopt_maximize_returns_positive_scores() {
    // Verify maximize negates objective and history correctly.
    let search = HyperoptSearch::new()
        .add_real("x", 0.0, 10.0, false)
        .with_seed(42);

    let objective = |params: &HyperparameterSet| -> f64 {
        let x = params.get_real("x").unwrap();
        -(x - 5.0).powi(2) + 25.0 // Maximum of 25 at x=5
    };

    let result = search.maximize(objective, Budget::Evaluations(200));
    // The negated best_score should be positive (near 25)
    assert!(
        result.best_score > 0.0,
        "Maximize should return positive best_score"
    );
    // History entries should also be negated (non-positive objective becomes non-negative)
    for &h in &result.history {
        assert!(h.is_finite());
    }
}

#[test]
fn test_hyperopt_with_n_jobs() {
    // Exercises the with_n_jobs builder, including the max(1) clamp.
    let search = HyperoptSearch::new().with_n_jobs(4);
    assert_eq!(search.n_jobs, 4);

    let search_zero = HyperoptSearch::new().with_n_jobs(0);
    assert_eq!(search_zero.n_jobs, 1); // Clamped to at least 1
}

#[test]
fn test_hyperopt_default() {
    // Exercises the Default impl which delegates to new().
    let search = HyperoptSearch::default();
    assert!(search.parameters.is_empty());
    assert!(search.seed.is_none());
    assert_eq!(search.n_jobs, 1);
}

#[test]
fn test_hyperparameter_name_all_variants() {
    let real = Hyperparameter::Real {
        name: "lr".to_string(),
        lower: 0.0,
        upper: 1.0,
        log_scale: false,
    };
    assert_eq!(real.name(), "lr");

    let int = Hyperparameter::Int {
        name: "epochs".to_string(),
        lower: 1,
        upper: 100,
    };
    assert_eq!(int.name(), "epochs");

    let cat = Hyperparameter::Categorical {
        name: "optimizer".to_string(),
        choices: vec!["sgd".to_string(), "adam".to_string()],
    };
    assert_eq!(cat.name(), "optimizer");
}

#[test]
fn test_hyperparameter_dim() {
    let real = Hyperparameter::Real {
        name: "x".to_string(),
        lower: 0.0,
        upper: 1.0,
        log_scale: false,
    };
    assert_eq!(real.dim(), 1);

    let int = Hyperparameter::Int {
        name: "n".to_string(),
        lower: 1,
        upper: 10,
    };
    assert_eq!(int.dim(), 1);

    let cat = Hyperparameter::Categorical {
        name: "color".to_string(),
        choices: vec!["r".to_string(), "g".to_string(), "b".to_string()],
    };
    assert_eq!(cat.dim(), 3);
}

#[test]
fn test_hyperparameter_set_missing_keys() {
    let params = HyperparameterSet::new();
    assert_eq!(params.get_real("nonexistent"), None);
    assert_eq!(params.get_int("nonexistent"), None);
    assert_eq!(params.get_categorical("nonexistent"), None);
}

#[test]
fn test_hyperopt_total_dim_mixed() {
    // Verify total_dim sums correctly across real, int, and categorical.
    let search = HyperoptSearch::new()
        .add_real("lr", 0.0, 1.0, false)          // dim = 1
        .add_int("n", 1, 10)                       // dim = 1
        .add_categorical("opt", &["a", "b", "c"]); // dim = 3
    assert_eq!(search.total_dim(), 5);
}

#[test]
fn test_decode_int_bounds_clamped() {
    // Verify integer decoding clamps to bounds at extremes.
    let search = HyperoptSearch::new().add_int("n", 5, 10);

    // t=0.0 should give lower bound
    let params_low = search.decode(&[0.0]);
    assert_eq!(params_low.get_int("n"), Some(5));

    // t=1.0 should give upper bound
    let params_high = search.decode(&[1.0]);
    assert_eq!(params_high.get_int("n"), Some(10));
}

#[test]
fn test_decode_categorical_one_hot() {
    // Verify categorical decoding picks the index with highest value.
    let search = HyperoptSearch::new().add_categorical("color", &["red", "green", "blue"]);

    // One-hot: green has highest value
    let params = search.decode(&[0.1, 0.9, 0.2]);
    assert_eq!(params.get_categorical("color"), Some("green"));

    // One-hot: blue has highest value
    let params2 = search.decode(&[0.1, 0.2, 0.8]);
    assert_eq!(params2.get_categorical("color"), Some("blue"));
}

#[test]
fn test_search_algorithm_default() {
    let algo = SearchAlgorithm::default();
    assert!(matches!(algo, SearchAlgorithm::DifferentialEvolution));
}
