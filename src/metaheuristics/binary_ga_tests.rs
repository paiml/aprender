use super::*;

#[test]
fn test_binary_ga_feature_selection() {
    // Objective: minimize selected features while keeping at least one
    let objective = |bits: &[f64]| {
        let count = bits.iter().filter(|&&b| b > 0.5).count();
        if count == 0 {
            100.0 // Penalty for no features
        } else {
            count as f64
        }
    };

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(50);
    let space = SearchSpace::binary(10);
    let result = ga.optimize(&objective, &space, Budget::Evaluations(2000));

    // Should select minimal features (ideally 1)
    let selected = BinaryGA::selected_features(&result.solution);
    assert!(!selected.is_empty(), "Should select at least one feature");
    assert!(
        result.objective_value < 5.0,
        "Should minimize features, got: {}",
        result.objective_value
    );
}

#[test]
fn test_binary_ga_onemax() {
    // OneMax: maximize number of 1s (minimize negative count)
    let objective = |bits: &[f64]| {
        let ones = bits.iter().filter(|&&b| b > 0.5).count();
        -(ones as f64) // Negate for minimization
    };

    let mut ga = BinaryGA::default().with_seed(123).with_population_size(100);
    let space = SearchSpace::binary(20);
    let result = ga.optimize(&objective, &space, Budget::Evaluations(5000));

    // Should find all 1s (objective = -20)
    assert!(
        result.objective_value < -15.0,
        "OneMax should find mostly 1s, got: {}",
        result.objective_value
    );
}

#[test]
fn test_binary_ga_builder() {
    let ga = BinaryGA::default()
        .with_population_size(200)
        .with_mutation_prob(0.05)
        .with_seed(999);
    assert_eq!(ga.population_size, 200);
    assert!((ga.mutation_prob - 0.05).abs() < 1e-10);
}

#[test]
fn test_selected_features_helper() {
    let solution = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let selected = BinaryGA::selected_features(&solution);
    assert_eq!(selected, vec![1, 3, 4]);
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_binary_ga_with_crossover_prob() {
    let ga = BinaryGA::default().with_crossover_prob(0.7);
    assert!((ga.crossover_prob - 0.7).abs() < 1e-10);

    // Test clamping
    let ga2 = BinaryGA::default().with_crossover_prob(1.5);
    assert!((ga2.crossover_prob - 1.0).abs() < 1e-10);

    let ga3 = BinaryGA::default().with_crossover_prob(-0.5);
    assert!((ga3.crossover_prob - 0.0).abs() < 1e-10);
}

#[test]
fn test_binary_ga_mutation_prob_clamping() {
    let ga = BinaryGA::default().with_mutation_prob(2.0);
    assert!((ga.mutation_prob - 1.0).abs() < 1e-10);

    let ga2 = BinaryGA::default().with_mutation_prob(-1.0);
    assert!((ga2.mutation_prob - 0.0).abs() < 1e-10);
}

#[test]
fn test_binary_ga_population_size_min() {
    let ga = BinaryGA::default().with_population_size(1);
    assert_eq!(ga.population_size, 4); // Minimum is 4
}

#[test]
fn test_binary_ga_best_empty() {
    let ga = BinaryGA::default();
    assert!(ga.best().is_none());
}

#[test]
fn test_binary_ga_best_after_optimize() {
    let objective = |bits: &[f64]| bits.iter().sum::<f64>();

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
    let space = SearchSpace::binary(5);
    ga.optimize(&objective, &space, Budget::Evaluations(100));

    assert!(ga.best().is_some());
}

#[test]
fn test_binary_ga_history_after_optimize() {
    let objective = |bits: &[f64]| bits.iter().sum::<f64>();

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
    let space = SearchSpace::binary(5);
    ga.optimize(&objective, &space, Budget::Evaluations(100));

    let history = ga.history();
    assert!(!history.is_empty());
    // History should be monotonically non-increasing (we're minimizing)
    for w in history.windows(2) {
        assert!(
            w[1] <= w[0] + 0.001,
            "History should not increase: {} > {}",
            w[1],
            w[0]
        );
    }
}

#[test]
fn test_binary_ga_reset() {
    let objective = |bits: &[f64]| bits.iter().sum::<f64>();

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
    let space = SearchSpace::binary(5);
    ga.optimize(&objective, &space, Budget::Evaluations(100));

    assert!(ga.best().is_some());
    assert!(!ga.history().is_empty());

    ga.reset();

    assert!(ga.best().is_none());
    assert!(ga.history().is_empty());
}

#[test]
fn test_binary_ga_debug() {
    let ga = BinaryGA::default();
    let debug_str = format!("{:?}", ga);
    assert!(debug_str.contains("BinaryGA"));
}

#[test]
fn test_binary_ga_clone() {
    let ga = BinaryGA::default().with_seed(42).with_population_size(50);
    let cloned = ga.clone();
    assert_eq!(cloned.population_size, 50);
}

#[test]
fn test_binary_ga_termination_converged() {
    // Use a simple objective that converges quickly
    let objective = |_bits: &[f64]| 0.0; // Always returns 0

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
    let space = SearchSpace::binary(5);
    let result = ga.optimize(&objective, &space, Budget::Evaluations(1000));

    // With constant objective, should converge
    assert!(
        result.termination == TerminationReason::Converged
            || result.termination == TerminationReason::BudgetExhausted
    );
}

#[test]
fn test_binary_ga_no_seed() {
    // Test without seed (uses entropy)
    let objective = |bits: &[f64]| bits.iter().sum::<f64>();

    let mut ga = BinaryGA::default().with_population_size(20);
    let space = SearchSpace::binary(5);
    let result = ga.optimize(&objective, &space, Budget::Evaluations(100));

    assert!(result.objective_value.is_finite());
}

#[test]
fn test_binary_ga_continuous_space_compatibility() {
    // BinaryGA can also work with continuous space (treated as binary)
    let objective = |bits: &[f64]| bits.iter().sum::<f64>();

    let mut ga = BinaryGA::default().with_seed(42).with_population_size(20);
    let space = SearchSpace::continuous(5, 0.0, 1.0);
    let result = ga.optimize(&objective, &space, Budget::Evaluations(100));

    assert!(result.objective_value.is_finite());
}

#[test]
fn test_selected_features_all_zeros() {
    let solution = vec![0.0, 0.0, 0.0, 0.0];
    let selected = BinaryGA::selected_features(&solution);
    assert!(selected.is_empty());
}

#[test]
fn test_selected_features_all_ones() {
    let solution = vec![1.0, 1.0, 1.0];
    let selected = BinaryGA::selected_features(&solution);
    assert_eq!(selected, vec![0, 1, 2]);
}
