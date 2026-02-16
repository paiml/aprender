//\! AutoML Search Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

pub(crate) use crate::automl::params::RandomForestParam as RF;

#[test]
fn test_search_space_builder() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add(RF::NEstimators, 10..500)
        .add(RF::MaxDepth, 2..20);

    assert_eq!(space.len(), 2);
    assert!(space.get(&RF::NEstimators).is_some());
    assert!(space.get(&RF::MaxDepth).is_some());
}

#[test]
fn test_search_space_continuous() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 0.0, 1.0)
        .add_log_scale(RF::MaxDepth, (1e-4..1.0).log_scale());

    assert_eq!(space.len(), 2);
}

#[test]
fn test_search_space_categorical() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_categorical(RF::MaxFeatures, ["sqrt", "log2", "0.5"])
        .add_bool(RF::Bootstrap, [true, false]);

    assert_eq!(space.len(), 2);
}

#[test]
fn test_random_search_deterministic() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add(RF::NEstimators, 10..500)
        .add(RF::MaxDepth, 2..20);

    let mut search1 = RandomSearch::new(10).with_seed(42);
    let mut search2 = RandomSearch::new(10).with_seed(42);

    let trials1 = search1.suggest(&space, 5);
    let trials2 = search2.suggest(&space, 5);

    for (t1, t2) in trials1.iter().zip(trials2.iter()) {
        assert_eq!(
            t1.get(&RF::NEstimators),
            t2.get(&RF::NEstimators),
            "Same seed should produce same results"
        );
    }
}

#[test]
fn test_random_search_respects_budget() {
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..500);

    let mut search = RandomSearch::new(5);

    let trials1 = search.suggest(&space, 3);
    assert_eq!(trials1.len(), 3);
    assert_eq!(search.remaining(), 2);

    let trials2 = search.suggest(&space, 10);
    assert_eq!(trials2.len(), 2);
    assert_eq!(search.remaining(), 0);
}

#[test]
fn test_grid_search_cartesian_product() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_categorical(RF::Bootstrap, [true, false])
        .add_categorical(RF::MaxFeatures, ["sqrt", "log2"]);

    let mut search = GridSearch::new(10);
    let trials = search.suggest(&space, 100);

    // 2 x 2 = 4 combinations
    assert_eq!(trials.len(), 4);
}

#[test]
fn test_trial_accessors() {
    let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 100..101) // Single value: 100
            .add_bool(RF::Bootstrap, [true, false]);

    let mut rng = XorShift64::new(42);
    let trial = space.sample(&mut rng);

    assert_eq!(trial.get_i64(&RF::NEstimators), Some(100));
    assert!(trial.get_bool(&RF::Bootstrap).is_some());
}

#[test]
fn test_param_value_conversions() {
    assert_eq!(ParamValue::from(42_i32).as_i64(), Some(42));
    assert_eq!(ParamValue::from(1.234_f64).as_f64(), Some(1.234));
    assert_eq!(ParamValue::from(true).as_bool(), Some(true));
    assert_eq!(ParamValue::from("test").as_str(), Some("test"));
}

#[test]
fn test_hyperparam_sampling() {
    let mut rng = XorShift64::new(42);

    let continuous = HyperParam::continuous(0.0, 1.0);
    for _ in 0..100 {
        let v = continuous
            .sample(&mut rng)
            .as_f64()
            .expect("continuous param should return float");
        assert!((0.0..=1.0).contains(&v));
    }

    let integer = HyperParam::integer(10, 20);
    for _ in 0..100 {
        let v = integer
            .sample(&mut rng)
            .as_i64()
            .expect("integer param should return int");
        assert!((10..=20).contains(&v));
    }
}

#[test]
fn test_log_scale_sampling() {
    let mut rng = XorShift64::new(42);
    let param = HyperParam::continuous_log(1e-4, 1.0);

    let mut samples = Vec::new();
    for _ in 0..1000 {
        let v = param
            .sample(&mut rng)
            .as_f64()
            .expect("log scale param should return float");
        assert!((1e-4..=1.0).contains(&v));
        samples.push(v);
    }

    // Log scale should have more samples near lower end
    let below_01 = samples.iter().filter(|&&v| v < 0.1).count();
    let above_01 = samples.iter().filter(|&&v| v >= 0.1).count();
    assert!(
        below_01 > above_01 / 2,
        "Log scale should sample more from lower range"
    );
}

#[test]
fn test_grid_points_continuous() {
    let param = HyperParam::continuous(0.0, 1.0);
    let points = param.grid_points(5);

    assert_eq!(points.len(), 5);
    assert_eq!(points[0].as_f64(), Some(0.0));
    assert_eq!(points[4].as_f64(), Some(1.0));
}

#[test]
fn test_grid_points_integer() {
    let param = HyperParam::integer(1, 10);
    let points = param.grid_points(5);

    assert!(points.len() <= 5);
    let first = points[0]
        .as_i64()
        .expect("integer grid point should be int");
    assert!(first >= 1);
}

#[test]
fn test_trial_display() {
    let mut values = HashMap::new();
    values.insert(RF::NEstimators, ParamValue::Int(100));
    values.insert(RF::MaxDepth, ParamValue::Int(5));

    let trial = Trial { values };
    let s = format!("{trial}");

    assert!(s.contains("n_estimators=100"));
    assert!(s.contains("max_depth=5"));
}

#[test]
fn test_xorshift_rng() {
    let mut rng = XorShift64::new(12345);

    // Should produce different values
    let v1 = rng.gen_f64();
    let v2 = rng.gen_f64();
    assert_ne!(v1, v2);

    // Values should be in [0, 1)
    for _ in 0..1000 {
        let v = rng.gen_f64();
        assert!((0.0..1.0).contains(&v));
    }
}

#[test]
fn test_de_search_basic() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let mut search = DESearch::new(50).with_seed(42);
    let trials = search.suggest(&space, 20);

    assert_eq!(trials.len(), 20);

    // All trials should have valid values
    for trial in &trials {
        let n_est = trial
            .get_f64(&RF::NEstimators)
            .expect("NEstimators should exist");
        let depth = trial.get_f64(&RF::MaxDepth).expect("MaxDepth should exist");

        assert!((10.0..=500.0).contains(&n_est));
        assert!((2.0..=20.0).contains(&depth));
    }
}

#[test]
fn test_de_search_deterministic() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let mut search1 = DESearch::new(50).with_seed(42);
    let mut search2 = DESearch::new(50).with_seed(42);

    let trials1 = search1.suggest(&space, 10);
    let trials2 = search2.suggest(&space, 10);

    for (t1, t2) in trials1.iter().zip(trials2.iter()) {
        assert_eq!(
            t1.get_f64(&RF::NEstimators),
            t2.get_f64(&RF::NEstimators),
            "Same seed should produce same results"
        );
    }
}

#[test]
fn test_de_search_respects_budget() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let mut search = DESearch::new(5).with_seed(42);

    // First batch
    let t1 = search.suggest(&space, 3);
    assert_eq!(t1.len(), 3);
    assert_eq!(search.remaining(), 2);

    // Second batch
    let t2 = search.suggest(&space, 10);
    assert_eq!(t2.len(), 2); // Only 2 remaining
    assert_eq!(search.remaining(), 0);
}

#[test]
fn test_de_search_with_integers() {
    let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500) // Integer range
            .add(RF::MaxDepth, 2..20);

    let mut search = DESearch::new(50).with_seed(42);
    let trials = search.suggest(&space, 10);

    for trial in &trials {
        let n_est = trial
            .get_i64(&RF::NEstimators)
            .expect("NEstimators should exist");
        let depth = trial.get_i64(&RF::MaxDepth).expect("MaxDepth should exist");

        assert!((10..=500).contains(&n_est));
        assert!((2..=20).contains(&depth));
    }
}

#[test]
fn test_de_search_strategies() {
    use crate::metaheuristics::DEStrategy;

    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    for strategy in [
        DEStrategy::Rand1Bin,
        DEStrategy::Best1Bin,
        DEStrategy::CurrentToBest1Bin,
    ] {
        let mut search = DESearch::new(20).with_strategy(strategy).with_seed(42);
        let trials = search.suggest(&space, 10);
        assert_eq!(trials.len(), 10, "Strategy {strategy:?} should work");
    }
}

// =========================================================================
// Active Learning Tests (RED PHASE - These should fail initially)
// =========================================================================

#[test]
fn test_active_learning_basic() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let base = RandomSearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base).with_uncertainty_threshold(0.1);

    let trials = search.suggest(&space, 10);
    assert_eq!(trials.len(), 10);
}

#[test]
fn test_active_learning_stops_on_confidence() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(1000).with_seed(42);
    let mut search = ActiveLearningSearch::new(base)
        .with_uncertainty_threshold(0.5)
        .with_min_samples(5);

    // First batch
    let trials1 = search.suggest(&space, 10);
    assert!(!trials1.is_empty());

    // Simulate high-confidence results (low variance)
    let results: Vec<TrialResult<RF>> = trials1
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.95, // All same score = low uncertainty
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Should stop early due to low uncertainty
    assert!(
        search.should_stop(),
        "Should stop when uncertainty is below threshold"
    );
}

#[test]
fn test_active_learning_continues_on_uncertainty() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(1000).with_seed(42);
    let mut search = ActiveLearningSearch::new(base)
            .with_uncertainty_threshold(0.01) // Very low threshold
            .with_min_samples(3);

    let trials = search.suggest(&space, 10);

    // Simulate high-uncertainty results (high variance)
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: if i % 2 == 0 { 0.1 } else { 0.9 }, // High variance
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Should NOT stop - still uncertain
    assert!(
        !search.should_stop(),
        "Should continue when uncertainty is high"
    );
}

#[test]
fn test_active_learning_uncertainty_score() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base);

    let trials = search.suggest(&space, 5);

    // Low variance results
    let low_var_results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&low_var_results);
    let low_uncertainty = search.uncertainty();

    // Reset and try high variance
    let base2 = RandomSearch::new(100).with_seed(42);
    let mut search2 = ActiveLearningSearch::new(base2);
    let trials2 = search2.suggest(&space, 5);

    let high_var_results: Vec<TrialResult<RF>> = trials2
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: i as f64 / 5.0, // 0.0, 0.2, 0.4, 0.6, 0.8
            metrics: HashMap::new(),
        })
        .collect();

    search2.update(&high_var_results);
    let high_uncertainty = search2.uncertainty();

    assert!(
        high_uncertainty > low_uncertainty,
        "High variance should have higher uncertainty: {high_uncertainty} vs {low_uncertainty}"
    );
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
