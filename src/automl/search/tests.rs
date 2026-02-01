//\! AutoML Search Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

use crate::automl::params::RandomForestParam as RF;

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

#[test]
fn test_active_learning_with_de_search() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    // Active learning wrapping DE
    let base = DESearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base).with_uncertainty_threshold(0.1);

    let trials = search.suggest(&space, 20);
    assert_eq!(trials.len(), 20);

    // All values should be within bounds
    for trial in &trials {
        let n_est = trial
            .get_f64(&RF::NEstimators)
            .expect("NEstimators should exist");
        assert!((10.0..=500.0).contains(&n_est));
    }
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_log_scale_struct() {
    let ls = LogScale {
        low: 1e-4,
        high: 1.0,
    };
    assert_eq!(ls.low, 1e-4);
    assert_eq!(ls.high, 1.0);

    // Test Clone/Copy via Debug
    let _cloned = ls;
    let debug_str = format!("{:?}", ls);
    assert!(debug_str.contains("LogScale"));
}

#[test]
fn test_generic_param_display() {
    let param = GenericParam("learning_rate");
    let display = format!("{}", param);
    assert_eq!(display, "learning_rate");
}

#[test]
fn test_search_space_iter() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add(RF::NEstimators, 10..500)
        .add(RF::MaxDepth, 2..20);

    let count = space.iter().count();
    assert_eq!(count, 2);
}

#[test]
fn test_search_space_is_empty() {
    let empty: SearchSpace<RF> = SearchSpace::new();
    assert!(empty.is_empty());

    let non_empty = empty.add(RF::NEstimators, 10..100);
    assert!(!non_empty.is_empty());
}

#[test]
fn test_search_space_add_param() {
    let param = HyperParam::continuous(0.0, 1.0);
    let space: SearchSpace<RF> = SearchSpace::new().add_param(RF::NEstimators, param);

    assert!(space.get(&RF::NEstimators).is_some());
}

#[test]
fn test_trial_get_usize() {
    let mut values = HashMap::new();
    values.insert(RF::NEstimators, ParamValue::Int(100));

    let trial = Trial { values };
    assert_eq!(trial.get_usize(&RF::NEstimators), Some(100));
}

#[test]
fn test_param_value_as_str_none() {
    let pv = ParamValue::Int(42);
    assert!(pv.as_str().is_none());

    let pv2 = ParamValue::Float(1.0);
    assert!(pv2.as_str().is_none());
}

#[test]
fn test_param_value_as_bool_none() {
    let pv = ParamValue::Int(42);
    assert!(pv.as_bool().is_none());
}

#[test]
fn test_param_value_display() {
    assert_eq!(format!("{}", ParamValue::Float(1.5)), "1.500000");
    assert_eq!(format!("{}", ParamValue::Int(42)), "42");
    assert_eq!(format!("{}", ParamValue::Bool(true)), "true");
    assert_eq!(
        format!("{}", ParamValue::String("test".to_string())),
        "test"
    );
}

#[test]
fn test_param_value_from_usize() {
    let pv = ParamValue::from(42_usize);
    assert_eq!(pv.as_i64(), Some(42));
}

#[test]
fn test_param_value_from_string() {
    let pv = ParamValue::from("hello".to_string());
    assert_eq!(pv.as_str(), Some("hello"));
}

#[test]
fn test_hyperparam_clone() {
    let hp = HyperParam::continuous(0.0, 1.0);
    let cloned = hp.clone();
    if let HyperParam::Continuous { low, high, .. } = cloned {
        assert_eq!(low, 0.0);
        assert_eq!(high, 1.0);
    } else {
        panic!("Should be continuous");
    }
}

#[test]
fn test_grid_points_single_point() {
    let param = HyperParam::continuous(5.0, 10.0);
    let points = param.grid_points(1);

    assert_eq!(points.len(), 1);
    assert_eq!(points[0].as_f64(), Some(5.0));
}

#[test]
fn test_grid_points_log_scale() {
    let param = HyperParam::continuous_log(1.0, 100.0);
    let points = param.grid_points(3);

    assert_eq!(points.len(), 3);
    // First and last should be at bounds
    assert!((points[0].as_f64().unwrap() - 1.0).abs() < 0.01);
    assert!((points[2].as_f64().unwrap() - 100.0).abs() < 0.01);
}

#[test]
fn test_grid_empty_space() {
    let space: SearchSpace<RF> = SearchSpace::new();
    let grid = space.grid(5);

    assert_eq!(grid.len(), 1);
    assert!(grid[0].values.is_empty());
}

#[test]
fn test_de_search_with_jade() {
    let search = DESearch::new(50).with_jade();
    assert!(search.use_jade);
}

#[test]
fn test_de_search_with_mutation_factor() {
    let search = DESearch::new(50).with_mutation_factor(0.5);
    assert_eq!(search.mutation_factor, 0.5);
}

#[test]
fn test_de_search_with_crossover_rate() {
    let search = DESearch::new(50).with_crossover_rate(0.7);
    assert_eq!(search.crossover_rate, 0.7);
}

#[test]
fn test_de_search_with_population_size() {
    let search = DESearch::new(50).with_population_size(100);
    assert_eq!(search.population_size, 100);
}

#[test]
fn test_de_search_rand2bin_strategy() {
    use crate::metaheuristics::DEStrategy;

    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let mut search = DESearch::new(100)
        .with_strategy(DEStrategy::Rand2Bin)
        .with_seed(42);

    let trials = search.suggest(&space, 20);
    assert_eq!(trials.len(), 20);

    // Simulate results to trigger update with Rand2Bin mutation
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: i as f64,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Get next batch after evolution
    let trials2 = search.suggest(&space, 20);
    assert!(trials2.len() <= 20);
}

#[test]
fn test_de_search_with_categorical() {
    let space: SearchSpace<RF> =
        SearchSpace::new().add_categorical(RF::MaxFeatures, ["sqrt", "log2", "auto"]);

    let mut search = DESearch::new(50).with_seed(42);
    let trials = search.suggest(&space, 10);

    for trial in &trials {
        let val = trial.get(&RF::MaxFeatures).expect("should have value");
        let s = val.as_str().expect("should be string");
        assert!(["sqrt", "log2", "auto"].contains(&s));
    }
}

#[test]
fn test_active_learning_sample_count() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base);

    assert_eq!(search.sample_count(), 0);

    let trials = search.suggest(&space, 5);
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);
    assert_eq!(search.sample_count(), 5);
}

#[test]
fn test_active_learning_returns_empty_when_should_stop() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(1000).with_seed(42);
    let mut search = ActiveLearningSearch::new(base)
            .with_uncertainty_threshold(1.0) // Very high threshold - easy to satisfy
            .with_min_samples(5);

    // Get enough samples
    let trials = search.suggest(&space, 10);
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5, // Same score = zero variance
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Now should_stop() returns true, so suggest() should return empty
    assert!(search.should_stop());
    let empty_trials = search.suggest(&space, 10);
    assert!(empty_trials.is_empty());
}

#[test]
fn test_active_learning_near_zero_mean() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base);

    let trials = search.suggest(&space, 5);

    // Scores near zero to trigger the mean.abs() < 1e-10 branch
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: (i as f64) * 1e-12, // Very small values
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Should handle near-zero mean without dividing by zero
    let uncertainty = search.uncertainty();
    assert!(uncertainty.is_finite());
}

#[test]
fn test_active_learning_single_sample() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let base = RandomSearch::new(100).with_seed(42);
    let mut search = ActiveLearningSearch::new(base);

    let trials = search.suggest(&space, 1);
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // With only 1 sample, uncertainty should be infinite
    assert!(search.uncertainty().is_infinite());
}

#[test]
fn test_xorshift_edge_cases() {
    // Seed of 0 should be treated as 1
    let mut rng = XorShift64::new(0);
    let v = rng.gen_f64();
    assert!((0.0..1.0).contains(&v));

    // gen_i64_range with low >= high should return low
    let mut rng2 = XorShift64::new(42);
    assert_eq!(rng2.gen_i64_range(10, 10), 10);
    assert_eq!(rng2.gen_i64_range(10, 5), 10);

    // gen_usize with len 0 should return 0
    assert_eq!(rng2.gen_usize(0), 0);
}

#[test]
fn test_trial_result_clone() {
    let mut values = HashMap::new();
    values.insert(RF::NEstimators, ParamValue::Int(100));
    let trial = Trial { values };

    let result = TrialResult {
        trial,
        score: 0.95,
        metrics: HashMap::new(),
    };

    let cloned = result.clone();
    assert_eq!(cloned.score, 0.95);
}

#[test]
fn test_de_update_empty_results() {
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let mut search = DESearch::new(50).with_seed(42);
    let _trials = search.suggest(&space, 10);

    // Update with empty results - should not panic
    let empty: Vec<TrialResult<RF>> = vec![];
    search.update(&empty);
}

#[test]
fn test_grid_search_position_tracking() {
    let space: SearchSpace<RF> = SearchSpace::new().add_categorical(RF::Bootstrap, [true, false]);

    let mut search = GridSearch::new(5);

    // First batch
    let t1 = search.suggest(&space, 1);
    assert_eq!(t1.len(), 1);

    // Second batch from same position
    let t2 = search.suggest(&space, 1);
    assert_eq!(t2.len(), 1);

    // Total grid is 2, so we've exhausted it
    let t3 = search.suggest(&space, 10);
    assert!(t3.is_empty());
}

#[test]
fn test_search_space_default() {
    let space: SearchSpace<RF> = SearchSpace::default();
    assert!(space.is_empty());
}

#[test]
fn test_random_search_clone() {
    let search = RandomSearch::new(50).with_seed(42);
    let cloned = search.clone();
    assert_eq!(cloned.n_iter, 50);
    assert_eq!(cloned.seed, 42);
}

mod proptests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use proptest::prelude::*;

    proptest! {
        /// Random search should always respect budget constraint.
        #[test]
        fn prop_random_search_respects_budget(
            n_iter in 1_usize..100,
            seed in any::<u64>(),
            request in 1_usize..200
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut search = RandomSearch::new(n_iter).with_seed(seed);
            let trials = search.suggest(&space, request);

            prop_assert!(trials.len() <= n_iter);
            prop_assert!(trials.len() <= request);
        }

        /// Continuous parameters should always sample within bounds.
        #[test]
        fn prop_continuous_within_bounds(
            low in -1000.0_f64..1000.0,
            high_offset in 0.01_f64..1000.0,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let param = HyperParam::continuous(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_f64().expect("should be float");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Integer parameters should always sample within bounds.
        #[test]
        fn prop_integer_within_bounds(
            low in -1000_i64..1000,
            high_offset in 1_i64..100,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let param = HyperParam::integer(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_i64().expect("should be int");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Log scale should produce values in valid range.
        #[test]
        fn prop_log_scale_within_bounds(
            low_exp in -6_i32..0,
            high_exp in 0_i32..3,
            seed in any::<u64>()
        ) {
            let low = 10_f64.powi(low_exp);
            let high = 10_f64.powi(high_exp);
            let param = HyperParam::continuous_log(low, high);
            let mut rng = XorShift64::new(seed);

            for _ in 0..100 {
                let v = param.sample(&mut rng).as_f64().expect("should be float");
                prop_assert!((low..=high).contains(&v), "Value {} not in [{}, {}]", v, low, high);
            }
        }

        /// Same seed should produce same results (determinism).
        #[test]
        fn prop_random_search_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500)
                .add(RF::MaxDepth, 2..20);

            let mut s1 = RandomSearch::new(10).with_seed(seed);
            let mut s2 = RandomSearch::new(10).with_seed(seed);

            let t1 = s1.suggest(&space, 5);
            let t2 = s2.suggest(&space, 5);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
                prop_assert_eq!(a.get(&RF::MaxDepth), b.get(&RF::MaxDepth));
            }
        }

        /// Grid points for continuous params should be evenly spaced.
        #[test]
        fn prop_grid_points_count(n_points in 2_usize..20) {
            let param = HyperParam::continuous(0.0, 1.0);
            let points = param.grid_points(n_points);
            prop_assert_eq!(points.len(), n_points);
        }

        /// XorShift64 should always produce values in [0, 1).
        #[test]
        fn prop_xorshift_range(seed in 1_u64..u64::MAX) {
            let mut rng = XorShift64::new(seed);
            for _ in 0..1000 {
                let v = rng.gen_f64();
                prop_assert!((0.0..1.0).contains(&v));
            }
        }

        /// ParamValue conversions should be consistent.
        #[test]
        fn prop_param_value_int_roundtrip(v in any::<i32>()) {
            let pv = ParamValue::from(v);
            prop_assert_eq!(pv.as_i64(), Some(i64::from(v)));
        }

        /// ParamValue float conversions should preserve value.
        #[test]
        fn prop_param_value_float_roundtrip(v in any::<f32>()) {
            let pv = ParamValue::from(v);
            let result = pv.as_f64().expect("float param should convert");
            prop_assert!((result - f64::from(v)).abs() < 1e-10);
        }

        /// DESearch should respect budget constraint.
        #[test]
        fn prop_de_search_respects_budget(
            n_iter in 1_usize..50,
            seed in any::<u64>(),
            request in 1_usize..100
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let mut search = DESearch::new(n_iter).with_seed(seed);
            let trials = search.suggest(&space, request);

            prop_assert!(trials.len() <= n_iter);
            prop_assert!(trials.len() <= request);
        }

        /// DESearch continuous parameters should be within bounds.
        #[test]
        fn prop_de_search_continuous_bounds(
            low in 0.0_f64..100.0,
            high_offset in 1.0_f64..100.0,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, low, high);

            let mut search = DESearch::new(50).with_seed(seed);
            let trials = search.suggest(&space, 20);

            for trial in trials {
                let v = trial.get_f64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    v >= low && v <= high,
                    "Value {} not in [{}, {}]", v, low, high
                );
            }
        }

        /// DESearch should be deterministic with same seed.
        #[test]
        fn prop_de_search_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0)
                .add_continuous(RF::MaxDepth, 2.0, 20.0);

            let mut s1 = DESearch::new(50).with_seed(seed);
            let mut s2 = DESearch::new(50).with_seed(seed);

            let t1 = s1.suggest(&space, 10);
            let t2 = s2.suggest(&space, 10);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get_f64(&RF::NEstimators), b.get_f64(&RF::NEstimators));
                prop_assert_eq!(a.get_f64(&RF::MaxDepth), b.get_f64(&RF::MaxDepth));
            }
        }

        /// DESearch integer parameters should be integers within bounds.
        #[test]
        fn prop_de_search_integer_bounds(
            low in 1_i64..100,
            high_offset in 1_i64..100,
            seed in any::<u64>()
        ) {
            let high = low + high_offset;
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, low..high);

            let mut search = DESearch::new(50).with_seed(seed);
            let trials = search.suggest(&space, 20);

            for trial in trials {
                let v = trial.get_i64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    v >= low && v <= high,
                    "Value {} not in [{}, {}]", v, low, high
                );
            }
        }

        /// DESearch population should remain valid after update.
        #[test]
        fn prop_de_search_update_valid(seed in 1_u64..u64::MAX) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let mut search = DESearch::new(100).with_seed(seed);

            // Get initial population
            let trials1 = search.suggest(&space, 20);

            // Simulate results (higher n_estimators = better score)
            let results: Vec<TrialResult<RF>> = trials1.iter().map(|t| {
                let score = t.get_f64(&RF::NEstimators).unwrap_or(0.0);
                TrialResult {
                    trial: t.clone(),
                    score,
                    metrics: HashMap::new(),
                }
            }).collect();

            // Update with results
            search.update(&results);

            // Get next batch - should still be valid
            let trials2 = search.suggest(&space, 20);

            // All values should still be within bounds
            for trial in trials2 {
                let v = trial.get_f64(&RF::NEstimators).expect("should have value");
                prop_assert!(
                    (10.0..=500.0).contains(&v),
                    "Value {v} not in bounds after evolution"
                );
            }
        }

        /// Active learning should stop when all scores are identical (zero variance).
        #[test]
        fn prop_active_learning_stops_zero_variance(
            n_samples in 10_usize..50,
            score in 0.1_f64..0.9
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let base = RandomSearch::new(1000).with_seed(42);
            let mut search = ActiveLearningSearch::new(base)
                .with_uncertainty_threshold(0.01)
                .with_min_samples(n_samples);

            let trials = search.suggest(&space, n_samples);

            // All same score = zero variance = should stop
            let results: Vec<TrialResult<RF>> = trials
                .iter()
                .map(|t| TrialResult {
                    trial: t.clone(),
                    score,
                    metrics: HashMap::new(),
                })
                .collect();

            search.update(&results);

            // Zero variance means uncertainty should be very low
            prop_assert!(
                search.uncertainty() < 0.01,
                "Zero variance should give near-zero uncertainty, got {}",
                search.uncertainty()
            );
        }

        /// Active learning uncertainty should increase with score variance.
        #[test]
        fn prop_active_learning_uncertainty_increases_with_variance(
            base_score in 0.3_f64..0.7,
            seed in any::<u64>()
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            // Low variance case
            let base1 = RandomSearch::new(100).with_seed(seed);
            let mut search1 = ActiveLearningSearch::new(base1);
            let trials1 = search1.suggest(&space, 20);

            let low_var_results: Vec<TrialResult<RF>> = trials1
                .iter()
                .enumerate()
                .map(|(i, t)| TrialResult {
                    trial: t.clone(),
                    score: base_score + (i as f64) * 0.001, // Very small variance
                    metrics: HashMap::new(),
                })
                .collect();
            search1.update(&low_var_results);

            // High variance case
            let base2 = RandomSearch::new(100).with_seed(seed);
            let mut search2 = ActiveLearningSearch::new(base2);
            let trials2 = search2.suggest(&space, 20);

            let high_var_results: Vec<TrialResult<RF>> = trials2
                .iter()
                .enumerate()
                .map(|(i, t)| TrialResult {
                    trial: t.clone(),
                    score: if i % 2 == 0 { 0.1 } else { 0.9 }, // High variance
                    metrics: HashMap::new(),
                })
                .collect();
            search2.update(&high_var_results);

            prop_assert!(
                search2.uncertainty() > search1.uncertainty(),
                "High variance ({}) should have higher uncertainty than low variance ({})",
                search2.uncertainty(),
                search1.uncertainty()
            );
        }

        /// Active learning should not stop before min_samples.
        #[test]
        fn prop_active_learning_respects_min_samples(
            min_samples in 5_usize..20,
            seed in any::<u64>()
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add_continuous(RF::NEstimators, 10.0, 500.0);

            let base = RandomSearch::new(1000).with_seed(seed);
            let mut search = ActiveLearningSearch::new(base)
                .with_uncertainty_threshold(1.0) // High threshold = easy to satisfy
                .with_min_samples(min_samples);

            // Get fewer than min_samples
            let trials = search.suggest(&space, min_samples - 1);

            let results: Vec<TrialResult<RF>> = trials
                .iter()
                .map(|t| TrialResult {
                    trial: t.clone(),
                    score: 0.5, // Same score = low uncertainty
                    metrics: HashMap::new(),
                })
                .collect();

            search.update(&results);

            // Should NOT stop - not enough samples yet
            prop_assert!(
                !search.should_stop(),
                "Should not stop with {} samples (min={})",
                results.len(),
                min_samples
            );
        }
    }
}
