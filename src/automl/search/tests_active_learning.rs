use super::*;

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

// =========================================================================
// Coverage gap tests (targeting 32 missed lines)
// =========================================================================

#[test]
fn test_param_value_as_f64_from_int() {
    // Covers ParamValue::as_f64 for Int variant (line 160)
    let pv = ParamValue::Int(42);
    assert_eq!(pv.as_f64(), Some(42.0));

    let pv_neg = ParamValue::Int(-7);
    assert_eq!(pv_neg.as_f64(), Some(-7.0));
}

#[test]
fn test_param_value_as_f64_returns_none_for_non_numeric() {
    // Covers the None branch of as_f64 for Bool and String
    assert_eq!(ParamValue::Bool(true).as_f64(), None);
    assert_eq!(ParamValue::String("x".to_string()).as_f64(), None);
}

#[test]
fn test_param_value_as_i64_from_float() {
    // Covers ParamValue::as_i64 for Float variant (line 170)
    let pv = ParamValue::Float(3.7);
    assert_eq!(pv.as_i64(), Some(3)); // truncation

    let pv_neg = ParamValue::Float(-2.9);
    assert_eq!(pv_neg.as_i64(), Some(-2));
}
