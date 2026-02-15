
#[test]
fn test_param_value_as_i64_returns_none_for_non_numeric() {
    // Covers the None branch of as_i64 for Bool and String
    assert_eq!(ParamValue::Bool(false).as_i64(), None);
    assert_eq!(ParamValue::String("y".to_string()).as_i64(), None);
}

#[test]
fn test_param_value_from_f32() {
    // Covers From<f32> for ParamValue (lines 200-203)
    let pv = ParamValue::from(2.5_f32);
    assert_eq!(pv.as_f64(), Some(f64::from(2.5_f32)));

    let pv_neg = ParamValue::from(-0.5_f32);
    assert!(pv_neg.as_f64().is_some());
}

#[test]
fn test_rng_gen_f64_range_default_impl() {
    // Covers the default Rng::gen_f64_range implementation (lines 528-530)
    let mut rng = XorShift64::new(42);
    for _ in 0..100 {
        let val = rng.gen_f64_range(5.0, 10.0);
        assert!(val >= 5.0 && val < 10.0, "got {val}");
    }
    // Edge case: same low and high
    let val = rng.gen_f64_range(3.0, 3.0);
    assert!((val - 3.0).abs() < 1e-10);
}

#[test]
fn test_grid_points_categorical() {
    // Covers grid_points for Categorical variant (line 140)
    let param = HyperParam::categorical(["a", "b", "c"]);
    let points = param.grid_points(10);
    // Categorical returns all choices regardless of n_points
    assert_eq!(points.len(), 3);
    assert_eq!(points[0].as_str(), Some("a"));
    assert_eq!(points[1].as_str(), Some("b"));
    assert_eq!(points[2].as_str(), Some("c"));
}

#[test]
fn test_grid_points_integer_large_n_points() {
    // Covers integer grid_points with n_points larger than range
    let param = HyperParam::integer(1, 3);
    let points = param.grid_points(100);
    // Range is 3, step max(1, ceil(3/100)) = 1, so we get all 3 values
    assert!(points.len() <= 3);
    for p in &points {
        let v = p.as_i64().expect("should be int");
        assert!(v >= 1 && v <= 3);
    }
}

#[test]
fn test_grid_points_integer_exact_range() {
    // Integer grid points should cover the range
    let param = HyperParam::integer(5, 5);
    let points = param.grid_points(3);
    // Range is 1 (5..=5), step is max(1, ceil(1/3)) = 1
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].as_i64(), Some(5));
}

#[test]
fn test_de_update_not_initialized() {
    // Covers update when initialized=false (line 961)
    let mut search = DESearch::new(50).with_seed(42);
    assert!(!search.initialized);

    let mut values = HashMap::new();
    values.insert(RF::NEstimators, ParamValue::Float(100.0));
    let trial = Trial { values };
    let results = vec![TrialResult {
        trial,
        score: 0.5,
        metrics: HashMap::new(),
    }];

    // Update before suggest (not initialized) should return early
    search.update(&results);
    assert!(!search.initialized);
}

#[test]
fn test_de_update_results_exceed_fitness_len() {
    // Covers the i < self.fitness.len() check (line 968)
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let mut search = DESearch::new(200)
        .with_population_size(5) // Small population
        .with_seed(42);

    let trials = search.suggest(&space, 5);

    // Create more results than population size
    let mut results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5,
            metrics: HashMap::new(),
        })
        .collect();

    // Add extra results beyond population
    for _ in 0..10 {
        let mut values = HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Float(50.0));
        results.push(TrialResult {
            trial: Trial { values },
            score: 0.8,
            metrics: HashMap::new(),
        });
    }

    // Should not panic even with extra results
    search.update(&results);
}

#[test]
fn test_generic_param_name() {
    // Covers GenericParam::name (line 438-440)
    let gp = GenericParam("learning_rate");
    assert_eq!(gp.name(), "learning_rate");
    assert_eq!(gp.0, "learning_rate");
}

#[test]
fn test_generic_param_debug_clone_eq_hash() {
    let gp1 = GenericParam("lr");
    let gp2 = gp1; // Copy
    assert_eq!(gp1, gp2);

    let debug = format!("{gp1:?}");
    assert!(debug.contains("GenericParam"));

    // Hash via HashMap
    let mut map = HashMap::new();
    map.insert(gp1, 1);
    assert_eq!(map.get(&gp2), Some(&1));
}

#[test]
fn test_search_space_clone_and_debug() {
    let space: SearchSpace<RF> = SearchSpace::new()
        .add(RF::NEstimators, 10..100)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let cloned = space.clone();
    assert_eq!(cloned.len(), 2);
    assert!(cloned.get(&RF::NEstimators).is_some());

    let debug = format!("{space:?}");
    assert!(debug.contains("SearchSpace"));
}

#[test]
fn test_trial_get_missing_key() {
    // Covers Trial get methods returning None for missing keys
    let trial: Trial<RF> = Trial {
        values: HashMap::new(),
    };
    assert_eq!(trial.get(&RF::NEstimators), None);
    assert_eq!(trial.get_f64(&RF::NEstimators), None);
    assert_eq!(trial.get_i64(&RF::NEstimators), None);
    assert_eq!(trial.get_usize(&RF::NEstimators), None);
    assert_eq!(trial.get_bool(&RF::NEstimators), None);
}

#[test]
fn test_trial_get_bool_from_bool_param() {
    let mut values = HashMap::new();
    values.insert(RF::Bootstrap, ParamValue::Bool(true));
    let trial = Trial { values };
    assert_eq!(trial.get_bool(&RF::Bootstrap), Some(true));
}

#[test]
fn test_trial_display_empty() {
    let trial: Trial<RF> = Trial {
        values: HashMap::new(),
    };
    let s = format!("{trial}");
    assert_eq!(s, "{}");
}

#[test]
fn test_trial_result_debug() {
    let mut values = HashMap::new();
    values.insert(RF::NEstimators, ParamValue::Int(100));
    let result = TrialResult {
        trial: Trial { values },
        score: 0.95,
        metrics: HashMap::new(),
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("TrialResult"));
    assert!(debug.contains("0.95"));
}

#[test]
fn test_hyperparam_debug() {
    let hp_cont = HyperParam::continuous(0.0, 1.0);
    let hp_int = HyperParam::integer(1, 10);
    let hp_cat = HyperParam::categorical(["a", "b"]);

    let d1 = format!("{hp_cont:?}");
    let d2 = format!("{hp_int:?}");
    let d3 = format!("{hp_cat:?}");

    assert!(d1.contains("Continuous"));
    assert!(d2.contains("Integer"));
    assert!(d3.contains("Categorical"));
}

#[test]
fn test_grid_search_min_granularity() {
    // GridSearch::new clamps points_per_param to at least 2 (line 692)
    let gs = GridSearch::new(1);
    assert_eq!(gs.points_per_param, 2);

    let gs0 = GridSearch::new(0);
    assert_eq!(gs0.points_per_param, 2);
}

#[test]
fn test_grid_search_debug_clone() {
    let gs = GridSearch::new(5);
    let cloned = gs.clone();
    assert_eq!(cloned.points_per_param, 5);

    let debug = format!("{gs:?}");
    assert!(debug.contains("GridSearch"));
}

#[test]
fn test_de_search_clone_debug() {
    let de = DESearch::new(50).with_seed(123);
    let cloned = de.clone();
    assert_eq!(cloned.n_iter, 50);
    assert_eq!(cloned.seed, 123);

    let debug = format!("{de:?}");
    assert!(debug.contains("DESearch"));
}

#[test]
fn test_de_search_auto_population_size_various_dims() {
    // Auto pop size: (10 * dim).clamp(20, 100) (lines 874-876)
    // 1 dim -> 10, clamped to 20
    let space_1d: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 0.0, 1.0);
    let mut de = DESearch::new(100).with_seed(42);
    let trials = de.suggest(&space_1d, 20);
    assert_eq!(trials.len(), 20); // pop_size = max(20, 10*1) = 20

    // 5 dims -> 50
    let space_5d: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 0.0, 1.0)
        .add_continuous(RF::MaxDepth, 0.0, 1.0)
        .add(RF::MinSamplesSplit, 1..10)
        .add(RF::MinSamplesLeaf, 1..10)
        .add_categorical(RF::MaxFeatures, ["a", "b"]);
    let mut de2 = DESearch::new(200).with_seed(42);
    let trials2 = de2.suggest(&space_5d, 50);
    assert_eq!(trials2.len(), 50); // pop_size = 10*5 = 50
}

#[test]
fn test_de_search_update_best1bin_strategy() {
    use crate::metaheuristics::DEStrategy;

    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let mut search = DESearch::new(100)
        .with_strategy(DEStrategy::Best1Bin)
        .with_seed(42);

    let trials = search.suggest(&space, 20);

    // Simulate results with varying scores
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: (i as f64) * 0.05,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);

    // Next batch should still produce valid values
    let trials2 = search.suggest(&space, 20);
    for trial in &trials2 {
        let n = trial.get_f64(&RF::NEstimators).expect("should exist");
        assert!((10.0..=500.0).contains(&n));
    }
}

#[test]
fn test_de_search_update_current_to_best1bin_strategy() {
    use crate::metaheuristics::DEStrategy;

    let space: SearchSpace<RF> = SearchSpace::new()
        .add_continuous(RF::NEstimators, 10.0, 500.0)
        .add_continuous(RF::MaxDepth, 2.0, 20.0);

    let mut search = DESearch::new(100)
        .with_strategy(DEStrategy::CurrentToBest1Bin)
        .with_seed(42);

    let trials = search.suggest(&space, 20);
    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: (i as f64) * 0.1,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&results);
    let trials2 = search.suggest(&space, 20);
    assert!(!trials2.is_empty());
}

#[test]
fn test_active_learning_debug_clone() {
    let base = RandomSearch::new(100).with_seed(42);
    let al = ActiveLearningSearch::new(base)
        .with_uncertainty_threshold(0.05)
        .with_min_samples(10);

    let cloned = al.clone();
    assert_eq!(cloned.sample_count(), 0);
    assert!(cloned.uncertainty().is_infinite());

    let debug = format!("{al:?}");
    assert!(debug.contains("ActiveLearningSearch"));
}

#[test]
fn test_search_strategy_default_update() {
    // The default SearchStrategy::update is a no-op (line 519)
    // RandomSearch uses the default update
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
    let mut search = RandomSearch::new(10).with_seed(42);
    let trials = search.suggest(&space, 5);

    let results: Vec<TrialResult<RF>> = trials
        .iter()
        .map(|t| TrialResult {
            trial: t.clone(),
            score: 0.5,
            metrics: HashMap::new(),
        })
        .collect();

    // Default update should not panic
    search.update(&results);
    assert_eq!(search.remaining(), 5);
}

#[test]
fn test_param_value_equality() {
    assert_eq!(ParamValue::Float(1.0), ParamValue::Float(1.0));
    assert_ne!(ParamValue::Float(1.0), ParamValue::Float(2.0));
    assert_eq!(ParamValue::Int(42), ParamValue::Int(42));
    assert_ne!(ParamValue::Int(1), ParamValue::Int(2));
    assert_eq!(ParamValue::Bool(true), ParamValue::Bool(true));
    assert_ne!(ParamValue::Bool(true), ParamValue::Bool(false));
    assert_eq!(
        ParamValue::String("a".to_string()),
        ParamValue::String("a".to_string())
    );
    assert_ne!(
        ParamValue::String("a".to_string()),
        ParamValue::String("b".to_string())
    );
    // Cross-variant inequality
    assert_ne!(ParamValue::Float(1.0), ParamValue::Int(1));
}

#[test]
fn test_xorshift_clone_debug() {
    let rng = XorShift64::new(42);
    let cloned = rng.clone();
    let debug = format!("{rng:?}");
    assert!(debug.contains("XorShift64"));
    // Clone should produce same state
    assert_eq!(format!("{cloned:?}"), debug);
}

#[test]
fn test_de_search_log_scale_initialization() {
    // Covers the is_log branch during population initialization (line 887-890)
    let space: SearchSpace<RF> = SearchSpace::new().add_log_scale(
        RF::NEstimators,
        LogScale {
            low: 1e-4,
            high: 1.0,
        },
    );

    let mut search = DESearch::new(50).with_seed(42);
    let trials = search.suggest(&space, 20);
    assert_eq!(trials.len(), 20);

    for trial in &trials {
        let v = trial.get_f64(&RF::NEstimators).expect("should have value");
        assert!(v >= 1e-4 && v <= 1.0, "Log scale value {v} out of range");
    }
}

#[test]
fn test_de_search_fitness_infinity_replacement() {
    // Covers the fitness[i] == f64::INFINITY branch during update (line 1061)
    let space: SearchSpace<RF> = SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

    let mut search = DESearch::new(200).with_seed(42);

    // Suggest initial population
    let trials = search.suggest(&space, 20);

    // Only provide results for some members (fitness stays INFINITY for rest)
    let partial_results: Vec<TrialResult<RF>> = trials[..5]
        .iter()
        .enumerate()
        .map(|(i, t)| TrialResult {
            trial: t.clone(),
            score: i as f64,
            metrics: HashMap::new(),
        })
        .collect();

    search.update(&partial_results);

    // Next suggest should still work
    let trials2 = search.suggest(&space, 20);
    assert!(!trials2.is_empty());
}
