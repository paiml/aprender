use super::*;

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
