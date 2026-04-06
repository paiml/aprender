//! Tests for TPE optimizer

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::optim::hpo::error::HPOError;
    use crate::optim::hpo::tpe::TPEOptimizer;
    use crate::optim::hpo::types::{HyperparameterSpace, ParameterDomain};

    #[test]
    fn test_tpe_new() {
        let space = HyperparameterSpace::new();
        let tpe = TPEOptimizer::new(space);
        assert_eq!(tpe.n_trials(), 0);
        assert!(tpe.best_trial().is_none());
    }

    #[test]
    fn test_tpe_suggest_empty_space() {
        let space = HyperparameterSpace::new();
        let mut tpe = TPEOptimizer::new(space);
        let result = tpe.suggest();
        assert!(matches!(result, Err(HPOError::EmptySpace)));
    }

    #[test]
    fn test_tpe_suggest_startup() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 1e-5, high: 1e-1, log_scale: true });

        let mut tpe = TPEOptimizer::new(space).with_startup(5);

        // First suggestions should work (startup phase)
        for _i in 0..5 {
            let trial = tpe.suggest().expect("operation should succeed");
            assert!(trial.config.contains_key("lr"));
        }
    }

    #[test]
    fn test_tpe_record_and_best() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let mut tpe = TPEOptimizer::new(space);

        let trial1 = tpe.suggest().expect("operation should succeed");
        tpe.record(trial1, 0.5, 10);

        let trial2 = tpe.suggest().expect("operation should succeed");
        tpe.record(trial2, 0.3, 10);

        assert_eq!(tpe.n_trials(), 2);
        let best = tpe.best_trial().expect("operation should succeed");
        assert_eq!(best.score, 0.3);
    }

    #[test]
    fn test_tpe_with_gamma() {
        let space = HyperparameterSpace::new();
        let tpe = TPEOptimizer::new(space).with_gamma(0.15);
        assert!((tpe.gamma - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_tpe_guided_sampling() {
        let mut space = HyperparameterSpace::new();
        space.add("x", ParameterDomain::Continuous { low: 0.0, high: 10.0, log_scale: false });

        let mut tpe = TPEOptimizer::new(space).with_startup(5);

        // Run startup phase
        for _i in 0..5 {
            let trial = tpe.suggest().expect("operation should succeed");
            // Lower x values get better scores
            let score = trial
                .config
                .get("x")
                .expect("key should exist")
                .as_float()
                .expect("key should exist");
            tpe.record(trial, score, 10);
        }

        // After startup, TPE should suggest values closer to 0
        // (where scores are better in our mock objective)
        assert_eq!(tpe.n_trials(), 5);
    }

    #[test]
    fn test_tpe_record_failed() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let mut tpe = TPEOptimizer::new(space);
        let trial = tpe.suggest().expect("operation should succeed");
        tpe.record_failed(trial);

        // Failed trials shouldn't count as completed
        assert_eq!(tpe.n_trials(), 0);
    }

    #[test]
    fn test_tpe_tpe_sampling_with_trials() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 32 });
        space.add(
            "activation",
            ParameterDomain::Categorical { choices: vec!["relu".to_string(), "gelu".to_string()] },
        );

        let mut tpe = TPEOptimizer::new(space).with_startup(3);

        // Run startup phase
        for _ in 0..3 {
            let trial = tpe.suggest().expect("operation should succeed");
            let lr = trial
                .config
                .get("lr")
                .expect("key should exist")
                .as_float()
                .expect("key should exist");
            tpe.record(trial, lr, 10); // Score equals lr
        }

        // Now TPE sampling kicks in
        for _ in 0..5 {
            let trial = tpe.suggest().expect("operation should succeed");
            assert!(trial.config.contains_key("lr"));
            assert!(trial.config.contains_key("batch_size"));
            assert!(trial.config.contains_key("activation"));
            let lr = trial
                .config
                .get("lr")
                .expect("key should exist")
                .as_float()
                .expect("key should exist");
            tpe.record(trial, lr, 10);
        }

        assert_eq!(tpe.n_trials(), 8);
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::super::sampling::{
        count_categorical, sample_ei_ratio_continuous, sample_ei_ratio_discrete,
    };
    use crate::optim::hpo::types::{ParameterValue, Trial};

    #[test]
    fn test_sample_ei_ratio_continuous_empty() {
        let mut rng = rand::rng();
        let value = sample_ei_ratio_continuous(&[], &[], 0.0, 1.0, 1.0, &mut rng);
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn test_sample_ei_ratio_continuous_with_data() {
        let mut rng = rand::rng();
        let good = vec![0.1, 0.2, 0.15];
        let bad = vec![0.8, 0.9, 0.85];
        let value = sample_ei_ratio_continuous(&good, &bad, 0.0, 1.0, 0.1, &mut rng);
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn test_sample_ei_ratio_discrete_empty() {
        let mut rng = rand::rng();
        let value = sample_ei_ratio_discrete(&[], &[], 1, 10, &mut rng);
        assert!((1..=10).contains(&value));
    }

    #[test]
    fn test_sample_ei_ratio_discrete_with_data() {
        let mut rng = rand::rng();
        let good = vec![2, 3, 2];
        let bad = vec![8, 9, 10];
        let value = sample_ei_ratio_discrete(&good, &bad, 1, 10, &mut rng);
        assert!((1..=10).contains(&value));
    }

    #[test]
    fn test_count_categorical_empty() {
        let trials: Vec<&Trial> = vec![];
        let choices = vec!["a".to_string(), "b".to_string()];
        let counts = count_categorical("opt", &trials, &choices);
        assert_eq!(counts, vec![0, 0]);
    }

    #[test]
    fn test_count_categorical_with_trials() {
        use std::collections::HashMap;

        let mut config1 = HashMap::new();
        config1.insert("opt".to_string(), ParameterValue::Categorical("a".to_string()));
        let trial1 = Trial::new(0, config1);

        let mut config2 = HashMap::new();
        config2.insert("opt".to_string(), ParameterValue::Categorical("b".to_string()));
        let trial2 = Trial::new(1, config2);

        let mut config3 = HashMap::new();
        config3.insert("opt".to_string(), ParameterValue::Categorical("a".to_string()));
        let trial3 = Trial::new(2, config3);

        let trials: Vec<&Trial> = vec![&trial1, &trial2, &trial3];
        let choices = vec!["a".to_string(), "b".to_string()];
        let counts = count_categorical("opt", &trials, &choices);
        assert_eq!(counts, vec![2, 1]);
    }
}

#[cfg(test)]
mod property_tests {
    use crate::optim::hpo::tpe::TPEOptimizer;
    use crate::optim::hpo::types::{HyperparameterSpace, ParameterDomain};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_tpe_trials_increment(n_trials in 1usize..20) {
            let mut space = HyperparameterSpace::new();
            space.add("x", ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            });

            let mut tpe = TPEOptimizer::new(space);
            for i in 0..n_trials {
                let trial = tpe.suggest().expect("operation should succeed");
                let score = (i as f64) / 10.0;
                tpe.record(trial, score, 10);
            }
            prop_assert_eq!(tpe.n_trials(), n_trials);
        }
    }
}
