//! Tests for HPO types

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::optim::hpo::types::{
        AcquisitionFunction, HyperparameterSpace, ParameterDomain, ParameterValue, SearchStrategy,
        SurrogateModel, Trial, TrialStatus,
    };

    // -------------------------------------------------------------------------
    // ParameterValue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_float() {
        let v = ParameterValue::Float(0.5);
        assert_eq!(v.as_float(), Some(0.5));
        assert_eq!(v.as_int(), Some(0));
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_parameter_value_int() {
        let v = ParameterValue::Int(42);
        assert_eq!(v.as_float(), Some(42.0));
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_parameter_value_categorical() {
        let v = ParameterValue::Categorical("relu".to_string());
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_str(), Some("relu"));
    }

    // -------------------------------------------------------------------------
    // ParameterDomain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_domain_continuous_sample() {
        let domain = ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_continuous_log_scale() {
        let domain = ParameterDomain::Continuous { low: 1e-5, high: 1e-1, log_scale: true };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_discrete_sample() {
        let domain = ParameterDomain::Discrete { low: 8, high: 128 };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_categorical_sample() {
        let domain = ParameterDomain::Categorical {
            choices: vec!["relu".to_string(), "gelu".to_string(), "swish".to_string()],
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let value = domain.sample(&mut rng);
            assert!(domain.is_valid(&value));
        }
    }

    #[test]
    fn test_domain_is_valid() {
        let domain = ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false };

        assert!(domain.is_valid(&ParameterValue::Float(0.5)));
        assert!(!domain.is_valid(&ParameterValue::Float(1.5)));
        assert!(!domain.is_valid(&ParameterValue::Int(0)));
    }

    // -------------------------------------------------------------------------
    // HyperparameterSpace Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_space_new() {
        let space = HyperparameterSpace::new();
        assert!(space.is_empty());
        assert_eq!(space.len(), 0);
    }

    #[test]
    fn test_space_add_and_get() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 1e-5, high: 1e-1, log_scale: true });

        assert!(!space.is_empty());
        assert_eq!(space.len(), 1);
        assert!(space.get("lr").is_some());
        assert!(space.get("unknown").is_none());
    }

    #[test]
    fn test_space_sample_random() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 1e-5, high: 1e-1, log_scale: true });
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 64 });

        let mut rng = rand::rng();
        let config = space.sample_random(&mut rng);

        assert!(config.contains_key("lr"));
        assert!(config.contains_key("batch_size"));
        assert!(space.validate(&config).is_ok());
    }

    #[test]
    fn test_space_validate() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let mut valid_config = HashMap::new();
        valid_config.insert("lr".to_string(), ParameterValue::Float(0.5));
        assert!(space.validate(&valid_config).is_ok());

        let mut invalid_config = HashMap::new();
        invalid_config.insert("lr".to_string(), ParameterValue::Float(2.0));
        assert!(space.validate(&invalid_config).is_err());

        let missing_config = HashMap::new();
        assert!(space.validate(&missing_config).is_err());
    }

    // -------------------------------------------------------------------------
    // Trial Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trial_new() {
        let config = HashMap::new();
        let trial = Trial::new(0, config);
        assert_eq!(trial.id, 0);
        assert_eq!(trial.status, TrialStatus::Pending);
        assert_eq!(trial.score, f64::INFINITY);
    }

    #[test]
    fn test_trial_complete() {
        let mut trial = Trial::new(0, HashMap::new());
        trial.complete(0.5, 100);
        assert_eq!(trial.status, TrialStatus::Completed);
        assert_eq!(trial.score, 0.5);
        assert_eq!(trial.iterations, 100);
    }

    #[test]
    fn test_trial_fail() {
        let mut trial = Trial::new(0, HashMap::new());
        trial.fail();
        assert_eq!(trial.status, TrialStatus::Failed);
    }

    // -------------------------------------------------------------------------
    // Additional Space Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_space_iter() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 32 });

        let param_names: Vec<_> = space.iter().map(|(name, _)| name.clone()).collect();
        assert_eq!(param_names.len(), 2);
        assert!(param_names.contains(&"lr".to_string()));
        assert!(param_names.contains(&"batch_size".to_string()));
    }

    // -------------------------------------------------------------------------
    // Domain Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_domain_is_valid_type_mismatch() {
        let domain = ParameterDomain::Discrete { low: 0, high: 10 };
        // Float value for discrete domain
        assert!(!domain.is_valid(&ParameterValue::Float(5.0)));

        let domain = ParameterDomain::Categorical { choices: vec!["a".to_string()] };
        // Int value for categorical domain
        assert!(!domain.is_valid(&ParameterValue::Int(0)));
    }

    // -------------------------------------------------------------------------
    // Serde Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_serde() {
        let v = ParameterValue::Float(0.5);
        let json = serde_json::to_string(&v).expect("JSON serialization should succeed");
        let parsed: ParameterValue =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(v, parsed);

        let v = ParameterValue::Int(42);
        let json = serde_json::to_string(&v).expect("JSON serialization should succeed");
        let parsed: ParameterValue =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(v, parsed);

        let v = ParameterValue::Categorical("relu".to_string());
        let json = serde_json::to_string(&v).expect("JSON serialization should succeed");
        let parsed: ParameterValue =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(v, parsed);
    }

    #[test]
    fn test_parameter_domain_serde() {
        let domain = ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: true };
        let json = serde_json::to_string(&domain).expect("JSON serialization should succeed");
        let parsed: ParameterDomain =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        match parsed {
            ParameterDomain::Continuous { log_scale, .. } => assert!(log_scale),
            _ => panic!("Wrong domain type"),
        }

        let domain = ParameterDomain::Discrete { low: 8, high: 128 };
        let json = serde_json::to_string(&domain).expect("JSON serialization should succeed");
        let _parsed: ParameterDomain =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        let domain =
            ParameterDomain::Categorical { choices: vec!["a".to_string(), "b".to_string()] };
        let json = serde_json::to_string(&domain).expect("JSON serialization should succeed");
        let _parsed: ParameterDomain =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
    }

    #[test]
    fn test_hyperparameter_space_serde() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let json = serde_json::to_string(&space).expect("JSON serialization should succeed");
        let parsed: HyperparameterSpace =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_trial_serde() {
        let mut config = HashMap::new();
        config.insert("lr".to_string(), ParameterValue::Float(0.01));
        let mut trial = Trial::new(0, config);
        trial.complete(0.5, 100);

        let json = serde_json::to_string(&trial).expect("JSON serialization should succeed");
        let parsed: Trial =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.id, 0);
        assert_eq!(parsed.score, 0.5);
        assert_eq!(parsed.status, TrialStatus::Completed);
    }

    #[test]
    fn test_trial_status_serde() {
        for status in [
            TrialStatus::Pending,
            TrialStatus::Running,
            TrialStatus::Completed,
            TrialStatus::Failed,
            TrialStatus::Pruned,
        ] {
            let json = serde_json::to_string(&status).expect("JSON serialization should succeed");
            let parsed: TrialStatus =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
            assert_eq!(status, parsed);
        }
    }

    #[test]
    fn test_search_strategy_serde() {
        let strategy = SearchStrategy::Grid;
        let json = serde_json::to_string(&strategy).expect("JSON serialization should succeed");
        let _parsed: SearchStrategy =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        let strategy = SearchStrategy::Random { n_samples: 100 };
        let json = serde_json::to_string(&strategy).expect("JSON serialization should succeed");
        let _parsed: SearchStrategy =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        let strategy = SearchStrategy::Bayesian {
            n_initial: 10,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            surrogate: SurrogateModel::TPE,
        };
        let json = serde_json::to_string(&strategy).expect("JSON serialization should succeed");
        let _parsed: SearchStrategy =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        let strategy = SearchStrategy::Hyperband { max_iter: 81, eta: 3.0 };
        let json = serde_json::to_string(&strategy).expect("JSON serialization should succeed");
        let _parsed: SearchStrategy =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
    }

    #[test]
    fn test_acquisition_function_serde() {
        for acq in [
            AcquisitionFunction::ExpectedImprovement,
            AcquisitionFunction::UpperConfidenceBound { kappa: 2.576 },
            AcquisitionFunction::ProbabilityOfImprovement,
        ] {
            let json = serde_json::to_string(&acq).expect("JSON serialization should succeed");
            let _parsed: AcquisitionFunction =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
        }
    }

    #[test]
    fn test_surrogate_model_serde() {
        for surrogate in [
            SurrogateModel::TPE,
            SurrogateModel::GaussianProcess,
            SurrogateModel::RandomForest { n_trees: 100 },
        ] {
            let json =
                serde_json::to_string(&surrogate).expect("JSON serialization should succeed");
            let _parsed: SurrogateModel =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
        }
    }
}

#[cfg(test)]
mod property_tests {
    use crate::optim::hpo::types::{HyperparameterSpace, ParameterDomain};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_continuous_domain_valid(low in -100.0f64..0.0, high in 0.0f64..100.0) {
            let domain = ParameterDomain::Continuous {
                low,
                high,
                log_scale: false,
            };
            let mut rng = rand::rng();
            let value = domain.sample(&mut rng);
            prop_assert!(domain.is_valid(&value));
        }

        #[test]
        fn prop_discrete_domain_valid(low in -100i64..0, high in 0i64..100) {
            let domain = ParameterDomain::Discrete { low, high };
            let mut rng = rand::rng();
            let value = domain.sample(&mut rng);
            prop_assert!(domain.is_valid(&value));
        }

        #[test]
        fn prop_space_sample_validates(
            lr_low in 1e-6f64..1e-4,
            lr_high in 1e-2f64..1.0,
            bs_low in 1i64..16,
            bs_high in 32i64..256
        ) {
            let mut space = HyperparameterSpace::new();
            space.add("lr", ParameterDomain::Continuous {
                low: lr_low,
                high: lr_high,
                log_scale: true,
            });
            space.add("batch_size", ParameterDomain::Discrete {
                low: bs_low,
                high: bs_high,
            });

            let mut rng = rand::rng();
            let config = space.sample_random(&mut rng);
            prop_assert!(space.validate(&config).is_ok());
        }
    }
}
