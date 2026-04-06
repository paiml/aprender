//! Grid search for hyperparameter optimization

use std::collections::HashMap;

use super::types::{HyperparameterSpace, ParameterDomain, ParameterValue};

/// Grid search generator
#[derive(Debug, Clone)]
pub struct GridSearch {
    space: HyperparameterSpace,
    /// Grid points per continuous parameter
    pub(crate) n_points: usize,
}

/// Generate grid values for a single parameter domain.
fn domain_grid_values(domain: &ParameterDomain, n_points: usize) -> Vec<ParameterValue> {
    match domain {
        ParameterDomain::Continuous { low, high, log_scale } => {
            let divisor = (n_points - 1) as f64;
            if *log_scale {
                let log_low = low.max(f64::MIN_POSITIVE).ln();
                let log_high = high.max(f64::MIN_POSITIVE).ln();
                (0..n_points)
                    .map(|i| {
                        let t = i as f64 / divisor;
                        ParameterValue::Float((log_low + t * (log_high - log_low)).exp())
                    })
                    .collect()
            } else {
                (0..n_points)
                    .map(|i| {
                        let t = i as f64 / divisor;
                        ParameterValue::Float(low + t * (high - low))
                    })
                    .collect()
            }
        }
        ParameterDomain::Discrete { low, high } => {
            (*low..=*high).map(ParameterValue::Int).collect()
        }
        ParameterDomain::Categorical { choices } => {
            choices.iter().map(|c| ParameterValue::Categorical(c.clone())).collect()
        }
    }
}

impl GridSearch {
    /// Create new grid search
    pub fn new(space: HyperparameterSpace, n_points: usize) -> Self {
        Self { space, n_points: n_points.max(2) }
    }

    /// Generate all grid configurations
    pub fn configurations(&self) -> Vec<HashMap<String, ParameterValue>> {
        let param_values: Vec<(String, Vec<ParameterValue>)> = self
            .space
            .iter()
            .map(|(name, domain)| (name.clone(), domain_grid_values(domain, self.n_points)))
            .collect();

        // Generate cartesian product
        Self::cartesian_product(&param_values)
    }

    fn cartesian_product(
        param_values: &[(String, Vec<ParameterValue>)],
    ) -> Vec<HashMap<String, ParameterValue>> {
        if param_values.is_empty() {
            return vec![HashMap::new()];
        }

        let (name, values) = &param_values[0];
        let rest = param_values.get(1..).unwrap_or_default();
        let rest_configs = Self::cartesian_product(rest);

        values
            .iter()
            .flat_map(|v| {
                rest_configs.iter().map(move |config| {
                    let mut new_config = config.clone();
                    new_config.insert(name.clone(), v.clone());
                    new_config
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_search_new() {
        let space = HyperparameterSpace::new();
        let grid = GridSearch::new(space, 5);
        assert_eq!(grid.n_points, 5);
    }

    #[test]
    fn test_grid_search_empty_space() {
        let space = HyperparameterSpace::new();
        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        assert_eq!(configs.len(), 1); // One empty config
    }

    #[test]
    fn test_grid_search_single_param() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        assert_eq!(configs.len(), 5);

        // Check values are evenly spaced
        let values: Vec<f64> = configs
            .iter()
            .map(|c| c.get("lr").expect("key should exist").as_float().expect("key should exist"))
            .collect();
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_search_multiple_params() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });
        space.add(
            "act",
            ParameterDomain::Categorical { choices: vec!["relu".to_string(), "gelu".to_string()] },
        );

        let grid = GridSearch::new(space, 3);
        let configs = grid.configurations();
        // 3 lr values * 2 activation functions = 6
        assert_eq!(configs.len(), 6);
    }

    #[test]
    fn test_grid_search_discrete() {
        let mut space = HyperparameterSpace::new();
        space.add("batch_size", ParameterDomain::Discrete { low: 8, high: 10 });

        let grid = GridSearch::new(space, 5);
        let configs = grid.configurations();
        // Discrete [8,9,10] = 3 values
        assert_eq!(configs.len(), 3);
    }

    #[test]
    fn test_grid_search_log_scale() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 1e-4, high: 1e-1, log_scale: true });

        let grid = GridSearch::new(space, 4);
        let configs = grid.configurations();

        let values: Vec<f64> = configs
            .iter()
            .map(|c| c.get("lr").expect("key should exist").as_float().expect("key should exist"))
            .collect();

        // Log scale should give approximately: 1e-4, 1e-3, 1e-2, 1e-1
        assert!(values[0] < 1e-3);
        assert!(values[3] > 1e-2);
    }

    #[test]
    fn test_grid_search_min_n_points() {
        let space = HyperparameterSpace::new();
        let grid = GridSearch::new(space, 1); // Should be clamped to 2
        assert_eq!(grid.n_points, 2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_grid_search_size(n_points in 2usize..10) {
            let mut space = HyperparameterSpace::new();
            space.add("x", ParameterDomain::Continuous {
                low: 0.0,
                high: 1.0,
                log_scale: false,
            });

            let grid = GridSearch::new(space, n_points);
            let configs = grid.configurations();
            prop_assert_eq!(configs.len(), n_points);
        }
    }
}
