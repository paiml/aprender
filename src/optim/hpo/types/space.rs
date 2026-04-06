//! Hyperparameter search space

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::optim::hpo::error::{HPOError, Result};

use super::parameter::{ParameterDomain, ParameterValue};

/// Hyperparameter search space
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Parameter name -> domain mapping
    params: HashMap<String, ParameterDomain>,
}

impl HyperparameterSpace {
    /// Create an empty search space
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter to the search space
    pub fn add(&mut self, name: &str, domain: ParameterDomain) {
        self.params.insert(name.to_string(), domain);
    }

    /// Get a parameter domain
    pub fn get(&self, name: &str) -> Option<&ParameterDomain> {
        self.params.get(name)
    }

    /// Check if space is empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get number of parameters
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Iterate over parameters
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParameterDomain)> {
        self.params.iter()
    }

    /// Sample a random configuration
    pub fn sample_random<R: Rng>(&self, rng: &mut R) -> HashMap<String, ParameterValue> {
        self.params.iter().map(|(name, domain)| (name.clone(), domain.sample(rng))).collect()
    }

    /// Validate a configuration
    pub fn validate(&self, config: &HashMap<String, ParameterValue>) -> Result<()> {
        for (name, domain) in &self.params {
            match config.get(name) {
                Some(value) if domain.is_valid(value) => {}
                Some(value) => {
                    return Err(HPOError::InvalidValue(name.clone(), format!("{value:?}")))
                }
                None => return Err(HPOError::ParameterNotFound(name.clone())),
            }
        }
        Ok(())
    }
}
