//! Parameter value and domain types

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Parameter value (sampled from domain)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    Categorical(String),
}

impl ParameterValue {
    /// Get as float (converts int to float if needed)
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            ParameterValue::Categorical(_) => None,
        }
    }

    /// Get as int
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            ParameterValue::Categorical(_) => None,
        }
    }

    /// Get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParameterValue::Categorical(s) => Some(s),
            _ => None,
        }
    }
}

/// Parameter domain (search space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterDomain {
    /// Continuous range [low, high], optionally log-scaled
    Continuous { low: f64, high: f64, log_scale: bool },
    /// Discrete integer range [low, high]
    Discrete { low: i64, high: i64 },
    /// Categorical choices
    Categorical { choices: Vec<String> },
}

impl ParameterDomain {
    /// Sample a random value from this domain
    pub fn sample<R: Rng>(&self, rng: &mut R) -> ParameterValue {
        contract_pre_sample!();
        match self {
            ParameterDomain::Continuous { low, high, log_scale } => {
                let value = if *log_scale {
                    let log_low = low.max(f64::MIN_POSITIVE).ln();
                    let log_high = high.max(f64::MIN_POSITIVE).ln();
                    let log_val = log_low + rng.random::<f64>() * (log_high - log_low);
                    log_val.exp()
                } else {
                    low + rng.random::<f64>() * (high - low)
                };
                ParameterValue::Float(value)
            }
            ParameterDomain::Discrete { low, high } => {
                let range = (*high - *low + 1) as usize;
                let offset = (rng.random::<f64>() * range as f64).floor() as i64;
                let value = (*low + offset).min(*high);
                ParameterValue::Int(value)
            }
            ParameterDomain::Categorical { choices } => {
                let idx = (rng.random::<f64>() * choices.len() as f64).floor() as usize;
                let idx = idx.min(choices.len() - 1);
                ParameterValue::Categorical(choices[idx].clone())
            }
        }
    }

    /// Check if a value is valid for this domain
    pub fn is_valid(&self, value: &ParameterValue) -> bool {
        match (self, value) {
            (ParameterDomain::Continuous { low, high, .. }, ParameterValue::Float(v)) => {
                *v >= *low && *v <= *high
            }
            (ParameterDomain::Discrete { low, high }, ParameterValue::Int(v)) => {
                *v >= *low && *v <= *high
            }
            (ParameterDomain::Categorical { choices }, ParameterValue::Categorical(s)) => {
                choices.contains(s)
            }
            (
                ParameterDomain::Continuous { .. } | ParameterDomain::Categorical { .. },
                ParameterValue::Int(_),
            )
            | (
                ParameterDomain::Continuous { .. } | ParameterDomain::Discrete { .. },
                ParameterValue::Categorical(_),
            )
            | (
                ParameterDomain::Discrete { .. } | ParameterDomain::Categorical { .. },
                ParameterValue::Float(_),
            ) => false,
        }
    }
}
