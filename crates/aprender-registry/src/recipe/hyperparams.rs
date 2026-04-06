//! Hyperparameter types for training recipes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: usize,
    /// Weight decay (L2 regularization).
    #[serde(default)]
    pub weight_decay: f64,
    /// Gradient clipping norm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f64>,
    /// Warmup steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_steps: Option<usize>,
    /// Custom parameters.
    #[serde(default)]
    pub custom: HashMap<String, HyperparamValue>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 32,
            epochs: 10,
            weight_decay: 0.0,
            max_grad_norm: None,
            warmup_steps: None,
            custom: HashMap::new(),
        }
    }
}

impl Hyperparameters {
    /// Create a new hyperparameters builder.
    #[must_use]
    pub fn builder() -> HyperparametersBuilder {
        HyperparametersBuilder::new()
    }

    /// Set a custom parameter.
    pub fn set_custom(&mut self, name: impl Into<String>, value: HyperparamValue) {
        self.custom.insert(name.into(), value);
    }

    /// Get a custom parameter.
    #[must_use]
    pub fn get_custom(&self, name: &str) -> Option<&HyperparamValue> {
        self.custom.get(name)
    }
}

/// A hyperparameter value that can be one of several types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HyperparamValue {
    /// Floating point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
    /// List of values.
    List(Vec<HyperparamValue>),
}

impl HyperparamValue {
    /// Try to get as float.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as integer.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to get as boolean.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as string.
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as list.
    #[must_use]
    pub fn as_list(&self) -> Option<&[HyperparamValue]> {
        match self {
            Self::List(l) => Some(l),
            _ => None,
        }
    }
}

impl From<f64> for HyperparamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<i64> for HyperparamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for HyperparamValue {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<bool> for HyperparamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<String> for HyperparamValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for HyperparamValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl<T: Into<HyperparamValue>> From<Vec<T>> for HyperparamValue {
    fn from(v: Vec<T>) -> Self {
        Self::List(v.into_iter().map(Into::into).collect())
    }
}

/// Builder for hyperparameters.
#[derive(Debug, Default)]
pub struct HyperparametersBuilder {
    params: Hyperparameters,
}

impl HyperparametersBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self { params: Hyperparameters::default() }
    }

    /// Set learning rate.
    #[must_use]
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.params.learning_rate = lr;
        self
    }

    /// Set batch size.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.params.batch_size = size;
        self
    }

    /// Set epochs.
    #[must_use]
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.params.epochs = epochs;
        self
    }

    /// Set weight decay.
    #[must_use]
    pub fn weight_decay(mut self, decay: f64) -> Self {
        self.params.weight_decay = decay;
        self
    }

    /// Set max gradient norm.
    #[must_use]
    pub fn max_grad_norm(mut self, norm: f64) -> Self {
        self.params.max_grad_norm = Some(norm);
        self
    }

    /// Set warmup steps.
    #[must_use]
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.params.warmup_steps = Some(steps);
        self
    }

    /// Add a custom parameter.
    #[must_use]
    pub fn custom(mut self, name: impl Into<String>, value: impl Into<HyperparamValue>) -> Self {
        self.params.custom.insert(name.into(), value.into());
        self
    }

    /// Build the hyperparameters.
    #[must_use]
    pub fn build(self) -> Hyperparameters {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameters_default() {
        let params = Hyperparameters::default();
        assert!((params.learning_rate - 1e-3).abs() < 1e-10);
        assert_eq!(params.batch_size, 32);
        assert_eq!(params.epochs, 10);
    }

    #[test]
    fn test_hyperparameters_builder() {
        let params = Hyperparameters::builder()
            .learning_rate(2e-5)
            .batch_size(64)
            .epochs(3)
            .weight_decay(0.01)
            .max_grad_norm(1.0)
            .warmup_steps(100)
            .custom("dropout", 0.1)
            .build();

        assert!((params.learning_rate - 2e-5).abs() < 1e-10);
        assert_eq!(params.batch_size, 64);
        assert_eq!(params.epochs, 3);
        assert!((params.weight_decay - 0.01).abs() < 1e-10);
        assert_eq!(params.max_grad_norm, Some(1.0));
        assert_eq!(params.warmup_steps, Some(100));
        assert_eq!(params.get_custom("dropout").and_then(|v| v.as_float()), Some(0.1));
    }

    #[test]
    fn test_hyperparam_value_types() {
        let float_val = HyperparamValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_int(), Some(3));

        let int_val = HyperparamValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let bool_val = HyperparamValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let string_val = HyperparamValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));

        let list_val =
            HyperparamValue::List(vec![HyperparamValue::Int(1), HyperparamValue::Int(2)]);
        assert_eq!(list_val.as_list().map(|l| l.len()), Some(2));
    }

    #[test]
    fn test_hyperparam_value_from() {
        let from_float: HyperparamValue = 3.14.into();
        assert!(matches!(from_float, HyperparamValue::Float(_)));

        let from_int: HyperparamValue = 42i64.into();
        assert!(matches!(from_int, HyperparamValue::Int(_)));

        let from_bool: HyperparamValue = true.into();
        assert!(matches!(from_bool, HyperparamValue::Bool(_)));

        let from_str: HyperparamValue = "test".into();
        assert!(matches!(from_str, HyperparamValue::String(_)));

        let from_vec: HyperparamValue = vec![1i64, 2i64, 3i64].into();
        assert!(matches!(from_vec, HyperparamValue::List(_)));
    }

    #[test]
    fn test_hyperparameters_serialization() {
        let params = Hyperparameters::builder()
            .learning_rate(1e-4)
            .batch_size(16)
            .custom("hidden_size", 768i64)
            .build();

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: Hyperparameters = serde_json::from_str(&json).unwrap();

        assert!((params.learning_rate - deserialized.learning_rate).abs() < 1e-10);
        assert_eq!(params.batch_size, deserialized.batch_size);
    }

    #[test]
    fn test_set_get_custom() {
        let mut params = Hyperparameters::default();
        params.set_custom("test_param", HyperparamValue::Float(0.5));

        let value = params.get_custom("test_param");
        assert_eq!(value.and_then(|v| v.as_float()), Some(0.5));

        assert!(params.get_custom("nonexistent").is_none());
    }
}
