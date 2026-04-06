//! Parameter Logging API (GH-73)
//!
//! Provides structured parameter tracking for training experiments.
//! Parameters are stored as typed key-value pairs with JSON serialization
//! and diff support for comparing experiment configurations.
//!
//! # Example
//!
//! ```
//! use entrenar::monitor::params::{ParamLogger, ParamValue};
//!
//! let mut logger = ParamLogger::new();
//! logger.log_param("learning_rate", 1e-4_f64);
//! logger.log_param("epochs", 10_i64);
//! logger.log_param("model", "llama-7b");
//! logger.log_param("use_lora", true);
//!
//! assert_eq!(
//!     logger.get_param("learning_rate"),
//!     Some(&ParamValue::Float(1e-4))
//! );
//!
//! let json = logger.to_json();
//! assert!(json.contains("learning_rate"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// ParamValue
// =============================================================================

/// A typed parameter value supporting common training hyperparameter types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    /// String parameter (e.g., model name, optimizer type)
    String(String),
    /// Floating-point parameter (e.g., learning rate, weight decay)
    Float(f64),
    /// Integer parameter (e.g., epochs, batch size, seed)
    Int(i64),
    /// Boolean parameter (e.g., use_lora, freeze_base)
    Bool(bool),
}

impl ParamValue {
    /// Returns the type name of this value as a static string.
    pub fn type_name(&self) -> &'static str {
        match self {
            ParamValue::String(_) => "string",
            ParamValue::Float(_) => "float",
            ParamValue::Int(_) => "int",
            ParamValue::Bool(_) => "bool",
        }
    }
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::String(s) => write!(f, "{s}"),
            ParamValue::Float(v) => write!(f, "{v}"),
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Bool(v) => write!(f, "{v}"),
        }
    }
}

// -- From impls for ergonomic `log_param` calls --

impl From<&str> for ParamValue {
    fn from(s: &str) -> Self {
        ParamValue::String(s.to_string())
    }
}

impl From<String> for ParamValue {
    fn from(s: String) -> Self {
        ParamValue::String(s)
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::Float(v)
    }
}

impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        ParamValue::Float(f64::from(v))
    }
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::Int(v)
    }
}

impl From<i32> for ParamValue {
    fn from(v: i32) -> Self {
        ParamValue::Int(i64::from(v))
    }
}

impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        ParamValue::Bool(v)
    }
}

// =============================================================================
// ParamDiff
// =============================================================================

/// Result of comparing two `ParamLogger` instances.
///
/// Captures which parameters were changed, added, or removed between
/// two experiment configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDiff {
    /// Parameters present in both loggers but with different values.
    /// Maps key -> (old_value, new_value).
    pub changed: HashMap<String, (ParamValue, ParamValue)>,
    /// Parameters present only in the *other* logger (new additions).
    pub added: HashMap<String, ParamValue>,
    /// Parameters present only in *self* (removed in other).
    pub removed: HashMap<String, ParamValue>,
}

impl ParamDiff {
    /// Returns `true` if there are no differences.
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.added.is_empty() && self.removed.is_empty()
    }

    /// Total number of differences (changed + added + removed).
    pub fn len(&self) -> usize {
        self.changed.len() + self.added.len() + self.removed.len()
    }
}

// =============================================================================
// ParamLogger
// =============================================================================

/// Structured parameter logger for training experiments.
///
/// Stores hyperparameters, configuration flags, and other experiment metadata
/// as typed key-value pairs. Supports JSON serialization and diff comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamLogger {
    params: HashMap<String, ParamValue>,
}

impl ParamLogger {
    /// Create a new, empty parameter logger.
    pub fn new() -> Self {
        Self { params: HashMap::new() }
    }

    /// Log a single parameter. Overwrites any existing value for the key.
    pub fn log_param(&mut self, key: &str, value: impl Into<ParamValue>) {
        self.params.insert(key.to_string(), value.into());
    }

    /// Log multiple parameters at once. Overwrites existing values.
    pub fn log_params(&mut self, params: HashMap<String, ParamValue>) {
        self.params.extend(params);
    }

    /// Retrieve a parameter by key.
    pub fn get_param(&self, key: &str) -> Option<&ParamValue> {
        self.params.get(key)
    }

    /// Retrieve all parameters as a reference to the underlying map.
    pub fn get_all_params(&self) -> &HashMap<String, ParamValue> {
        &self.params
    }

    /// Returns the number of logged parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns `true` if no parameters have been logged.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Serialize all parameters to a JSON string.
    ///
    /// Keys are sorted for deterministic output.
    pub fn to_json(&self) -> String {
        // Use BTreeMap for sorted keys -> deterministic JSON
        let sorted: std::collections::BTreeMap<&String, &ParamValue> = self.params.iter().collect();
        serde_json::to_string_pretty(&sorted).unwrap_or_else(|e| {
            eprintln!("ParamLogger JSON serialization failed: {e}");
            "{}".to_string()
        })
    }

    /// Compute the diff between `self` and `other`.
    ///
    /// - **changed**: keys present in both with different values
    /// - **added**: keys in `other` but not in `self`
    /// - **removed**: keys in `self` but not in `other`
    pub fn diff(&self, other: &ParamLogger) -> ParamDiff {
        let mut changed = HashMap::new();
        let mut added = HashMap::new();
        let mut removed = HashMap::new();

        // Find changed and removed
        for (key, self_val) in &self.params {
            match other.params.get(key) {
                Some(other_val) if self_val != other_val => {
                    changed.insert(key.clone(), (self_val.clone(), other_val.clone()));
                }
                None => {
                    removed.insert(key.clone(), self_val.clone());
                }
                _ => {} // Same value, no diff
            }
        }

        // Find added (in other but not in self)
        for (key, other_val) in &other.params {
            if !self.params.contains_key(key) {
                added.insert(key.clone(), other_val.clone());
            }
        }

        ParamDiff { changed, added, removed }
    }
}

impl Default for ParamLogger {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_logger_new_is_empty() {
        let logger = ParamLogger::new();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_log_param_string() {
        let mut logger = ParamLogger::new();
        logger.log_param("model", "llama-7b");
        assert_eq!(logger.get_param("model"), Some(&ParamValue::String("llama-7b".to_string())));
    }

    #[test]
    fn test_log_param_float() {
        let mut logger = ParamLogger::new();
        logger.log_param("lr", 1e-4_f64);
        assert_eq!(logger.get_param("lr"), Some(&ParamValue::Float(1e-4)));
    }

    #[test]
    fn test_log_param_f32_converts_to_f64() {
        let mut logger = ParamLogger::new();
        logger.log_param("weight_decay", 0.01_f32);
        assert_eq!(logger.get_param("weight_decay"), Some(&ParamValue::Float(f64::from(0.01_f32))));
    }

    #[test]
    fn test_log_param_int() {
        let mut logger = ParamLogger::new();
        logger.log_param("epochs", 10_i64);
        assert_eq!(logger.get_param("epochs"), Some(&ParamValue::Int(10)));
    }

    #[test]
    fn test_log_param_i32_converts_to_i64() {
        let mut logger = ParamLogger::new();
        logger.log_param("batch_size", 32_i32);
        assert_eq!(logger.get_param("batch_size"), Some(&ParamValue::Int(32)));
    }

    #[test]
    fn test_log_param_bool() {
        let mut logger = ParamLogger::new();
        logger.log_param("use_lora", true);
        assert_eq!(logger.get_param("use_lora"), Some(&ParamValue::Bool(true)));
    }

    #[test]
    fn test_log_param_owned_string() {
        let mut logger = ParamLogger::new();
        logger.log_param("optimizer", String::from("adamw"));
        assert_eq!(logger.get_param("optimizer"), Some(&ParamValue::String("adamw".to_string())));
    }

    #[test]
    fn test_log_param_overwrites() {
        let mut logger = ParamLogger::new();
        logger.log_param("lr", 1e-3_f64);
        logger.log_param("lr", 1e-4_f64);
        assert_eq!(logger.get_param("lr"), Some(&ParamValue::Float(1e-4)));
        assert_eq!(logger.len(), 1);
    }

    #[test]
    fn test_get_param_missing_returns_none() {
        let logger = ParamLogger::new();
        assert_eq!(logger.get_param("nonexistent"), None);
    }

    #[test]
    fn test_log_params_bulk() {
        let mut logger = ParamLogger::new();
        let mut params = HashMap::new();
        params.insert("lr".to_string(), ParamValue::Float(1e-4));
        params.insert("epochs".to_string(), ParamValue::Int(10));
        params.insert("model".to_string(), ParamValue::String("gpt2".to_string()));
        logger.log_params(params);

        assert_eq!(logger.len(), 3);
        assert_eq!(logger.get_param("lr"), Some(&ParamValue::Float(1e-4)));
        assert_eq!(logger.get_param("epochs"), Some(&ParamValue::Int(10)));
    }

    #[test]
    fn test_get_all_params() {
        let mut logger = ParamLogger::new();
        logger.log_param("a", 1_i64);
        logger.log_param("b", 2_i64);

        let all = logger.get_all_params();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("a"));
        assert!(all.contains_key("b"));
    }

    #[test]
    fn test_to_json_deterministic() {
        let mut logger = ParamLogger::new();
        logger.log_param("z_param", 1_i64);
        logger.log_param("a_param", 2_i64);
        logger.log_param("m_param", 3_i64);

        let json = logger.to_json();
        // Keys should be sorted alphabetically
        let a_pos = json.find("a_param").expect("a_param not found");
        let m_pos = json.find("m_param").expect("m_param not found");
        let z_pos = json.find("z_param").expect("z_param not found");
        assert!(a_pos < m_pos, "a_param should come before m_param");
        assert!(m_pos < z_pos, "m_param should come before z_param");
    }

    #[test]
    fn test_to_json_contains_values() {
        let mut logger = ParamLogger::new();
        logger.log_param("lr", 0.001_f64);
        logger.log_param("use_lora", true);
        logger.log_param("model", "gpt2");

        let json = logger.to_json();
        assert!(json.contains("0.001"));
        assert!(json.contains("true"));
        assert!(json.contains("gpt2"));
    }

    #[test]
    fn test_to_json_empty() {
        let logger = ParamLogger::new();
        let json = logger.to_json();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_to_json_roundtrip() {
        let mut logger = ParamLogger::new();
        logger.log_param("lr", 1e-4_f64);
        logger.log_param("epochs", 10_i64);
        logger.log_param("model", "llama");
        logger.log_param("lora", true);

        let json = logger.to_json();
        let deserialized: std::collections::BTreeMap<String, ParamValue> =
            serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(deserialized.len(), 4);
        assert_eq!(deserialized.get("lr"), Some(&ParamValue::Float(1e-4)));
        assert_eq!(deserialized.get("epochs"), Some(&ParamValue::Int(10)));
        assert_eq!(deserialized.get("model"), Some(&ParamValue::String("llama".to_string())));
        assert_eq!(deserialized.get("lora"), Some(&ParamValue::Bool(true)));
    }

    // =========================================================================
    // Diff tests
    // =========================================================================

    #[test]
    fn test_diff_identical_is_empty() {
        let mut a = ParamLogger::new();
        a.log_param("lr", 1e-4_f64);
        a.log_param("epochs", 10_i64);

        let mut b = ParamLogger::new();
        b.log_param("lr", 1e-4_f64);
        b.log_param("epochs", 10_i64);

        let diff = a.diff(&b);
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_diff_empty_loggers() {
        let a = ParamLogger::new();
        let b = ParamLogger::new();
        let diff = a.diff(&b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_changed_values() {
        let mut a = ParamLogger::new();
        a.log_param("lr", 1e-3_f64);
        a.log_param("epochs", 10_i64);

        let mut b = ParamLogger::new();
        b.log_param("lr", 1e-4_f64);
        b.log_param("epochs", 10_i64);

        let diff = a.diff(&b);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(
            diff.changed.get("lr"),
            Some(&(ParamValue::Float(1e-3), ParamValue::Float(1e-4)))
        );
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_added_params() {
        let mut a = ParamLogger::new();
        a.log_param("lr", 1e-4_f64);

        let mut b = ParamLogger::new();
        b.log_param("lr", 1e-4_f64);
        b.log_param("warmup", 100_i64);

        let diff = a.diff(&b);
        assert!(diff.changed.is_empty());
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added.get("warmup"), Some(&ParamValue::Int(100)));
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_removed_params() {
        let mut a = ParamLogger::new();
        a.log_param("lr", 1e-4_f64);
        a.log_param("warmup", 100_i64);

        let mut b = ParamLogger::new();
        b.log_param("lr", 1e-4_f64);

        let diff = a.diff(&b);
        assert!(diff.changed.is_empty());
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed.get("warmup"), Some(&ParamValue::Int(100)));
    }

    #[test]
    fn test_diff_mixed_changes() {
        let mut a = ParamLogger::new();
        a.log_param("lr", 1e-3_f64);
        a.log_param("old_param", "remove_me");
        a.log_param("same", 42_i64);

        let mut b = ParamLogger::new();
        b.log_param("lr", 1e-4_f64);
        b.log_param("new_param", true);
        b.log_param("same", 42_i64);

        let diff = a.diff(&b);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.len(), 3);
        assert!(!diff.is_empty());

        assert!(diff.changed.contains_key("lr"));
        assert!(diff.added.contains_key("new_param"));
        assert!(diff.removed.contains_key("old_param"));
    }

    #[test]
    fn test_diff_type_change_counts_as_changed() {
        let mut a = ParamLogger::new();
        a.log_param("value", 10_i64);

        let mut b = ParamLogger::new();
        b.log_param("value", 10.0_f64);

        let diff = a.diff(&b);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(
            diff.changed.get("value"),
            Some(&(ParamValue::Int(10), ParamValue::Float(10.0)))
        );
    }

    // =========================================================================
    // ParamValue tests
    // =========================================================================

    #[test]
    fn test_param_value_type_name() {
        assert_eq!(ParamValue::String("x".into()).type_name(), "string");
        assert_eq!(ParamValue::Float(1.0).type_name(), "float");
        assert_eq!(ParamValue::Int(1).type_name(), "int");
        assert_eq!(ParamValue::Bool(true).type_name(), "bool");
    }

    #[test]
    fn test_param_value_display() {
        assert_eq!(format!("{}", ParamValue::String("hello".into())), "hello");
        assert_eq!(format!("{}", ParamValue::Float(3.14)), "3.14");
        assert_eq!(format!("{}", ParamValue::Int(42)), "42");
        assert_eq!(format!("{}", ParamValue::Bool(false)), "false");
    }

    #[test]
    fn test_param_value_serde_roundtrip() {
        let values = vec![
            ParamValue::String("test".into()),
            ParamValue::Float(1.23),
            ParamValue::Int(-5),
            ParamValue::Bool(true),
        ];
        for val in &values {
            let json = serde_json::to_string(val).expect("serialize");
            let back: ParamValue = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&back, val);
        }
    }

    #[test]
    fn test_param_diff_is_empty_and_len() {
        let diff =
            ParamDiff { changed: HashMap::new(), added: HashMap::new(), removed: HashMap::new() };
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);

        let mut diff2 =
            ParamDiff { changed: HashMap::new(), added: HashMap::new(), removed: HashMap::new() };
        diff2.added.insert("x".to_string(), ParamValue::Int(1));
        assert!(!diff2.is_empty());
        assert_eq!(diff2.len(), 1);
    }

    #[test]
    fn test_default_impl() {
        let logger = ParamLogger::default();
        assert!(logger.is_empty());
    }
}
