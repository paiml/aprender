//! Label validation for metrics cardinality control
//!
//! Implements Prometheus best practices for label management:
//! - Allowlist known labels to prevent explosion
//! - Reject high-cardinality values (user IDs, timestamps)
//! - Enforce naming conventions
//!
//! # Peer-Reviewed Foundation
//! - Prometheus Authors. "Metric and Label Naming."
//!   https://prometheus.io/docs/practices/naming/

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Type alias for label key-value pairs
pub type Labels = HashMap<String, String>;

/// Single label key-value pair
pub type LabelPair = (String, String);

/// Label validation errors
#[derive(Debug, Error, PartialEq)]
pub enum LabelError {
    #[error("unknown label: {0} (not in allowlist)")]
    UnknownLabel(String),

    #[error("label value too long: {key} ({len} > {max})")]
    ValueTooLong { key: String, len: usize, max: usize },

    #[error("numeric label value rejected: {key}={value} (high cardinality risk)")]
    NumericValue { key: String, value: String },

    #[error("invalid label name: {0} (must match [a-zA-Z_][a-zA-Z0-9_]*)")]
    InvalidName(String),

    #[error("reserved label prefix: {0} (__ is reserved)")]
    ReservedPrefix(String),

    #[error("too many labels: {count} (max {max})")]
    TooManyLabels { count: usize, max: usize },
}

/// Label validator with configurable rules
///
/// Pattern: Prometheus depguard-style allowlist enforcement
#[derive(Debug, Clone)]
pub struct LabelValidator {
    /// Allowed label names (empty = allow all)
    allowed_labels: HashSet<String>,
    /// Maximum label value length
    max_value_length: usize,
    /// Maximum labels per metric
    max_labels: usize,
    /// Labels that can have numeric values
    numeric_allowed: HashSet<String>,
    /// Whether to enforce allowlist (false = warn only)
    strict: bool,
}

impl Default for LabelValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelValidator {
    /// Create validator with default settings
    pub fn new() -> Self {
        let mut numeric_allowed = HashSet::new();
        // These labels commonly have numeric values
        numeric_allowed.insert("status_code".to_string());
        numeric_allowed.insert("exit_code".to_string());
        numeric_allowed.insert("port".to_string());
        numeric_allowed.insert("fd".to_string());
        numeric_allowed.insert("pid".to_string());
        numeric_allowed.insert("cpu".to_string());

        Self {
            allowed_labels: HashSet::new(), // Empty = allow all
            max_value_length: 128,
            max_labels: 10,
            numeric_allowed,
            strict: false,
        }
    }

    /// Create strict validator with explicit allowlist
    pub fn strict(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_labels: allowed.into_iter().map(Into::into).collect(),
            max_value_length: 128,
            max_labels: 10,
            numeric_allowed: HashSet::new(),
            strict: true,
        }
    }

    /// Add allowed labels
    pub fn allow_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed_labels.extend(labels.into_iter().map(Into::into));
        self
    }

    /// Add labels that can have numeric values
    pub fn allow_numeric(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.numeric_allowed.extend(labels.into_iter().map(Into::into));
        self
    }

    /// Set maximum value length
    pub fn max_value_length(mut self, max: usize) -> Self {
        self.max_value_length = max;
        self
    }

    /// Set maximum labels per metric
    pub fn max_labels(mut self, max: usize) -> Self {
        self.max_labels = max;
        self
    }

    /// Validate a set of labels
    pub fn validate(&self, labels: &Labels) -> Result<(), LabelError> {
        // Check label count
        if labels.len() > self.max_labels {
            return Err(LabelError::TooManyLabels { count: labels.len(), max: self.max_labels });
        }

        for (key, value) in labels {
            self.validate_name(key)?;
            self.validate_value(key, value)?;
        }

        Ok(())
    }

    /// Validate label name
    fn validate_name(&self, name: &str) -> Result<(), LabelError> {
        // Check reserved prefix
        if name.starts_with("__") {
            return Err(LabelError::ReservedPrefix(name.to_string()));
        }

        // Check name format: [a-zA-Z_][a-zA-Z0-9_]*
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return Err(LabelError::InvalidName(name.to_string())),
        }
        for c in chars {
            if !c.is_ascii_alphanumeric() && c != '_' {
                return Err(LabelError::InvalidName(name.to_string()));
            }
        }

        // Check allowlist (if strict mode and non-empty allowlist)
        if self.strict && !self.allowed_labels.is_empty() && !self.allowed_labels.contains(name) {
            return Err(LabelError::UnknownLabel(name.to_string()));
        }

        Ok(())
    }

    /// Validate label value
    fn validate_value(&self, key: &str, value: &str) -> Result<(), LabelError> {
        // Check length
        if value.len() > self.max_value_length {
            return Err(LabelError::ValueTooLong {
                key: key.to_string(),
                len: value.len(),
                max: self.max_value_length,
            });
        }

        // Check for numeric values (high cardinality risk)
        if !self.numeric_allowed.contains(key) && looks_numeric(value) {
            return Err(LabelError::NumericValue {
                key: key.to_string(),
                value: value.to_string(),
            });
        }

        Ok(())
    }
}

/// Check if a string looks like a number (potential high cardinality)
fn looks_numeric(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    // Check if it parses as a number
    s.parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create labels from tuples
    fn labels_from_pairs<I, K, V>(pairs: I) -> Labels
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        pairs.into_iter().map(|(k, v)| (k.into(), v.into())).collect()
    }

    #[test]
    fn test_valid_labels() {
        let validator = LabelValidator::new();
        let labels = labels_from_pairs([("method", "GET"), ("path", "/api/users")]);
        assert!(validator.validate(&labels).is_ok());
    }

    #[test]
    fn test_reserved_prefix() {
        let validator = LabelValidator::new();
        let labels = labels_from_pairs([("__internal", "value")]);
        assert_eq!(
            validator.validate(&labels),
            Err(LabelError::ReservedPrefix("__internal".to_string()))
        );
    }

    #[test]
    fn test_invalid_name_starts_with_digit() {
        let validator = LabelValidator::new();
        let labels = labels_from_pairs([("123invalid", "value")]);
        assert_eq!(
            validator.validate(&labels),
            Err(LabelError::InvalidName("123invalid".to_string()))
        );
    }

    #[test]
    fn test_value_too_long() {
        let validator = LabelValidator::new().max_value_length(10);
        let labels = labels_from_pairs([("key", "this is way too long")]);
        match validator.validate(&labels) {
            Err(LabelError::ValueTooLong { key, len, max }) => {
                assert_eq!(key, "key");
                assert_eq!(len, 20); // "this is way too long" is 20 chars
                assert_eq!(max, 10);
            }
            _ => panic!("expected ValueTooLong error"),
        }
    }

    #[test]
    fn test_numeric_value_rejected() {
        let validator = LabelValidator::new();
        let labels = labels_from_pairs([("user_id", "12345")]);
        assert!(matches!(validator.validate(&labels), Err(LabelError::NumericValue { .. })));
    }

    #[test]
    fn test_numeric_value_allowed_for_specific_labels() {
        let validator = LabelValidator::new();
        let labels = labels_from_pairs([("status_code", "200")]);
        assert!(validator.validate(&labels).is_ok());
    }

    #[test]
    fn test_too_many_labels() {
        let validator = LabelValidator::new().max_labels(2);
        let labels = labels_from_pairs([("a", "1"), ("b", "2"), ("c", "3")]);
        assert_eq!(
            validator.validate(&labels),
            Err(LabelError::TooManyLabels { count: 3, max: 2 })
        );
    }

    #[test]
    fn test_strict_mode_unknown_label() {
        let validator = LabelValidator::strict(["method", "path"]);
        let labels = labels_from_pairs([("unknown", "value")]);
        assert_eq!(
            validator.validate(&labels),
            Err(LabelError::UnknownLabel("unknown".to_string()))
        );
    }

    #[test]
    fn test_strict_mode_allowed_label() {
        let validator = LabelValidator::strict(["method", "path"]);
        let labels = labels_from_pairs([("method", "GET")]);
        assert!(validator.validate(&labels).is_ok());
    }

    #[test]
    fn test_looks_numeric() {
        assert!(looks_numeric("123"));
        assert!(looks_numeric("12.34"));
        assert!(looks_numeric("-5"));
        assert!(looks_numeric("1e10"));
        assert!(!looks_numeric("abc"));
        assert!(!looks_numeric("12abc"));
        assert!(!looks_numeric(""));
    }
}
