//! Type definitions for SQLite storage backend.
//!
//! Contains parameter values, filter operations, and metadata structures.

use crate::storage::RunStatus;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameter value types for log_param
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParameterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ParameterValue>),
    Dict(HashMap<String, ParameterValue>),
}

impl ParameterValue {
    /// Get type name for storage
    pub fn type_name(&self) -> &'static str {
        match self {
            ParameterValue::String(_) => "string",
            ParameterValue::Int(_) => "int",
            ParameterValue::Float(_) => "float",
            ParameterValue::Bool(_) => "bool",
            ParameterValue::List(_) => "list",
            ParameterValue::Dict(_) => "dict",
        }
    }

    /// Serialize to JSON string for storage
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Deserialize from JSON string
    pub fn from_json(s: &str) -> Option<Self> {
        serde_json::from_str(s).ok()
    }
}

/// Filter operations for parameter search
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
    StartsWith,
}

/// Parameter filter for searching runs
#[derive(Debug, Clone)]
pub struct ParamFilter {
    pub key: String,
    pub op: FilterOp,
    pub value: ParameterValue,
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Run metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub params: HashMap<String, ParameterValue>,
    pub tags: HashMap<String, String>,
}

/// Artifact reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub run_id: String,
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ParameterValue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_type_name() {
        assert_eq!(ParameterValue::String("test".to_string()).type_name(), "string");
        assert_eq!(ParameterValue::Int(42).type_name(), "int");
        assert_eq!(ParameterValue::Float(3.14).type_name(), "float");
        assert_eq!(ParameterValue::Bool(true).type_name(), "bool");
        assert_eq!(ParameterValue::List(vec![]).type_name(), "list");
        assert_eq!(ParameterValue::Dict(HashMap::new()).type_name(), "dict");
    }

    #[test]
    fn test_parameter_value_json_roundtrip() {
        let values = vec![
            ParameterValue::String("hello".to_string()),
            ParameterValue::Int(42),
            ParameterValue::Float(3.14),
            ParameterValue::Bool(true),
            ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]),
        ];

        for value in values {
            let json = value.to_json();
            let parsed = ParameterValue::from_json(&json).expect("parsing should succeed");
            assert_eq!(value, parsed);
        }
    }

    #[test]
    fn test_parameter_value_dict() {
        let mut dict = HashMap::new();
        dict.insert("nested".to_string(), ParameterValue::Int(42));
        let param = ParameterValue::Dict(dict);
        assert_eq!(param.type_name(), "dict");

        let json = param.to_json();
        let parsed = ParameterValue::from_json(&json).expect("parsing should succeed");
        assert_eq!(param, parsed);
    }

    #[test]
    fn test_parameter_value_from_invalid_json() {
        let result = ParameterValue::from_json("invalid json {{{");
        assert!(result.is_none());
    }
}
