use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("Failed to read contract file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("Schema violation: {0}")]
    Schema(String),

    #[error("Missing required field: {section}.{field}")]
    MissingField { section: String, field: String },

    #[error("Invalid reference: {from} references non-existent {to}")]
    InvalidReference { from: String, to: String },

    #[error("Duplicate ID: {id} in {section}")]
    DuplicateId { id: String, section: String },
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub severity: Severity,
    pub rule: String,
    pub message: String,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.severity {
            Severity::Error => "ERROR",
            Severity::Warning => "WARN",
            Severity::Info => "INFO",
        };
        write!(f, "[{prefix}] {}: {}", self.rule, self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violation_display_error() {
        let v = Violation {
            severity: Severity::Error,
            rule: "SCHEMA-001".to_string(),
            message: "test error".to_string(),
            location: Some("metadata".to_string()),
        };
        let s = v.to_string();
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("SCHEMA-001"));
        assert!(s.contains("test error"));
    }

    #[test]
    fn violation_display_warning() {
        let v = Violation {
            severity: Severity::Warning,
            rule: "SCHEMA-006".to_string(),
            message: "test warning".to_string(),
            location: None,
        };
        let s = v.to_string();
        assert!(s.contains("[WARN]"));
    }

    #[test]
    fn violation_display_info() {
        let v = Violation {
            severity: Severity::Info,
            rule: "INFO-001".to_string(),
            message: "informational".to_string(),
            location: None,
        };
        let s = v.to_string();
        assert!(s.contains("[INFO]"));
    }

    #[test]
    fn contract_error_io() {
        let err = ContractError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
        let s = err.to_string();
        assert!(s.contains("Failed to read"));
    }

    #[test]
    fn contract_error_schema() {
        let err = ContractError::Schema("bad schema".to_string());
        assert!(err.to_string().contains("Schema violation"));
    }

    #[test]
    fn contract_error_missing_field() {
        let err = ContractError::MissingField {
            section: "metadata".to_string(),
            field: "version".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("metadata"));
        assert!(s.contains("version"));
    }

    #[test]
    fn contract_error_invalid_reference() {
        let err = ContractError::InvalidReference {
            from: "harness".to_string(),
            to: "obligation".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("harness"));
        assert!(s.contains("obligation"));
    }

    #[test]
    fn contract_error_duplicate_id() {
        let err = ContractError::DuplicateId {
            id: "KANI-001".to_string(),
            section: "kani_harnesses".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("KANI-001"));
    }
}
