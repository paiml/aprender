//! Failure types and context structures.

use serde::{Deserialize, Serialize};

/// Categories of training failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureCategory {
    /// Data quality issues (corrupt files, missing features, invalid formats)
    DataQuality,

    /// Model convergence failures (NaN loss, exploding gradients, divergence)
    ModelConvergence,

    /// Resource exhaustion (OOM, disk full, timeout)
    ResourceExhaustion,

    /// Dependency failures (missing crates, version conflicts, build errors)
    DependencyFailure,

    /// Configuration errors (invalid hyperparameters, missing required fields)
    ConfigurationError,

    /// Unknown or uncategorized failure
    Unknown,
}

impl FailureCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::DataQuality => "Data quality issue",
            Self::ModelConvergence => "Model convergence failure",
            Self::ResourceExhaustion => "Resource exhaustion",
            Self::DependencyFailure => "Dependency failure",
            Self::ConfigurationError => "Configuration error",
            Self::Unknown => "Unknown failure",
        }
    }

    /// Pattern table: each entry maps keywords to a failure category.
    /// Checked in priority order (first match wins).
    const CATEGORY_PATTERNS: &'static [(&'static [&'static str], FailureCategory)] = &[
        (&["nan", "inf", "exploding", "diverge", "gradient"], FailureCategory::ModelConvergence),
        (
            &["out of memory", "oom", "memory", "timeout", "disk full", "no space"],
            FailureCategory::ResourceExhaustion,
        ),
        (
            &[
                "corrupt",
                "invalid data",
                "missing feature",
                "data format",
                "parse error",
                "invalid shape",
            ],
            FailureCategory::DataQuality,
        ),
        (
            &["dependency", "crate", "version", "build error", "compile"],
            FailureCategory::DependencyFailure,
        ),
        (
            &["config", "parameter", "invalid value", "missing field", "required"],
            FailureCategory::ConfigurationError,
        ),
    ];

    /// Attempt to categorize from error message patterns
    pub fn from_error_message(message: &str) -> Self {
        let lower = message.to_lowercase();

        for (patterns, category) in Self::CATEGORY_PATTERNS {
            if patterns.iter().any(|p| lower.contains(p)) {
                return *category;
            }
        }

        Self::Unknown
    }
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Structured failure context for a training run
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureContext {
    /// Error code (e.g., "E001", "NAN_LOSS")
    pub error_code: String,

    /// Human-readable error message
    pub message: String,

    /// Failure category for aggregation
    pub category: FailureCategory,

    /// Optional stack trace
    pub stack_trace: Option<String>,

    /// Suggested fix or remediation
    pub suggested_fix: Option<String>,

    /// Related run IDs that may have similar issues
    pub related_runs: Vec<String>,
}

impl FailureContext {
    /// Create a new failure context
    pub fn new(error_code: impl Into<String>, message: impl Into<String>) -> Self {
        let message_str = message.into();
        let category = FailureCategory::from_error_message(&message_str);

        Self {
            error_code: error_code.into(),
            message: message_str,
            category,
            stack_trace: None,
            suggested_fix: None,
            related_runs: Vec::new(),
        }
    }

    /// Create with explicit category
    pub fn with_category(
        error_code: impl Into<String>,
        message: impl Into<String>,
        category: FailureCategory,
    ) -> Self {
        Self {
            error_code: error_code.into(),
            message: message.into(),
            category,
            stack_trace: None,
            suggested_fix: None,
            related_runs: Vec::new(),
        }
    }

    /// Add a stack trace
    pub fn with_stack_trace(mut self, trace: impl Into<String>) -> Self {
        self.stack_trace = Some(trace.into());
        self
    }

    /// Add a suggested fix
    pub fn with_suggested_fix(mut self, fix: impl Into<String>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Add related run IDs
    pub fn with_related_runs(mut self, runs: Vec<String>) -> Self {
        self.related_runs = runs;
        self
    }

    /// Generate a suggested fix based on the category
    pub fn generate_suggested_fix(&self) -> String {
        match self.category {
            FailureCategory::ModelConvergence => {
                "Try reducing the learning rate, enabling gradient clipping, \
                 or checking for NaN values in input data."
                    .to_string()
            }
            FailureCategory::ResourceExhaustion => {
                "Try reducing batch size, using gradient checkpointing, \
                 or enabling mixed-precision training."
                    .to_string()
            }
            FailureCategory::DataQuality => {
                "Validate input data format, check for missing values, \
                 and verify data preprocessing pipeline."
                    .to_string()
            }
            FailureCategory::DependencyFailure => {
                "Run `cargo update`, check Cargo.lock for version conflicts, \
                 and verify all required features are enabled."
                    .to_string()
            }
            FailureCategory::ConfigurationError => {
                "Review configuration file for typos, missing required fields, \
                 and invalid parameter values."
                    .to_string()
            }
            FailureCategory::Unknown => {
                "Review the error message and stack trace for more details. \
                 Consider enabling debug logging."
                    .to_string()
            }
        }
    }
}

impl<E: std::error::Error> From<&E> for FailureContext {
    fn from(error: &E) -> Self {
        let message = error.to_string();
        let category = FailureCategory::from_error_message(&message);

        let mut context = Self::new("ERR_GENERIC", message);
        context.category = category;

        // Try to get source chain for stack trace
        let mut trace = String::new();
        let mut source = error.source();
        while let Some(s) = source {
            trace.push_str(&format!("Caused by: {s}\n"));
            source = s.source();
        }
        if !trace.is_empty() {
            context.stack_trace = Some(trace);
        }

        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_category_description() {
        assert_eq!(FailureCategory::DataQuality.description(), "Data quality issue");
        assert_eq!(FailureCategory::ModelConvergence.description(), "Model convergence failure");
        assert_eq!(FailureCategory::ResourceExhaustion.description(), "Resource exhaustion");
        assert_eq!(FailureCategory::DependencyFailure.description(), "Dependency failure");
        assert_eq!(FailureCategory::ConfigurationError.description(), "Configuration error");
        assert_eq!(FailureCategory::Unknown.description(), "Unknown failure");
    }

    #[test]
    fn test_failure_category_display() {
        assert_eq!(format!("{}", FailureCategory::DataQuality), "Data quality issue");
    }

    #[test]
    fn test_from_error_message_model_convergence() {
        assert_eq!(
            FailureCategory::from_error_message("NaN loss detected"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("inf value in tensor"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("exploding gradients"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("model diverged"),
            FailureCategory::ModelConvergence
        );
        assert_eq!(
            FailureCategory::from_error_message("gradient overflow"),
            FailureCategory::ModelConvergence
        );
    }

    #[test]
    fn test_from_error_message_resource_exhaustion() {
        assert_eq!(
            FailureCategory::from_error_message("out of memory"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("OOM killed"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("memory allocation failed"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("timeout exceeded"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("disk full"),
            FailureCategory::ResourceExhaustion
        );
        assert_eq!(
            FailureCategory::from_error_message("no space left"),
            FailureCategory::ResourceExhaustion
        );
    }

    #[test]
    fn test_from_error_message_data_quality() {
        assert_eq!(
            FailureCategory::from_error_message("corrupt file"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("invalid data format"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("missing feature: X"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("data format error"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("parse error"),
            FailureCategory::DataQuality
        );
        assert_eq!(
            FailureCategory::from_error_message("invalid shape"),
            FailureCategory::DataQuality
        );
    }

    #[test]
    fn test_from_error_message_dependency() {
        assert_eq!(
            FailureCategory::from_error_message("dependency not found"),
            FailureCategory::DependencyFailure
        );
        assert_eq!(
            FailureCategory::from_error_message("crate version conflict"),
            FailureCategory::DependencyFailure
        );
        assert_eq!(
            FailureCategory::from_error_message("version mismatch"),
            FailureCategory::DependencyFailure
        );
        assert_eq!(
            FailureCategory::from_error_message("build error"),
            FailureCategory::DependencyFailure
        );
        assert_eq!(
            FailureCategory::from_error_message("compile failed"),
            FailureCategory::DependencyFailure
        );
    }

    #[test]
    fn test_from_error_message_configuration() {
        assert_eq!(
            FailureCategory::from_error_message("config error"),
            FailureCategory::ConfigurationError
        );
        assert_eq!(
            FailureCategory::from_error_message("invalid parameter"),
            FailureCategory::ConfigurationError
        );
        assert_eq!(
            FailureCategory::from_error_message("invalid value for field"),
            FailureCategory::ConfigurationError
        );
        assert_eq!(
            FailureCategory::from_error_message("missing field: name"),
            FailureCategory::ConfigurationError
        );
        assert_eq!(
            FailureCategory::from_error_message("required field missing"),
            FailureCategory::ConfigurationError
        );
    }

    #[test]
    fn test_from_error_message_unknown() {
        assert_eq!(
            FailureCategory::from_error_message("something weird happened"),
            FailureCategory::Unknown
        );
        assert_eq!(FailureCategory::from_error_message(""), FailureCategory::Unknown);
    }

    #[test]
    fn test_failure_context_new() {
        let ctx = FailureContext::new("E001", "NaN loss detected");
        assert_eq!(ctx.error_code, "E001");
        assert_eq!(ctx.message, "NaN loss detected");
        assert_eq!(ctx.category, FailureCategory::ModelConvergence);
        assert!(ctx.stack_trace.is_none());
        assert!(ctx.suggested_fix.is_none());
        assert!(ctx.related_runs.is_empty());
    }

    #[test]
    fn test_failure_context_with_category() {
        let ctx =
            FailureContext::with_category("E002", "Custom error", FailureCategory::DataQuality);
        assert_eq!(ctx.error_code, "E002");
        assert_eq!(ctx.category, FailureCategory::DataQuality);
    }

    #[test]
    fn test_failure_context_with_stack_trace() {
        let ctx = FailureContext::new("E001", "error").with_stack_trace("at line 42");
        assert_eq!(ctx.stack_trace, Some("at line 42".to_string()));
    }

    #[test]
    fn test_failure_context_with_suggested_fix() {
        let ctx = FailureContext::new("E001", "error").with_suggested_fix("Try rebooting");
        assert_eq!(ctx.suggested_fix, Some("Try rebooting".to_string()));
    }

    #[test]
    fn test_failure_context_with_related_runs() {
        let ctx = FailureContext::new("E001", "error")
            .with_related_runs(vec!["run1".to_string(), "run2".to_string()]);
        assert_eq!(ctx.related_runs.len(), 2);
    }

    #[test]
    fn test_generate_suggested_fix_all_categories() {
        let categories = [
            FailureCategory::ModelConvergence,
            FailureCategory::ResourceExhaustion,
            FailureCategory::DataQuality,
            FailureCategory::DependencyFailure,
            FailureCategory::ConfigurationError,
            FailureCategory::Unknown,
        ];
        for category in categories {
            let ctx = FailureContext::with_category("E001", "error", category);
            let fix = ctx.generate_suggested_fix();
            assert!(!fix.is_empty());
        }
    }

    #[test]
    fn test_failure_context_from_error() {
        use std::io;
        let err = io::Error::new(io::ErrorKind::OutOfMemory, "out of memory");
        let ctx = FailureContext::from(&err);
        assert_eq!(ctx.error_code, "ERR_GENERIC");
        assert!(ctx.message.contains("memory"));
        assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
    }

    #[test]
    fn test_failure_category_serialization() {
        let cat = FailureCategory::DataQuality;
        let json = serde_json::to_string(&cat).expect("JSON serialization should succeed");
        let deserialized: FailureCategory =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(cat, deserialized);
    }

    #[test]
    fn test_failure_context_serialization() {
        let ctx = FailureContext::new("E001", "test error")
            .with_stack_trace("trace")
            .with_suggested_fix("fix it");
        let json = serde_json::to_string(&ctx).expect("JSON serialization should succeed");
        let deserialized: FailureContext =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(ctx.error_code, deserialized.error_code);
        assert_eq!(ctx.stack_trace, deserialized.stack_trace);
    }

    #[test]
    fn test_failure_category_clone_copy() {
        let cat = FailureCategory::ModelConvergence;
        let cloned = cat;
        let copied = cat;
        assert_eq!(cat, cloned);
        assert_eq!(cat, copied);
    }

    #[test]
    fn test_failure_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FailureCategory::DataQuality);
        set.insert(FailureCategory::ModelConvergence);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_failure_context_builder_chain() {
        let ctx = FailureContext::new("E001", "error")
            .with_stack_trace("trace")
            .with_suggested_fix("fix")
            .with_related_runs(vec!["run1".to_string()]);
        assert!(ctx.stack_trace.is_some());
        assert!(ctx.suggested_fix.is_some());
        assert_eq!(ctx.related_runs.len(), 1);
    }
}
