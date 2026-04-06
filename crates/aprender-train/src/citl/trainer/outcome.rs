//! Compilation outcome types for CITL trainer

use super::SourceSpan;
use serde::{Deserialize, Serialize};

/// Outcome of a compilation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilationOutcome {
    /// Compilation succeeded
    Success,
    /// Compilation failed with errors
    Failure {
        /// Error codes encountered
        error_codes: Vec<String>,
        /// Error spans
        error_spans: Vec<SourceSpan>,
        /// Error messages
        messages: Vec<String>,
    },
}

impl CompilationOutcome {
    /// Create a success outcome
    #[must_use]
    pub fn success() -> Self {
        Self::Success
    }

    /// Create a failure outcome
    #[must_use]
    pub fn failure(
        error_codes: Vec<String>,
        error_spans: Vec<SourceSpan>,
        messages: Vec<String>,
    ) -> Self {
        Self::Failure { error_codes, error_spans, messages }
    }

    /// Check if the outcome is success
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Get error codes if failure
    #[must_use]
    pub fn error_codes(&self) -> Vec<&str> {
        match self {
            Self::Success => vec![],
            Self::Failure { error_codes, .. } => error_codes.iter().map(String::as_str).collect(),
        }
    }

    /// Get error spans if failure
    #[must_use]
    pub fn error_spans(&self) -> Vec<&SourceSpan> {
        match self {
            Self::Success => vec![],
            Self::Failure { error_spans, .. } => error_spans.iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_outcome_success() {
        let outcome = CompilationOutcome::success();
        assert!(outcome.is_success());
        assert!(outcome.error_codes().is_empty());
    }

    #[test]
    fn test_compilation_outcome_failure() {
        let outcome = CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("main.rs", 5)],
            vec!["type mismatch".to_string()],
        );
        assert!(!outcome.is_success());
        assert_eq!(outcome.error_codes(), vec!["E0308"]);
        assert_eq!(outcome.error_spans().len(), 1);
    }
}
