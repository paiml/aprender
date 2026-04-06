//! Check result types for preflight validation.

use serde::{Deserialize, Serialize};

/// Result of a single preflight check
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CheckResult {
    /// Check passed
    Passed { message: String },
    /// Check failed
    Failed { message: String, details: Option<String> },
    /// Check was skipped
    Skipped { reason: String },
    /// Check produced a warning (non-fatal)
    Warning { message: String },
}

impl CheckResult {
    /// Create a passed result
    pub fn passed(message: impl Into<String>) -> Self {
        Self::Passed { message: message.into() }
    }

    /// Create a failed result
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed { message: message.into(), details: None }
    }

    /// Create a failed result with details
    pub fn failed_with_details(message: impl Into<String>, details: impl Into<String>) -> Self {
        Self::Failed { message: message.into(), details: Some(details.into()) }
    }

    /// Create a skipped result
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self::Skipped { reason: reason.into() }
    }

    /// Create a warning result
    pub fn warning(message: impl Into<String>) -> Self {
        Self::Warning { message: message.into() }
    }

    /// Check if the result is passed
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    /// Check if the result is failed
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// Check if the result is a warning
    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warning { .. })
    }

    /// Check if the result is skipped
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CheckResult Tests
    // =========================================================================

    #[test]
    fn test_check_result_passed() {
        let result = CheckResult::passed("test passed");
        assert!(result.is_passed());
        assert!(!result.is_failed());
    }

    #[test]
    fn test_check_result_failed() {
        let result = CheckResult::failed("test failed");
        assert!(result.is_failed());
        assert!(!result.is_passed());
    }

    #[test]
    fn test_check_result_failed_with_details() {
        let result = CheckResult::failed_with_details("failed", "some details");
        assert!(result.is_failed());
        if let CheckResult::Failed { details, .. } = result {
            assert_eq!(details, Some("some details".to_string()));
        }
    }

    #[test]
    fn test_check_result_warning() {
        let result = CheckResult::warning("warning message");
        assert!(result.is_warning());
        assert!(!result.is_failed());
    }

    #[test]
    fn test_check_result_skipped() {
        let result = CheckResult::skipped("skipped reason");
        assert!(result.is_skipped());
    }
}
