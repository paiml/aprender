//! Error types for presentar-terminal.

use presentar_core::BrickVerification;
use thiserror::Error;

/// Errors that can occur in the TUI application.
#[derive(Debug, Error)]
pub enum TuiError {
    /// IO error from terminal operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Brick verification failed (Jidoka gate).
    #[error("Brick verification failed: {0}")]
    VerificationFailed(VerificationError),

    /// Invalid Brick configuration.
    #[error("Invalid brick: {0}")]
    InvalidBrick(String),

    /// Budget exceeded during rendering.
    #[error("Budget exceeded: {phase} took {elapsed_ms}ms (budget: {budget_ms}ms)")]
    BudgetExceeded {
        phase: String,
        elapsed_ms: u64,
        budget_ms: u64,
    },

    /// Terminal not available.
    #[error("Terminal not available")]
    TerminalNotAvailable,
}

/// Verification error with details.
#[derive(Debug)]
pub struct VerificationError {
    /// The verification result.
    pub verification: BrickVerification,
    /// Human-readable summary.
    pub summary: String,
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary)
    }
}

impl From<BrickVerification> for VerificationError {
    fn from(v: BrickVerification) -> Self {
        let summary = if v.is_valid() {
            "Verification passed".to_string()
        } else {
            format!(
                "Verification failed: {} assertion(s) failed",
                v.failed.len()
            )
        };
        Self {
            verification: v,
            summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::BrickAssertion;
    use std::time::Duration;

    #[test]
    fn test_tui_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let tui_err: TuiError = io_err.into();
        assert!(matches!(tui_err, TuiError::Io(_)));
        assert!(tui_err.to_string().contains("IO error"));
    }

    #[test]
    fn test_tui_error_invalid_brick() {
        let err = TuiError::InvalidBrick("test error".to_string());
        assert!(err.to_string().contains("Invalid brick"));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_tui_error_budget_exceeded() {
        let err = TuiError::BudgetExceeded {
            phase: "render".to_string(),
            elapsed_ms: 50,
            budget_ms: 16,
        };
        let msg = err.to_string();
        assert!(msg.contains("Budget exceeded"));
        assert!(msg.contains("render"));
        assert!(msg.contains("50ms"));
        assert!(msg.contains("16ms"));
    }

    #[test]
    fn test_tui_error_terminal_not_available() {
        let err = TuiError::TerminalNotAvailable;
        assert_eq!(err.to_string(), "Terminal not available");
    }

    #[test]
    fn test_verification_error_display() {
        let verification = BrickVerification {
            passed: vec![],
            failed: vec![(BrickAssertion::max_latency_ms(16), "too slow".to_string())],
            verification_time: Duration::from_micros(10),
        };
        let err = VerificationError::from(verification);
        assert!(err.to_string().contains("1 assertion(s) failed"));
    }

    #[test]
    fn test_verification_error_passed() {
        let verification = BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(16)],
            failed: vec![],
            verification_time: Duration::from_micros(10),
        };
        let err = VerificationError::from(verification);
        assert_eq!(err.to_string(), "Verification passed");
    }

    #[test]
    fn test_tui_error_verification_failed() {
        let verification = BrickVerification {
            passed: vec![],
            failed: vec![(BrickAssertion::max_latency_ms(16), "too slow".to_string())],
            verification_time: Duration::from_micros(10),
        };
        let err = TuiError::VerificationFailed(VerificationError::from(verification));
        assert!(err.to_string().contains("Brick verification failed"));
    }

    #[test]
    fn test_verification_error_debug() {
        let verification = BrickVerification {
            passed: vec![],
            failed: vec![(BrickAssertion::max_latency_ms(16), "too slow".to_string())],
            verification_time: Duration::from_micros(10),
        };
        let err = VerificationError::from(verification);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("VerificationError"));
    }

    #[test]
    fn test_verification_error_multiple_failures() {
        let verification = BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(100)],
            failed: vec![
                (BrickAssertion::max_latency_ms(16), "too slow".to_string()),
                (
                    BrickAssertion::max_latency_ms(8),
                    "way too slow".to_string(),
                ),
            ],
            verification_time: Duration::from_micros(50),
        };
        let err = VerificationError::from(verification);
        assert!(err.to_string().contains("2 assertion(s) failed"));
        assert!(err.verification.passed.len() == 1);
        assert!(err.verification.failed.len() == 2);
    }

    #[test]
    fn test_tui_error_debug() {
        let err = TuiError::TerminalNotAvailable;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("TerminalNotAvailable"));
    }

    #[test]
    fn test_tui_error_io_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let tui_err = TuiError::from(io_err);
        assert!(tui_err.to_string().contains("IO error"));
        assert!(tui_err.to_string().contains("access denied"));
    }

    #[test]
    fn test_budget_exceeded_all_fields() {
        let err = TuiError::BudgetExceeded {
            phase: "layout".to_string(),
            elapsed_ms: 100,
            budget_ms: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("Budget exceeded"));
        assert!(msg.contains("layout"));
        assert!(msg.contains("100ms"));
        assert!(msg.contains("50ms"));
    }

    #[test]
    fn test_invalid_brick_with_details() {
        let err = TuiError::InvalidBrick("Missing required field: title".to_string());
        assert!(err.to_string().contains("Invalid brick"));
        assert!(err.to_string().contains("Missing required field"));
    }
}
