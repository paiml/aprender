//! Error correlation types for CITL trainer

use super::{DecisionTrace, SourceSpan};
use crate::citl::FixSuggestion;

/// Result of error correlation analysis
#[derive(Debug, Clone)]
pub struct ErrorCorrelation {
    /// The error code being analyzed
    pub error_code: String,
    /// Error span where the error occurred
    pub error_span: SourceSpan,
    /// Decisions that may have contributed to the error (sorted by suspiciousness)
    pub suspicious_decisions: Vec<SuspiciousDecision>,
    /// Suggested fixes from the pattern store
    pub fix_suggestions: Vec<FixSuggestion>,
}

/// A decision suspected of contributing to an error
#[derive(Debug, Clone)]
pub struct SuspiciousDecision {
    /// The decision trace
    pub decision: DecisionTrace,
    /// Suspiciousness score (0.0 to 1.0)
    pub suspiciousness: f32,
    /// Reason for suspicion
    pub reason: String,
}

impl SuspiciousDecision {
    /// Create a new suspicious decision
    #[must_use]
    pub fn new(decision: DecisionTrace, suspiciousness: f32, reason: impl Into<String>) -> Self {
        Self { decision, suspiciousness: suspiciousness.clamp(0.0, 1.0), reason: reason.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suspicious_decision_new() {
        let trace = DecisionTrace::new("d1", "type", "desc");
        let suspicious = SuspiciousDecision::new(trace, 0.8, "high suspicion");
        assert_eq!(suspicious.suspiciousness, 0.8);
    }

    #[test]
    fn test_suspicious_decision_clamped() {
        let trace = DecisionTrace::new("d1", "type", "desc");
        let suspicious = SuspiciousDecision::new(trace, 1.5, "over max");
        assert_eq!(suspicious.suspiciousness, 1.0);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_suspiciousness_clamped(score in -10.0f32..10.0) {
            let trace = DecisionTrace::new("d", "type", "desc");
            let suspicious = SuspiciousDecision::new(trace, score, "reason");
            prop_assert!(suspicious.suspiciousness >= 0.0);
            prop_assert!(suspicious.suspiciousness <= 1.0);
        }
    }
}
