//! Decision trace types for CITL trainer

use super::SourceSpan;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single decision in the compiler trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Unique ID for this decision
    pub id: String,
    /// Type of decision (e.g., "type_inference", "borrow_check", "lifetime_resolution")
    pub decision_type: String,
    /// Description of the decision
    pub description: String,
    /// Source span where this decision was made
    pub span: Option<SourceSpan>,
    /// Timestamp (nanoseconds since session start)
    pub timestamp_ns: u64,
    /// Dependencies on other decisions
    pub depends_on: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl DecisionTrace {
    /// Create a new decision trace
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        decision_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            decision_type: decision_type.into(),
            description: description.into(),
            span: None,
            timestamp_ns: 0,
            depends_on: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the source span
    #[must_use]
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Set the timestamp
    #[must_use]
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// Add a dependency
    #[must_use]
    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.depends_on.push(dep_id.into());
        self
    }

    /// Add dependencies
    #[must_use]
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.depends_on.extend(deps);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_trace_new() {
        let trace = DecisionTrace::new("d1", "type_inference", "Inferred type i32");
        assert_eq!(trace.id, "d1");
        assert_eq!(trace.decision_type, "type_inference");
        assert_eq!(trace.description, "Inferred type i32");
    }

    #[test]
    fn test_decision_trace_with_span() {
        let trace =
            DecisionTrace::new("d1", "type", "desc").with_span(SourceSpan::line("main.rs", 5));
        assert!(trace.span.is_some());
    }

    #[test]
    fn test_decision_trace_with_timestamp() {
        let trace = DecisionTrace::new("d1", "type", "desc").with_timestamp(1000);
        assert_eq!(trace.timestamp_ns, 1000);
    }

    #[test]
    fn test_decision_trace_with_dependencies() {
        let trace = DecisionTrace::new("d1", "type", "desc")
            .with_dependency("d0")
            .with_dependencies(vec!["d2".to_string(), "d3".to_string()]);
        assert_eq!(trace.depends_on.len(), 3);
    }
}
