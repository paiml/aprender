//! Fix pattern representation for error adjusts.

use super::ChunkId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A pattern representing a successful fix for an error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixPattern {
    /// Unique identifier for this pattern
    pub id: ChunkId,
    /// The error code this pattern fixes (e.g., "E0308", "E0382")
    pub error_code: String,
    /// Sequence of decisions that led to this fix
    pub decision_sequence: Vec<String>,
    /// The actual fix diff (unified diff format)
    pub fix_diff: String,
    /// Number of times this pattern was successfully applied
    pub success_count: u32,
    /// Number of times this pattern was attempted
    pub attempt_count: u32,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl FixPattern {
    /// Create a new fix pattern
    #[must_use]
    pub fn new(error_code: impl Into<String>, fix_diff: impl Into<String>) -> Self {
        Self {
            id: ChunkId::new(),
            error_code: error_code.into(),
            decision_sequence: Vec::new(),
            fix_diff: fix_diff.into(),
            success_count: 0,
            attempt_count: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add a decision to the sequence
    #[must_use]
    pub fn with_decision(mut self, decision: impl Into<String>) -> Self {
        self.decision_sequence.push(decision.into());
        self
    }

    /// Add multiple decisions to the sequence
    #[must_use]
    pub fn with_decisions(mut self, decisions: Vec<String>) -> Self {
        self.decision_sequence.extend(decisions);
        self
    }

    /// Record a successful application
    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.attempt_count += 1;
    }

    /// Record a failed application
    pub fn record_failure(&mut self) {
        self.attempt_count += 1;
    }

    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        if self.attempt_count == 0 {
            0.0
        } else {
            self.success_count as f32 / self.attempt_count as f32
        }
    }

    /// Convert to searchable text for indexing
    #[must_use]
    pub fn to_searchable_text(&self) -> String {
        let decisions = self.decision_sequence.join(" ");
        format!("{} {} {}", self.error_code, decisions, self.fix_diff)
    }
}
