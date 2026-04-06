//! Lamport logical timestamp implementation

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Lamport logical timestamp for causal ordering
///
/// Implements Lamport's logical clock algorithm for establishing
/// happens-before relationships without synchronized wall clocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LamportTimestamp {
    /// Logical counter value
    pub counter: u64,
    /// Node identifier for tie-breaking
    pub node_id: String,
}

impl LamportTimestamp {
    /// Create a new timestamp for a node
    pub fn new(node_id: &str) -> Self {
        Self { counter: 0, node_id: node_id.to_string() }
    }

    /// Create a timestamp with specific counter value
    pub fn with_counter(node_id: &str, counter: u64) -> Self {
        Self { counter, node_id: node_id.to_string() }
    }

    /// Increment the timestamp for a local event
    ///
    /// Returns a copy of the new timestamp value.
    pub fn increment(&mut self) -> Self {
        self.counter += 1;
        self.clone()
    }

    /// Merge with another timestamp (on message receive)
    ///
    /// Sets counter to max(self.counter, other.counter) + 1
    /// Returns a copy of the new timestamp value.
    pub fn merge(&mut self, other: &Self) -> Self {
        self.counter = self.counter.max(other.counter) + 1;
        self.clone()
    }

    /// Check if this timestamp happens-before another
    ///
    /// Returns true if:
    /// - self.counter < other.counter, OR
    /// - self.counter == other.counter AND self.node_id < other.node_id
    ///
    /// Note: If neither happens_before the other, events are concurrent.
    pub fn happens_before(&self, other: &Self) -> bool {
        match self.counter.cmp(&other.counter) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.node_id < other.node_id,
        }
    }

    /// Check if events are concurrent (neither happens-before the other)
    pub fn is_concurrent_with(&self, other: &Self) -> bool {
        self.counter == other.counter && self.node_id != other.node_id
    }
}

impl PartialOrd for LamportTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LamportTimestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.counter.cmp(&other.counter) {
            Ordering::Equal => self.node_id.cmp(&other.node_id),
            other => other,
        }
    }
}
