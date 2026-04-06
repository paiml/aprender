//! Causal lineage tracking

use super::event::{LineageEvent, LineageEventType};
use serde::{Deserialize, Serialize};

/// Causal lineage tracking for a set of events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalLineage {
    /// Events in causal order
    pub events: Vec<LineageEvent>,
}

impl CausalLineage {
    /// Create a new empty lineage
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the lineage
    pub fn add_event(&mut self, event: LineageEvent) {
        self.events.push(event);
        self.events.sort();
    }

    /// Get events in causal order
    pub fn events_in_order(&self) -> &[LineageEvent] {
        &self.events
    }

    /// Get events for a specific run
    pub fn events_for_run(&self, run_id: &str) -> Vec<&LineageEvent> {
        self.events.iter().filter(|e| e.run_id == run_id).collect()
    }

    /// Get events of a specific type
    pub fn events_of_type(&self, event_type: LineageEventType) -> Vec<&LineageEvent> {
        self.events.iter().filter(|e| e.event_type == event_type).collect()
    }

    /// Find the latest event for a run
    pub fn latest_event_for_run(&self, run_id: &str) -> Option<&LineageEvent> {
        self.events.iter().rev().find(|e| e.run_id == run_id)
    }

    /// Check if run A causally precedes run B
    pub fn run_precedes(&self, run_a: &str, run_b: &str) -> bool {
        let a_events = self.events_for_run(run_a);
        let b_events = self.events_for_run(run_b);

        if a_events.is_empty() || b_events.is_empty() {
            return false;
        }

        // A precedes B if all of A's events happen before B's first event
        let b_first = b_events.first().expect("b_events is non-empty (checked above)");
        a_events.iter().all(|a| a.timestamp.happens_before(&b_first.timestamp))
    }

    /// Get the total number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if lineage is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
