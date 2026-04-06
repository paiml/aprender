//! Lineage event types and structures

use super::timestamp::LamportTimestamp;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Type of lineage event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LineageEventType {
    /// Training run started
    RunStarted,
    /// Metric was logged
    MetricLogged,
    /// Artifact was saved
    ArtifactSaved,
    /// Training run completed
    RunCompleted,
    /// Model was promoted to production
    ModelPromoted,
    /// Model was rolled back
    ModelRolledBack,
}

impl LineageEventType {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::RunStarted => "Run started",
            Self::MetricLogged => "Metric logged",
            Self::ArtifactSaved => "Artifact saved",
            Self::RunCompleted => "Run completed",
            Self::ModelPromoted => "Model promoted",
            Self::ModelRolledBack => "Model rolled back",
        }
    }
}

impl std::fmt::Display for LineageEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A single event in the causal lineage
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEvent {
    /// Lamport timestamp for causal ordering
    pub timestamp: LamportTimestamp,
    /// Type of event
    pub event_type: LineageEventType,
    /// Associated run ID
    pub run_id: String,
    /// Optional additional context
    pub context: Option<String>,
}

impl LineageEvent {
    /// Create a new lineage event
    pub fn new(timestamp: LamportTimestamp, event_type: LineageEventType, run_id: &str) -> Self {
        Self { timestamp, event_type, run_id: run_id.to_string(), context: None }
    }

    /// Add context to the event
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl PartialOrd for LineageEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LineageEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_event_type_description_run_started() {
        assert_eq!(LineageEventType::RunStarted.description(), "Run started");
    }

    #[test]
    fn test_lineage_event_type_description_metric_logged() {
        assert_eq!(LineageEventType::MetricLogged.description(), "Metric logged");
    }

    #[test]
    fn test_lineage_event_type_description_artifact_saved() {
        assert_eq!(LineageEventType::ArtifactSaved.description(), "Artifact saved");
    }

    #[test]
    fn test_lineage_event_type_description_run_completed() {
        assert_eq!(LineageEventType::RunCompleted.description(), "Run completed");
    }

    #[test]
    fn test_lineage_event_type_description_model_promoted() {
        assert_eq!(LineageEventType::ModelPromoted.description(), "Model promoted");
    }

    #[test]
    fn test_lineage_event_type_description_model_rolled_back() {
        assert_eq!(LineageEventType::ModelRolledBack.description(), "Model rolled back");
    }

    #[test]
    fn test_lineage_event_type_display() {
        assert_eq!(LineageEventType::RunStarted.to_string(), "Run started");
        assert_eq!(LineageEventType::MetricLogged.to_string(), "Metric logged");
        assert_eq!(LineageEventType::ArtifactSaved.to_string(), "Artifact saved");
        assert_eq!(LineageEventType::RunCompleted.to_string(), "Run completed");
        assert_eq!(LineageEventType::ModelPromoted.to_string(), "Model promoted");
        assert_eq!(LineageEventType::ModelRolledBack.to_string(), "Model rolled back");
    }

    #[test]
    fn test_lineage_event_type_clone() {
        let et = LineageEventType::RunStarted;
        let cloned = et;
        assert_eq!(et, cloned);
    }

    #[test]
    fn test_lineage_event_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(LineageEventType::RunStarted);
        set.insert(LineageEventType::RunStarted);
        assert_eq!(set.len(), 1);
        set.insert(LineageEventType::RunCompleted);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_lineage_event_new() {
        let ts = LamportTimestamp::new("node1");
        let event = LineageEvent::new(ts, LineageEventType::RunStarted, "run-123");
        assert_eq!(event.run_id, "run-123");
        assert_eq!(event.event_type, LineageEventType::RunStarted);
        assert!(event.context.is_none());
    }

    #[test]
    fn test_lineage_event_with_context() {
        let ts = LamportTimestamp::new("node1");
        let event = LineageEvent::new(ts, LineageEventType::MetricLogged, "run-456")
            .with_context("loss=0.5");
        assert_eq!(event.context, Some("loss=0.5".to_string()));
    }

    #[test]
    fn test_lineage_event_ordering() {
        let ts1 = LamportTimestamp::with_counter("node1", 1);
        let ts2 = LamportTimestamp::with_counter("node1", 2);

        let event1 = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-1");
        let event2 = LineageEvent::new(ts2, LineageEventType::MetricLogged, "run-1");

        assert!(event1 < event2);
        assert!(event2 > event1);
        assert_eq!(event1.cmp(&event2), Ordering::Less);
    }

    #[test]
    fn test_lineage_event_partial_ord() {
        let ts1 = LamportTimestamp::with_counter("node1", 1);
        let ts2 = LamportTimestamp::with_counter("node1", 2);

        let event1 = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-1");
        let event2 = LineageEvent::new(ts2, LineageEventType::MetricLogged, "run-1");

        assert_eq!(event1.partial_cmp(&event2), Some(Ordering::Less));
    }

    #[test]
    fn test_lineage_event_clone() {
        let ts = LamportTimestamp::new("node1");
        let event =
            LineageEvent::new(ts, LineageEventType::RunStarted, "run-123").with_context("test");
        let cloned = event.clone();
        assert_eq!(event.run_id, cloned.run_id);
        assert_eq!(event.event_type, cloned.event_type);
        assert_eq!(event.context, cloned.context);
    }

    #[test]
    fn test_lineage_event_eq() {
        let ts = LamportTimestamp::new("node1");
        let event1 = LineageEvent::new(ts.clone(), LineageEventType::RunStarted, "run-123");
        let event2 = LineageEvent::new(ts, LineageEventType::RunStarted, "run-123");
        assert_eq!(event1, event2);
    }

    #[test]
    fn test_lineage_event_serde() {
        let ts = LamportTimestamp::new("node1");
        let event = LineageEvent::new(ts, LineageEventType::ArtifactSaved, "run-789")
            .with_context("model.gguf");

        let json = serde_json::to_string(&event).expect("JSON serialization should succeed");
        let deserialized: LineageEvent =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(event.run_id, deserialized.run_id);
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.context, deserialized.context);
    }

    #[test]
    fn test_lineage_event_type_serde() {
        let et = LineageEventType::ModelPromoted;
        let json = serde_json::to_string(&et).expect("JSON serialization should succeed");
        let deserialized: LineageEventType =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(et, deserialized);
    }

    #[test]
    fn test_lineage_event_type_debug() {
        assert_eq!(format!("{:?}", LineageEventType::RunStarted), "RunStarted");
        assert_eq!(format!("{:?}", LineageEventType::MetricLogged), "MetricLogged");
        assert_eq!(format!("{:?}", LineageEventType::ArtifactSaved), "ArtifactSaved");
        assert_eq!(format!("{:?}", LineageEventType::RunCompleted), "RunCompleted");
        assert_eq!(format!("{:?}", LineageEventType::ModelPromoted), "ModelPromoted");
        assert_eq!(format!("{:?}", LineageEventType::ModelRolledBack), "ModelRolledBack");
    }
}
