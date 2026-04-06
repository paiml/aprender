//! Provenance edge types for causal relationships.

use super::node::NodeId;
use serde::{Deserialize, Serialize};

/// Causal relation between nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRelation {
    /// Data flowed from source to sink
    DataFlow,
    /// Inference triggered by input
    Triggered,
    /// Decision influenced by
    Influenced,
    /// Action caused by decision
    Caused,
}

/// Edge in provenance graph (directed, acyclic)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    /// Source node
    pub from: NodeId,
    /// Destination node
    pub to: NodeId,
    /// Causal relation
    pub relation: CausalRelation,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}
