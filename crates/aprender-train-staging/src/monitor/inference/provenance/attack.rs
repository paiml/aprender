//! Attack path and anomaly types for incident analysis.

use super::edge::ProvenanceEdge;
use super::node::{NodeId, ProvenanceNode};

/// Reconstructed attack/incident path
#[derive(Debug, Clone)]
pub struct AttackPath {
    /// Nodes in causal order (root cause → incident)
    pub nodes: Vec<(NodeId, ProvenanceNode)>,
    /// Edges connecting the path
    pub edges: Vec<ProvenanceEdge>,
    /// Time span of the incident in nanoseconds
    pub duration_ns: u64,
    /// Identified anomaly indices (in nodes vector)
    pub anomaly_indices: Vec<usize>,
}

impl AttackPath {
    /// Number of nodes in the path
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if path contains anomalies
    pub fn has_anomalies(&self) -> bool {
        !self.anomaly_indices.is_empty()
    }
}

/// Anomaly detected in provenance path
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Node ID where anomaly was detected
    pub node_id: NodeId,
    /// Description of the anomaly
    pub description: String,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
}
