//! Provenance node types for incident reconstruction.

use serde::{Deserialize, Serialize};

/// Unique identifier for a node in the provenance graph
pub type NodeId = u64;

/// Provenance graph node types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProvenanceNode {
    /// Raw sensor/input data
    Input {
        /// Source identifier
        source: String,
        /// Timestamp in nanoseconds
        timestamp_ns: u64,
        /// Hash of the input data
        hash: u64,
    },

    /// Preprocessing transformation
    Transform {
        /// Operation name
        operation: String,
        /// References to input nodes
        input_refs: Vec<NodeId>,
    },

    /// Model inference
    Inference {
        /// Model identifier
        model_id: String,
        /// Model version
        model_version: String,
        /// Confidence of the inference
        confidence: f32,
        /// Output value
        output: f32,
    },

    /// Post-processing or sensor fusion
    Fusion {
        /// Fusion strategy
        strategy: String,
        /// References to input nodes
        input_refs: Vec<NodeId>,
    },

    /// Final action/output
    Action {
        /// Action type
        action_type: String,
        /// Confidence
        confidence: f32,
        /// Alternatives considered
        alternatives: Vec<(String, f32)>,
    },
}

impl ProvenanceNode {
    /// Get the timestamp if available
    pub fn timestamp_ns(&self) -> Option<u64> {
        match self {
            ProvenanceNode::Input { timestamp_ns, .. } => Some(*timestamp_ns),
            _ => None,
        }
    }

    /// Get node type as string
    pub fn type_name(&self) -> &'static str {
        match self {
            ProvenanceNode::Input { .. } => "Input",
            ProvenanceNode::Transform { .. } => "Transform",
            ProvenanceNode::Inference { .. } => "Inference",
            ProvenanceNode::Fusion { .. } => "Fusion",
            ProvenanceNode::Action { .. } => "Action",
        }
    }
}
