//! Provenance Graph for Incident Reconstruction (ENT-111)
//!
//! Captures causal relationships between system entities for forensic analysis.

mod attack;
mod edge;
mod graph;
mod node;
mod reconstructor;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

// Re-export all public types for API compatibility
pub use attack::{Anomaly, AttackPath};
pub use edge::{CausalRelation, ProvenanceEdge};
pub use graph::ProvenanceGraph;
pub use node::{NodeId, ProvenanceNode};
pub use reconstructor::IncidentReconstructor;
