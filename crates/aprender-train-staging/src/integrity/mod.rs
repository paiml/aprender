//! Behavioral Integrity & Lineage Module (ENT-013, ENT-014, ENT-015)
//!
//! Provides behavioral integrity verification, causal lineage tracking,
//! and trace storage policies for experiment reproducibility.
//!
//! # Components
//!
//! - [`lineage`] - Lamport timestamps and causal event ordering
//! - [`trace_storage`] - Trace compression and retention policies
//! - [`behavioral`] - Behavioral integrity metrics and promotion gates

pub mod behavioral;
pub mod lineage;
pub mod trace_storage;

pub use behavioral::{BehavioralIntegrity, MetamorphicViolation};
pub use lineage::{CausalLineage, LamportTimestamp, LineageEvent, LineageEventType};
pub use trace_storage::{CompressionAlgorithm, TraceStoragePolicy};
