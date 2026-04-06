//! Oracle Mode - Intelligent query interface for the Sovereign AI Stack
//!
//! Provides:
//! - Knowledge graph of stack components
//! - Natural language query interface
//! - Component recommendations
//! - Integration pattern discovery
//! - RAG-based documentation retrieval (APR-Powered)
//! - Cookbook with practical recipes
//! - Local workspace discovery and multi-project intelligence
//! - Popperian falsification test generation

pub mod arxiv;
pub mod cookbook;
pub mod coursera;
pub mod falsify;
mod knowledge_graph;
#[cfg(feature = "native")]
pub mod local_workspace;
mod query_engine;
pub mod rag;
mod recommender;
pub mod svg;
mod types;

// Re-export public API - these are used by library consumers
#[allow(unused_imports)]
pub use knowledge_graph::*;
#[allow(unused_imports)]
pub use query_engine::QueryEngine;
pub use recommender::*;
#[allow(unused_imports)]
pub use types::*;

// Re-export RAG oracle - will be used by CLI integration
#[allow(unused_imports)]
pub use rag::{RagOracle, RagOracleConfig};

// Re-export local workspace oracle
#[cfg(feature = "native")]
#[allow(unused_imports)]
pub use local_workspace::{
    DependencyInfo, DevState, DriftType, GitStatus, LocalProject, LocalWorkspaceOracle,
    PublishOrder, PublishStep, VersionDrift, WorkspaceSummary,
};
