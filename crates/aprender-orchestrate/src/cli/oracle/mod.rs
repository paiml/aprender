//! Oracle command implementations
//!
//! This module contains all Oracle-related CLI commands extracted from main.rs.
//! The module is organized into submodules for better maintainability:
//!
//! - `types`: Shared types and enums (OracleOutputFormat)
//! - `rag`: RAG query commands
//! - `rag_index`: RAG indexing commands
//! - `local`: Local workspace discovery
//! - `cookbook`: Cookbook recipe commands

#![cfg(feature = "native")]

mod cookbook;
mod coursera;
mod local;
mod pmat_query;
mod rag;
mod rag_index;
mod types;

// Re-export all public items to maintain API compatibility
pub use cookbook::cmd_oracle_cookbook;
pub use coursera::{cmd_oracle_asset, CourseraAssetType};
pub use local::cmd_oracle_local;
pub use pmat_query::cmd_oracle_pmat_query;
#[allow(unused_imports)]
pub use rag::{
    cmd_oracle_rag, cmd_oracle_rag_answer, cmd_oracle_rag_dashboard, cmd_oracle_rag_stats,
    cmd_oracle_rag_with_profile,
};
pub use rag_index::cmd_oracle_rag_index;
pub use types::OracleOutputFormat;

// Re-export classic oracle functions from oracle_classic.rs
pub use super::oracle_classic::{cmd_oracle, OracleOptions};
