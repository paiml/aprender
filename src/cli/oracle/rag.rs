//! RAG Oracle commands
//!
//! This module contains RAG (Retrieval Augmented Generation) related commands
//! for indexing and querying stack documentation.
//!
//! When the `rag` feature is enabled, queries are dispatched to SQLite+FTS5
//! via `trueno_rag::sqlite::SqliteIndex` for BM25-ranked results.
//! Otherwise falls back to in-memory `HybridRetriever` loaded from JSON.
//!
//! ## Submodule layout (QA-002 compliance)
//!
//! - `rag_helpers`: Shared formatting helpers
//! - `rag_sqlite`: SQLite+FTS5 backend types and search functions
//! - `rag_json_fallback`: JSON fallback backend (non-rag builds)
//! - `rag_display`: Result display and formatting
//! - `rag_commands`: Public command entry points
//! - `rag_stats`: Index statistics and TUI dashboard

#[path = "rag_helpers.rs"]
mod rag_helpers;

#[path = "rag_sqlite.rs"]
mod rag_sqlite;

#[path = "rag_json_fallback.rs"]
mod rag_json_fallback;

#[path = "rag_display.rs"]
mod rag_display;

#[path = "rag_commands.rs"]
mod rag_commands;

#[path = "rag_stats.rs"]
mod rag_stats;

// Re-export public command entry points
pub use rag_commands::{cmd_oracle_rag, cmd_oracle_rag_answer, cmd_oracle_rag_with_profile};
pub use rag_stats::{cmd_oracle_rag_dashboard, cmd_oracle_rag_stats};

// Re-export items used by sibling modules (pmat_query, rag_index)
#[cfg(not(feature = "rag"))]
pub(super) use rag_json_fallback::rag_load_index;
#[cfg(feature = "rag")]
pub(super) use rag_sqlite::{
    extract_component, rag_load_sqlite, rag_search_sqlite, sqlite_index_path,
};

#[cfg(all(test, feature = "rag"))]
#[path = "rag_tests.rs"]
mod tests;

#[cfg(all(test, feature = "rag"))]
#[path = "rag_tests_integration.rs"]
mod tests_integration;
