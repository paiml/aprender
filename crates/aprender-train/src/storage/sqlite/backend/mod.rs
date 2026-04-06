//! SQLite Backend implementation for Experiment Storage.
//!
//! Sovereign, local-first storage using SQLite with WAL mode.
//!
//! # Toyota Way: (Heijunka)
//!
//! SQLite provides consistent, predictable performance without external dependencies.

pub(crate) mod schema;
mod sqlite_backend;

pub use sqlite_backend::SqliteBackend;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
