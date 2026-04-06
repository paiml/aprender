//! Metrics Storage Module
//!
//! Persists training metrics to Parquet files using trueno-db.
//! Feature-gated behind `monitor` feature flag.

mod error;
mod in_memory;
mod json_file;
mod traits;

pub use error::{StorageError, StorageResult};
pub use in_memory::InMemoryStore;
pub use json_file::JsonFileStore;
pub use traits::MetricsStore;

#[cfg(test)]
mod tests;
