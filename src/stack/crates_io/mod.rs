//! Crates.io API Client
//!
//! Provides functionality to query crates.io for version information
//! and verify published crate status.
//!
//! Features:
//! - In-memory caching with TTL
//! - Persistent file-based cache for offline mode
//! - Configurable cache TTL

mod cache;
mod client;
mod mock;
#[cfg(test)]
mod tests;
mod types;

// Re-export all public types from types module
pub use types::{
    CacheEntry, CrateData, CrateResponse, DependencyData, DependencyResponse, PersistentCacheEntry,
    VersionData,
};

// Re-export cache module
pub use cache::PersistentCache;

// Re-export client
pub use client::CratesIoClient;

// Re-export mock client
pub use mock::MockCratesIoClient;
