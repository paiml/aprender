//! Offline Model Registry (ENT-017)
//!
//! Provides local model storage and verification for air-gapped deployments.

mod manifest;
mod offline;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types to preserve the public API
pub use manifest::RegistryManifest;
pub use offline::OfflineModelRegistry;
pub use types::{ModelEntry, ModelSource};
