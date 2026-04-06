//! Cloud Storage Backends (#72)
//!
//! Artifact storage abstraction supporting local, S3, Azure, and GCS backends.
//!
//! # Toyota Principle: Heijunka (平準化)
//!
//! Level workloads across storage tiers - artifacts are stored content-addressably
//! to enable efficient caching and deduplication across storage backends.
//!
//! # Example
//!
//! ```
//! use entrenar::storage::cloud::{ArtifactBackend, LocalBackend};
//! use std::path::PathBuf;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let backend = LocalBackend::new(PathBuf::from("/tmp/artifacts"));
//!     let hash = backend.put("model.safetensors", b"test data")?;
//!     let data = backend.get(&hash)?;
//!     Ok(())
//! }
//! ```

mod azure;
mod config;
mod error;
mod gcs;
mod local;
mod memory;
mod metadata;
mod s3;
mod traits;

// Re-export all public types for API compatibility
pub use azure::AzureConfig;
pub use config::BackendConfig;
pub use error::{CloudError, Result};
pub use gcs::GCSConfig;
pub use local::LocalBackend;
pub use memory::InMemoryBackend;
pub use metadata::ArtifactMetadata;
pub use s3::{MockS3Backend, S3Config};
pub use traits::{compute_hash, ArtifactBackend};
