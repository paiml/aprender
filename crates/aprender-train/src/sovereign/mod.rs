//! Sovereign Deployment Module (ENT-016 through ENT-018)
//!
//! Enables air-gapped deployment scenarios for university and enterprise environments.
//!
//! # Components
//!
//! - [`distribution`] - Distribution manifest and packaging (ENT-016)
//! - [`registry`] - Offline model registry for local model storage (ENT-017)
//! - [`nix`] - Nix flake generation for reproducible deployments (ENT-018)
//!
//! # Example
//!
//! ```rust
//! use entrenar::sovereign::{
//!     SovereignDistribution, DistributionTier, DistributionFormat,
//!     OfflineModelRegistry, ModelSource,
//!     NixFlakeConfig,
//! };
//!
//! // Create a core distribution
//! let dist = SovereignDistribution::core()
//!     .with_format(DistributionFormat::Iso);
//!
//! // Generate Nix flake
//! let config = NixFlakeConfig::sovereign_stack();
//! let flake_nix = config.generate_flake_nix();
//! ```

pub mod distribution;
pub mod governance;
pub mod nix;
pub mod registry;

pub use distribution::{
    ComponentManifest, DistributionFormat, DistributionTier, SovereignDistribution,
};
pub use governance::{
    ApiAllowlist, AuditEntry, AuditTrail, DataClassification, DeletionCascade, KeySource,
    ResidencyConfig, WeightSovereigntyConfig,
};
pub use nix::{CrateSpec, NixFlakeConfig, NixSystem};
pub use registry::{ModelEntry, ModelSource, OfflineModelRegistry, RegistryManifest};
