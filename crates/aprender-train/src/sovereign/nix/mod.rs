//! Nix Flake Configuration (ENT-018)
//!
//! Generates Nix flake configurations for reproducible sovereign deployments.

mod config;
mod crate_spec;
mod system;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use config::NixFlakeConfig;
pub use crate_spec::CrateSpec;
pub use system::NixSystem;
