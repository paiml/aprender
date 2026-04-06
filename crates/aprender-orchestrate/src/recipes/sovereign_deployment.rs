//! Sovereign deployment recipe implementation.

use crate::experiment::{ExperimentError, SovereignArtifact, SovereignDistribution};
use crate::recipes::RecipeResult;
use serde::{Deserialize, Serialize};

/// Sovereign deployment recipe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignDeploymentConfig {
    /// Distribution name
    pub name: String,
    /// Version
    pub version: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Require signatures
    pub require_signatures: bool,
    /// Nix flake support
    pub enable_nix_flake: bool,
    /// Offline registry path
    pub offline_registry_path: Option<String>,
}

impl Default for SovereignDeploymentConfig {
    fn default() -> Self {
        Self {
            name: "sovereign-model".to_string(),
            version: "0.1.0".to_string(),
            platforms: vec!["linux-x86_64".to_string()],
            require_signatures: true,
            enable_nix_flake: true,
            offline_registry_path: None,
        }
    }
}

/// Sovereign deployment recipe
#[derive(Debug)]
pub struct SovereignDeploymentRecipe {
    config: SovereignDeploymentConfig,
    distribution: SovereignDistribution,
}

impl SovereignDeploymentRecipe {
    /// Create a new sovereign deployment recipe
    pub fn new(config: SovereignDeploymentConfig) -> Self {
        let mut distribution = SovereignDistribution::new(&config.name, &config.version);
        for platform in &config.platforms {
            distribution.add_platform(platform);
        }
        Self { config, distribution }
    }

    /// Shared implementation for adding artifacts of any type
    fn add_artifact_impl(
        &mut self,
        name: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
        artifact_type: crate::experiment::ArtifactType,
    ) {
        self.distribution.add_artifact(SovereignArtifact {
            name: name.into(),
            artifact_type,
            sha256: sha256.into(),
            size_bytes,
            source_url: None,
        });
    }

    /// Add a model artifact
    pub fn add_model(
        &mut self,
        name: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) {
        self.add_artifact_impl(name, sha256, size_bytes, crate::experiment::ArtifactType::Model);
    }

    /// Add a binary artifact
    pub fn add_binary(
        &mut self,
        name: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) {
        self.add_artifact_impl(name, sha256, size_bytes, crate::experiment::ArtifactType::Binary);
    }

    /// Add a dataset artifact
    pub fn add_dataset(
        &mut self,
        name: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) {
        self.add_artifact_impl(name, sha256, size_bytes, crate::experiment::ArtifactType::Dataset);
    }

    /// Sign an artifact (placeholder - would use real crypto in production)
    pub fn sign_artifact(&mut self, artifact_name: impl Into<String>, key_id: impl Into<String>) {
        let name = artifact_name.into();
        // In production, this would actually compute the signature
        let signature = format!("sig_placeholder_{}", &name);
        self.distribution.signatures.push(crate::experiment::ArtifactSignature {
            artifact_name: name,
            algorithm: crate::experiment::SignatureAlgorithm::Ed25519,
            signature,
            key_id: key_id.into(),
        });
    }

    /// Validate and build the distribution
    pub fn build(&self) -> Result<RecipeResult, ExperimentError> {
        // Validate signatures if required
        if self.config.require_signatures {
            self.distribution.validate_signatures()?;
        }

        let mut result = RecipeResult::success("sovereign-deployment");
        result = result.with_metric("artifact_count", self.distribution.artifacts.len() as f64);
        result =
            result.with_metric("total_size_bytes", self.distribution.total_size_bytes() as f64);
        result = result.with_metric("platform_count", self.distribution.platforms.len() as f64);

        // Add artifacts to result
        for artifact in &self.distribution.artifacts {
            result = result.with_artifact(&artifact.name);
        }

        Ok(result)
    }

    /// Get the distribution
    pub fn distribution(&self) -> &SovereignDistribution {
        &self.distribution
    }

    /// Export distribution manifest as JSON
    pub fn export_manifest(&self) -> Result<String, ExperimentError> {
        serde_json::to_string_pretty(&self.distribution)
            .map_err(|e| ExperimentError::StorageError(e.to_string()))
    }
}
