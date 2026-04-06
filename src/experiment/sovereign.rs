//! Sovereign distribution manifest for air-gapped deployments
//!
//! This module contains sovereign distribution types extracted from experiment/mod.rs.

use super::ExperimentError;
use serde::{Deserialize, Serialize};

/// Sovereign distribution manifest for air-gapped deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignDistribution {
    /// Distribution name
    pub name: String,
    /// Version
    pub version: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Required artifacts
    pub artifacts: Vec<SovereignArtifact>,
    /// Cryptographic signatures
    pub signatures: Vec<ArtifactSignature>,
    /// Offline registry configuration
    pub offline_registry: Option<OfflineRegistryConfig>,
    /// Nix flake hash for reproducibility
    pub nix_flake_hash: Option<String>,
}

/// Artifact in a sovereign distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignArtifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// SHA-256 hash
    pub sha256: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Download URL (for pre-staging)
    pub source_url: Option<String>,
}

/// Types of distributable artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    Binary,
    Model,
    Dataset,
    Config,
    Documentation,
    Container,
    NixDerivation,
}

/// Cryptographic signature for artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSignature {
    /// Artifact name this signature is for
    pub artifact_name: String,
    /// Signature algorithm
    pub algorithm: SignatureAlgorithm,
    /// Base64-encoded signature
    pub signature: String,
    /// Public key identifier
    pub key_id: String,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    RSA4096,
    EcdsaP256,
}

/// Offline model registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineRegistryConfig {
    /// Registry path
    pub path: String,
    /// Index file location
    pub index_path: String,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Last sync timestamp
    pub last_sync: Option<String>,
}

impl SovereignDistribution {
    /// Create a new sovereign distribution
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            platforms: Vec::new(),
            artifacts: Vec::new(),
            signatures: Vec::new(),
            offline_registry: None,
            nix_flake_hash: None,
        }
    }

    /// Add a platform target
    pub fn add_platform(&mut self, platform: impl Into<String>) {
        self.platforms.push(platform.into());
    }

    /// Add an artifact
    pub fn add_artifact(&mut self, artifact: SovereignArtifact) {
        self.artifacts.push(artifact);
    }

    /// Validate all artifacts have signatures
    pub fn validate_signatures(&self) -> Result<(), ExperimentError> {
        for artifact in &self.artifacts {
            let has_sig = self.signatures.iter().any(|s| s.artifact_name == artifact.name);
            if !has_sig {
                return Err(ExperimentError::SovereignValidationFailed(format!(
                    "Missing signature for artifact: {}",
                    artifact.name
                )));
            }
        }
        Ok(())
    }

    /// Calculate total distribution size
    pub fn total_size_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_distribution_new() {
        let dist = SovereignDistribution::new("my-dist", "1.0.0");
        assert_eq!(dist.name, "my-dist");
        assert_eq!(dist.version, "1.0.0");
        assert!(dist.platforms.is_empty());
        assert!(dist.artifacts.is_empty());
        assert!(dist.signatures.is_empty());
        assert!(dist.offline_registry.is_none());
        assert!(dist.nix_flake_hash.is_none());
    }

    #[test]
    fn test_add_platform() {
        let mut dist = SovereignDistribution::new("test", "1.0");
        dist.add_platform("linux-x86_64");
        dist.add_platform("darwin-aarch64");
        assert_eq!(dist.platforms.len(), 2);
        assert!(dist.platforms.contains(&"linux-x86_64".to_string()));
        assert!(dist.platforms.contains(&"darwin-aarch64".to_string()));
    }

    #[test]
    fn test_add_artifact() {
        let mut dist = SovereignDistribution::new("test", "1.0");
        let artifact = SovereignArtifact {
            name: "model.apr".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "abc123".to_string(),
            size_bytes: 1024,
            source_url: None,
        };
        dist.add_artifact(artifact);
        assert_eq!(dist.artifacts.len(), 1);
        assert_eq!(dist.artifacts[0].name, "model.apr");
    }

    #[test]
    fn test_total_size_bytes() {
        let mut dist = SovereignDistribution::new("test", "1.0");
        dist.add_artifact(SovereignArtifact {
            name: "a.bin".to_string(),
            artifact_type: ArtifactType::Binary,
            sha256: "hash1".to_string(),
            size_bytes: 1000,
            source_url: None,
        });
        dist.add_artifact(SovereignArtifact {
            name: "b.model".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "hash2".to_string(),
            size_bytes: 2000,
            source_url: None,
        });
        assert_eq!(dist.total_size_bytes(), 3000);
    }

    #[test]
    fn test_total_size_bytes_empty() {
        let dist = SovereignDistribution::new("test", "1.0");
        assert_eq!(dist.total_size_bytes(), 0);
    }

    #[test]
    fn test_validate_signatures_missing() {
        let mut dist = SovereignDistribution::new("test", "1.0");
        dist.add_artifact(SovereignArtifact {
            name: "unsigned.bin".to_string(),
            artifact_type: ArtifactType::Binary,
            sha256: "hash".to_string(),
            size_bytes: 100,
            source_url: None,
        });
        let result = dist.validate_signatures();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_signatures_valid() {
        let mut dist = SovereignDistribution::new("test", "1.0");
        dist.add_artifact(SovereignArtifact {
            name: "signed.bin".to_string(),
            artifact_type: ArtifactType::Binary,
            sha256: "hash".to_string(),
            size_bytes: 100,
            source_url: None,
        });
        dist.signatures.push(ArtifactSignature {
            artifact_name: "signed.bin".to_string(),
            algorithm: SignatureAlgorithm::Ed25519,
            signature: "base64sig".to_string(),
            key_id: "key123".to_string(),
        });
        assert!(dist.validate_signatures().is_ok());
    }

    #[test]
    fn test_validate_signatures_empty_dist() {
        let dist = SovereignDistribution::new("test", "1.0");
        assert!(dist.validate_signatures().is_ok());
    }

    #[test]
    fn test_artifact_type_serialize() {
        assert_eq!(
            serde_json::to_string(&ArtifactType::Binary).expect("json serialize failed"),
            "\"Binary\""
        );
        assert_eq!(
            serde_json::to_string(&ArtifactType::Model).expect("json serialize failed"),
            "\"Model\""
        );
        assert_eq!(
            serde_json::to_string(&ArtifactType::NixDerivation).expect("json serialize failed"),
            "\"NixDerivation\""
        );
    }

    #[test]
    fn test_signature_algorithm_equality() {
        assert_eq!(SignatureAlgorithm::Ed25519, SignatureAlgorithm::Ed25519);
        assert_ne!(SignatureAlgorithm::Ed25519, SignatureAlgorithm::RSA4096);
    }

    #[test]
    fn test_sovereign_artifact_source_url() {
        let artifact = SovereignArtifact {
            name: "model.apr".to_string(),
            artifact_type: ArtifactType::Model,
            sha256: "abc123".to_string(),
            size_bytes: 1024,
            source_url: Some("https://example.com/model.apr".to_string()),
        };
        assert!(artifact.source_url.is_some());
        assert_eq!(
            artifact.source_url.expect("unexpected failure"),
            "https://example.com/model.apr"
        );
    }

    #[test]
    fn test_offline_registry_config() {
        let config = OfflineRegistryConfig {
            path: "/opt/registry".to_string(),
            index_path: "/opt/registry/index.json".to_string(),
            platforms: vec!["linux-x86_64".to_string()],
            last_sync: Some("2025-01-01T00:00:00Z".to_string()),
        };
        assert_eq!(config.path, "/opt/registry");
        assert_eq!(config.platforms.len(), 1);
    }

    #[test]
    fn test_distribution_with_registry() {
        let mut dist = SovereignDistribution::new("air-gapped", "2.0");
        dist.offline_registry = Some(OfflineRegistryConfig {
            path: "/mnt/data".to_string(),
            index_path: "/mnt/data/index.json".to_string(),
            platforms: vec!["linux-x86_64".to_string(), "darwin-aarch64".to_string()],
            last_sync: None,
        });
        assert!(dist.offline_registry.is_some());
    }

    #[test]
    fn test_distribution_with_nix_hash() {
        let mut dist = SovereignDistribution::new("reproducible", "1.0");
        dist.nix_flake_hash = Some("sha256-abc123".to_string());
        assert_eq!(dist.nix_flake_hash, Some("sha256-abc123".to_string()));
    }
}
