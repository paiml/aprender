//! Sovereign distribution manifest

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ComponentManifest, DistributionFormat, DistributionTier};

/// Sovereign distribution manifest
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SovereignDistribution {
    /// Distribution name
    pub name: String,
    /// Distribution version
    pub version: String,
    /// Distribution tier
    pub tier: DistributionTier,
    /// Distribution format
    pub format: DistributionFormat,
    /// Component manifests
    pub components: Vec<ComponentManifest>,
    /// SHA-256 checksum of bundle
    pub checksum: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl SovereignDistribution {
    /// Create a new distribution manifest
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        tier: DistributionTier,
        format: DistributionFormat,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tier,
            format,
            components: Vec::new(),
            checksum: String::new(),
            created_at: Utc::now(),
        }
    }

    /// Create Core tier distribution (~50MB)
    pub fn core() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-core",
            version,
            DistributionTier::Core,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2"),
            ComponentManifest::aprender("0.1"),
        ];

        dist
    }

    /// Create Standard tier distribution (~200MB)
    pub fn standard() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-standard",
            version,
            DistributionTier::Standard,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2"),
            ComponentManifest::aprender("0.1"),
            ComponentManifest::renacer("0.1"),
            ComponentManifest::new("trueno-db", "0.1", "trueno-db"),
            ComponentManifest::new("ruchy", "0.1", "ruchy"),
        ];

        dist
    }

    /// Create Full tier distribution (~500MB)
    pub fn full() -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let mut dist = Self::new(
            "entrenar-sovereign-full",
            version,
            DistributionTier::Full,
            DistributionFormat::Tarball,
        );

        dist.components = vec![
            ComponentManifest::entrenar_core(version),
            ComponentManifest::trueno("0.2").with_features(["gpu", "cuda"]),
            ComponentManifest::aprender("0.1"),
            ComponentManifest::renacer("0.1"),
            ComponentManifest::new("trueno-db", "0.1", "trueno-db"),
            ComponentManifest::new("ruchy", "0.1", "ruchy"),
            ComponentManifest::new("entrenar-gpu", version, "entrenar-gpu")
                .with_features(["cuda", "rocm"]),
            ComponentManifest::new("entrenar-bench", version, "entrenar-bench"),
            ComponentManifest::new("entrenar-inspect", version, "entrenar-inspect"),
            ComponentManifest::new("entrenar-lora", version, "entrenar-lora"),
            ComponentManifest::new("entrenar-shell", version, "entrenar-shell"),
        ];

        dist
    }

    /// Set the distribution format
    pub fn with_format(mut self, format: DistributionFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the checksum
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = checksum.into();
        self
    }

    /// Calculate and set checksum from data
    pub fn compute_checksum(&mut self, data: &[u8]) {
        let mut hasher = Sha256::new();
        hasher.update(data);
        self.checksum = format!("{:x}", hasher.finalize());
    }

    /// Verify checksum against data
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        if self.checksum.is_empty() {
            return false;
        }

        let mut hasher = Sha256::new();
        hasher.update(data);
        let computed = format!("{:x}", hasher.finalize());

        computed == self.checksum
    }

    /// Serialize to JSON manifest
    pub fn to_manifest_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_err| "{}".to_string())
    }

    /// Parse from JSON manifest
    pub fn from_manifest_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the suggested filename for this distribution
    pub fn suggested_filename(&self) -> String {
        format!("{}-{}.{}", self.name, self.version, self.format.extension())
    }

    /// Get total component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Check if distribution includes a specific component
    pub fn has_component(&self, name: &str) -> bool {
        self.components.iter().any(|c| c.name == name)
    }
}

impl Default for SovereignDistribution {
    fn default() -> Self {
        Self::core()
    }
}
