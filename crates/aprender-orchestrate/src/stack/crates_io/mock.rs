//! Mock client for testing without network calls.

use super::types::{CrateData, CrateResponse, VersionData};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Mock client for testing without network calls
#[derive(Debug, Default)]
pub struct MockCratesIoClient {
    /// Predefined responses
    pub responses: HashMap<String, Result<CrateResponse, String>>,
}

impl MockCratesIoClient {
    /// Create a new mock client
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mock response for a crate
    pub fn add_crate(&mut self, name: impl Into<String>, version: impl Into<String>) -> &mut Self {
        let name = name.into();
        let version = version.into();

        let response = CrateResponse {
            krate: CrateData {
                max_stable_version: Some(version.clone()),
                downloads: 1000,
                ..CrateData::new(name.clone(), version.clone())
            },
            versions: vec![VersionData::new(version, 1000)],
        };

        self.responses.insert(name, Ok(response));
        self
    }

    /// Add a "not found" response for a crate
    pub fn add_not_found(&mut self, name: impl Into<String>) -> &mut Self {
        self.responses.insert(name.into(), Err("Not found".to_string()));
        self
    }

    /// Get crate (mock implementation)
    pub fn get_crate(&self, name: &str) -> Result<CrateResponse> {
        match self.responses.get(name) {
            Some(Ok(response)) => Ok(response.clone()),
            Some(Err(e)) => Err(anyhow!("{}", e)),
            None => Err(anyhow!("Crate '{}' not found", name)),
        }
    }

    /// Get latest version (mock implementation)
    pub fn get_latest_version(&self, name: &str) -> Result<semver::Version> {
        let response = self.get_crate(name)?;
        response.krate.max_version.parse().map_err(|e| anyhow!("Failed to parse version: {}", e))
    }

    /// Check if version is published (mock implementation)
    pub fn is_version_published(&self, name: &str, version: &semver::Version) -> Result<bool> {
        let response = self.get_crate(name)?;
        let version_str = version.to_string();
        Ok(response.versions.iter().any(|v| v.num == version_str && !v.yanked))
    }
}
