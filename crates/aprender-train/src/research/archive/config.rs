//! Archive provider configuration.

use serde::{Deserialize, Serialize};

/// Zenodo-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenodoConfig {
    /// API access token
    pub token: String,
    /// Use sandbox environment
    pub sandbox: bool,
    /// Community to submit to (optional)
    pub community: Option<String>,
}

impl ZenodoConfig {
    /// Create a new Zenodo configuration
    pub fn new(token: impl Into<String>) -> Self {
        Self { token: token.into(), sandbox: false, community: None }
    }

    /// Use sandbox environment
    pub fn with_sandbox(mut self, sandbox: bool) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Set community
    pub fn with_community(mut self, community: impl Into<String>) -> Self {
        self.community = Some(community.into());
        self
    }

    /// Get the appropriate base URL
    pub fn base_url(&self) -> &'static str {
        if self.sandbox {
            "https://sandbox.zenodo.org"
        } else {
            "https://zenodo.org"
        }
    }
}

/// Figshare-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigshareConfig {
    /// API access token
    pub token: String,
    /// Project ID (optional)
    pub project_id: Option<u64>,
}

impl FigshareConfig {
    /// Create a new Figshare configuration
    pub fn new(token: impl Into<String>) -> Self {
        Self { token: token.into(), project_id: None }
    }

    /// Set project ID
    pub fn with_project(mut self, project_id: u64) -> Self {
        self.project_id = Some(project_id);
        self
    }
}
