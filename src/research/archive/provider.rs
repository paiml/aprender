//! Archive provider definitions.

use serde::{Deserialize, Serialize};

/// Archive provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchiveProvider {
    /// Zenodo (CERN)
    Zenodo,
    /// Figshare
    Figshare,
    /// Dryad
    Dryad,
    /// Dataverse
    Dataverse,
}

impl std::fmt::Display for ArchiveProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zenodo => write!(f, "Zenodo"),
            Self::Figshare => write!(f, "Figshare"),
            Self::Dryad => write!(f, "Dryad"),
            Self::Dataverse => write!(f, "Dataverse"),
        }
    }
}

impl ArchiveProvider {
    /// Get the base URL for the provider
    pub fn base_url(&self) -> &'static str {
        match self {
            Self::Zenodo => "https://zenodo.org",
            Self::Figshare => "https://figshare.com",
            Self::Dryad => "https://datadryad.org",
            Self::Dataverse => "https://dataverse.harvard.edu",
        }
    }

    /// Get the sandbox URL (if available)
    pub fn sandbox_url(&self) -> Option<&'static str> {
        match self {
            Self::Zenodo => Some("https://sandbox.zenodo.org"),
            Self::Figshare => None,
            Self::Dryad => None,
            Self::Dataverse => None,
        }
    }

    /// Get the API endpoint
    pub fn api_endpoint(&self) -> &'static str {
        match self {
            Self::Zenodo => "https://zenodo.org/api/deposit/depositions",
            Self::Figshare => "https://api.figshare.com/v2/account/articles",
            Self::Dryad => "https://datadryad.org/api/v2/datasets",
            Self::Dataverse => "https://dataverse.harvard.edu/api/dataverses",
        }
    }
}
