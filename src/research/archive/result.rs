//! Deposit result and error types.

use serde::{Deserialize, Serialize};

use super::provider::ArchiveProvider;
use super::DOI_REGEX;

/// Result of a successful deposit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositResult {
    /// Assigned DOI
    pub doi: String,
    /// Provider-specific record ID
    pub record_id: String,
    /// URL to the deposited record
    pub url: String,
    /// Provider that received the deposit
    pub provider: ArchiveProvider,
}

impl DepositResult {
    /// Generate URL for the deposit
    pub fn generate_url(&self) -> String {
        self.url.clone()
    }

    /// Generate DOI URL
    pub fn doi_url(&self) -> String {
        format!("https://doi.org/{}", self.doi)
    }
}

/// Deposit errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum DepositError {
    #[error("No files provided for deposit")]
    NoFiles,
    #[error("Authentication failed")]
    AuthenticationFailed,
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
    #[error("Upload failed: {0}")]
    UploadFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
}

/// Validate DOI format
pub fn validate_doi(doi: &str) -> bool {
    DOI_REGEX.is_match(doi)
}
