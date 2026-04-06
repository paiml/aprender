//! Archive deposit functionality.

use crate::research::artifact::ResearchArtifact;

use super::metadata::DepositMetadata;
use super::provider::ArchiveProvider;
use super::result::{DepositError, DepositResult};

/// Archive deposit request
#[derive(Debug, Clone)]
pub struct ArchiveDeposit {
    /// Archive provider
    pub provider: ArchiveProvider,
    /// Artifact to deposit
    pub artifact: ResearchArtifact,
    /// Deposit metadata
    pub metadata: DepositMetadata,
    /// Files to upload (path -> content)
    pub files: Vec<(String, Vec<u8>)>,
}

impl ArchiveDeposit {
    /// Create a new deposit
    pub fn new(provider: ArchiveProvider, artifact: ResearchArtifact) -> Self {
        let metadata = DepositMetadata::from_artifact(&artifact);
        Self { provider, artifact, metadata, files: Vec::new() }
    }

    /// Add a file to upload
    pub fn with_file(mut self, filename: impl Into<String>, content: Vec<u8>) -> Self {
        self.files.push((filename.into(), content));
        self
    }

    /// Add a text file
    pub fn with_text_file(self, filename: impl Into<String>, content: impl Into<String>) -> Self {
        self.with_file(filename, content.into().into_bytes())
    }

    /// Perform deposit (mock implementation for testing)
    pub fn deposit(&self) -> Result<DepositResult, DepositError> {
        // This is a mock implementation - real implementation would use HTTP client
        // to interact with the archive's API

        // Validate we have at least one file
        if self.files.is_empty() {
            return Err(DepositError::NoFiles);
        }

        // Generate mock DOI and record ID
        let record_id = format!("{}", rand::random::<u64>() % 10_000_000);
        let doi = format!("10.5281/zenodo.{record_id}");

        let url = format!("{}/record/{}", self.provider.base_url(), record_id);

        Ok(DepositResult { doi, record_id, url, provider: self.provider })
    }
}
