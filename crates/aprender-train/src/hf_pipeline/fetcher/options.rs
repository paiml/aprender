//! Fetch options for HuggingFace model downloads.
//!
//! Provides configuration for model fetching including revision, files, and security settings.

use std::path::PathBuf;

/// Options for model fetching
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Git revision (branch, tag, or commit)
    pub revision: String,
    /// Specific files to download
    pub files: Vec<String>,
    /// Allow PyTorch pickle files (SECURITY RISK)
    pub allow_pytorch_pickle: bool,
    /// Expected SHA256 hash for verification
    pub verify_sha256: Option<String>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            files: vec![],
            allow_pytorch_pickle: false,
            verify_sha256: None,
            cache_dir: None,
        }
    }
}

impl FetchOptions {
    /// Create new options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set revision
    #[must_use]
    pub fn revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = rev.into();
        self
    }

    /// Set files to download
    #[must_use]
    pub fn files(mut self, files: &[&str]) -> Self {
        self.files = files.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Allow PyTorch pickle files (SECURITY RISK)
    #[must_use]
    pub fn allow_pytorch_pickle(mut self, allow: bool) -> Self {
        self.allow_pytorch_pickle = allow;
        self
    }

    /// Set SHA256 hash for verification
    #[must_use]
    pub fn verify_sha256(mut self, hash: impl Into<String>) -> Self {
        self.verify_sha256 = Some(hash.into());
        self
    }

    /// Set cache directory
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
}
