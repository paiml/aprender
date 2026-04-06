//! HuggingFace dataset fetcher

use std::path::PathBuf;

use crate::hf_pipeline::error::{FetchError, Result};

use super::dataset_impl::Dataset;
use super::options::DatasetOptions;

/// HuggingFace dataset fetcher
pub struct HfDatasetFetcher {
    /// HuggingFace token
    #[allow(dead_code)]
    token: Option<String>,
    /// Cache directory
    cache_dir: PathBuf,
}

impl HfDatasetFetcher {
    /// Create new fetcher
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("huggingface")
            .join("datasets");

        Ok(Self { token: std::env::var("HF_TOKEN").ok(), cache_dir })
    }

    /// Set cache directory
    #[must_use]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }

    /// Fetch dataset from HuggingFace
    ///
    /// # Arguments
    ///
    /// * `dataset_id` - Dataset ID (e.g., "wikitext", "squad")
    /// * `options` - Fetch options
    pub fn fetch(&self, dataset_id: &str, options: DatasetOptions) -> Result<Dataset> {
        // Validate dataset ID
        if dataset_id.is_empty() {
            return Err(FetchError::InvalidRepoId { repo_id: dataset_id.into() });
        }

        // For now, create mock dataset (actual HF API integration later)
        let num_examples = options.max_examples.unwrap_or(1000);
        let mut dataset = Dataset::mock(num_examples, 128);

        if options.shuffle {
            if let Some(seed) = options.seed {
                dataset.shuffle(seed);
            }
        }

        Ok(dataset)
    }

    /// Load dataset from local parquet file
    pub fn load_parquet(&self, path: &std::path::Path) -> Result<Dataset> {
        if !path.exists() {
            return Err(FetchError::FileNotFound {
                repo: path.parent().unwrap_or(path).display().to_string(),
                file: path.file_name().unwrap_or_default().to_string_lossy().into(),
            });
        }

        // Mock implementation - actual parquet parsing later
        Ok(Dataset::mock(100, 128))
    }
}

impl Default for HfDatasetFetcher {
    fn default() -> Self {
        Self::new().expect("Failed to create dataset fetcher")
    }
}
