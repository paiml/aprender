//! HuggingFace Hub dataset download functionality.

use std::path::{Path, PathBuf};

use crate::{
    backend::{HttpBackend, StorageBackend},
    dataset::ArrowDataset,
    error::{Error, Result},
};

/// Base URL for HuggingFace Hub datasets API.
const HF_HUB_URL: &str = "https://huggingface.co";

/// HuggingFace Hub dataset configuration and loader.
///
/// This struct provides a builder pattern for configuring and downloading
/// datasets from the HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct HfDataset {
    /// Dataset repository name (e.g., "squad", "glue", "openai/gsm8k")
    repo_id: String,
    /// Git revision (branch, tag, or commit hash)
    revision: String,
    /// Dataset subset/config (optional)
    subset: Option<String>,
    /// Data split (train, validation, test)
    split: Option<String>,
    /// Local cache directory
    cache_dir: PathBuf,
}

impl HfDataset {
    /// Creates a new builder for a HuggingFace dataset.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - The dataset repository ID (e.g., "squad", "openai/gsm8k")
    pub fn builder(repo_id: impl Into<String>) -> HfDatasetBuilder {
        HfDatasetBuilder::new(repo_id)
    }

    /// Returns the repository ID.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Returns the revision being used.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the subset/config if set.
    pub fn subset(&self) -> Option<&str> {
        self.subset.as_deref()
    }

    /// Returns the split if set.
    pub fn split(&self) -> Option<&str> {
        self.split.as_deref()
    }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Downloads the dataset and returns an ArrowDataset.
    ///
    /// This method:
    /// 1. Checks the local cache for existing data
    /// 2. Downloads parquet files from HuggingFace Hub if not cached
    /// 3. Loads the parquet files into an ArrowDataset
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dataset cannot be found on HuggingFace Hub
    /// - The download fails
    /// - The parquet files cannot be parsed
    pub fn download(&self) -> Result<ArrowDataset> {
        // Build the URL path for the parquet file
        let parquet_path = self.build_parquet_path();
        let cache_file = self.cache_path_for(&parquet_path);

        // Check cache first
        if cache_file.exists() {
            return ArrowDataset::from_parquet(&cache_file);
        }

        // Download from HF Hub
        let url = self.build_download_url(&parquet_path);
        let http = HttpBackend::with_timeout(&url, 300)?;

        // The key is empty since we've already built the full URL
        let data = http.get("")?;

        // Ensure cache directory exists
        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(e, parent))?;
        }

        // Write to cache
        std::fs::write(&cache_file, &data).map_err(|e| Error::io(e, &cache_file))?;

        // Load from cache
        ArrowDataset::from_parquet(&cache_file)
    }

    /// Downloads the dataset to a specific output path.
    ///
    /// # Arguments
    ///
    /// * `output` - Path where the dataset should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the download or save fails.
    pub fn download_to(&self, output: impl AsRef<Path>) -> Result<ArrowDataset> {
        let output = output.as_ref();
        let parquet_path = self.build_parquet_path();
        let url = self.build_download_url(&parquet_path);

        let http = HttpBackend::with_timeout(&url, 300)?;
        let data = http.get("")?;

        // Ensure parent directory exists
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(e, parent))?;
        }

        // Write to output
        std::fs::write(output, &data).map_err(|e| Error::io(e, output))?;

        // Load and return
        ArrowDataset::from_parquet(output)
    }

    /// Builds the parquet file path within the repository.
    pub(crate) fn build_parquet_path(&self) -> String {
        let mut path_parts = Vec::new();

        // Add subset/config if present
        if let Some(ref subset) = self.subset {
            path_parts.push(subset.clone());
        } else {
            path_parts.push("default".to_string());
        }

        // Add split
        let split = self.split.as_deref().unwrap_or("train");
        path_parts.push(format!("{split}.parquet"));

        path_parts.join("/")
    }

    /// Builds the download URL for a parquet file.
    pub(crate) fn build_download_url(&self, parquet_path: &str) -> String {
        format!(
            "{}/datasets/{}/resolve/{}/data/{}",
            HF_HUB_URL, self.repo_id, self.revision, parquet_path
        )
    }

    /// Returns the cache path for a given parquet path.
    pub(crate) fn cache_path_for(&self, parquet_path: &str) -> PathBuf {
        self.cache_dir
            .join("huggingface")
            .join("datasets")
            .join(&self.repo_id)
            .join(&self.revision)
            .join(parquet_path)
    }

    /// Clears the local cache for this dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache cannot be deleted.
    pub fn clear_cache(&self) -> Result<()> {
        let cache_path = self
            .cache_dir
            .join("huggingface")
            .join("datasets")
            .join(&self.repo_id);

        if cache_path.exists() {
            std::fs::remove_dir_all(&cache_path).map_err(|e| Error::io(e, &cache_path))?;
        }

        Ok(())
    }
}

/// Builder for configuring HuggingFace dataset downloads.
#[derive(Debug, Clone)]
pub struct HfDatasetBuilder {
    repo_id: String,
    revision: String,
    subset: Option<String>,
    split: Option<String>,
    cache_dir: Option<PathBuf>,
}

impl HfDatasetBuilder {
    /// Creates a new builder with the given repository ID.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: "main".to_string(),
            subset: None,
            split: None,
            cache_dir: None,
        }
    }

    /// Sets the Git revision (branch, tag, or commit hash).
    ///
    /// Default is "main".
    #[must_use]
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Sets the dataset subset/configuration.
    ///
    /// Some datasets have multiple configurations (e.g., "glue" has "cola",
    /// "sst2", etc.)
    #[must_use]
    pub fn subset(mut self, subset: impl Into<String>) -> Self {
        self.subset = Some(subset.into());
        self
    }

    /// Sets the data split to download.
    ///
    /// Common values: "train", "validation", "test"
    #[must_use]
    pub fn split(mut self, split: impl Into<String>) -> Self {
        self.split = Some(split.into());
        self
    }

    /// Sets the local cache directory.
    ///
    /// Default is `~/.cache/alimentar` on Unix or `%LOCALAPPDATA%/alimentar` on
    /// Windows.
    #[must_use]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Builds the HfDataset configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or invalid.
    pub fn build(self) -> Result<HfDataset> {
        if self.repo_id.is_empty() {
            return Err(Error::invalid_config("Repository ID cannot be empty"));
        }

        let cache_dir = self.cache_dir.unwrap_or_else(default_cache_dir);

        Ok(HfDataset {
            repo_id: self.repo_id,
            revision: self.revision,
            subset: self.subset,
            split: self.split,
            cache_dir,
        })
    }
}

/// Returns the default cache directory for the current platform.
pub(crate) fn default_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local_app_data)
                .join("alimentar")
                .join("cache");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg_cache).join("alimentar");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("alimentar");
        }
    }

    // System scratch directory as last-resort cache location
    std::env::temp_dir().join("alimentar").join("cache")
}

/// Lists available parquet files for a dataset on HuggingFace Hub.
///
/// This function queries the HuggingFace Hub API to list available
/// parquet files for a given dataset.
///
/// # Arguments
///
/// * `repo_id` - The dataset repository ID
/// * `revision` - Git revision (default "main")
///
/// # Errors
///
/// Returns an error if the HTTP request fails or the response cannot be parsed.
///
/// # Note
///
/// This function requires the HF Hub API and may be rate-limited.
pub fn list_dataset_files(repo_id: &str, revision: Option<&str>) -> Result<Vec<String>> {
    let revision = revision.unwrap_or("main");
    let url = format!("{}/api/datasets/{}/tree/{}", HF_HUB_URL, repo_id, revision);

    let http = HttpBackend::with_timeout(&url, 30)?;
    let data = http.get("")?;

    // Parse JSON response
    let json: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| Error::storage(format!("Failed to parse HF Hub response: {e}")))?;

    let mut parquet_files = Vec::new();

    if let Some(items) = json.as_array() {
        for item in items {
            if let Some(path) = item.get("path").and_then(|p| p.as_str()) {
                if path.ends_with(".parquet") {
                    parquet_files.push(path.to_string());
                }
            }
        }
    }

    Ok(parquet_files)
}

/// Information about a HuggingFace dataset.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Dataset repository ID
    pub repo_id: String,
    /// Available splits
    pub splits: Vec<String>,
    /// Available subsets/configs
    pub subsets: Vec<String>,
    /// Total download size in bytes (if known)
    pub download_size: Option<u64>,
    /// Description
    pub description: Option<String>,
}
