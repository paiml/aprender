//! HuggingFace Hub dataset upload functionality.

use std::path::Path;

use crate::error::{Error, Result};

/// HuggingFace Hub API URL for uploads
pub(crate) const HF_API_URL: &str = "https://huggingface.co/api";

/// Publisher for uploading datasets to HuggingFace Hub.
///
/// # Example
///
/// ```no_run
/// use alimentar::hf_hub::HfPublisher;
/// use arrow::record_batch::RecordBatch;
///
/// let publisher = HfPublisher::new("paiml/my-dataset")
///     .with_token(std::env::var("HF_TOKEN").unwrap())
///     .with_private(false);
///
/// // publisher.upload_parquet("train.parquet", &batch).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HfPublisher {
    /// Repository ID (e.g., "paiml/depyler-citl")
    repo_id: String,
    /// HuggingFace API token
    token: Option<String>,
    /// Whether the dataset should be private
    private: bool,
    /// Commit message for uploads
    commit_message: String,
}

impl HfPublisher {
    /// Creates a new publisher for a HuggingFace dataset repository.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: std::env::var("HF_TOKEN").ok(),
            private: false,
            commit_message: "Upload via alimentar".to_string(),
        }
    }

    /// Sets the HuggingFace API token.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets whether the dataset should be private.
    #[must_use]
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Sets the commit message for uploads.
    #[must_use]
    pub fn with_commit_message(mut self, message: impl Into<String>) -> Self {
        self.commit_message = message.into();
        self
    }

    /// Returns the repository ID.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Creates the dataset repository on HuggingFace Hub if it doesn't exist.
    #[cfg(feature = "http")]
    pub async fn create_repo(&self) -> Result<()> {
        let token = self.token.as_ref().ok_or_else(|| {
            Error::io_no_path(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "HF_TOKEN required for upload",
            ))
        })?;

        // Split repo_id into org/name components
        let (org, name) = if let Some(slash_pos) = self.repo_id.find('/') {
            let org = &self.repo_id[..slash_pos];
            let name = &self.repo_id[slash_pos + 1..];
            (Some(org), name)
        } else {
            (None, self.repo_id.as_str())
        };

        let client = reqwest::Client::new();
        let url = format!("{}/repos/create", HF_API_URL);

        let mut body = serde_json::json!({
            "type": "dataset",
            "name": name,
            "private": self.private
        });

        // Add organization if present
        if let Some(org_name) = org {
            body["organization"] = serde_json::json!(org_name);
        }

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        // 409 Conflict means repo already exists, which is fine
        if response.status().is_success() || response.status().as_u16() == 409 {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(Error::io_no_path(std::io::Error::other(format!(
                "Failed to create repo: {} - {}",
                status, body
            ))))
        }
    }

    /// Uploads a file to the repository.
    ///
    /// This method automatically selects the appropriate upload method:
    /// - **Binary files** (parquet, images, etc.): Uses LFS preupload API
    /// - **Text files** (README.md, JSON, etc.): Uses direct NDJSON commit API
    ///
    /// The official `hf-hub` crate only supports downloads, making this upload
    /// capability a key differentiator for alimentar.
    #[cfg(feature = "hf-hub")]
    pub async fn upload_file(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        if is_binary_file(path_in_repo) {
            self.upload_file_lfs(path_in_repo, data).await
        } else {
            self.upload_file_direct(path_in_repo, data).await
        }
    }

    /// Uploads a text file directly using the NDJSON commit API.
    #[cfg(feature = "hf-hub")]
    async fn upload_file_direct(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        let token = self.token.as_ref().ok_or_else(|| {
            Error::io_no_path(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "HF_TOKEN required for upload",
            ))
        })?;

        let client = reqwest::Client::new();
        let url = format!("{}/datasets/{}/commit/main", HF_API_URL, self.repo_id);

        let ndjson_payload = build_ndjson_upload_payload(&self.commit_message, path_in_repo, data);

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/x-ndjson")
            .body(ndjson_payload)
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(Error::io_no_path(std::io::Error::other(format!(
                "Failed to upload: {} - {}",
                status, body
            ))))
        }
    }

    /// Uploads a binary file using LFS batch API.
    ///
    /// Flow:
    /// 1. Compute SHA256 hash of the file content (OID)
    /// 2. POST to LFS batch API to get presigned S3 upload URL
    /// 3. PUT binary content to the S3 URL
    /// 4. POST commit with lfsFile reference
    #[cfg(feature = "hf-hub")]
    async fn upload_file_lfs(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        let token = self.token.as_ref().ok_or_else(|| {
            Error::io_no_path(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "HF_TOKEN required for upload",
            ))
        })?;

        let client = reqwest::Client::new();

        // Step 1: Compute SHA256 hash (LFS Object ID)
        let oid = compute_sha256(data);
        let size = data.len();

        // Step 2: Call LFS batch API to get presigned S3 upload URL
        let batch_url = format!(
            "https://huggingface.co/datasets/{}.git/info/lfs/objects/batch",
            self.repo_id
        );
        let batch_body = build_lfs_batch_request(&oid, size);

        let batch_response = client
            .post(&batch_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/vnd.git-lfs+json")
            .body(batch_body)
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        if !batch_response.status().is_success() {
            let status = batch_response.status();
            let body = batch_response.text().await.unwrap_or_default();
            return Err(Error::io_no_path(std::io::Error::other(format!(
                "LFS batch API failed: {} - {}",
                status, body
            ))));
        }

        let batch_json: serde_json::Value = batch_response
            .json()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        // Extract S3 upload URL from response: objects[0].actions.upload.href
        let objects = batch_json["objects"].as_array().ok_or_else(|| {
            Error::io_no_path(std::io::Error::other("Invalid LFS batch response"))
        })?;

        let object = objects
            .first()
            .ok_or_else(|| Error::io_no_path(std::io::Error::other("No object in LFS response")))?;

        // Check if upload is needed (object might already exist)
        let upload_action = object.get("actions").and_then(|a| a.get("upload"));

        if let Some(upload) = upload_action {
            let upload_url = upload["href"].as_str().ok_or_else(|| {
                Error::io_no_path(std::io::Error::other("No upload URL in LFS response"))
            })?;

            // Step 3: Upload binary content to S3 (presigned URL, no auth header needed)
            let upload_response = client
                .put(upload_url)
                .header("Content-Type", "application/octet-stream")
                .body(data.to_vec())
                .send()
                .await
                .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

            if !upload_response.status().is_success() {
                let status = upload_response.status();
                let body = upload_response.text().await.unwrap_or_default();
                return Err(Error::io_no_path(std::io::Error::other(format!(
                    "LFS S3 upload failed: {} - {}",
                    status, body
                ))));
            }
        }
        // If no upload action, object already exists in LFS - proceed to commit

        // Step 4: Commit with LFS file reference
        let commit_url = format!("{}/datasets/{}/commit/main", HF_API_URL, self.repo_id);
        let commit_payload =
            build_ndjson_lfs_commit(&self.commit_message, path_in_repo, &oid, size);

        let commit_response = client
            .post(&commit_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/x-ndjson")
            .body(commit_payload)
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        if commit_response.status().is_success() {
            Ok(())
        } else {
            let status = commit_response.status();
            let body = commit_response.text().await.unwrap_or_default();
            Err(Error::io_no_path(std::io::Error::other(format!(
                "LFS commit failed: {} - {}",
                status, body
            ))))
        }
    }

    /// Uploads a RecordBatch as a parquet file.
    #[cfg(feature = "hf-hub")]
    pub async fn upload_batch(
        &self,
        path_in_repo: &str,
        batch: &arrow::record_batch::RecordBatch,
    ) -> Result<()> {
        use parquet::arrow::ArrowWriter;

        // Write batch to parquet in memory
        let mut buffer = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buffer, batch.schema(), None).map_err(Error::Parquet)?;
            writer.write(batch).map_err(Error::Parquet)?;
            writer.close().map_err(Error::Parquet)?;
        }

        self.upload_file(path_in_repo, &buffer).await
    }

    /// Uploads a local parquet file to the repository.
    #[cfg(feature = "hf-hub")]
    pub async fn upload_parquet_file(&self, local_path: &Path, path_in_repo: &str) -> Result<()> {
        let data = std::fs::read(local_path).map_err(|e| Error::io(e, local_path))?;
        self.upload_file(path_in_repo, &data).await
    }

    /// Synchronous wrapper for creating repo (for CLI use).
    #[cfg(all(feature = "http", feature = "tokio-runtime"))]
    pub fn create_repo_sync(&self) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.create_repo())
    }

    /// Synchronous wrapper for uploading file (for CLI use).
    #[cfg(all(feature = "hf-hub", feature = "tokio-runtime"))]
    pub fn upload_file_sync(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_file(path_in_repo, data))
    }

    /// Synchronous wrapper for uploading parquet file (for CLI use).
    #[cfg(all(feature = "hf-hub", feature = "tokio-runtime"))]
    pub fn upload_parquet_file_sync(&self, local_path: &Path, path_in_repo: &str) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_parquet_file(local_path, path_in_repo))
    }

    /// Uploads a README.md with validation.
    ///
    /// Validates the dataset card metadata before upload to catch issues like
    /// invalid `task_categories` before they cause HuggingFace warnings.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails or upload fails.
    #[cfg(feature = "hf-hub")]
    pub async fn upload_readme_validated(&self, content: &str) -> Result<()> {
        super::validation::DatasetCardValidator::validate_readme_strict(content)?;
        self.upload_file("README.md", content.as_bytes()).await
    }

    /// Synchronous wrapper for validated README upload.
    #[cfg(all(feature = "hf-hub", feature = "tokio-runtime"))]
    pub fn upload_readme_validated_sync(&self, content: &str) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_readme_validated(content))
    }
}

/// Builder for HfPublisher with fluent interface.
#[derive(Debug, Clone)]
pub struct HfPublisherBuilder {
    repo_id: String,
    token: Option<String>,
    private: bool,
    commit_message: String,
}

impl HfPublisherBuilder {
    /// Creates a new builder.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: None,
            private: false,
            commit_message: "Upload via alimentar".to_string(),
        }
    }

    /// Sets the token.
    #[must_use]
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets private flag.
    #[must_use]
    pub fn private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Sets commit message.
    #[must_use]
    pub fn commit_message(mut self, message: impl Into<String>) -> Self {
        self.commit_message = message.into();
        self
    }

    /// Builds the publisher.
    pub fn build(self) -> HfPublisher {
        HfPublisher {
            repo_id: self.repo_id,
            token: self.token.or_else(|| std::env::var("HF_TOKEN").ok()),
            private: self.private,
            commit_message: self.commit_message,
        }
    }
}

// ============================================================================
// NDJSON Upload Payload Builder
// ============================================================================

/// Builds an NDJSON payload for the HuggingFace Hub commit API.
///
/// The HuggingFace commit API uses NDJSON (Newline-Delimited JSON) format:
/// - Line 1: Header with commit message
/// - Line 2+: File operations with base64-encoded content
///
/// # Arguments
///
/// * `commit_message` - The commit summary message
/// * `path_in_repo` - The file path within the repository
/// * `data` - The raw file content to upload
///
/// # Returns
///
/// A string containing the NDJSON payload ready for upload.
///
/// # Example
///
/// ```ignore
/// let payload = build_ndjson_upload_payload(
///     "Upload training data",
///     "train.parquet",
///     &parquet_bytes
/// );
/// ```
#[cfg(feature = "hf-hub")]
pub fn build_ndjson_upload_payload(
    commit_message: &str,
    path_in_repo: &str,
    data: &[u8],
) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // Line 1: Header with commit message
    let header = serde_json::json!({
        "key": "header",
        "value": {
            "summary": commit_message,
            "description": ""
        }
    });

    // Line 2: File operation with base64 content
    let file_op = serde_json::json!({
        "key": "file",
        "value": {
            "content": STANDARD.encode(data),
            "path": path_in_repo,
            "encoding": "base64"
        }
    });

    format!("{}\n{}", header, file_op)
}

// ============================================================================
// LFS Upload Support for Binary Files
// ============================================================================

/// Binary file extensions that require LFS upload.
const BINARY_EXTENSIONS: &[&str] = &[
    "parquet",
    "arrow",
    "bin",
    "safetensors",
    "pt",
    "pth",
    "onnx",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "tiff",
    "mp3",
    "wav",
    "flac",
    "ogg",
    "mp4",
    "webm",
    "avi",
    "mkv",
    "zip",
    "tar",
    "gz",
    "bz2",
    "xz",
    "7z",
    "rar",
    "pdf",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "npy",
    "npz",
    "h5",
    "hdf5",
    "pkl",
    "pickle",
];

/// Checks if a file path is a binary file that requires LFS upload.
///
/// HuggingFace Hub requires binary files to be uploaded via LFS/XET storage.
/// This function detects common binary file extensions.
pub fn is_binary_file(path: &str) -> bool {
    path.rsplit('.')
        .next()
        .map(|ext| BINARY_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Computes SHA256 hash of data for LFS.
///
/// LFS uses SHA256 hashes as object identifiers (OIDs).
#[cfg(feature = "hf-hub")]
pub fn compute_sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Builds a preupload request for the HuggingFace LFS API.
///
/// # Arguments
///
/// * `path` - The file path in the repository
/// * `data` - The binary file content
///
/// # Returns
///
/// JSON string for the preupload API request.
#[cfg(feature = "hf-hub")]
pub fn build_lfs_preupload_request(path: &str, data: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // Sample is first 512 bytes, base64 encoded
    let sample_size = std::cmp::min(512, data.len());
    let sample = STANDARD.encode(&data[..sample_size]);

    let request = serde_json::json!({
        "files": [{
            "path": path,
            "size": data.len(),
            "sample": sample
        }]
    });

    request.to_string()
}

/// Builds a request for the LFS batch API.
///
/// The LFS batch API is the Git LFS standard endpoint for uploading large
/// files. It returns presigned S3 URLs for actual binary upload.
///
/// # Arguments
///
/// * `oid` - The SHA256 hash of the file content
/// * `size` - The file size in bytes
///
/// # Returns
///
/// JSON string for the LFS batch API request.
#[cfg(feature = "hf-hub")]
pub fn build_lfs_batch_request(oid: &str, size: usize) -> String {
    let request = serde_json::json!({
        "operation": "upload",
        "transfers": ["basic"],
        "objects": [{
            "oid": oid,
            "size": size
        }]
    });

    request.to_string()
}

/// Builds an NDJSON commit payload for LFS files.
///
/// Unlike regular files (which use base64 content), LFS files use
/// the `lfsFile` key with SHA256 OID and size.
///
/// # Arguments
///
/// * `commit_message` - The commit summary message
/// * `path_in_repo` - The file path within the repository
/// * `oid` - The SHA256 hash of the file content
/// * `size` - The file size in bytes
#[cfg(feature = "hf-hub")]
pub fn build_ndjson_lfs_commit(
    commit_message: &str,
    path_in_repo: &str,
    oid: &str,
    size: usize,
) -> String {
    // Line 1: Header with commit message (same as regular commits)
    let header = serde_json::json!({
        "key": "header",
        "value": {
            "summary": commit_message,
            "description": ""
        }
    });

    // Line 2: LFS file operation with OID instead of content
    let file_op = serde_json::json!({
        "key": "lfsFile",
        "value": {
            "path": path_in_repo,
            "algo": "sha256",
            "oid": oid,
            "size": size
        }
    });

    format!("{}\n{}", header, file_op)
}
