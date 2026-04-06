//! HuggingFace Hub publisher
//!
//! Uploads models, files, and model cards to HuggingFace Hub repositories
//! using the HF REST API.

use std::path::Path;

use super::config::PublishConfig;
use super::model_card::ModelCard;
use super::result::{PublishError, PublishResult};
use crate::hf_pipeline::HfModelFetcher;

const HF_API_BASE: &str = "https://huggingface.co/api";

/// HuggingFace Hub publisher
pub struct HfPublisher {
    config: PublishConfig,
    client: reqwest::blocking::Client,
    token: String,
}

impl HfPublisher {
    /// Create a new publisher with config
    pub fn new(config: PublishConfig) -> Result<Self, PublishError> {
        let token = config
            .token
            .clone()
            .or_else(HfModelFetcher::resolve_token)
            .ok_or(PublishError::AuthRequired)?;

        if config.repo_id.is_empty() || !config.repo_id.contains('/') {
            return Err(PublishError::InvalidRepoId { repo_id: config.repo_id.clone() });
        }

        let client =
            reqwest::blocking::Client::builder().user_agent("entrenar/0.5").build().map_err(
                |e| PublishError::Http { message: format!("Failed to create HTTP client: {e}") },
            )?;

        Ok(Self { config, client, token })
    }

    /// Create the HuggingFace repository
    ///
    /// POST <https://huggingface.co/api/repos/create>
    pub fn create_repo(&self) -> Result<String, PublishError> {
        let url = format!("{HF_API_BASE}/repos/create");

        let mut body = serde_json::json!({
            "name": self.repo_name(),
            "type": self.config.repo_type.to_string(),
            "private": self.config.private,
        });

        // Add organization if repo_id contains one
        if let Some(org) = self.repo_org() {
            body["organization"] = serde_json::Value::String(org);
        }

        let response =
            self.client.post(&url).bearer_auth(&self.token).json(&body).send().map_err(|e| {
                PublishError::Http { message: format!("Create repo request failed: {e}") }
            })?;

        if response.status().is_success() || response.status().as_u16() == 409 {
            // 409 = already exists, which is fine
            let repo_url = format!(
                "https://huggingface.co/{}/{}",
                self.config.repo_type.api_path(),
                self.config.repo_id
            );
            Ok(repo_url)
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(PublishError::RepoCreationFailed {
                repo_id: self.config.repo_id.clone(),
                message: format!("HTTP {status}: {body}"),
            })
        }
    }

    /// Upload a local file to the repository
    ///
    /// PUT <https://huggingface.co/api/{type}s/{repo_id}/upload/{path}>
    pub fn upload_file(&self, local_path: &Path, path_in_repo: &str) -> Result<(), PublishError> {
        let content = std::fs::read(local_path).map_err(PublishError::Io)?;
        self.upload_bytes(&content, path_in_repo)
    }

    /// Upload bytes directly to the repository
    pub fn upload_bytes(&self, content: &[u8], path_in_repo: &str) -> Result<(), PublishError> {
        let url = format!(
            "{HF_API_BASE}/{}/{}/upload/main/{}",
            self.config.repo_type.api_path(),
            self.config.repo_id,
            path_in_repo
        );

        let response = self
            .client
            .put(&url)
            .bearer_auth(&self.token)
            .header("Content-Type", "application/octet-stream")
            .body(content.to_vec())
            .send()
            .map_err(|e| PublishError::UploadFailed {
                path: path_in_repo.to_string(),
                message: format!("Upload request failed: {e}"),
            })?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(PublishError::UploadFailed {
                path: path_in_repo.to_string(),
                message: format!("HTTP {status}: {body}"),
            })
        }
    }

    /// Full publish flow: create repo → upload files → upload model card
    pub fn publish(
        &self,
        files: &[(&Path, &str)],
        model_card: Option<&ModelCard>,
    ) -> Result<PublishResult, PublishError> {
        let repo_url = self.create_repo()?;

        let mut files_uploaded = 0;

        // Upload all files
        for (local_path, remote_path) in files {
            self.upload_file(local_path, remote_path)?;
            files_uploaded += 1;
        }

        // Upload model card
        let model_card_generated = if let Some(card) = model_card {
            let markdown = card.to_markdown();
            self.upload_bytes(markdown.as_bytes(), "README.md")?;
            true
        } else {
            false
        };

        Ok(PublishResult {
            repo_url,
            repo_id: self.config.repo_id.clone(),
            files_uploaded,
            model_card_generated,
        })
    }

    /// Extract the repository name (part after the last '/')
    fn repo_name(&self) -> &str {
        self.config.repo_id.rsplit('/').next().unwrap_or(&self.config.repo_id)
    }

    /// Extract the organization (part before '/')
    fn repo_org(&self) -> Option<String> {
        let parts: Vec<&str> = self.config.repo_id.splitn(2, '/').collect();
        if parts.len() == 2 {
            Some(parts[0].to_string())
        } else {
            None
        }
    }
}

impl std::fmt::Debug for HfPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfPublisher")
            .field("repo_id", &self.config.repo_id)
            .field("repo_type", &self.config.repo_type)
            .field("private", &self.config.private)
            .finish_non_exhaustive()
    }
}
