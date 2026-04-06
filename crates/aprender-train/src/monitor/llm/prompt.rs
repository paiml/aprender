//! Prompt versioning with content-addressable IDs.

use crate::monitor::llm::error::{LLMError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Prompt identifier (content-addressable)
pub type PromptId = String;

/// Prompt version with content-addressable ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVersion {
    /// Content-addressable ID (SHA-256 of template)
    pub id: PromptId,
    /// Prompt template (with {variable} placeholders)
    pub template: String,
    /// Variable names in the template
    pub variables: Vec<String>,
    /// Version number (monotonically increasing per template family)
    pub version: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// SHA-256 hash of the template
    pub sha256: String,
    /// Optional description
    pub description: Option<String>,
    /// Optional tags
    pub tags: HashMap<String, String>,
}

impl PromptVersion {
    /// Create a new prompt version
    pub fn new(template: &str, variables: Vec<String>) -> Self {
        let sha256 = Self::compute_hash(template);
        let id = sha256[..16].to_string(); // Short ID from hash

        Self {
            id,
            template: template.to_string(),
            variables,
            version: 1,
            created_at: Utc::now(),
            sha256,
            description: None,
            tags: HashMap::new(),
        }
    }

    /// Create with specific version number
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Add description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute SHA-256 hash of template
    fn compute_hash(template: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(template.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Render template with variables
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String> {
        let mut result = self.template.clone();
        for var in &self.variables {
            let placeholder = format!("{{{var}}}");
            if let Some(value) = vars.get(var) {
                result = result.replace(&placeholder, value);
            } else {
                return Err(LLMError::EvaluationFailed(format!("Missing variable: {var}")));
            }
        }
        Ok(result)
    }

    /// Extract variables from template
    pub fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut in_var = false;
        let mut current = String::new();

        for c in template.chars() {
            match c {
                '{' => {
                    in_var = true;
                    current.clear();
                }
                '}' if in_var => {
                    if !current.is_empty() && !vars.contains(&current) {
                        vars.push(current.clone());
                    }
                    in_var = false;
                }
                _ if in_var => {
                    current.push(c);
                }
                _non_brace => {}
            }
        }
        vars
    }
}
