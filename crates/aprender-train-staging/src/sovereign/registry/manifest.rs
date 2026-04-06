//! Registry manifest containing all model entries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::types::ModelEntry;

/// Registry manifest containing all model entries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryManifest {
    /// List of model entries
    pub models: Vec<ModelEntry>,
    /// Last sync timestamp
    pub last_sync: Option<DateTime<Utc>>,
    /// Registry version
    pub version: String,
}

impl RegistryManifest {
    /// Create a new empty manifest
    pub fn new() -> Self {
        Self { models: Vec::new(), last_sync: None, version: "1.0".to_string() }
    }

    /// Add a model entry
    pub fn add(&mut self, entry: ModelEntry) {
        // Update or insert
        if let Some(existing) = self.models.iter_mut().find(|m| m.name == entry.name) {
            *existing = entry;
        } else {
            self.models.push(entry);
        }
    }

    /// Find a model by name
    pub fn find(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Find a model by name (mutable)
    pub fn find_mut(&mut self, name: &str) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|m| m.name == name)
    }

    /// List all available models (those with local paths)
    pub fn available(&self) -> Vec<&ModelEntry> {
        self.models.iter().filter(|m| m.is_local()).collect()
    }

    /// Update sync timestamp
    pub fn mark_synced(&mut self) {
        self.last_sync = Some(Utc::now());
    }

    /// Get total size of all models
    pub fn total_size_bytes(&self) -> u64 {
        self.models.iter().map(|m| m.size_bytes).sum()
    }

    /// Get count of models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if manifest is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
