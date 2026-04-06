//! System prompt presets for Banco.
//!
//! Save named system prompts and reference them in chat via `@preset:name`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// A saved system prompt preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPreset {
    pub id: String,
    pub name: String,
    pub content: String,
    pub created: u64,
}

/// In-memory preset store.
pub struct PromptStore {
    presets: RwLock<HashMap<String, PromptPreset>>,
    counter: std::sync::atomic::AtomicU64,
}

impl PromptStore {
    #[must_use]
    pub fn new() -> Self {
        let store = Self {
            presets: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        };
        // Seed with built-in presets
        store.seed_defaults();
        store
    }

    fn seed_defaults(&self) {
        let defaults = [
            ("coding", "Coding Assistant", "You are an expert software engineer. Write clean, tested, idiomatic code. Explain your reasoning."),
            ("concise", "Concise", "You are a helpful assistant. Be concise and direct. No filler."),
            ("tutor", "Tutor", "You are a patient tutor. Explain concepts step by step. Ask the student questions to check understanding."),
        ];
        for (id, name, content) in defaults {
            if let Ok(mut store) = self.presets.write() {
                store.insert(
                    id.to_string(),
                    PromptPreset {
                        id: id.to_string(),
                        name: name.to_string(),
                        content: content.to_string(),
                        created: 0,
                    },
                );
            }
        }
    }

    /// Create or update a preset.
    pub fn save(&self, id: &str, name: &str, content: &str) -> PromptPreset {
        let preset = PromptPreset {
            id: id.to_string(),
            name: name.to_string(),
            content: content.to_string(),
            created: epoch_secs(),
        };
        if let Ok(mut store) = self.presets.write() {
            store.insert(id.to_string(), preset.clone());
        }
        preset
    }

    /// Create a preset with auto-generated ID.
    pub fn create(&self, name: &str, content: &str) -> PromptPreset {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = format!("preset-{seq}");
        self.save(&id, name, content)
    }

    /// Get a preset by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<PromptPreset> {
        self.presets.read().ok()?.get(id).cloned()
    }

    /// List all presets.
    #[must_use]
    pub fn list(&self) -> Vec<PromptPreset> {
        self.presets
            .read()
            .map(|s| {
                let mut v: Vec<_> = s.values().cloned().collect();
                v.sort_by(|a, b| a.id.cmp(&b.id));
                v
            })
            .unwrap_or_default()
    }

    /// Delete a preset by ID. Returns true if it existed.
    pub fn delete(&self, id: &str) -> bool {
        self.presets.write().map(|mut s| s.remove(id).is_some()).unwrap_or(false)
    }

    /// Expand `@preset:id` references in a message content string.
    /// Returns the expanded content, or the original if no preset found.
    #[must_use]
    pub fn expand(&self, content: &str) -> String {
        if let Some(preset_id) = content.strip_prefix("@preset:") {
            let preset_id = preset_id.trim();
            if let Some(preset) = self.get(preset_id) {
                return preset.content;
            }
        }
        content.to_string()
    }
}

impl Default for PromptStore {
    fn default() -> Self {
        Self::new()
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
