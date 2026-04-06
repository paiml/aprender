//! Batch inference — process multiple prompts in a single request.
//!
//! Accepts a list of prompt items, processes each through the chat pipeline,
//! and returns all results. Useful for dataset evaluation, bulk classification,
//! and generating training data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A single item in a batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItem {
    pub id: String,
    pub messages: Vec<crate::serve::templates::ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_tokens() -> u32 {
    256
}

/// Result for a single batch item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemResult {
    pub id: String,
    pub content: String,
    pub finish_reason: String,
    pub tokens: u32,
}

/// A batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub batch_id: String,
    pub status: BatchStatus,
    pub total_items: usize,
    pub completed_items: usize,
    pub results: Vec<BatchItemResult>,
}

/// Batch job status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Processing,
    Complete,
    Failed,
}

/// Batch store — tracks batch jobs.
pub struct BatchStore {
    jobs: RwLock<HashMap<String, BatchJob>>,
    counter: std::sync::atomic::AtomicU64,
}

impl BatchStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create and immediately run a batch job (synchronous for Phase 3).
    pub fn run(
        &self,
        items: Vec<BatchItem>,
        process_fn: impl Fn(&BatchItem) -> BatchItemResult,
    ) -> BatchJob {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let batch_id = format!("batch-{}-{seq}", epoch_secs());
        let total = items.len();

        let results: Vec<BatchItemResult> = items.iter().map(&process_fn).collect();

        let job = BatchJob {
            batch_id: batch_id.clone(),
            status: BatchStatus::Complete,
            total_items: total,
            completed_items: results.len(),
            results,
        };

        if let Ok(mut store) = self.jobs.write() {
            store.insert(batch_id, job.clone());
        }

        job
    }

    /// Get a batch job by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<BatchJob> {
        self.jobs.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// List all batch jobs.
    #[must_use]
    pub fn list(&self) -> Vec<BatchJob> {
        let store = self.jobs.read().unwrap_or_else(|e| e.into_inner());
        store.values().cloned().collect()
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
