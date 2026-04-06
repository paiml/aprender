//! Training run management — start, stop, list, metrics, export.
//!
//! Types and store for training runs. Presets and the training engine
//! live in `training_engine.rs`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Re-export engine types so callers use `training::TrainingPreset`
pub use super::training_engine::{run_lora_training, TrainingPreset};

// ============================================================================
// Training types
// ============================================================================

/// Training run metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub dataset_id: String,
    pub method: TrainingMethod,
    pub config: TrainingConfig,
    pub status: TrainingStatus,
    pub created_at: u64,
    pub metrics: Vec<TrainingMetric>,
    /// True when metrics are from simulated cosine schedule, not real gradients.
    /// Honest labeling per Jidoka — stop-the-line on false claims.
    #[serde(default)]
    pub simulated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Training method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMethod {
    Lora,
    Qlora,
    FullFinetune,
}

/// Optimizer type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerType {
    Adam,
    AdamW,
    Sgd,
}

/// Learning rate scheduler type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    Constant,
    Cosine,
    Linear,
    StepDecay,
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_max_seq_length")]
    pub max_seq_length: u32,
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default = "default_optimizer")]
    pub optimizer: OptimizerType,
    #[serde(default = "default_scheduler")]
    pub scheduler: SchedulerType,
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u32,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: u32,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
}

fn default_lora_r() -> u32 {
    16
}
fn default_lora_alpha() -> u32 {
    32
}
fn default_learning_rate() -> f64 {
    2e-4
}
fn default_epochs() -> u32 {
    3
}
fn default_batch_size() -> u32 {
    4
}
fn default_max_seq_length() -> u32 {
    2048
}
fn default_optimizer() -> OptimizerType {
    OptimizerType::AdamW
}
fn default_scheduler() -> SchedulerType {
    SchedulerType::Cosine
}
fn default_warmup_steps() -> u32 {
    100
}
fn default_grad_accum() -> u32 {
    4
}
fn default_max_grad_norm() -> f64 {
    1.0
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            max_seq_length: default_max_seq_length(),
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
            ],
            optimizer: default_optimizer(),
            scheduler: default_scheduler(),
            warmup_steps: default_warmup_steps(),
            gradient_accumulation_steps: default_grad_accum(),
            max_grad_norm: default_max_grad_norm(),
        }
    }
}

/// Training run status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    Queued,
    Running,
    Complete,
    Failed,
    Stopped,
}

/// A single training metric snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetric {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_sec: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_secs: Option<u64>,
}

/// Export format for trained adapters/models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Safetensors,
    Gguf,
    Apr,
}

/// Export request configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    #[serde(default = "default_export_format")]
    pub format: ExportFormat,
    #[serde(default)]
    pub merge: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

fn default_export_format() -> ExportFormat {
    ExportFormat::Safetensors
}

/// Export result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub run_id: String,
    pub format: ExportFormat,
    pub merged: bool,
    pub path: String,
    pub size_bytes: u64,
}

// ============================================================================
// Training store
// ============================================================================

/// Training run store.
pub struct TrainingStore {
    runs: RwLock<HashMap<String, TrainingRun>>,
    counter: std::sync::atomic::AtomicU64,
}

impl TrainingStore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            runs: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Start a training run.
    pub fn start(
        &self,
        dataset_id: &str,
        method: TrainingMethod,
        config: TrainingConfig,
    ) -> TrainingRun {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let run = TrainingRun {
            id: format!("run-{}-{seq}", epoch_secs()),
            dataset_id: dataset_id.to_string(),
            method,
            config,
            status: TrainingStatus::Queued,
            created_at: epoch_secs(),
            metrics: Vec::new(),
            simulated: true, // No real gradient-based training yet
            export_path: None,
            error: None,
        };
        if let Ok(mut store) = self.runs.write() {
            store.insert(run.id.clone(), run.clone());
        }
        run
    }

    /// Push a metric snapshot for a run.
    pub fn push_metric(&self, run_id: &str, metric: TrainingMetric) {
        if let Ok(mut store) = self.runs.write() {
            if let Some(run) = store.get_mut(run_id) {
                run.metrics.push(metric);
            }
        }
    }

    /// Update run status.
    pub fn set_status(&self, run_id: &str, status: TrainingStatus) {
        if let Ok(mut store) = self.runs.write() {
            if let Some(run) = store.get_mut(run_id) {
                run.status = status;
            }
        }
    }

    /// Mark run as failed with error message.
    pub fn fail(&self, run_id: &str, error: &str) {
        if let Ok(mut store) = self.runs.write() {
            if let Some(run) = store.get_mut(run_id) {
                run.status = TrainingStatus::Failed;
                run.error = Some(error.to_string());
            }
        }
    }

    /// Set export path for a completed run.
    pub fn set_export_path(&self, run_id: &str, path: &str) {
        if let Ok(mut store) = self.runs.write() {
            if let Some(run) = store.get_mut(run_id) {
                run.export_path = Some(path.to_string());
            }
        }
    }

    /// List all runs.
    #[must_use]
    pub fn list(&self) -> Vec<TrainingRun> {
        let store = self.runs.read().unwrap_or_else(|e| e.into_inner());
        let mut runs: Vec<TrainingRun> = store.values().cloned().collect();
        runs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        runs
    }

    /// Get a run by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<TrainingRun> {
        self.runs.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Stop a run.
    pub fn stop(&self, id: &str) -> Result<(), TrainingError> {
        let mut store = self.runs.write().map_err(|_| TrainingError::LockPoisoned)?;
        let run = store.get_mut(id).ok_or(TrainingError::NotFound(id.to_string()))?;
        run.status = TrainingStatus::Stopped;
        Ok(())
    }

    /// Delete a run.
    pub fn delete(&self, id: &str) -> Result<(), TrainingError> {
        let mut store = self.runs.write().map_err(|_| TrainingError::LockPoisoned)?;
        store.remove(id).ok_or(TrainingError::NotFound(id.to_string()))?;
        Ok(())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Training errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    NotFound(String),
    NoModel,
    NoDataset(String),
    LockPoisoned,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Training run not found: {id}"),
            Self::NoModel => write!(f, "No model loaded — load a model first"),
            Self::NoDataset(id) => write!(f, "Dataset not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for TrainingError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
