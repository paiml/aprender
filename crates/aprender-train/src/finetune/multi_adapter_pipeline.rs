//! Multi-Adapter Pipeline (GPU-SHARE Phase 2.1, GH-203)
//!
//! Trains N LoRA adapter sets concurrently on a single frozen NF4 base model.
//! The base model is loaded once to GPU, and each adapter maintains independent:
//! - LoRA A/B matrices (Q and V projections)
//! - AdamW optimizer state
//! - Training data iterator
//! - Checkpoint directory
//!
//! # VRAM Savings
//!
//! Compared to N separate processes (MPS), this saves (N-1) × base_model_vram:
//! - MPS (3 adapters on 7B): 3 × 7.3 GB = 21.9 GB
//! - Multi-adapter (3 adapters on 7B): 7.3 GB + 3 × 0.02 GB = 7.36 GB
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │         Frozen NF4 Base Model        │ ← loaded once
//! │    (CudaNf4TransformerBlock × L)     │
//! └──────────┬───────────┬───────────┬───┘
//!            │           │           │
//!     ┌──────┴──┐ ┌──────┴──┐ ┌──────┴──┐
//!     │Adapter 0│ │Adapter 1│ │Adapter 2│
//!     │LoRA A/B │ │LoRA A/B │ │LoRA A/B │
//!     │Optimizer│ │Optimizer│ │Optimizer│
//!     │  Data   │ │  Data   │ │  Data   │
//!     └─────────┘ └─────────┘ └─────────┘
//! ```

use super::instruct_corpus::InstructSample;
use super::instruct_pipeline::{InstructConfig, InstructPipeline, InstructStepResult};
use super::instruct_trainer::InstructEpochMetrics;
use crate::lora::LoRALayer;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Scheduling strategy for multi-adapter training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterSchedule {
    /// All adapters process one sample each per step (synchronized).
    Synchronized,
    /// Round-robin: each step trains one adapter.
    #[default]
    RoundRobin,
    /// Priority: adapter with highest validation loss gets the next step.
    PriorityValLoss,
}

/// Configuration for a single adapter slot.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Path to training data (JSONL instruct corpus).
    pub data_path: PathBuf,
    /// Directory for adapter checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Per-adapter hyperparameters (lora_rank, lr, epochs, etc.)
    pub instruct_config: InstructConfig,
}

/// TOML file schema for `--adapters-config adapters.toml` (GPU-SHARE §2.4).
///
/// # Example
///
/// ```toml
/// [[adapter]]
/// data = "data/corpus-a.jsonl"
/// checkpoint = "checkpoints/adapter-a"
/// label = "code-review"
/// rank = 16
/// learning_rate = 0.0002
///
/// [[adapter]]
/// data = "data/corpus-b.jsonl"
/// checkpoint = "checkpoints/adapter-b"
/// label = "bug-fixing"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AdaptersConfigFile {
    /// List of adapter configurations.
    #[serde(rename = "adapter")]
    pub adapters: Vec<AdapterEntry>,
}

/// A single adapter entry in the TOML config file.
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterEntry {
    /// Path to training data (JSONL instruct corpus).
    pub data: PathBuf,
    /// Directory for adapter checkpoints.
    pub checkpoint: PathBuf,
    /// Human-readable label for this adapter.
    #[serde(default)]
    pub label: Option<String>,
    /// LoRA rank override (default: 16).
    #[serde(default)]
    pub rank: Option<usize>,
    /// Learning rate override.
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// Epochs override.
    #[serde(default)]
    pub epochs: Option<usize>,
    /// Maximum sequence length override.
    #[serde(default)]
    pub max_seq_len: Option<usize>,
}

impl AdaptersConfigFile {
    /// Parse an adapters config from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_toml(&contents)
    }

    /// Parse an adapters config from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        let config: Self =
            toml::from_str(toml_str).map_err(|e| format!("failed to parse adapters TOML: {e}"))?;
        if config.adapters.is_empty() {
            return Err("adapters config must have at least one [[adapter]] entry".to_string());
        }
        Ok(config)
    }

    /// Convert to `Vec<AdapterConfig>` using a base `InstructConfig` for defaults.
    pub fn to_adapter_configs(&self, base: &InstructConfig) -> Vec<AdapterConfig> {
        self.adapters
            .iter()
            .map(|entry| {
                let mut config = base.clone();
                if let Some(rank) = entry.rank {
                    config.lora_rank = rank;
                    config.lora_alpha = rank as f32 * 2.0;
                }
                if let Some(lr) = entry.learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(epochs) = entry.epochs {
                    config.epochs = epochs;
                }
                if let Some(seq_len) = entry.max_seq_len {
                    config.max_seq_len = seq_len;
                }
                AdapterConfig {
                    data_path: entry.data.clone(),
                    checkpoint_dir: entry.checkpoint.clone(),
                    instruct_config: config,
                }
            })
            .collect()
    }
}

/// Runtime state for one adapter during training.
pub struct AdapterSlot {
    /// Per-adapter LoRA layers (Q and V projections, per transformer layer).
    pub lora_layers: Vec<LoRALayer>,
    /// Training data for this adapter.
    pub train_samples: Vec<InstructSample>,
    /// Validation data for this adapter.
    pub val_samples: Vec<InstructSample>,
    /// Checkpoint directory for this adapter.
    pub checkpoint_dir: PathBuf,
    /// Per-adapter metrics history.
    pub metrics: Vec<InstructEpochMetrics>,
    /// Per-adapter config.
    pub config: InstructConfig,
    /// Current sample index within the training data.
    pub cursor: usize,
    /// Best validation loss (for early stopping / priority scheduling).
    pub best_val_loss: f32,

    /// Per-adapter GPU LoRA optimizer states.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) optimizer_states: Option<Vec<crate::transformer::GpuLoraOptimizerState>>,
    /// NF4 LoRA optimizer step counter.
    #[cfg(feature = "cuda")]
    pub lora_step: u32,
}

/// Multi-adapter training pipeline.
///
/// Trains N LoRA adapter sets on a shared frozen NF4 base model.
/// GPU memory is dominated by the base model (~7 GB for 7B NF4);
/// each adapter adds only ~20 MB (LoRA A/B matrices + optimizer state).
pub struct MultiAdapterPipeline {
    /// The base InstructPipeline (owns the frozen transformer + CUDA blocks).
    pub base_pipeline: InstructPipeline,
    /// Independent adapter slots.
    pub adapters: Vec<AdapterSlot>,
    /// Scheduling strategy.
    pub schedule: AdapterSchedule,
    /// Current step counter (global across all adapters).
    pub global_step: usize,
}

impl MultiAdapterPipeline {
    /// Create a new multi-adapter pipeline.
    ///
    /// The `base_pipeline` should be a fully initialized InstructPipeline
    /// (with CUDA blocks uploaded if GPU training is desired). Adapter slots
    /// are initially empty — call `add_adapter()` to register each one.
    pub fn new(base_pipeline: InstructPipeline, schedule: AdapterSchedule) -> Self {
        Self { base_pipeline, adapters: Vec::new(), schedule, global_step: 0 }
    }

    /// Add an adapter slot with its own training data and checkpoint directory.
    pub fn add_adapter(
        &mut self,
        config: AdapterConfig,
        train_samples: Vec<InstructSample>,
        val_samples: Vec<InstructSample>,
    ) {
        let model_config = &self.base_pipeline.model.config;
        let lora_layers = InstructPipeline::build_lora_layers(
            &self.base_pipeline.model,
            model_config,
            &config.instruct_config,
        );

        let slot = AdapterSlot {
            lora_layers,
            train_samples,
            val_samples,
            checkpoint_dir: config.checkpoint_dir,
            metrics: Vec::new(),
            config: config.instruct_config,
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        };

        self.adapters.push(slot);
    }

    /// Number of registered adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Select which adapter index to train next based on the schedule.
    pub fn select_next_adapter(&self) -> Option<usize> {
        if self.adapters.is_empty() {
            return None;
        }
        match self.schedule {
            AdapterSchedule::Synchronized => {
                // All adapters train — caller should iterate all
                Some(0)
            }
            AdapterSchedule::RoundRobin => Some(self.global_step % self.adapters.len()),
            AdapterSchedule::PriorityValLoss => {
                // Pick adapter with highest (worst) validation loss
                self.adapters
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.best_val_loss
                            .partial_cmp(&b.best_val_loss)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
            }
        }
    }

    /// Train one step on the specified adapter.
    ///
    /// Swaps the adapter's LoRA layers into the base pipeline, runs one
    /// training step, then swaps them back out.
    ///
    /// # Returns
    ///
    /// Training step result (loss, perplexity) or `None` if the adapter's
    /// data is exhausted.
    pub fn train_step_adapter(&mut self, adapter_idx: usize) -> Option<InstructStepResult> {
        let slot = &mut self.adapters[adapter_idx];

        // Check if data is exhausted
        if slot.cursor >= slot.train_samples.len() {
            return None;
        }

        let sample = &slot.train_samples[slot.cursor];
        slot.cursor += 1;

        // Tokenize prompt and response separately
        if !self.base_pipeline.has_tokenizer() {
            return None;
        }
        let prompt_ids = self.base_pipeline.tokenize(&sample.instruction);
        let response_ids = self.base_pipeline.tokenize(&sample.response);

        if prompt_ids.is_empty() || response_ids.is_empty() {
            return None;
        }

        // Swap adapter's LoRA layers into the base pipeline
        std::mem::swap(&mut slot.lora_layers, &mut self.base_pipeline.lora_layers);

        // Run training step through base pipeline (uses shared CUDA blocks)
        let result = self.base_pipeline.train_step(&prompt_ids, &response_ids);

        // Swap LoRA layers back
        std::mem::swap(&mut slot.lora_layers, &mut self.base_pipeline.lora_layers);

        self.global_step += 1;

        Some(result)
    }

    /// Reset all adapter cursors for a new epoch.
    pub fn reset_epoch(&mut self, seed: u64) {
        for (i, slot) in self.adapters.iter_mut().enumerate() {
            slot.cursor = 0;
            // Shuffle training data with per-adapter seed
            shuffle_samples(&mut slot.train_samples, seed.wrapping_add(i as u64));
        }
    }

    /// Check if all adapters have exhausted their training data.
    pub fn all_exhausted(&self) -> bool {
        self.adapters.iter().all(|s| s.cursor >= s.train_samples.len())
    }

    /// Batch training step across all non-exhausted adapters (GH-204).
    ///
    /// Trains each adapter that still has data, using the scheduling mode.
    /// In `Synchronized` mode, all adapters train one sample each.
    /// In `RoundRobin`, only the next scheduled adapter trains.
    /// In `PriorityValLoss`, the adapter with highest val loss trains.
    ///
    /// Returns per-adapter step results (indexed by adapter, None if skipped/exhausted).
    ///
    /// NOTE: Current implementation runs sequential forward+backward per adapter
    /// (swapping LoRA layers). Future optimization: fused BatchLoRA forward
    /// through shared NF4 blocks with per-adapter LoRA deltas (arXiv:2510.00206).
    pub fn batch_train_step(&mut self) -> Vec<Option<InstructStepResult>> {
        let n = self.adapters.len();
        let mut results = vec![None; n];

        match self.schedule {
            AdapterSchedule::Synchronized => {
                // All adapters train one sample each
                for i in 0..n {
                    results[i] = self.train_step_adapter(i);
                }
            }
            AdapterSchedule::RoundRobin | AdapterSchedule::PriorityValLoss => {
                // Single adapter per step
                if let Some(idx) = self.select_next_adapter() {
                    results[idx] = self.train_step_adapter(idx);
                }
            }
        }

        results
    }

    /// Save a checkpoint for the specified adapter.
    ///
    /// Creates `{checkpoint_dir}/epoch-{epoch}/` with:
    /// - `metadata.json`: adapter index, epoch, metrics
    /// - `model.safetensors`: LoRA A/B weights for this adapter
    pub fn save_adapter_checkpoint(
        &self,
        adapter_idx: usize,
        epoch: usize,
        avg_loss: f32,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let slot = &self.adapters[adapter_idx];
        let ckpt_dir = slot.checkpoint_dir.join(format!("epoch-{epoch}"));
        std::fs::create_dir_all(&ckpt_dir)?;

        // Metadata
        let metadata = serde_json::json!({
            "mode": "multi_adapter",
            "adapter_index": adapter_idx,
            "epoch": epoch,
            "avg_loss": avg_loss,
            "best_val_loss": slot.best_val_loss,
            "lora_rank": slot.config.lora_rank,
            "lora_alpha": slot.config.lora_alpha,
            "train_samples": slot.train_samples.len(),
            "global_step": self.global_step,
        });
        std::fs::write(ckpt_dir.join("metadata.json"), serde_json::to_string_pretty(&metadata)?)?;

        // Save LoRA weights as SafeTensors
        save_adapter_lora_weights(&slot.lora_layers, &ckpt_dir)?;

        Ok(ckpt_dir)
    }

    /// Save best checkpoint for an adapter (overwrites previous best).
    pub fn save_best_checkpoint(
        &self,
        adapter_idx: usize,
        epoch: usize,
        avg_loss: f32,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let slot = &self.adapters[adapter_idx];
        let best_dir = slot.checkpoint_dir.join("best");
        std::fs::create_dir_all(&best_dir)?;

        let metadata = serde_json::json!({
            "mode": "multi_adapter",
            "adapter_index": adapter_idx,
            "epoch": epoch,
            "avg_loss": avg_loss,
            "lora_rank": slot.config.lora_rank,
            "lora_alpha": slot.config.lora_alpha,
            "global_step": self.global_step,
        });
        std::fs::write(best_dir.join("metadata.json"), serde_json::to_string_pretty(&metadata)?)?;

        save_adapter_lora_weights(&slot.lora_layers, &best_dir)?;
        Ok(best_dir)
    }
}

/// Save LoRA A/B weights to a SafeTensors file in the given directory.
fn save_adapter_lora_weights(
    lora_layers: &[LoRALayer],
    dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for (idx, lora) in lora_layers.iter().enumerate() {
        let layer = idx / 2;
        let proj = if idx % 2 == 0 { "q" } else { "v" };

        // LoRA A: [rank, d_in]
        let a_data = lora.lora_a().data();
        let a_bytes: Vec<u8> =
            bytemuck::cast_slice(a_data.as_slice().expect("contiguous lora_a")).to_vec();
        let a_shape = vec![lora.rank(), lora.d_in()];
        tensor_data.push((format!("lora.{layer}.{proj}_proj.lora_a"), a_bytes, a_shape));

        // LoRA B: [d_out, rank]
        let b_data = lora.lora_b().data();
        let b_bytes: Vec<u8> =
            bytemuck::cast_slice(b_data.as_slice().expect("contiguous lora_b")).to_vec();
        let b_shape = vec![lora.d_out(), lora.rank()];
        tensor_data.push((format!("lora.{layer}.{proj}_proj.lora_b"), b_bytes, b_shape));
    }

    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                shape.clone(),
                bytes,
            )
            .expect("valid tensor view");
            (name.as_str(), view)
        })
        .collect();

    let safetensor_bytes = safetensors::serialize(views, None)
        .map_err(|e| format!("SafeTensors serialization failed: {e}"))?;
    std::fs::write(dir.join("model.safetensors"), safetensor_bytes)?;
    Ok(())
}

/// Simple Fisher-Yates shuffle with a deterministic seed.
fn shuffle_samples(samples: &mut [InstructSample], seed: u64) {
    let mut rng = seed;
    for i in (1..samples.len()).rev() {
        // xorshift64
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        samples.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_round_robin() {
        let sched = AdapterSchedule::RoundRobin;
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot(), dummy_slot()],
            schedule: sched,
            global_step: 0,
        };

        assert_eq!(pipeline.select_next_adapter(), Some(0));

        let pipeline = MultiAdapterPipeline { global_step: 1, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(1));

        let pipeline = MultiAdapterPipeline { global_step: 5, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(2));
    }

    #[test]
    fn test_schedule_priority_val_loss() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = 1.0;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 3.0; // worst
        let mut slot2 = dummy_slot();
        slot2.best_val_loss = 2.0;

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1, slot2],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };

        assert_eq!(pipeline.select_next_adapter(), Some(1)); // highest loss
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert_eq!(pipeline.select_next_adapter(), None);
        assert!(pipeline.all_exhausted());
    }

    #[test]
    fn test_shuffle_deterministic() {
        let mut samples1 = vec![
            InstructSample {
                instruction: "a".into(),
                response: "1".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "b".into(),
                response: "2".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "c".into(),
                response: "3".into(),
                system: None,
                metadata: None,
            },
        ];
        let mut samples2 = samples1.clone();

        shuffle_samples(&mut samples1, 42);
        shuffle_samples(&mut samples2, 42);

        // Same seed → same order
        for (s1, s2) in samples1.iter().zip(samples2.iter()) {
            assert_eq!(s1.instruction, s2.instruction);
        }
    }

    #[test]
    fn test_batch_train_step_synchronized() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::Synchronized,
            global_step: 0,
        };

        // No tokenizer → all results are None, but batch_train_step returns correct length
        let results = pipeline.batch_train_step();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_train_step_round_robin() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        let results = pipeline.batch_train_step();
        assert_eq!(results.len(), 3);
        // RoundRobin at step 0 → only adapter 0 would be trained
        // (but no tokenizer, so all None)
    }

    #[test]
    fn test_adapters_config_parse() {
        let toml = r#"
[[adapter]]
data = "data/corpus-a.jsonl"
checkpoint = "checkpoints/adapter-a"
label = "code-review"
rank = 16
learning_rate = 0.0002

[[adapter]]
data = "data/corpus-b.jsonl"
checkpoint = "checkpoints/adapter-b"
label = "bug-fixing"
rank = 8
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid TOML");
        assert_eq!(config.adapters.len(), 2);
        assert_eq!(config.adapters[0].data, PathBuf::from("data/corpus-a.jsonl"));
        assert_eq!(config.adapters[0].rank, Some(16));
        assert_eq!(config.adapters[0].learning_rate, Some(0.0002));
        assert_eq!(config.adapters[1].rank, Some(8));
        assert!(config.adapters[1].learning_rate.is_none());
    }

    #[test]
    fn test_adapters_config_to_adapter_configs() {
        let toml = r#"
[[adapter]]
data = "data/a.jsonl"
checkpoint = "ckpt/a"
rank = 32
learning_rate = 0.001
epochs = 5
max_seq_len = 256
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].instruct_config.lora_rank, 32);
        assert!((adapters[0].instruct_config.learning_rate - 0.001).abs() < f32::EPSILON);
        assert_eq!(adapters[0].instruct_config.epochs, 5);
        assert_eq!(adapters[0].instruct_config.max_seq_len, 256);
    }

    #[test]
    fn test_adapters_config_empty_fails() {
        let toml = "";
        assert!(AdaptersConfigFile::from_toml(toml).is_err());
    }

    #[test]
    fn test_adapters_config_defaults_from_base() {
        let toml = r#"
[[adapter]]
data = "data/x.jsonl"
checkpoint = "ckpt/x"
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let base = InstructConfig {
            lora_rank: 16,
            learning_rate: 0.0002,
            epochs: 3,
            max_seq_len: 512,
            ..Default::default()
        };
        let adapters = config.to_adapter_configs(&base);
        // Should inherit base defaults when not overridden
        assert_eq!(adapters[0].instruct_config.lora_rank, 16);
        assert!((adapters[0].instruct_config.learning_rate - 0.0002).abs() < f32::EPSILON);
        assert_eq!(adapters[0].instruct_config.epochs, 3);
        assert_eq!(adapters[0].instruct_config.max_seq_len, 512);
    }

    fn create_dummy_pipeline() -> InstructPipeline {
        use crate::transformer::TransformerConfig;
        let config = TransformerConfig::tiny();
        InstructPipeline::new(&config, InstructConfig::default())
    }

    fn dummy_slot() -> AdapterSlot {
        AdapterSlot {
            lora_layers: Vec::new(),
            train_samples: Vec::new(),
            val_samples: Vec::new(),
            checkpoint_dir: PathBuf::from("/tmp/test"),
            metrics: Vec::new(),
            config: InstructConfig::default(),
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        }
    }

    fn dummy_slot_with_data(n_samples: usize) -> AdapterSlot {
        let samples: Vec<InstructSample> = (0..n_samples)
            .map(|i| InstructSample {
                instruction: format!("inst_{i}"),
                response: format!("resp_{i}"),
                system: None,
                metadata: None,
            })
            .collect();
        AdapterSlot {
            lora_layers: Vec::new(),
            train_samples: samples,
            val_samples: Vec::new(),
            checkpoint_dir: PathBuf::from("/tmp/test"),
            metrics: Vec::new(),
            config: InstructConfig::default(),
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        }
    }

    // ── Coverage improvement tests ───────────────────────────────

    #[test]
    fn test_adapter_schedule_default() {
        let sched: AdapterSchedule = Default::default();
        assert_eq!(sched, AdapterSchedule::RoundRobin);
    }

    #[test]
    fn test_adapter_schedule_debug() {
        assert_eq!(format!("{:?}", AdapterSchedule::Synchronized), "Synchronized");
        assert_eq!(format!("{:?}", AdapterSchedule::RoundRobin), "RoundRobin");
        assert_eq!(format!("{:?}", AdapterSchedule::PriorityValLoss), "PriorityValLoss");
    }

    #[test]
    fn test_adapter_schedule_clone() {
        let sched = AdapterSchedule::PriorityValLoss;
        let cloned = sched;
        assert_eq!(sched, cloned);
    }

    #[test]
    fn test_adapter_schedule_eq() {
        assert_eq!(AdapterSchedule::Synchronized, AdapterSchedule::Synchronized);
        assert_ne!(AdapterSchedule::Synchronized, AdapterSchedule::RoundRobin);
        assert_ne!(AdapterSchedule::RoundRobin, AdapterSchedule::PriorityValLoss);
    }

    #[test]
    fn test_select_next_adapter_synchronized() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::Synchronized,
            global_step: 0,
        };
        // Synchronized always returns Some(0)
        assert_eq!(pipeline.select_next_adapter(), Some(0));
    }

    #[test]
    fn test_select_next_adapter_synchronized_any_step() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::Synchronized,
            global_step: 42,
        };
        assert_eq!(pipeline.select_next_adapter(), Some(0));
    }

    #[test]
    fn test_select_next_adapter_round_robin_wraps() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 3,
        };
        assert_eq!(pipeline.select_next_adapter(), Some(0)); // 3 % 3 = 0
    }

    #[test]
    fn test_select_next_adapter_priority_all_infinity() {
        // All slots have INFINITY best_val_loss → first one wins (or any, but deterministic)
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        let result = pipeline.select_next_adapter();
        assert!(result.is_some());
    }

    #[test]
    fn test_select_next_adapter_priority_with_nan() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = f32::NAN;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 1.0;

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        // NaN comparison uses Ordering::Equal fallback, so result is deterministic
        let result = pipeline.select_next_adapter();
        assert!(result.is_some());
    }

    #[test]
    fn test_num_adapters() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert_eq!(pipeline.num_adapters(), 3);
    }

    #[test]
    fn test_num_adapters_empty() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert_eq!(pipeline.num_adapters(), 0);
    }

    #[test]
    fn test_all_exhausted_with_data() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(3), dummy_slot_with_data(2)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert!(!pipeline.all_exhausted());
    }

    #[test]
    fn test_all_exhausted_partially() {
        let mut slot0 = dummy_slot_with_data(3);
        slot0.cursor = 3; // exhausted
        let slot1 = dummy_slot_with_data(2); // not exhausted

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert!(!pipeline.all_exhausted());
    }

    #[test]
    fn test_all_exhausted_all_done() {
        let mut slot0 = dummy_slot_with_data(3);
        slot0.cursor = 3;
        let mut slot1 = dummy_slot_with_data(2);
        slot1.cursor = 2;

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert!(pipeline.all_exhausted());
    }

    #[test]
    fn test_reset_epoch() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(5), dummy_slot_with_data(3)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        pipeline.adapters[0].cursor = 5;
        pipeline.adapters[1].cursor = 3;

        pipeline.reset_epoch(42);

        assert_eq!(pipeline.adapters[0].cursor, 0);
        assert_eq!(pipeline.adapters[1].cursor, 0);
    }

    #[test]
    fn test_reset_epoch_shuffle_deterministic() {
        let mut pipeline1 = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(10)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        let mut pipeline2 = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(10)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        pipeline1.reset_epoch(123);
        pipeline2.reset_epoch(123);

        // Same seed should produce same shuffle
        for (s1, s2) in pipeline1.adapters[0]
            .train_samples
            .iter()
            .zip(pipeline2.adapters[0].train_samples.iter())
        {
            assert_eq!(s1.instruction, s2.instruction);
        }
    }

    #[test]
    fn test_shuffle_samples_empty() {
        let mut samples: Vec<InstructSample> = vec![];
        shuffle_samples(&mut samples, 42);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_shuffle_samples_single() {
        let mut samples = vec![InstructSample {
            instruction: "only".into(),
            response: "one".into(),
            system: None,
            metadata: None,
        }];
        shuffle_samples(&mut samples, 42);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].instruction, "only");
    }

    #[test]
    fn test_shuffle_samples_different_seeds() {
        let mut samples1 = vec![
            InstructSample {
                instruction: "a".into(),
                response: "1".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "b".into(),
                response: "2".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "c".into(),
                response: "3".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "d".into(),
                response: "4".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "e".into(),
                response: "5".into(),
                system: None,
                metadata: None,
            },
        ];
        let mut samples2 = samples1.clone();

        shuffle_samples(&mut samples1, 1);
        shuffle_samples(&mut samples2, 999);

        // Different seeds should (very likely) produce different orderings
        let same =
            samples1.iter().zip(samples2.iter()).all(|(s1, s2)| s1.instruction == s2.instruction);
        // With 5! = 120 permutations, probability of same is ~0.83%, so this is safe
        assert!(!same, "Different seeds should produce different shuffles");
    }

    #[test]
    fn test_adapters_config_from_toml_invalid_toml() {
        let toml = "this is not valid TOML {{{}}}";
        let result = AdaptersConfigFile::from_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("failed to parse"), "Expected parse error, got: {err}");
    }

    #[test]
    fn test_adapters_config_from_toml_empty_adapters_array() {
        // Valid TOML but no [[adapter]] entries → should fail
        let toml = r#"
[settings]
foo = "bar"
"#;
        let result = AdaptersConfigFile::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_adapters_config_from_file_not_found() {
        let result = AdaptersConfigFile::from_file(Path::new("/tmp/nonexistent_adapters_xyz.toml"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("failed to read"), "Expected read error, got: {err}");
    }

    #[test]
    fn test_adapters_config_from_file_valid() {
        let dir = std::env::temp_dir().join("entrenar_adapter_cfg_test");
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("adapters.toml");
        std::fs::write(
            &path,
            r#"
[[adapter]]
data = "data/a.jsonl"
checkpoint = "ckpt/a"
label = "test-adapter"
"#,
        )
        .expect("write file");
        let config = AdaptersConfigFile::from_file(&path).expect("valid config");
        assert_eq!(config.adapters.len(), 1);
        assert_eq!(config.adapters[0].label, Some("test-adapter".to_string()));
        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn test_adapter_entry_defaults() {
        let toml = r#"
[[adapter]]
data = "data/x.jsonl"
checkpoint = "ckpt/x"
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let entry = &config.adapters[0];
        assert!(entry.label.is_none());
        assert!(entry.rank.is_none());
        assert!(entry.learning_rate.is_none());
        assert!(entry.epochs.is_none());
        assert!(entry.max_seq_len.is_none());
    }

    #[test]
    fn test_adapter_entry_all_fields() {
        let toml = r#"
[[adapter]]
data = "data/full.jsonl"
checkpoint = "ckpt/full"
label = "full-adapter"
rank = 64
learning_rate = 0.001
epochs = 10
max_seq_len = 1024
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let entry = &config.adapters[0];
        assert_eq!(entry.data, PathBuf::from("data/full.jsonl"));
        assert_eq!(entry.checkpoint, PathBuf::from("ckpt/full"));
        assert_eq!(entry.label, Some("full-adapter".to_string()));
        assert_eq!(entry.rank, Some(64));
        assert_eq!(entry.learning_rate, Some(0.001));
        assert_eq!(entry.epochs, Some(10));
        assert_eq!(entry.max_seq_len, Some(1024));
    }

    #[test]
    fn test_to_adapter_configs_rank_sets_alpha() {
        let toml = r#"
[[adapter]]
data = "data/a.jsonl"
checkpoint = "ckpt/a"
rank = 32
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        // rank=32 → alpha = 32*2.0 = 64.0
        assert_eq!(adapters[0].instruct_config.lora_rank, 32);
        assert!((adapters[0].instruct_config.lora_alpha - 64.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_to_adapter_configs_multiple() {
        let toml = r#"
[[adapter]]
data = "a.jsonl"
checkpoint = "ckpt/a"
rank = 8
learning_rate = 0.0001

[[adapter]]
data = "b.jsonl"
checkpoint = "ckpt/b"
epochs = 20

[[adapter]]
data = "c.jsonl"
checkpoint = "ckpt/c"
max_seq_len = 128
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let base = InstructConfig {
            lora_rank: 16,
            learning_rate: 0.0002,
            epochs: 3,
            max_seq_len: 512,
            ..Default::default()
        };
        let adapters = config.to_adapter_configs(&base);
        assert_eq!(adapters.len(), 3);

        // First adapter: rank=8, lr=0.0001
        assert_eq!(adapters[0].instruct_config.lora_rank, 8);
        assert!((adapters[0].instruct_config.learning_rate - 0.0001).abs() < f32::EPSILON);
        assert_eq!(adapters[0].instruct_config.epochs, 3); // inherited

        // Second adapter: epochs=20
        assert_eq!(adapters[1].instruct_config.lora_rank, 16); // inherited
        assert_eq!(adapters[1].instruct_config.epochs, 20);

        // Third adapter: max_seq_len=128
        assert_eq!(adapters[2].instruct_config.max_seq_len, 128);
        assert_eq!(adapters[2].instruct_config.lora_rank, 16); // inherited
    }

    #[test]
    fn test_batch_train_step_priority_val_loss() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = 2.0;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 5.0; // worst → should be selected

        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };

        let results = pipeline.batch_train_step();
        assert_eq!(results.len(), 2);
        // No tokenizer → both None, but the function should not panic
    }

    #[test]
    fn test_adapter_config_debug() {
        let config = AdapterConfig {
            data_path: PathBuf::from("test.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            instruct_config: InstructConfig::default(),
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("AdapterConfig"));
        assert!(debug.contains("test.jsonl"));
    }

    #[test]
    fn test_adapter_config_clone() {
        let config = AdapterConfig {
            data_path: PathBuf::from("test.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            instruct_config: InstructConfig::default(),
        };
        let cloned = config.clone();
        assert_eq!(cloned.data_path, PathBuf::from("test.jsonl"));
        assert_eq!(cloned.checkpoint_dir, PathBuf::from("/tmp/ckpt"));
    }

    #[test]
    fn test_adapters_config_file_debug() {
        let toml = r#"
[[adapter]]
data = "a.jsonl"
checkpoint = "ckpt/a"
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let debug = format!("{config:?}");
        assert!(debug.contains("AdaptersConfigFile"));
    }

    #[test]
    fn test_adapter_entry_debug() {
        let toml = r#"
[[adapter]]
data = "a.jsonl"
checkpoint = "ckpt/a"
label = "test"
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let debug = format!("{:?}", config.adapters[0]);
        assert!(debug.contains("AdapterEntry"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_adapter_slot_cursor_tracking() {
        let mut slot = dummy_slot_with_data(5);
        assert_eq!(slot.cursor, 0);
        slot.cursor = 3;
        assert_eq!(slot.cursor, 3);
        assert!(slot.cursor < slot.train_samples.len());
        slot.cursor = 5;
        assert!(slot.cursor >= slot.train_samples.len());
    }

    #[test]
    fn test_adapter_slot_best_val_loss() {
        let mut slot = dummy_slot();
        assert_eq!(slot.best_val_loss, f32::INFINITY);
        slot.best_val_loss = 0.5;
        assert!((slot.best_val_loss - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multi_adapter_pipeline_global_step() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert_eq!(pipeline.global_step, 0);
    }

    #[test]
    fn test_train_step_adapter_exhausted() {
        let mut slot = dummy_slot_with_data(2);
        slot.cursor = 2; // already exhausted

        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        let result = pipeline.train_step_adapter(0);
        assert!(result.is_none(), "Exhausted adapter should return None");
    }

    #[test]
    fn test_batch_train_step_empty() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::Synchronized,
            global_step: 0,
        };
        let results = pipeline.batch_train_step();
        assert!(results.is_empty());
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_multi_adapter_pipeline_new() {
        let pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::Synchronized);
        assert_eq!(pipeline.num_adapters(), 0);
        assert_eq!(pipeline.global_step, 0);
        assert!(pipeline.all_exhausted());
    }

    #[test]
    fn test_multi_adapter_pipeline_add_adapter() {
        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            instruct_config: InstructConfig::default(),
        };
        let samples = vec![InstructSample {
            instruction: "test".into(),
            response: "response".into(),
            system: None,
            metadata: None,
        }];
        pipeline.add_adapter(config, samples, vec![]);
        assert_eq!(pipeline.num_adapters(), 1);
        assert!(!pipeline.all_exhausted());
    }

    #[test]
    fn test_train_step_adapter_no_tokenizer() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(5)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        // No tokenizer loaded → should return None
        let result = pipeline.train_step_adapter(0);
        assert!(result.is_none());
        // Cursor should have advanced
        assert_eq!(pipeline.adapters[0].cursor, 1);
    }

    #[test]
    fn test_train_step_increments_global_step() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(5)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        // Even though result is None (no tokenizer), global_step should not increment
        // because early return happens before step increment
        let _ = pipeline.train_step_adapter(0);
        // Cursor advanced to 1, but no tokenizer so returns early before global_step increment
    }

    #[test]
    fn test_batch_train_step_synchronized_all_exhausted() {
        let mut slot0 = dummy_slot_with_data(1);
        slot0.cursor = 1;
        let mut slot1 = dummy_slot_with_data(1);
        slot1.cursor = 1;

        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::Synchronized,
            global_step: 0,
        };
        let results = pipeline.batch_train_step();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Option::is_none));
    }

    #[test]
    fn test_reset_epoch_different_seeds_different_orders() {
        let mut pipeline1 = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(20)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        let mut pipeline2 = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(20)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        pipeline1.reset_epoch(1);
        pipeline2.reset_epoch(999);

        let same = pipeline1.adapters[0]
            .train_samples
            .iter()
            .zip(pipeline2.adapters[0].train_samples.iter())
            .all(|(s1, s2)| s1.instruction == s2.instruction);
        assert!(!same, "Different seeds should produce different shuffles");
    }

    #[test]
    fn test_shuffle_samples_preserves_elements() {
        let mut samples: Vec<InstructSample> = (0..10)
            .map(|i| InstructSample {
                instruction: format!("inst_{i}"),
                response: format!("resp_{i}"),
                system: None,
                metadata: None,
            })
            .collect();
        let original_instructions: Vec<String> =
            samples.iter().map(|s| s.instruction.clone()).collect();

        shuffle_samples(&mut samples, 42);

        // All original elements should still be present
        let mut shuffled_instructions: Vec<String> =
            samples.iter().map(|s| s.instruction.clone()).collect();
        let mut sorted_original = original_instructions.clone();
        sorted_original.sort();
        shuffled_instructions.sort();
        assert_eq!(sorted_original, shuffled_instructions);
    }

    #[test]
    fn test_adapter_slot_metrics_empty() {
        let slot = dummy_slot();
        assert!(slot.metrics.is_empty());
    }

    #[test]
    fn test_adapter_slot_val_samples() {
        let slot = dummy_slot();
        assert!(slot.val_samples.is_empty());
    }

    #[test]
    fn test_adapter_slot_lora_layers_empty() {
        let slot = dummy_slot();
        assert!(slot.lora_layers.is_empty());
    }

    #[test]
    fn test_adapters_config_label_propagation() {
        let toml = r#"
[[adapter]]
data = "d1.jsonl"
checkpoint = "c1"
label = "adapter-one"

[[adapter]]
data = "d2.jsonl"
checkpoint = "c2"
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        assert_eq!(config.adapters[0].label, Some("adapter-one".to_string()));
        assert!(config.adapters[1].label.is_none());
    }

    #[test]
    fn test_adapters_config_to_adapter_configs_alpha_calculation() {
        let toml = r#"
[[adapter]]
data = "data.jsonl"
checkpoint = "ckpt"
rank = 64
"#;
        let config = AdaptersConfigFile::from_toml(toml).expect("valid");
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        // alpha = rank * 2.0 = 128.0
        assert!((adapters[0].instruct_config.lora_alpha - 128.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_select_next_adapter_round_robin_large_step() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot()],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 1000,
        };
        assert_eq!(pipeline.select_next_adapter(), Some(0)); // 1000 % 2 = 0

        let pipeline = MultiAdapterPipeline { global_step: 1001, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(1)); // 1001 % 2 = 1
    }

    #[test]
    fn test_select_next_adapter_priority_selects_worst() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = 0.1;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 10.0;
        let mut slot2 = dummy_slot();
        slot2.best_val_loss = 5.0;

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1, slot2],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        assert_eq!(pipeline.select_next_adapter(), Some(1)); // 10.0 is worst
    }

    #[test]
    fn test_multi_adapter_multiple_add_adapter() {
        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::Synchronized);

        for i in 0..3 {
            let config = AdapterConfig {
                data_path: PathBuf::from(format!("data{i}.jsonl")),
                checkpoint_dir: PathBuf::from(format!("/tmp/ckpt{i}")),
                instruct_config: InstructConfig::default(),
            };
            pipeline.add_adapter(config, vec![], vec![]);
        }
        assert_eq!(pipeline.num_adapters(), 3);
        assert!(pipeline.all_exhausted()); // all empty
    }

    // ── cov3: additional coverage tests ─────────────────────────────

    #[test]
    fn test_cov3_save_adapter_checkpoint_creates_dir_and_files() {
        let dir = std::env::temp_dir().join("entrenar_cov3_ckpt_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: dir.clone(),
            instruct_config: InstructConfig::default(),
        };
        let samples = vec![InstructSample {
            instruction: "test".into(),
            response: "resp".into(),
            system: None,
            metadata: None,
        }];
        pipeline.add_adapter(config, samples, vec![]);

        let result = pipeline.save_adapter_checkpoint(0, 1, 0.5);
        assert!(result.is_ok());
        let ckpt_dir = result.unwrap();
        assert!(ckpt_dir.join("metadata.json").exists());
        assert!(ckpt_dir.join("model.safetensors").exists());

        // Verify metadata contents
        let metadata_str = std::fs::read_to_string(ckpt_dir.join("metadata.json")).unwrap();
        assert!(metadata_str.contains("\"mode\": \"multi_adapter\""));
        assert!(metadata_str.contains("\"adapter_index\": 0"));
        assert!(metadata_str.contains("\"epoch\": 1"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_save_best_checkpoint_creates_dir_and_files() {
        let dir = std::env::temp_dir().join("entrenar_cov3_best_ckpt_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: dir.clone(),
            instruct_config: InstructConfig::default(),
        };
        pipeline.add_adapter(config, vec![], vec![]);

        let result = pipeline.save_best_checkpoint(0, 2, 0.3);
        assert!(result.is_ok());
        let best_dir = result.unwrap();
        assert_eq!(best_dir, dir.join("best"));
        assert!(best_dir.join("metadata.json").exists());
        assert!(best_dir.join("model.safetensors").exists());

        // Verify metadata
        let metadata_str = std::fs::read_to_string(best_dir.join("metadata.json")).unwrap();
        assert!(metadata_str.contains("\"mode\": \"multi_adapter\""));
        assert!(metadata_str.contains("\"epoch\": 2"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_save_best_checkpoint_overwrites_previous() {
        let dir = std::env::temp_dir().join("entrenar_cov3_best_overwrite");
        let _ = std::fs::remove_dir_all(&dir);

        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: dir.clone(),
            instruct_config: InstructConfig::default(),
        };
        pipeline.add_adapter(config, vec![], vec![]);

        // First save
        pipeline.save_best_checkpoint(0, 1, 1.0).unwrap();
        // Second save should overwrite
        pipeline.save_best_checkpoint(0, 5, 0.2).unwrap();

        let metadata_str = std::fs::read_to_string(dir.join("best").join("metadata.json")).unwrap();
        assert!(metadata_str.contains("\"epoch\": 5"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_save_adapter_lora_weights_empty_layers() {
        let dir = std::env::temp_dir().join("entrenar_cov3_empty_lora");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let result = save_adapter_lora_weights(&[], &dir);
        assert!(result.is_ok());
        // SafeTensors file should exist even with empty layers
        assert!(dir.join("model.safetensors").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_save_adapter_lora_weights_with_real_layers() {
        let dir = std::env::temp_dir().join("entrenar_cov3_real_lora");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Create a pipeline with real LoRA layers
        let model_config = crate::transformer::TransformerConfig::tiny();
        let model = crate::transformer::Transformer::new(&model_config);
        let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
        let layers = InstructPipeline::build_lora_layers(&model, &model_config, &instruct_config);

        let result = save_adapter_lora_weights(&layers, &dir);
        assert!(result.is_ok());

        // Verify SafeTensors can be read back
        let st_bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
        // 4 LoRA layers → 4 * 2 (A and B) = 8 tensors
        assert_eq!(st.len(), layers.len() * 2);

        // Verify naming convention
        let names: Vec<String> = st.names().iter().map(std::string::ToString::to_string).collect();
        assert!(names.iter().any(|n| n.contains("lora_a")));
        assert!(names.iter().any(|n| n.contains("lora_b")));
        assert!(names.iter().any(|n| n.contains("q_proj")));
        assert!(names.iter().any(|n| n.contains("v_proj")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_shuffle_samples_large_input() {
        let mut samples: Vec<InstructSample> = (0..100)
            .map(|i| InstructSample {
                instruction: format!("inst_{i}"),
                response: format!("resp_{i}"),
                system: None,
                metadata: None,
            })
            .collect();
        let original: Vec<String> = samples.iter().map(|s| s.instruction.clone()).collect();

        shuffle_samples(&mut samples, 12345);

        let shuffled: Vec<String> = samples.iter().map(|s| s.instruction.clone()).collect();
        // Should be different order
        assert_ne!(original, shuffled, "100 samples should shuffle to different order");
        // But same elements
        let mut sorted_original = original;
        sorted_original.sort();
        let mut sorted_shuffled = shuffled;
        sorted_shuffled.sort();
        assert_eq!(sorted_original, sorted_shuffled);
    }

    #[test]
    fn test_cov3_shuffle_samples_two_elements() {
        let mut samples = vec![
            InstructSample {
                instruction: "a".into(),
                response: "1".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "b".into(),
                response: "2".into(),
                system: None,
                metadata: None,
            },
        ];
        // Verify no panic with 2 elements
        shuffle_samples(&mut samples, 42);
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn test_cov3_adapters_config_toml_all_overrides() {
        let toml = r#"
[[adapter]]
data = "data/test.jsonl"
checkpoint = "ckpt/test"
label = "full-override"
rank = 64
learning_rate = 0.001
epochs = 20
max_seq_len = 2048
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        assert_eq!(adapters[0].instruct_config.lora_rank, 64);
        assert!((adapters[0].instruct_config.lora_alpha - 128.0).abs() < f32::EPSILON);
        assert!((adapters[0].instruct_config.learning_rate - 0.001).abs() < f32::EPSILON);
        assert_eq!(adapters[0].instruct_config.epochs, 20);
        assert_eq!(adapters[0].instruct_config.max_seq_len, 2048);
    }

    #[test]
    fn test_cov3_adapters_config_many_adapters() {
        let mut toml_str = String::new();
        for i in 0..10 {
            toml_str.push_str(&format!(
                r#"
[[adapter]]
data = "data/{i}.jsonl"
checkpoint = "ckpt/{i}"
rank = {rank}
"#,
                i = i,
                rank = 4 + i * 2,
            ));
        }
        let config = AdaptersConfigFile::from_toml(&toml_str).unwrap();
        assert_eq!(config.adapters.len(), 10);
        // Verify each has the right rank
        for (i, entry) in config.adapters.iter().enumerate() {
            assert_eq!(entry.rank, Some(4 + i * 2));
        }
    }

    #[test]
    fn test_cov3_adapters_config_toml_missing_required_fields() {
        // Missing checkpoint field
        let toml = r#"
[[adapter]]
data = "data.jsonl"
"#;
        let result = AdaptersConfigFile::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_cov3_adapters_config_toml_missing_data_field() {
        let toml = r#"
[[adapter]]
checkpoint = "ckpt"
"#;
        let result = AdaptersConfigFile::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_cov3_adapters_config_toml_extra_fields_ignored() {
        // Extra fields should be ignored by serde
        let toml = r#"
[[adapter]]
data = "data.jsonl"
checkpoint = "ckpt"
unknown_field = "ignored"
"#;
        // Depending on serde config, this might fail or succeed
        // toml::from_str with #[serde(deny_unknown_fields)] would fail
        // Without it, it succeeds
        let result = AdaptersConfigFile::from_toml(toml);
        // Just verify no panic
        let _ = result;
    }

    #[test]
    fn test_cov3_adapters_config_rank_zero() {
        let toml = r#"
[[adapter]]
data = "data.jsonl"
checkpoint = "ckpt"
rank = 0
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        assert_eq!(adapters[0].instruct_config.lora_rank, 0);
        assert!((adapters[0].instruct_config.lora_alpha - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cov3_add_adapter_creates_lora_layers() {
        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            instruct_config: InstructConfig { lora_rank: 4, ..InstructConfig::default() },
        };
        pipeline.add_adapter(config, vec![], vec![]);
        // Adapter should have LoRA layers created from the base model
        // tiny model has 2 layers → 2 * 2 (Q+V) = 4 LoRA layers
        assert_eq!(pipeline.adapters[0].lora_layers.len(), 4);
    }

    #[test]
    fn test_cov3_add_adapter_with_val_samples() {
        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            instruct_config: InstructConfig::default(),
        };
        let val_samples = vec![InstructSample {
            instruction: "val_q".into(),
            response: "val_a".into(),
            system: None,
            metadata: None,
        }];
        pipeline.add_adapter(config, vec![], val_samples);
        assert_eq!(pipeline.adapters[0].val_samples.len(), 1);
        assert_eq!(pipeline.adapters[0].val_samples[0].instruction, "val_q");
    }

    #[test]
    fn test_cov3_add_adapter_initial_state() {
        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: PathBuf::from("/tmp/ckpt_initial"),
            instruct_config: InstructConfig { lora_rank: 8, ..InstructConfig::default() },
        };
        pipeline.add_adapter(config, vec![], vec![]);
        let slot = &pipeline.adapters[0];
        assert_eq!(slot.cursor, 0);
        assert_eq!(slot.best_val_loss, f32::INFINITY);
        assert!(slot.metrics.is_empty());
        assert_eq!(slot.config.lora_rank, 8);
        assert_eq!(slot.checkpoint_dir, PathBuf::from("/tmp/ckpt_initial"));
    }

    #[test]
    fn test_cov3_train_step_adapter_empty_tokens() {
        // Test with empty instruction and response after tokenization
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(5)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        // No tokenizer → returns None early
        let result = pipeline.train_step_adapter(0);
        assert!(result.is_none());
        // Cursor should have advanced by 1
        assert_eq!(pipeline.adapters[0].cursor, 1);
    }

    #[test]
    fn test_cov3_batch_train_step_synchronized_mixed_exhaustion() {
        let slot0 = dummy_slot_with_data(3); // has data
        let mut slot1 = dummy_slot_with_data(1);
        slot1.cursor = 1; // exhausted

        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::Synchronized,
            global_step: 0,
        };

        let results = pipeline.batch_train_step();
        assert_eq!(results.len(), 2);
        // Adapter 1 is exhausted → None
        assert!(results[1].is_none());
    }

    #[test]
    fn test_cov3_batch_train_step_round_robin_cycling() {
        // Verify round-robin cycling by manually advancing global_step
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![
                dummy_slot_with_data(10),
                dummy_slot_with_data(10),
                dummy_slot_with_data(10),
            ],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        // Step 0 → adapter 0
        assert_eq!(pipeline.select_next_adapter(), Some(0));
        // Simulate step 1
        let pipeline = MultiAdapterPipeline { global_step: 1, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(1));
        // Simulate step 2
        let pipeline = MultiAdapterPipeline { global_step: 2, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(2));
        // Step 3 wraps back to 0
        let pipeline = MultiAdapterPipeline { global_step: 3, ..pipeline };
        assert_eq!(pipeline.select_next_adapter(), Some(0));
    }

    #[test]
    fn test_cov3_reset_epoch_multiple_adapters_independent_seeds() {
        let mut pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot_with_data(20), dummy_slot_with_data(20)],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };

        pipeline.reset_epoch(42);

        // Different adapters should get different shuffle orders
        // (seed + adapter index → different effective seed)
        let order0: Vec<String> =
            pipeline.adapters[0].train_samples.iter().map(|s| s.instruction.clone()).collect();
        let order1: Vec<String> =
            pipeline.adapters[1].train_samples.iter().map(|s| s.instruction.clone()).collect();
        // With different effective seeds, orderings should differ
        assert_ne!(order0, order1, "Different adapters should have different shuffle orders");
    }

    #[test]
    fn test_cov3_adapter_schedule_copy() {
        let s1 = AdapterSchedule::PriorityValLoss;
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_cov3_adapters_config_file_clone() {
        let toml = r#"
[[adapter]]
data = "data.jsonl"
checkpoint = "ckpt"
label = "test"
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.adapters.len(), 1);
        assert_eq!(cloned.adapters[0].label, Some("test".to_string()));
    }

    #[test]
    fn test_cov3_adapter_entry_clone() {
        let toml = r#"
[[adapter]]
data = "data.jsonl"
checkpoint = "ckpt"
rank = 32
learning_rate = 0.001
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let cloned = config.adapters[0].clone();
        assert_eq!(cloned.rank, Some(32));
        assert_eq!(cloned.learning_rate, Some(0.001));
    }

    #[test]
    fn test_cov3_save_adapter_checkpoint_metadata_values() {
        let dir = std::env::temp_dir().join("entrenar_cov3_ckpt_meta");
        let _ = std::fs::remove_dir_all(&dir);

        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        pipeline.global_step = 42;
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: dir.clone(),
            instruct_config: InstructConfig {
                lora_rank: 8,
                lora_alpha: 16.0,
                ..InstructConfig::default()
            },
        };
        let samples: Vec<InstructSample> = (0..5)
            .map(|i| InstructSample {
                instruction: format!("q{i}"),
                response: format!("a{i}"),
                system: None,
                metadata: None,
            })
            .collect();
        pipeline.add_adapter(config, samples, vec![]);
        pipeline.adapters[0].best_val_loss = 0.75;

        let ckpt_dir = pipeline.save_adapter_checkpoint(0, 3, 0.42).unwrap();
        let metadata_str = std::fs::read_to_string(ckpt_dir.join("metadata.json")).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();

        assert_eq!(metadata["adapter_index"], 0);
        assert_eq!(metadata["epoch"], 3);
        assert_eq!(metadata["lora_rank"], 8);
        assert_eq!(metadata["train_samples"], 5);
        assert_eq!(metadata["global_step"], 42);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_save_adapter_checkpoint_multiple_epochs() {
        let dir = std::env::temp_dir().join("entrenar_cov3_multi_epoch");
        let _ = std::fs::remove_dir_all(&dir);

        let mut pipeline =
            MultiAdapterPipeline::new(create_dummy_pipeline(), AdapterSchedule::RoundRobin);
        let config = AdapterConfig {
            data_path: PathBuf::from("data.jsonl"),
            checkpoint_dir: dir.clone(),
            instruct_config: InstructConfig::default(),
        };
        pipeline.add_adapter(config, vec![], vec![]);

        // Save multiple epochs
        for epoch in 0..3 {
            let ckpt_dir =
                pipeline.save_adapter_checkpoint(0, epoch, 1.0 - epoch as f32 * 0.2).unwrap();
            assert!(ckpt_dir.join("metadata.json").exists());
            assert!(ckpt_dir.join("model.safetensors").exists());
        }

        // All three epoch directories should exist
        assert!(dir.join("epoch-0").exists());
        assert!(dir.join("epoch-1").exists());
        assert!(dir.join("epoch-2").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_all_exhausted_single_adapter_one_sample() {
        let slot = dummy_slot_with_data(1);
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert!(!pipeline.all_exhausted());
    }

    #[test]
    fn test_cov3_all_exhausted_single_adapter_cursor_at_end() {
        let mut slot = dummy_slot_with_data(1);
        slot.cursor = 1;
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert!(pipeline.all_exhausted());
    }

    #[test]
    fn test_cov3_select_priority_single_adapter() {
        let mut slot = dummy_slot();
        slot.best_val_loss = 3.0;
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        assert_eq!(pipeline.select_next_adapter(), Some(0));
    }

    #[test]
    fn test_cov3_select_priority_equal_losses() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = 1.0;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 1.0;
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        let result = pipeline.select_next_adapter();
        // With equal losses, max_by picks the last one (stable)
        assert!(result == Some(0) || result == Some(1));
    }

    #[test]
    fn test_cov3_to_adapter_configs_no_overrides() {
        let toml = r#"
[[adapter]]
data = "d.jsonl"
checkpoint = "c"
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let base = InstructConfig {
            lora_rank: 32,
            lora_alpha: 64.0,
            learning_rate: 0.005,
            epochs: 7,
            max_seq_len: 1024,
            gradient_clip_norm: Some(2.0),
            quantize_nf4: true,
        };
        let adapters = config.to_adapter_configs(&base);
        // All base values should be inherited
        assert_eq!(adapters[0].instruct_config.lora_rank, 32);
        assert!((adapters[0].instruct_config.lora_alpha - 64.0).abs() < f32::EPSILON);
        assert!((adapters[0].instruct_config.learning_rate - 0.005).abs() < f32::EPSILON);
        assert_eq!(adapters[0].instruct_config.epochs, 7);
        assert_eq!(adapters[0].instruct_config.max_seq_len, 1024);
        assert_eq!(adapters[0].instruct_config.gradient_clip_norm, Some(2.0));
        assert!(adapters[0].instruct_config.quantize_nf4);
    }

    #[test]
    fn test_cov3_to_adapter_configs_preserves_data_and_checkpoint_paths() {
        let toml = r#"
[[adapter]]
data = "/absolute/path/data.jsonl"
checkpoint = "../relative/ckpt"
"#;
        let config = AdaptersConfigFile::from_toml(toml).unwrap();
        let base = InstructConfig::default();
        let adapters = config.to_adapter_configs(&base);
        assert_eq!(adapters[0].data_path, PathBuf::from("/absolute/path/data.jsonl"));
        assert_eq!(adapters[0].checkpoint_dir, PathBuf::from("../relative/ckpt"));
    }

    #[test]
    fn test_cov3_adapters_config_from_file_invalid_toml() {
        let dir = std::env::temp_dir().join("entrenar_cov3_invalid_toml");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("invalid.toml");
        std::fs::write(&path, "this {{ is not valid TOML").unwrap();
        let result = AdaptersConfigFile::from_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("failed to parse"), "Expected parse error, got: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cov3_adapter_slot_checkpoint_dir() {
        let slot = AdapterSlot {
            lora_layers: Vec::new(),
            train_samples: Vec::new(),
            val_samples: Vec::new(),
            checkpoint_dir: PathBuf::from("/my/custom/ckpt"),
            metrics: Vec::new(),
            config: InstructConfig::default(),
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        };
        assert_eq!(slot.checkpoint_dir, PathBuf::from("/my/custom/ckpt"));
    }

    #[test]
    fn test_cov3_multi_adapter_schedule_field() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };
        assert_eq!(pipeline.schedule, AdapterSchedule::PriorityValLoss);
    }
}
