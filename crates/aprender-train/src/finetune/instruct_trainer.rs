//! Production training loop for instruction fine-tuning (GH-371)
//!
//! `InstructTrainer` wraps `InstructPipeline` with epoch management,
//! validation, checkpointing, LR scheduling, and early stopping.
//!
//! # Contract Invariants
//!
//! - F-INST-003: Perplexity reported per epoch
//! - F-LOOP-002: Validation computed every epoch
//! - F-LOOP-007: Data shuffled per epoch
//! - F-LOOP-008: Val split disjoint
//! - F-LOOP-010: Early stopping respects patience

use super::instruct_corpus::{format_chat_prompt, InstructSample};
use super::instruct_pipeline::InstructPipeline;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Training configuration for instruction trainer.
#[derive(Debug, Clone)]
pub struct InstructTrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Fraction of data reserved for validation (0.0, 0.5]
    pub val_split: f32,
    /// Save checkpoint every N epochs
    pub save_every: usize,
    /// Early stopping patience in epochs
    pub early_stopping_patience: usize,
    /// Directory for checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Log metrics every N epochs
    pub log_interval: usize,
    /// Warmup steps as fraction of total steps
    pub warmup_fraction: f32,
    /// Minimum learning rate for cosine decay
    pub lr_min: f32,
}

impl Default for InstructTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            val_split: 0.2,
            save_every: 1,
            early_stopping_patience: 5,
            checkpoint_dir: PathBuf::from("checkpoints"),
            seed: 42,
            log_interval: 1,
            warmup_fraction: 0.1,
            lr_min: 1e-6,
        }
    }
}

/// Metrics for a single training epoch.
#[derive(Debug, Clone)]
pub struct InstructEpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,
    /// Average training loss (response tokens only)
    pub train_loss: f32,
    /// Training perplexity
    pub train_perplexity: f32,
    /// Average validation loss
    pub val_loss: f32,
    /// Validation perplexity
    pub val_perplexity: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Epoch wall-clock time in milliseconds
    pub epoch_time_ms: u64,
    /// Training throughput (samples/second)
    pub samples_per_sec: f32,
}

/// Result of the full training run.
#[derive(Debug, Clone)]
pub struct InstructTrainResult {
    /// Per-epoch metrics
    pub epoch_metrics: Vec<InstructEpochMetrics>,
    /// Epoch with lowest validation loss
    pub best_epoch: usize,
    /// Lowest validation loss achieved
    pub best_val_loss: f32,
    /// Whether training stopped early
    pub stopped_early: bool,
    /// Total wall-clock training time in milliseconds
    pub total_time_ms: u64,
}

/// Prepared token sequences for training.
struct PreparedSample {
    prompt_ids: Vec<u32>,
    response_ids: Vec<u32>,
}

/// Production training loop for instruction fine-tuning.
pub struct InstructTrainer {
    /// The instruction pipeline (model + optimizer)
    pipeline: InstructPipeline,
    /// Training configuration
    config: InstructTrainingConfig,
    /// Training data (shuffled per epoch)
    train_data: Vec<InstructSample>,
    /// Validation data (frozen, never shuffled)
    val_data: Vec<InstructSample>,
    /// Base random seed
    rng_seed: u64,
    /// SHA-256 hash of training data for provenance
    data_hash: String,
}

impl InstructTrainer {
    /// Create a new trainer by splitting corpus into train/val sets.
    ///
    /// # Errors
    /// Returns error if corpus is empty, val_split is out of range, or epochs is 0.
    pub fn new(
        pipeline: InstructPipeline,
        corpus: Vec<InstructSample>,
        config: InstructTrainingConfig,
    ) -> crate::Result<Self> {
        if corpus.is_empty() {
            return Err(crate::Error::ConfigError("GH-371: corpus must not be empty".to_string()));
        }
        if config.val_split <= 0.0 || config.val_split > 0.5 {
            return Err(crate::Error::ConfigError(format!(
                "GH-371: val_split must be in (0.0, 0.5], got {}",
                config.val_split,
            )));
        }
        if config.epochs == 0 {
            return Err(crate::Error::ConfigError("GH-371: epochs must be > 0".to_string()));
        }

        let (train_data, val_data) = Self::split_dataset(&corpus, config.val_split, config.seed);

        if train_data.is_empty() || val_data.is_empty() {
            return Err(crate::Error::ConfigError(format!(
                "GH-371: split produced empty set (train={}, val={}). Need more samples.",
                train_data.len(),
                val_data.len(),
            )));
        }

        let rng_seed = config.seed;
        let data_hash = Self::compute_data_hash(&corpus);

        Ok(Self { pipeline, config, train_data, val_data, rng_seed, data_hash })
    }

    /// Run the full training loop.
    pub fn train(&mut self) -> InstructTrainResult {
        use crate::optim::{LRScheduler, WarmupCosineDecayLR};

        let total_start = std::time::Instant::now();
        let base_lr = self.pipeline.learning_rate();
        let total_steps = self.config.epochs * self.train_data.len();
        let warmup_steps = (total_steps as f32 * self.config.warmup_fraction) as usize;

        let mut scheduler =
            WarmupCosineDecayLR::new(base_lr, self.config.lr_min, warmup_steps, total_steps);

        let mut epoch_metrics = Vec::new();
        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch = 0usize;
        let mut patience_counter = 0usize;
        let mut stopped_early = false;

        // Pre-tokenize validation data (frozen across epochs)
        // KAIZEN-046: Removed unnecessary .clone() — both borrows are immutable.
        let val_prepared = self.prepare_samples(&self.val_data);

        // KAIZEN-046: Pre-collect validation token IDs once (was: re-cloned every epoch).
        let val_prompts: Vec<Vec<u32>> =
            val_prepared.iter().map(|s| s.prompt_ids.clone()).collect();
        let val_responses: Vec<Vec<u32>> =
            val_prepared.iter().map(|s| s.response_ids.clone()).collect();

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // Shuffle training data
            self.shuffle_train(epoch as u64);

            // Pre-tokenize training data for this epoch (after shuffle)
            // KAIZEN-046: Removed unnecessary .clone() — both borrows are immutable.
            let train_prepared = self.prepare_samples(&self.train_data);

            // ── Train ──
            let mut epoch_loss = 0.0f32;
            let mut epoch_tokens = 0usize;

            for sample in &train_prepared {
                let lr = scheduler.get_lr();
                self.pipeline.set_learning_rate(lr);

                let result = self.pipeline.train_step(&sample.prompt_ids, &sample.response_ids);
                epoch_loss += result.loss * result.num_response_tokens as f32;
                epoch_tokens += result.num_response_tokens;
                scheduler.step();
            }

            let train_loss = if epoch_tokens > 0 { epoch_loss / epoch_tokens as f32 } else { 0.0 };

            // PMAT-512: Emit per-epoch loss to stderr so canary parser can extract it.
            // Format matches WGPU path (finetune.rs:678) for parser compatibility.
            eprintln!(
                "  Epoch {} complete: avg_loss={:.4} tokens={} samples={} lr={:.2e}",
                epoch + 1,
                train_loss,
                epoch_tokens,
                train_prepared.len(),
                self.pipeline.learning_rate(),
            );

            // ── Validate ──
            // KAIZEN-046: val_prompts/val_responses hoisted outside epoch loop.
            let val_result = self.pipeline.evaluate(&val_prompts, &val_responses);

            let epoch_time_ms = epoch_start.elapsed().as_millis() as u64;
            let samples_per_sec = if epoch_time_ms > 0 {
                train_prepared.len() as f32 / (epoch_time_ms as f32 / 1000.0)
            } else {
                0.0
            };

            let metrics = InstructEpochMetrics {
                epoch,
                train_loss,
                train_perplexity: train_loss.exp().min(1e6),
                val_loss: val_result.avg_loss,
                val_perplexity: val_result.perplexity,
                learning_rate: self.pipeline.learning_rate(),
                epoch_time_ms,
                samples_per_sec,
            };

            // ── Checkpointing ──
            if val_result.avg_loss < best_val_loss {
                best_val_loss = val_result.avg_loss;
                best_epoch = epoch;
                patience_counter = 0;

                // Save best checkpoint
                let best_path = self.config.checkpoint_dir.join("best");
                let _ = self.save_checkpoint(&best_path, epoch, &metrics);
            } else {
                patience_counter += 1;
            }

            // Periodic checkpoint
            let effective_save_every = if self.config.epochs <= self.config.save_every {
                1
            } else {
                self.config.save_every
            };
            if effective_save_every > 0 && (epoch + 1) % effective_save_every == 0 {
                let epoch_path = self.config.checkpoint_dir.join(format!("epoch-{epoch}"));
                let _ = self.save_checkpoint(&epoch_path, epoch, &metrics);
            }

            epoch_metrics.push(metrics);

            // ── Early stopping ──
            if patience_counter >= self.config.early_stopping_patience {
                stopped_early = true;
                break;
            }
        }

        // PMAT-512: Print final training summary to stderr for canary parser.
        if let Some(last) = epoch_metrics.last() {
            eprintln!(
                "[training] Training complete: final_loss={:.4} best_val_loss={:.4} best_epoch={} epochs={} time={}s{}",
                last.train_loss,
                best_val_loss,
                best_epoch + 1,
                epoch_metrics.len(),
                total_start.elapsed().as_secs(),
                if stopped_early { " (early stopped)" } else { "" },
            );
        }

        // PMAT-483: Print profiler report at end of training (text + JSON)
        if self.pipeline.profiler.is_enabled() {
            self.pipeline.profiler.print_report();
            self.pipeline.profiler.print_json_report();
        }

        InstructTrainResult {
            epoch_metrics,
            best_epoch,
            best_val_loss,
            stopped_early,
            total_time_ms: total_start.elapsed().as_millis() as u64,
        }
    }

    /// Prepare samples by tokenizing prompt and response.
    fn prepare_samples(&self, samples: &[InstructSample]) -> Vec<PreparedSample> {
        samples
            .iter()
            .map(|sample| {
                let (prompt_text, response_text) = format_chat_prompt(sample);
                PreparedSample {
                    prompt_ids: self.pipeline.tokenize(&prompt_text),
                    response_ids: self.pipeline.tokenize(&response_text),
                }
            })
            .collect()
    }

    /// Split dataset into train/val with deterministic shuffling.
    fn split_dataset(
        corpus: &[InstructSample],
        val_split: f32,
        seed: u64,
    ) -> (Vec<InstructSample>, Vec<InstructSample>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut indices: Vec<usize> = (0..corpus.len()).collect();

        // Fisher-Yates shuffle with deterministic seed
        for i in (1..indices.len()).rev() {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let j = (hasher.finish() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let val_size = (corpus.len() as f32 * val_split).ceil() as usize;
        let val_size = val_size.max(1).min(corpus.len() - 1);

        let val_data: Vec<InstructSample> =
            indices[..val_size].iter().map(|&i| corpus[i].clone()).collect();
        let train_data: Vec<InstructSample> =
            indices[val_size..].iter().map(|&i| corpus[i].clone()).collect();

        (train_data, val_data)
    }

    /// Shuffle training data with epoch-specific seed.
    fn shuffle_train(&mut self, epoch: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = self.train_data.len();
        for i in (1..n).rev() {
            let mut hasher = DefaultHasher::new();
            self.rng_seed.hash(&mut hasher);
            epoch.hash(&mut hasher);
            i.hash(&mut hasher);
            let j = (hasher.finish() as usize) % (i + 1);
            self.train_data.swap(i, j);
        }
    }

    /// Compute SHA-256 hash of corpus for provenance.
    fn compute_data_hash(corpus: &[InstructSample]) -> String {
        let mut hasher = Sha256::new();
        for s in corpus {
            hasher.update(s.instruction.as_bytes());
            hasher.update([0u8]);
            hasher.update(s.response.as_bytes());
            hasher.update([0u8]);
        }
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Get data hash for provenance tracking.
    #[must_use]
    pub fn data_hash(&self) -> &str {
        &self.data_hash
    }

    /// Get training data size.
    #[must_use]
    pub fn train_size(&self) -> usize {
        self.train_data.len()
    }

    /// Get validation data size.
    #[must_use]
    pub fn val_size(&self) -> usize {
        self.val_data.len()
    }

    /// Save a checkpoint with LoRA adapter weights and training metadata.
    ///
    /// Creates a directory at `path` containing:
    /// - `metadata.json`: training metrics for this checkpoint
    /// - `model.safetensors`: LoRA adapter weights (Q/V projections per layer)
    pub fn save_checkpoint(
        &mut self,
        path: &std::path::Path,
        epoch: usize,
        metrics: &InstructEpochMetrics,
    ) -> crate::Result<()> {
        contract_pre_save_checkpoint!();
        // Sync GPU LoRA weights to CPU before saving
        #[cfg(feature = "cuda")]
        self.pipeline.sync_lora_to_cpu();

        std::fs::create_dir_all(path).map_err(|e| {
            crate::Error::Io(format!("Failed to create checkpoint dir {}: {e}", path.display()))
        })?;

        // Save metadata.json
        let metadata = serde_json::json!({
            "task": "instruct",
            "epoch": epoch,
            "train_loss": metrics.train_loss,
            "val_loss": metrics.val_loss,
            "train_perplexity": metrics.train_perplexity,
            "val_perplexity": metrics.val_perplexity,
            "learning_rate": metrics.learning_rate,
            "epoch_time_ms": metrics.epoch_time_ms,
            "samples_per_sec": metrics.samples_per_sec,
            "lora_rank": self.pipeline.config.lora_rank,
            "lora_alpha": self.pipeline.config.lora_alpha,
            "data_hash": self.data_hash,
        });

        let meta_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize metadata: {e}"))
        })?;
        std::fs::write(path.join("metadata.json"), meta_json)?;

        // Save LoRA adapter weights as SafeTensors
        let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

        for (idx, lora) in self.pipeline.lora_layers.iter().enumerate() {
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

        let mut st_metadata = std::collections::HashMap::new();
        st_metadata.insert("epoch".to_string(), epoch.to_string());
        st_metadata.insert("val_loss".to_string(), format!("{:.6}", metrics.val_loss));

        let safetensor_bytes = safetensors::serialize(views, Some(st_metadata)).map_err(|e| {
            crate::Error::Serialization(format!("SafeTensors serialization failed: {e}"))
        })?;
        std::fs::write(path.join("model.safetensors"), safetensor_bytes)?;

        contract_post_save_checkpoint!(());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finetune::instruct_pipeline::InstructConfig;
    use crate::transformer::TransformerConfig;

    fn make_corpus(n: usize) -> Vec<InstructSample> {
        (0..n)
            .map(|i| InstructSample {
                instruction: format!("Write function {i}"),
                response: format!("def func_{i}():\n    return {i}"),
                system: None,
                metadata: None,
            })
            .collect()
    }

    #[test]
    fn test_trainer_creation() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 32, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let corpus = make_corpus(20);
        let config = InstructTrainingConfig { epochs: 2, ..Default::default() };

        let trainer = InstructTrainer::new(pipeline, corpus, config);
        assert!(trainer.is_ok());

        let trainer = trainer.unwrap();
        assert!(trainer.train_size() > 0);
        assert!(trainer.val_size() > 0);
    }

    #[test]
    fn test_trainer_empty_corpus() {
        let model_config = TransformerConfig::tiny();
        let instruct_config = InstructConfig::default();
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let config = InstructTrainingConfig::default();

        let result = InstructTrainer::new(pipeline, vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_train() {
        let model_config = TransformerConfig::tiny();
        let instruct_config =
            InstructConfig { lora_rank: 4, max_seq_len: 32, ..InstructConfig::default() };
        let pipeline = InstructPipeline::new(&model_config, instruct_config);
        let corpus = make_corpus(10);
        let config = InstructTrainingConfig { epochs: 2, save_every: 1, ..Default::default() };

        let mut trainer = InstructTrainer::new(pipeline, corpus, config).unwrap();
        let result = trainer.train();

        assert_eq!(result.epoch_metrics.len(), 2);
        assert!(result.best_val_loss >= 0.0);
        assert!(result.total_time_ms > 0);
    }

    #[test]
    fn test_data_hash_deterministic() {
        let corpus = make_corpus(5);
        let hash1 = InstructTrainer::compute_data_hash(&corpus);
        let hash2 = InstructTrainer::compute_data_hash(&corpus);
        assert_eq!(hash1, hash2);
        assert!(hash1.starts_with("sha256:"));
    }

    #[test]
    fn test_split_disjoint() {
        let corpus = make_corpus(20);
        let (train, val) = InstructTrainer::split_dataset(&corpus, 0.2, 42);
        assert_eq!(train.len() + val.len(), 20);
        assert!(!train.is_empty());
        assert!(!val.is_empty());
    }
}
