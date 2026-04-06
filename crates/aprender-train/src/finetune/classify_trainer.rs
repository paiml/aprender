//! Production training loop for classification fine-tuning (SSC-026)
//!
//! `ClassifyTrainer` wraps `ClassifyPipeline` with epoch management,
//! validation, checkpointing, LR scheduling, and early stopping.
//!
//! # Contract Invariants
//!
//! - F-LOOP-001: EMA loss decreasing over training (alpha=0.1, 5-epoch window)
//! - F-LOOP-002: Validation computed every epoch
//! - F-LOOP-007: Data shuffled per epoch (different order)
//! - F-LOOP-008: Val split disjoint (zero overlap with training set)
//! - F-LOOP-009: Val set frozen (same composition across epochs)
//! - F-LOOP-010: Early stopping respects patience

use super::classification::{SafetySample, TokenizedSample};
use super::classify_eval_report::ClassifyEvalReport;
use super::classify_pipeline::ClassifyPipeline;
use super::distributed::DistributedConfig;
use crate::optim::LRScheduler;
use crate::optim::WarmupCosineDecayLR;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Training configuration for the classification trainer.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs (default: 50)
    pub epochs: usize,
    /// Fraction of data reserved for validation (default: 0.2)
    pub val_split: f32,
    /// Save checkpoint every N epochs (default: 5)
    pub save_every: usize,
    /// Early stopping patience in epochs (default: 10)
    pub early_stopping_patience: usize,
    /// Directory for checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Random seed for reproducibility (default: 42)
    pub seed: u64,
    /// Log metrics every N epochs (default: 1)
    pub log_interval: usize,
    /// Warmup steps as fraction of total steps (default: 0.1)
    pub warmup_fraction: f32,
    /// Minimum learning rate for cosine decay (default: 1e-6)
    pub lr_min: f32,
    /// Oversample minority classes to match majority count (default: false).
    /// When enabled, duplicates minority-class samples and skips auto class weights.
    pub oversample_minority: bool,
    /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (default: false).
    ///
    /// When enabled, transformer blocks use `CudaNf4TransformerBlock` instead of
    /// `CudaTransformerBlock`, achieving ~8x VRAM compression on frozen weights.
    /// Only LoRA adapters remain trainable in fp32.
    pub quantize_nf4: bool,
    /// Distributed training configuration (multi-node TCP gradient AllReduce).
    ///
    /// When set, the trainer operates in either coordinator or worker mode:
    /// - Coordinator: manages epochs, shards data, AllReduces gradients (F-DP-001)
    /// - Worker: receives shards, computes forward/backward, sends gradients
    pub distributed: Option<DistributedConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            val_split: 0.2,
            save_every: 5,
            early_stopping_patience: 10,
            checkpoint_dir: PathBuf::from("checkpoints"),
            seed: 42,
            log_interval: 1,
            warmup_fraction: 0.1,
            lr_min: 1e-6,
            oversample_minority: false,
            quantize_nf4: false,
            distributed: None,
        }
    }
}

/// Metrics for a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f32,
    /// Training accuracy (0.0-1.0)
    pub train_accuracy: f32,
    /// Average validation loss
    pub val_loss: f32,
    /// Validation accuracy (0.0-1.0)
    pub val_accuracy: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Epoch wall-clock time in milliseconds
    pub epoch_time_ms: u64,
    /// Training throughput (samples/second)
    pub samples_per_sec: f32,
}

/// Result of the full training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Per-epoch metrics
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Epoch with lowest validation loss
    pub best_epoch: usize,
    /// Lowest validation loss achieved
    pub best_val_loss: f32,
    /// Whether training stopped early
    pub stopped_early: bool,
    /// Total wall-clock training time in milliseconds
    pub total_time_ms: u64,
}

/// Production training loop for classification fine-tuning.
///
/// Wraps `ClassifyPipeline` with:
/// - Epoch management with per-epoch shuffling
/// - Validation on a disjoint, frozen split
/// - Warmup + cosine decay LR scheduling
/// - Periodic checkpointing (SafeTensors + metadata JSON)
/// - Early stopping with configurable patience
pub struct ClassifyTrainer {
    /// The classification pipeline (model + optimizer)
    pipeline: ClassifyPipeline,
    /// Training configuration
    config: TrainingConfig,
    /// Training data (shuffled per epoch)
    train_data: Vec<SafetySample>,
    /// Pre-tokenized training data — indices parallel `train_data` (KAIZEN-028).
    /// Token IDs computed once at construction; shuffled in sync with `train_data`.
    train_tokens: Vec<TokenizedSample>,
    /// Pre-tokenized validation data (frozen, KAIZEN-028).
    val_tokens: Vec<TokenizedSample>,
    /// Validation data (frozen, never shuffled)
    val_data: Vec<SafetySample>,
    /// Base random seed
    rng_seed: u64,
    /// Optional monitor writer for live TUI updates
    monitor_writer: Option<crate::monitor::tui::TrainingStateWriter>,
    /// SHA-256 hash of training data for provenance (F-CKPT-017)
    data_hash: String,
    /// Training start timestamp (ISO 8601) for provenance
    train_start: String,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for ClassifyTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassifyTrainer")
            .field("config", &self.config)
            .field("train_data_len", &self.train_data.len())
            .field("train_tokens_len", &self.train_tokens.len())
            .field("val_data_len", &self.val_data.len())
            .field("val_tokens_len", &self.val_tokens.len())
            .field("rng_seed", &self.rng_seed)
            .finish()
    }
}

impl ClassifyTrainer {
    /// Create a new trainer by splitting corpus into train/val sets.
    ///
    /// # Arguments
    /// * `pipeline` - Initialized `ClassifyPipeline`
    /// * `corpus` - Full dataset of labeled samples
    /// * `config` - Training configuration
    ///
    /// # Errors
    /// Returns error if corpus is empty, val_split is out of (0.0, 0.5],
    /// or epochs is 0.
    pub fn new(
        mut pipeline: ClassifyPipeline,
        corpus: Vec<SafetySample>,
        config: TrainingConfig,
    ) -> crate::Result<Self> {
        if corpus.is_empty() {
            return Err(crate::Error::ConfigError("SSC-026: corpus must not be empty".to_string()));
        }
        if config.val_split <= 0.0 || config.val_split > 0.5 {
            return Err(crate::Error::ConfigError(format!(
                "SSC-026: val_split must be in (0.0, 0.5], got {}",
                config.val_split,
            )));
        }
        if config.epochs == 0 {
            return Err(crate::Error::ConfigError("SSC-026: epochs must be > 0".to_string()));
        }

        // ── Auto-detect class imbalance and apply weights ────────────────
        // Skip when oversampling: data will be balanced, so weights are unnecessary.
        if !config.oversample_minority {
            Self::auto_balance_classes(&mut pipeline, &corpus);
        }

        let (mut train_data, val_data) =
            Self::split_dataset(&corpus, config.val_split, config.seed);

        if config.oversample_minority {
            Self::oversample_training_data(&mut train_data, config.seed);
        }

        if train_data.is_empty() || val_data.is_empty() {
            return Err(crate::Error::ConfigError(format!(
                "SSC-026: split produced empty set (train={}, val={}). Need more samples.",
                train_data.len(),
                val_data.len(),
            )));
        }

        let rng_seed = config.seed;

        // KAIZEN-028: Pre-tokenize all samples once at construction time.
        // Eliminates redundant BPE encoding across epochs and batches.
        // For 17,942 SSC samples × 50 epochs = 897,100 tokenizations reduced to 17,942.
        let train_tokens = pipeline.pre_tokenize(&train_data);
        let val_tokens = pipeline.pre_tokenize(&val_data);

        // F-CKPT-017: Compute data hash for provenance
        let data_hash = Self::compute_data_hash(&corpus);
        let train_start = chrono::Utc::now().to_rfc3339();

        Ok(Self {
            pipeline,
            config,
            train_data,
            train_tokens,
            val_tokens,
            val_data,
            rng_seed,
            monitor_writer: None,
            data_hash,
            train_start,
        })
    }

    /// Compute SHA-256 hash of training corpus for provenance (F-CKPT-017).
    ///
    /// Hash is computed over sorted (input, label) pairs for determinism.
    fn compute_data_hash(corpus: &[SafetySample]) -> String {
        let mut hasher = Sha256::new();
        let mut sorted: Vec<(&str, usize)> =
            corpus.iter().map(|s| (s.input.as_str(), s.label)).collect();
        sorted.sort_unstable();
        for (input, label) in &sorted {
            hasher.update(input.as_bytes());
            hasher.update([0u8]); // separator
            hasher.update(label.to_le_bytes());
        }
        let result = hasher.finalize();
        format!("sha256:{result:x}")
    }

    /// Auto-detect class imbalance and apply sqrt-inverse weights when no
    /// explicit weights are configured.
    ///
    /// World-class training frameworks (sklearn, HuggingFace Trainer) auto-balance
    /// by default. A training run on imbalanced data with uniform weights silently
    /// optimizes for majority-class accuracy — the model learns to never predict
    /// minority classes.
    ///
    /// Threshold: if max_count / min_count > 2.0, imbalance is detected.
    /// Strategy: `SqrtInverse` (moderate rebalancing, avoids overadjust).
    fn auto_balance_classes(pipeline: &mut ClassifyPipeline, corpus: &[SafetySample]) {
        use super::classification::{compute_class_weights, corpus_stats, ClassWeightStrategy};

        // Skip if user explicitly configured weights
        if pipeline.config.class_weights.is_some() {
            return;
        }

        let num_classes = pipeline.config.num_classes;
        let stats = corpus_stats(corpus, num_classes);

        // Check if any class is missing entirely
        let min_count = stats.class_counts.iter().copied().min().unwrap_or(0);
        let max_count = stats.class_counts.iter().copied().max().unwrap_or(1);

        if min_count == 0 {
            println!(
                "  Warning: class with zero samples detected. \
                 Class weights not applied (would produce Inf)."
            );
            return;
        }

        let imbalance_ratio = max_count as f64 / min_count as f64;

        if imbalance_ratio > 2.0 {
            let weights =
                compute_class_weights(&stats, ClassWeightStrategy::SqrtInverse, num_classes);
            println!(
                "  Auto-detected class imbalance (ratio {imbalance_ratio:.1}:1), \
                 applying sqrt-inverse weights: {weights:?}"
            );
            println!("  Class counts: {:?} (total: {})", stats.class_counts, stats.total);
            pipeline.config.class_weights = Some(weights);
        } else {
            println!("  Class balance OK (ratio {imbalance_ratio:.1}:1), using uniform weights");
        }
    }

    /// Oversample minority classes by duplicating samples until each class
    /// matches the majority count.
    ///
    /// This is a simple, effective strategy for moderate imbalance (e.g. 93/7 splits).
    /// After oversampling the training set is shuffled deterministically.
    fn oversample_training_data(train_data: &mut Vec<SafetySample>, seed: u64) {
        use std::collections::HashMap;

        // Count per-class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, sample) in train_data.iter().enumerate() {
            class_indices.entry(sample.label).or_default().push(i);
        }

        let majority_count = class_indices.values().map(std::vec::Vec::len).max().unwrap_or(0);
        let before = train_data.len();

        // Duplicate minority samples (cycling) to match majority
        for indices in class_indices.values() {
            let count = indices.len();
            if count < majority_count {
                let deficit = majority_count - count;
                for i in 0..deficit {
                    let src_idx = indices[i % count];
                    train_data.push(train_data[src_idx].clone());
                }
            }
        }

        // Deterministic shuffle via Fisher-Yates with simple LCG
        let n = train_data.len();
        let mut rng_state: u64 = seed.wrapping_mul(0x517cc1b727220a95).wrapping_add(1);
        for i in (1..n).rev() {
            rng_state =
                rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            train_data.swap(i, j);
        }

        println!(
            "  Oversampled minority classes: {before} \u{2192} {} training samples",
            train_data.len()
        );
    }

    /// Attach a monitor writer for live TUI updates.
    ///
    /// When set, training emits per-batch metrics to the experiment directory
    /// via atomic JSON writes, enabling `apr monitor <dir>` from another shell.
    pub fn set_monitor_writer(&mut self, writer: crate::monitor::tui::TrainingStateWriter) {
        self.monitor_writer = Some(writer);
    }

    /// Run the full training loop.
    ///
    /// For each epoch:
    /// 1. Shuffle training data (deterministic, seed varies per epoch)
    /// 2. Process batches via `pipeline.train_batch()`
    /// 3. Compute validation metrics (forward-only)
    /// 4. Step LR scheduler
    /// 5. Record metrics
    /// 6. Save checkpoint if `save_every` or new best val_loss
    /// 7. Check early stopping
    pub fn train(&mut self) -> TrainResult {
        // Dispatch to coordinator-mode training if distributed config is set
        if self.is_coordinator_mode() {
            return self.train_as_coordinator();
        }

        let total_start = std::time::Instant::now();
        let batch_size = self.pipeline.config.batch_size;
        let batches_per_epoch = self.train_data.len().div_ceil(batch_size);
        let total_steps = self.config.epochs * batches_per_epoch;
        let warmup_steps = (self.config.warmup_fraction * total_steps as f32) as usize;
        let lr_max = self.pipeline.optimizer_lr();

        let mut scheduler =
            WarmupCosineDecayLR::new(lr_max, self.config.lr_min, warmup_steps, total_steps);

        // Initialize monitor writer if attached
        if let Some(ref mut writer) = self.monitor_writer {
            writer.set_epochs(self.config.epochs, batches_per_epoch);
            let _ = writer.start();
        }

        let mut epoch_metrics_vec: Vec<EpochMetrics> = Vec::with_capacity(self.config.epochs);
        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch: usize = 0;
        let mut epochs_without_improvement: usize = 0;
        let mut stopped_early = false;
        let mut training_failed = false;

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // F-LOOP-007: Shuffle training data with epoch-specific seed
            self.shuffle_training_data(epoch);

            // Train one epoch
            let (train_loss, train_accuracy) = self.train_epoch(&mut scheduler, epoch);

            // F-LOOP-002: Validate every epoch
            let (val_loss, val_accuracy) = self.validate();

            let epoch_time = epoch_start.elapsed();
            let epoch_time_ms = epoch_time.as_millis() as u64;
            let samples_per_sec = if epoch_time_ms > 0 {
                self.train_data.len() as f32 / (epoch_time_ms as f32 / 1000.0)
            } else {
                0.0
            };

            let metrics = EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                learning_rate: scheduler.get_lr(),
                epoch_time_ms,
                samples_per_sec,
            };

            epoch_metrics_vec.push(metrics.clone());

            // Epoch summary via monitoring framework
            let is_best = val_loss < best_val_loss;
            if let Some(ref writer) = self.monitor_writer {
                writer.emit_epoch_summary(
                    epoch + 1,
                    self.config.epochs,
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                    epoch_time.as_secs_f32(),
                    scheduler.get_lr(),
                    is_best,
                );
            }

            // Track best validation loss
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;

                // Save best checkpoint
                let best_path = self.config.checkpoint_dir.join("best");
                let _ = self.save_checkpoint(&best_path, epoch, &metrics);
            } else {
                epochs_without_improvement += 1;
            }

            // Periodic checkpoint — when epochs <= save_every, save every epoch
            let effective_save_every = if self.config.epochs <= self.config.save_every {
                1
            } else {
                self.config.save_every
            };
            if effective_save_every > 0 && (epoch + 1) % effective_save_every == 0 {
                let epoch_path = self.config.checkpoint_dir.join(format!("epoch-{epoch}"));
                let _ = self.save_checkpoint(&epoch_path, epoch, &metrics);
            }

            // Detect NaN/Inf loss — signal failure to monitor
            if !train_loss.is_finite() || !val_loss.is_finite() {
                if let Some(ref mut writer) = self.monitor_writer {
                    let _ = writer.fail("NaN or Inf loss detected");
                }
                training_failed = true;
                stopped_early = true;
                break;
            }

            // F-LOOP-010: Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience {
                stopped_early = true;
                break;
            }
        }

        // Signal training completion to monitor (skip if already failed)
        if !training_failed {
            if let Some(ref mut writer) = self.monitor_writer {
                let _ = writer.complete();
            }
        }

        let total_time_ms = total_start.elapsed().as_millis() as u64;

        TrainResult {
            epoch_metrics: epoch_metrics_vec,
            best_epoch,
            best_val_loss,
            stopped_early,
            total_time_ms,
        }
    }

    /// Run training as the distributed coordinator.
    ///
    /// Starts a `GradientServer`, waits for workers, then runs the full
    /// training loop with distributed AllReduce gradient averaging.
    ///
    /// # Contract: F-DP-001 (Weight Consistency)
    ///
    /// After each AllReduce step, all workers receive identical averaged
    /// gradients and apply the same optimizer step.
    fn train_as_coordinator(&mut self) -> TrainResult {
        use super::gradient_server::GradientServer;

        let dist_config = self
            .config
            .distributed
            .clone()
            .expect("train_as_coordinator requires distributed config");

        let total_start = std::time::Instant::now();

        // Bind gradient server
        let mut server = match GradientServer::bind(dist_config) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[coordinator] Failed to bind: {e}");
                return TrainResult {
                    epoch_metrics: vec![],
                    best_epoch: 0,
                    best_val_loss: f32::INFINITY,
                    stopped_early: true,
                    total_time_ms: total_start.elapsed().as_millis() as u64,
                };
            }
        };

        // Wait for all workers to connect
        if let Err(e) = server.wait_for_workers() {
            eprintln!("[coordinator] Worker connection failed: {e}");
            return TrainResult {
                epoch_metrics: vec![],
                best_epoch: 0,
                best_val_loss: f32::INFINITY,
                stopped_early: true,
                total_time_ms: total_start.elapsed().as_millis() as u64,
            };
        }

        let num_workers = server.worker_count();
        server.set_total_samples(self.train_data.len());

        eprintln!(
            "[coordinator] Starting training: {} epochs, {} workers, {} samples",
            self.config.epochs,
            num_workers,
            self.train_data.len(),
        );

        let mut epoch_metrics_vec: Vec<EpochMetrics> = Vec::with_capacity(self.config.epochs);
        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch = 0usize;
        let mut stopped_early = false;

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            self.shuffle_training_data(epoch);

            let batch_size = self.pipeline.config.batch_size;
            let mut total_loss = 0.0f32;
            let mut total_correct = 0usize;
            let mut total_samples = 0usize;

            // KAIZEN-032: Borrow pre-tokenized data directly — no per-epoch clone.
            for (step_idx, chunk) in self.train_tokens.chunks(batch_size).enumerate() {
                let step =
                    epoch as u64 * (self.train_tokens.len() / batch_size) as u64 + step_idx as u64;

                // Send shard assignments to workers
                if let Err(e) = server.send_shard_assignments(step) {
                    eprintln!("[coordinator] Shard assignment failed at step {step}: {e}");
                    stopped_early = true;
                    break;
                }

                // Coordinator also computes its own shard (local forward/backward)
                let _local = self.pipeline.train_batch_tokenized(chunk);

                // Collect and average gradients from all workers (F-DP-001)
                match server.collect_and_reduce(step) {
                    Ok(allreduce) => {
                        // Apply averaged gradients locally
                        self.pipeline.apply_lora_gradients(&allreduce.avg_gradients);

                        // Broadcast to workers
                        if let Err(e) = server.broadcast_averaged(step, &allreduce) {
                            eprintln!("[coordinator] Broadcast failed at step {step}: {e}");
                            stopped_early = true;
                            break;
                        }

                        total_loss += allreduce.global_loss * allreduce.total_samples as f32;
                        total_correct += allreduce.total_correct;
                        total_samples += allreduce.total_samples;
                    }
                    Err(e) => {
                        eprintln!("[coordinator] AllReduce failed at step {step}: {e}");
                        stopped_early = true;
                        break;
                    }
                }
            }

            if stopped_early {
                break;
            }

            let avg_loss = if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };
            let accuracy =
                if total_samples > 0 { total_correct as f32 / total_samples as f32 } else { 0.0 };

            // Validate on coordinator's local val set
            let (val_loss, val_accuracy) = self.validate();

            let epoch_time_ms = epoch_start.elapsed().as_millis() as u64;
            let samples_per_sec = if epoch_time_ms > 0 {
                total_samples as f32 / (epoch_time_ms as f32 / 1000.0)
            } else {
                0.0
            };

            let metrics = EpochMetrics {
                epoch,
                train_loss: avg_loss,
                train_accuracy: accuracy,
                val_loss,
                val_accuracy,
                learning_rate: self.pipeline.optimizer_lr(),
                epoch_time_ms,
                samples_per_sec,
            };

            eprintln!(
                "[coordinator] Epoch {}: loss={:.4}, acc={:.1}%, val_loss={:.4}, val_acc={:.1}%",
                epoch + 1,
                avg_loss,
                accuracy * 100.0,
                val_loss,
                val_accuracy * 100.0,
            );

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;

                let best_path = self.config.checkpoint_dir.join("best");
                let _ = self.save_checkpoint(&best_path, epoch, &metrics);
            }

            epoch_metrics_vec.push(metrics);
        }

        server.shutdown_workers();

        TrainResult {
            epoch_metrics: epoch_metrics_vec,
            best_epoch,
            best_val_loss,
            stopped_early,
            total_time_ms: total_start.elapsed().as_millis() as u64,
        }
    }

    /// Train for one epoch, processing all training data in batches.
    ///
    /// Returns `(avg_loss, accuracy)` for the epoch.
    fn train_epoch(&mut self, scheduler: &mut WarmupCosineDecayLR, epoch: usize) -> (f32, f32) {
        let batch_size = self.pipeline.config.batch_size;
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        let epoch_start = std::time::Instant::now();

        // KAIZEN-032: Borrow pre-tokenized data directly — no per-epoch clone.
        for (batch_idx, chunk) in self.train_tokens.chunks(batch_size).enumerate() {
            // Apply current LR from scheduler
            self.pipeline.set_optimizer_lr(scheduler.get_lr());

            let result = self.pipeline.train_batch_tokenized(chunk);
            total_loss += result.avg_loss * result.total as f32;
            total_correct += result.correct;
            total_samples += result.total;

            let running_avg_loss =
                if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };
            let elapsed_secs = epoch_start.elapsed().as_secs_f32();
            let samples_per_sec =
                if elapsed_secs > 0.0 { total_samples as f32 / elapsed_secs } else { 0.0 };
            let current_lr = scheduler.get_lr();

            let step = batch_idx + 1;
            let acc =
                if total_samples > 0 { total_correct as f32 / total_samples as f32 } else { 0.0 };

            // Emit per-batch metrics to monitor (JSON state file + optional console)
            if let Some(ref mut writer) = self.monitor_writer {
                let _ = writer.update_step(
                    epoch + 1,
                    step,
                    running_avg_loss,
                    current_lr,
                    result.grad_norm,
                    samples_per_sec,
                    acc,
                );
            }

            // Step scheduler per batch
            scheduler.step();
        }

        let avg_loss = if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };
        let accuracy =
            if total_samples > 0 { total_correct as f32 / total_samples as f32 } else { 0.0 };

        (avg_loss, accuracy)
    }

    /// Compute validation metrics (forward-only, no gradient updates).
    ///
    /// F-LOOP-009: Val set is frozen — same samples every epoch.
    /// KAIZEN-013: Uses pre-tokenized cache and reports progress.
    fn validate(&mut self) -> (f32, f32) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let total = self.val_tokens.len();

        let val_start = std::time::Instant::now();

        // KAIZEN-028: Use pre-tokenized validation data — no BPE re-encoding.
        // KAIZEN-013: Progress reporting with timing and running accuracy.
        for (i, sample) in self.val_tokens.iter().enumerate() {
            let (loss, predicted) = self.pipeline.forward_only(&sample.token_ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
            // Progress reporting every 100 samples
            if (i + 1) % 100 == 0 || i + 1 == total {
                let elapsed = val_start.elapsed().as_secs_f32();
                let sam_per_sec = if elapsed > 0.0 { (i + 1) as f32 / elapsed } else { 0.0 };
                let running_acc = if i > 0 { correct as f32 / (i + 1) as f32 * 100.0 } else { 0.0 };
                eprint!(
                    "\r  Validating: {}/{} ({:.1} sam/s, acc={:.1}%)   ",
                    i + 1,
                    total,
                    sam_per_sec,
                    running_acc,
                );
            }
        }

        let val_elapsed = val_start.elapsed();
        let val_sam_per_sec = if val_elapsed.as_secs_f32() > 0.0 {
            total as f32 / val_elapsed.as_secs_f32()
        } else {
            0.0
        };
        eprintln!(
            "\r  Validation complete: {} samples in {:.1}s ({:.1} sam/s)              ",
            total,
            val_elapsed.as_secs_f32(),
            val_sam_per_sec,
        );

        let avg_loss = if total > 0 { total_loss / total as f32 } else { 0.0 };
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        (avg_loss, accuracy)
    }

    /// Shuffle training data using Fisher-Yates with epoch-dependent seed.
    ///
    /// F-LOOP-007: `seed = base_seed + epoch` ensures different order per epoch
    /// but deterministic across runs.
    fn shuffle_training_data(&mut self, epoch: usize) {
        let seed = self.rng_seed.wrapping_add(epoch as u64);
        let mut rng_state = seed;
        let n = self.train_data.len();

        // Fisher-Yates shuffle with LCG PRNG
        // KAIZEN-028: Shuffle train_tokens in sync with train_data
        for i in (1..n).rev() {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng_state >> 33) as usize % (i + 1);
            self.train_data.swap(i, j);
            self.train_tokens.swap(i, j);
        }
    }

    /// Save checkpoint with metadata JSON and SafeTensors model weights.
    ///
    /// When GPU training is active, downloads GPU-updated transformer weights
    /// to CPU before saving so checkpoints include all trained parameters.
    ///
    /// Creates: `{path}/metadata.json` and `{path}/model.safetensors`
    ///
    /// # Contract (C-CKPT-001)
    ///
    /// - **Precondition**: `path` is a writable directory (or will be created)
    /// - **Postcondition**: Checkpoint contains all trained parameters including
    ///   GPU-updated transformer block weights (if GPU training active)
    /// - **Invariant**: CPU model state is consistent with GPU state after save
    pub fn save_checkpoint(
        &mut self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        contract_pre_save_checkpoint!();
        // Sync GPU weights to CPU before saving (no-op if GPU training inactive)
        #[cfg(feature = "cuda")]
        self.pipeline.sync_weights_to_cpu();
        std::fs::create_dir_all(path).map_err(|e| {
            crate::Error::Io(format!("Failed to create checkpoint dir {}: {e}", path.display()))
        })?;

        // Save metadata.json
        let metadata = serde_json::json!({
            "epoch": epoch,
            "train_loss": metrics.train_loss,
            "train_accuracy": metrics.train_accuracy,
            "val_loss": metrics.val_loss,
            "val_accuracy": metrics.val_accuracy,
            "learning_rate": metrics.learning_rate,
            "epoch_time_ms": metrics.epoch_time_ms,
            "samples_per_sec": metrics.samples_per_sec,
            "class_weights": self.pipeline.config.class_weights,
        });

        let meta_path = path.join("metadata.json");
        let meta_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize metadata: {e}"))
        })?;
        std::fs::write(&meta_path, meta_json)?;

        // Save model weights as SafeTensors (classifier head + LoRA adapters)
        let params = self.pipeline.classifier.parameters();
        let st_path = path.join("model.safetensors");

        // Collect classifier head tensor data
        let tensor_names = ["classifier.weight", "classifier.bias"];
        let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = params
            .iter()
            .zip(tensor_names.iter())
            .map(|(tensor, name)| {
                let data = tensor.data();
                let bytes: Vec<u8> =
                    bytemuck::cast_slice(data.as_slice().expect("contiguous")).to_vec();
                let shape = vec![tensor.len()];
                (name.to_string(), bytes, shape)
            })
            .collect();

        // Collect LoRA adapter weights (F-CLASS-008: Q/V projections)
        // Convention: 2 adapters per layer (Q=even, V=odd)
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
        std::fs::write(&st_path, safetensor_bytes)?;

        // Save APR format (full training state)
        self.save_apr_checkpoint(path, epoch, metrics)?;

        // Save adapter-only APR (F-CKPT-003: no __training__.* tensors)
        self.save_adapter_apr(path, epoch, metrics)?;

        // ── HuggingFace-compatible metadata (config.json, adapter_config.json, tokenizer.json) ──

        // config.json: HF model architecture config
        let model_config = &self.pipeline.model.config;
        let hf_config = serde_json::json!({
            "architectures": ["Qwen2ForSequenceClassification"],
            "model_type": "qwen2",
            "hidden_size": model_config.hidden_size,
            "num_attention_heads": model_config.num_attention_heads,
            "num_key_value_heads": model_config.num_kv_heads,
            "intermediate_size": model_config.intermediate_size,
            "num_hidden_layers": model_config.num_hidden_layers,
            "vocab_size": model_config.vocab_size,
            "max_position_embeddings": model_config.max_position_embeddings,
            "rms_norm_eps": model_config.rms_norm_eps,
            "rope_theta": model_config.rope_theta,
            "use_cache": true,
            "torch_dtype": "float32",
            "num_labels": self.pipeline.config.num_classes,
            "problem_type": "single_label_classification",
        });
        let config_json = serde_json::to_string_pretty(&hf_config).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize config.json: {e}"))
        })?;
        std::fs::write(path.join("config.json"), config_json)?;

        // adapter_config.json: PEFT adapter configuration
        let lora_config = crate::lora::LoRAConfig::new(
            self.pipeline.config.lora_rank,
            self.pipeline.config.lora_alpha,
        )
        .target_qv_projections();

        let base_model = self.pipeline.model_dir().map(|p| p.display().to_string());

        let peft_config =
            crate::lora::PeftAdapterConfig::from_lora_config(&lora_config, base_model.as_deref())
                .with_task_type("SEQ_CLS");

        let adapter_json = peft_config.to_json().map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize adapter_config.json: {e}"))
        })?;
        std::fs::write(path.join("adapter_config.json"), adapter_json)?;

        // tokenizer.json: copy from base model directory (if available)
        if let Some(model_dir) = self.pipeline.model_dir() {
            let src = model_dir.join("tokenizer.json");
            if src.exists() {
                std::fs::copy(&src, path.join("tokenizer.json"))
                    .map_err(|e| crate::Error::Io(format!("Failed to copy tokenizer.json: {e}")))?;
            }
        }

        contract_post_save_checkpoint!(());
        Ok(())
    }

    /// Save model in APR format with full training state.
    ///
    /// # Contract (F-CKPT-001, F-CKPT-004, F-CKPT-005)
    ///
    /// - **F-CKPT-001**: All adapter tensors (classifier + LoRA A/B)
    /// - **F-CKPT-004**: Optimizer state (`__training__.optimizer.*`)
    /// - **F-CKPT-005**: Training metadata (epoch, LR, step count)
    ///
    /// Inference readers skip `__training__.*` via `AprReader::open_filtered()`.
    fn save_apr_checkpoint(
        &self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        use aprender::serialization::apr::AprWriter;

        let mut writer = AprWriter::new();

        // ── Schema version (F-CKPT-002) ─────────────────────────────────
        writer
            .set_metadata("__checkpoint__.schema_version".to_string(), serde_json::json!("1.2.0"));

        // ── Rich metadata ────────────────────────────────────────────────
        writer.set_metadata("model_type".to_string(), serde_json::json!("adapter"));
        writer.set_metadata("epoch".to_string(), serde_json::json!(epoch));
        writer.set_metadata("val_loss".to_string(), serde_json::json!(metrics.val_loss));
        writer.set_metadata("val_accuracy".to_string(), serde_json::json!(metrics.val_accuracy));
        writer.set_metadata("train_loss".to_string(), serde_json::json!(metrics.train_loss));
        writer
            .set_metadata("train_accuracy".to_string(), serde_json::json!(metrics.train_accuracy));
        writer.set_metadata("architecture".to_string(), serde_json::json!("qwen2_classify"));
        writer.set_metadata(
            "num_classes".to_string(),
            serde_json::json!(self.pipeline.config.num_classes),
        );
        writer.set_metadata(
            "lora_rank".to_string(),
            serde_json::json!(self.pipeline.config.lora_rank),
        );
        writer.set_metadata(
            "lora_alpha".to_string(),
            serde_json::json!(self.pipeline.config.lora_alpha),
        );
        writer.set_metadata(
            "hidden_size".to_string(),
            serde_json::json!(self.pipeline.model.config.hidden_size),
        );
        writer.set_metadata(
            "num_layers".to_string(),
            serde_json::json!(self.pipeline.model.config.num_hidden_layers),
        );

        // ── Provenance chain (F-CKPT-017) ───────────────────────────────
        writer.set_metadata("data_hash".to_string(), serde_json::json!(self.data_hash));
        if let Some(model_dir) = self.pipeline.model_dir() {
            writer.set_metadata(
                "base_model_source".to_string(),
                serde_json::json!(model_dir.display().to_string()),
            );
        }
        writer.set_metadata(
            "provenance".to_string(),
            serde_json::json!({
                "tool": format!("entrenar v{}", env!("CARGO_PKG_VERSION")),
                "started_at": self.train_start,
            }),
        );

        // ── Classifier head tensors ──────────────────────────────────────
        let weight = &self.pipeline.classifier.weight;
        let weight_data = weight.data();
        let weight_slice = weight_data.as_slice().expect("contiguous weight");
        writer.add_tensor_f32("classifier.weight", vec![weight.len()], weight_slice);

        let bias = &self.pipeline.classifier.bias;
        let bias_data = bias.data();
        let bias_slice = bias_data.as_slice().expect("contiguous bias");
        writer.add_tensor_f32("classifier.bias", vec![bias.len()], bias_slice);

        // ── LoRA adapter tensors (F-CKPT-001: adapter completeness) ──────
        for (idx, lora) in self.pipeline.lora_layers.iter().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            let a_data = lora.lora_a().data();
            let a_slice = a_data.as_slice().expect("contiguous lora_a");
            writer.add_tensor_f32(
                format!("lora.{layer}.{proj}_proj.lora_a"),
                vec![lora.rank(), lora.d_in()],
                a_slice,
            );

            let b_data = lora.lora_b().data();
            let b_slice = b_data.as_slice().expect("contiguous lora_b");
            writer.add_tensor_f32(
                format!("lora.{layer}.{proj}_proj.lora_b"),
                vec![lora.d_out(), lora.rank()],
                b_slice,
            );
        }

        // ── Training state (F-CKPT-004: optimizer moments) ──────────────
        let optimizer = self.pipeline.optimizer();

        // Save AdamW step counter as 1-element tensor
        writer.add_tensor_f32(
            "__training__.optimizer.step",
            vec![1],
            &[optimizer.step_count() as f32],
        );

        // Save first moments (m) and second moments (v)
        for (i, (m_opt, v_opt)) in
            optimizer.first_moments().iter().zip(optimizer.second_moments().iter()).enumerate()
        {
            if let Some(m) = m_opt {
                let m_slice = m.as_slice().expect("contiguous moment m");
                writer.add_tensor_f32(
                    format!("__training__.optimizer.m.{i}"),
                    vec![m.len()],
                    m_slice,
                );
            }
            if let Some(v) = v_opt {
                let v_slice = v.as_slice().expect("contiguous moment v");
                writer.add_tensor_f32(
                    format!("__training__.optimizer.v.{i}"),
                    vec![v.len()],
                    v_slice,
                );
            }
        }

        // ── Training metadata (F-CKPT-005) ──────────────────────────────
        writer.add_tensor_f32("__training__.epoch", vec![1], &[epoch as f32]);
        writer.add_tensor_f32("__training__.learning_rate", vec![1], &[metrics.learning_rate]);

        // ── NaN/Inf check (F-CKPT-007) ──────────────────────────────────
        if !weight_slice.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::Serialization(
                "F-CKPT-007: classifier.weight contains NaN or Inf".to_string(),
            ));
        }
        if !bias_slice.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::Serialization(
                "F-CKPT-007: classifier.bias contains NaN or Inf".to_string(),
            ));
        }
        for (idx, lora) in self.pipeline.lora_layers.iter().enumerate() {
            let a = lora.lora_a().data();
            let b = lora.lora_b().data();
            if !a.iter().all(|v| v.is_finite()) {
                return Err(crate::Error::Serialization(format!(
                    "F-CKPT-007: lora[{idx}].lora_a contains NaN or Inf"
                )));
            }
            if !b.iter().all(|v| v.is_finite()) {
                return Err(crate::Error::Serialization(format!(
                    "F-CKPT-007: lora[{idx}].lora_b contains NaN or Inf"
                )));
            }
        }

        // ── Shape validation (F-CKPT-008) ────────────────────────────────
        let expected_weight_len =
            self.pipeline.config.num_classes * self.pipeline.model.config.hidden_size;
        if weight_slice.len() != expected_weight_len {
            return Err(crate::Error::Serialization(format!(
                "F-CKPT-008: classifier.weight shape mismatch: \
                 expected {} ({}×{}), got {}",
                expected_weight_len,
                self.pipeline.config.num_classes,
                self.pipeline.model.config.hidden_size,
                weight_slice.len(),
            )));
        }
        if bias_slice.len() != self.pipeline.config.num_classes {
            return Err(crate::Error::Serialization(format!(
                "F-CKPT-008: classifier.bias shape mismatch: \
                 expected {}, got {}",
                self.pipeline.config.num_classes,
                bias_slice.len(),
            )));
        }

        let apr_path = path.join("model.apr");
        writer
            .write(&apr_path)
            .map_err(|e| crate::Error::Serialization(format!("APR serialization failed: {e}")))?;

        Ok(())
    }

    /// Save adapter-only APR (no training state) (F-CKPT-003).
    ///
    /// Produces a `.adapter.apr` with zero `__training__.*` tensors.
    /// Used for publishing and inference deployment.
    fn save_adapter_apr(
        &self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        use aprender::serialization::apr::AprWriter;

        let mut writer = AprWriter::new();

        writer
            .set_metadata("__checkpoint__.schema_version".to_string(), serde_json::json!("1.3.0"));
        writer.set_metadata("model_type".to_string(), serde_json::json!("adapter"));
        writer.set_metadata("epoch".to_string(), serde_json::json!(epoch));
        writer.set_metadata("val_loss".to_string(), serde_json::json!(metrics.val_loss));
        writer.set_metadata("val_accuracy".to_string(), serde_json::json!(metrics.val_accuracy));
        writer.set_metadata("architecture".to_string(), serde_json::json!("qwen2_classify"));
        writer.set_metadata(
            "num_classes".to_string(),
            serde_json::json!(self.pipeline.config.num_classes),
        );
        writer.set_metadata(
            "lora_rank".to_string(),
            serde_json::json!(self.pipeline.config.lora_rank),
        );
        writer.set_metadata(
            "lora_alpha".to_string(),
            serde_json::json!(self.pipeline.config.lora_alpha),
        );
        writer.set_metadata(
            "hidden_size".to_string(),
            serde_json::json!(self.pipeline.model.config.hidden_size),
        );
        writer.set_metadata("data_hash".to_string(), serde_json::json!(self.data_hash));
        writer.set_metadata(
            "provenance".to_string(),
            serde_json::json!({
                "tool": format!("entrenar v{}", env!("CARGO_PKG_VERSION")),
                "started_at": self.train_start,
            }),
        );

        // Classifier head
        let weight = &self.pipeline.classifier.weight;
        let weight_data = weight.data();
        let weight_slice = weight_data.as_slice().expect("contiguous weight");
        writer.add_tensor_f32("classifier.weight", vec![weight.len()], weight_slice);

        let bias = &self.pipeline.classifier.bias;
        let bias_data = bias.data();
        let bias_slice = bias_data.as_slice().expect("contiguous bias");
        writer.add_tensor_f32("classifier.bias", vec![bias.len()], bias_slice);

        // LoRA adapters (NO __training__.* tensors — F-CKPT-003)
        for (idx, lora) in self.pipeline.lora_layers.iter().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            let a_data = lora.lora_a().data();
            let a_slice = a_data.as_slice().expect("contiguous lora_a");
            writer.add_tensor_f32(
                format!("lora.{layer}.{proj}_proj.lora_a"),
                vec![lora.rank(), lora.d_in()],
                a_slice,
            );

            let b_data = lora.lora_b().data();
            let b_slice = b_data.as_slice().expect("contiguous lora_b");
            writer.add_tensor_f32(
                format!("lora.{layer}.{proj}_proj.lora_b"),
                vec![lora.d_out(), lora.rank()],
                b_slice,
            );
        }

        let adapter_path = path.join("model.adapter.apr");
        writer.write(&adapter_path).map_err(|e| {
            crate::Error::Serialization(format!("APR adapter serialization failed: {e}"))
        })?;

        Ok(())
    }

    /// Resume training state from an APR checkpoint (F-CKPT-006).
    ///
    /// Loads model weights (classifier + LoRA) and optimizer state
    /// (`__training__.*` tensors) from a `.ckpt.apr` or `model.apr` file.
    ///
    /// Returns the epoch number stored in the checkpoint so the training
    /// loop can resume from the next epoch.
    ///
    /// # Errors
    /// Returns error if checkpoint is invalid or tensors are missing.
    pub fn resume_from_apr_checkpoint(&mut self, apr_path: &Path) -> crate::Result<usize> {
        use aprender::serialization::apr::AprReader;

        let reader = AprReader::open(apr_path).map_err(|e| {
            crate::Error::Serialization(format!("Failed to open APR checkpoint: {e}"))
        })?;

        // ── Data hash verification (F-CKPT-006) ─────────────────────────
        if let Some(saved_hash) = reader.get_metadata("data_hash").and_then(|v| v.as_str()) {
            if saved_hash != self.data_hash {
                return Err(crate::Error::ConfigError(format!(
                    "F-CKPT-006: training data hash mismatch. \
                     Checkpoint: {saved_hash}, current: {}. \
                     Use --allow-data-mismatch to override.",
                    self.data_hash,
                )));
            }
        }

        // ── Shape-config validation (F-CKPT-014) ────────────────────────
        let expected_weight =
            self.pipeline.config.num_classes * self.pipeline.model.config.hidden_size;
        reader
            .validate_tensor_shape("classifier.weight", expected_weight)
            .map_err(|e| crate::Error::Serialization(e))?;
        reader
            .validate_tensor_shape("classifier.bias", self.pipeline.config.num_classes)
            .map_err(|e| crate::Error::Serialization(e))?;

        // ── Restore classifier head (F-CKPT-013: NaN scan) ──────────────
        let weight_data = reader
            .read_tensor_f32_checked("classifier.weight")
            .map_err(|e| crate::Error::Serialization(e))?;
        let bias_data = reader
            .read_tensor_f32_checked("classifier.bias")
            .map_err(|e| crate::Error::Serialization(e))?;

        self.pipeline
            .classifier
            .weight
            .data_mut()
            .as_slice_mut()
            .expect("contiguous weight")
            .copy_from_slice(&weight_data);
        self.pipeline
            .classifier
            .bias
            .data_mut()
            .as_slice_mut()
            .expect("contiguous bias")
            .copy_from_slice(&bias_data);

        // ── Restore LoRA adapters ───────────────────────────────────────
        for (idx, lora) in self.pipeline.lora_layers.iter_mut().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            let a_name = format!("lora.{layer}.{proj}_proj.lora_a");
            let b_name = format!("lora.{layer}.{proj}_proj.lora_b");

            if let Ok(a_data) = reader.read_tensor_f32(&a_name) {
                let a_tensor = lora.lora_a_mut();
                let a_buf = a_tensor.data_mut();
                a_buf.as_slice_mut().expect("contiguous lora_a").copy_from_slice(&a_data);
            }
            if let Ok(b_data) = reader.read_tensor_f32(&b_name) {
                let b_tensor = lora.lora_b_mut();
                let b_buf = b_tensor.data_mut();
                b_buf.as_slice_mut().expect("contiguous lora_b").copy_from_slice(&b_data);
            }
        }

        // ── Restore optimizer state (F-CKPT-004) ────────────────────────
        let optimizer = self.pipeline.optimizer_mut();

        // Restore step counter
        if let Ok(step_data) = reader.read_tensor_f32("__training__.optimizer.step") {
            optimizer.set_step_count(step_data[0] as u64);
        }

        // Restore first and second moments
        for i in 0..256 {
            let m_name = format!("__training__.optimizer.m.{i}");
            let v_name = format!("__training__.optimizer.v.{i}");

            let m_exists = reader.read_tensor_f32(&m_name);
            let v_exists = reader.read_tensor_f32(&v_name);

            match (m_exists, v_exists) {
                (Ok(m_data), Ok(v_data)) => {
                    optimizer.set_first_moment(i, ndarray::Array1::from_vec(m_data));
                    optimizer.set_second_moment(i, ndarray::Array1::from_vec(v_data));
                }
                _ => break, // No more moment buffers
            }
        }

        // ── Restore training metadata (F-CKPT-005) ─────────────────────
        let epoch = if let Ok(epoch_data) = reader.read_tensor_f32("__training__.epoch") {
            epoch_data[0] as usize
        } else {
            // Fall back to metadata
            reader
                .get_metadata("epoch")
                .and_then(serde_json::Value::as_u64)
                .map_or(0, |e| e as usize)
        };

        if let Ok(lr_data) = reader.read_tensor_f32("__training__.learning_rate") {
            self.pipeline.set_optimizer_lr(lr_data[0]);
        }

        println!(
            "  Resumed from APR checkpoint: epoch {epoch}, optimizer step {}",
            self.pipeline.optimizer().step_count(),
        );

        Ok(epoch)
    }

    /// Split dataset into disjoint train/val sets.
    ///
    /// F-LOOP-008: Guarantees zero overlap between train and val.
    /// F-LOOP-009: Val set is deterministic given the same seed.
    ///
    /// # Arguments
    /// * `data` - Full dataset
    /// * `val_ratio` - Fraction for validation (0.0, 0.5]
    /// * `seed` - Random seed for deterministic shuffling
    pub fn split_dataset(
        data: &[SafetySample],
        val_ratio: f32,
        seed: u64,
    ) -> (Vec<SafetySample>, Vec<SafetySample>) {
        if data.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut indices: Vec<usize> = (0..data.len()).collect();

        // Fisher-Yates shuffle with LCG PRNG for determinism
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let val_count = ((data.len() as f32) * val_ratio).ceil() as usize;
        let val_count = val_count.min(data.len() - 1).max(1);

        let val_indices = &indices[..val_count];
        let train_indices = &indices[val_count..];

        let val_data: Vec<SafetySample> = val_indices.iter().map(|&i| data[i].clone()).collect();
        let train_data: Vec<SafetySample> =
            train_indices.iter().map(|&i| data[i].clone()).collect();

        (train_data, val_data)
    }

    /// Get a reference to the training data.
    #[must_use]
    pub fn train_data(&self) -> &[SafetySample] {
        &self.train_data
    }

    /// Get a reference to the validation data.
    #[must_use]
    pub fn val_data(&self) -> &[SafetySample] {
        &self.val_data
    }

    /// Get a reference to the training config.
    #[must_use]
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get a mutable reference to the underlying pipeline.
    pub fn pipeline_mut(&mut self) -> &mut ClassifyPipeline {
        &mut self.pipeline
    }

    /// Check if distributed coordinator mode is configured.
    fn is_coordinator_mode(&self) -> bool {
        self.config
            .distributed
            .as_ref()
            .is_some_and(|d| matches!(d.role, super::distributed::NodeRole::Coordinator))
    }

    /// Run as a distributed worker node.
    ///
    /// Connects to the coordinator, then enters a loop:
    /// 1. Receive shard assignment (or shutdown)
    /// 2. Compute forward/backward on assigned shard
    /// 3. Collect LoRA gradients and send to coordinator
    /// 4. Receive averaged gradients and apply optimizer step
    ///
    /// # Contract: F-DP-001 (Weight Consistency)
    ///
    /// After applying averaged gradients, worker weights match coordinator weights.
    ///
    /// # Errors
    ///
    /// Returns error on connection failure or protocol violation.
    pub fn run_worker(&mut self) -> crate::Result<TrainResult> {
        let dist_config = self.config.distributed.clone().ok_or_else(|| {
            crate::Error::ConfigError("distributed config required for worker mode".into())
        })?;

        let gpu_count = 1u32; // single GPU per worker for now
        let backend = "cpu"; // will be wgpu/cuda when GPU training wired

        let client =
            super::worker_client::WorkerClient::connect(dist_config, gpu_count, backend)
                .map_err(|e| crate::Error::ConfigError(format!("worker connect failed: {e}")))?;

        eprintln!(
            "[worker {}] Connected (total workers: {})",
            client.worker_id(),
            client.total_workers(),
        );

        let total_start = std::time::Instant::now();
        let epoch_metrics_vec: Vec<EpochMetrics> = Vec::new();
        let best_val_loss = f32::INFINITY;
        let best_epoch = 0usize;

        // Clone training data so we can index into it by shard range
        let all_samples: Vec<SafetySample> = self.train_data.clone();

        loop {
            let shard = match client.receive_shard() {
                Ok(Some(s)) => s,
                Ok(None) => {
                    eprintln!("[worker {}] Received shutdown", client.worker_id());
                    break;
                }
                Err(e) => {
                    return Err(crate::Error::ConfigError(format!("shard receive failed: {e}")));
                }
            };

            let step = shard.step;
            let shard_start = shard.shard_start.min(all_samples.len());
            let shard_end = shard.shard_end.min(all_samples.len());
            let shard_data = &all_samples[shard_start..shard_end];

            // Forward + backward on our shard
            let batch_result = self.pipeline.train_batch(shard_data);

            // Collect LoRA gradients
            let gradients = self.pipeline.collect_lora_gradients();

            // Send gradients to coordinator
            client
                .send_gradients(
                    step,
                    gradients,
                    batch_result.avg_loss,
                    batch_result.correct,
                    batch_result.total,
                )
                .map_err(|e| crate::Error::ConfigError(format!("gradient send failed: {e}")))?;

            // Receive averaged gradients
            let averaged = client
                .receive_averaged()
                .map_err(|e| crate::Error::ConfigError(format!("averaged receive failed: {e}")))?;

            // Apply averaged gradients via optimizer step
            self.pipeline.apply_lora_gradients(&averaged.gradients);

            eprintln!(
                "[worker {}] step {step}: loss={:.4}, global_loss={:.4}",
                client.worker_id(),
                batch_result.avg_loss,
                averaged.global_loss,
            );
        }

        Ok(TrainResult {
            epoch_metrics: epoch_metrics_vec,
            best_epoch,
            best_val_loss,
            stopped_early: false,
            total_time_ms: total_start.elapsed().as_millis() as u64,
        })
    }

    /// Evaluate the model on a dataset, returning structured per-class metrics.
    ///
    /// Runs forward-only on every sample, collects predictions, and computes
    /// precision/recall/F1/confusion matrix via `ConfusionMatrix` and `MultiClassMetrics`.
    ///
    /// # Arguments
    /// * `data` - Labeled samples to evaluate on
    /// * `label_names` - Human-readable class names (length must match num_classes)
    pub fn evaluate(
        &mut self,
        data: &[SafetySample],
        label_names: &[String],
    ) -> ClassifyEvalReport {
        let start = std::time::Instant::now();
        let num_classes = self.pipeline.config.num_classes;

        let mut y_true: Vec<usize> = Vec::with_capacity(data.len());
        let mut y_pred: Vec<usize> = Vec::with_capacity(data.len());
        let mut all_probs: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        let mut total_loss = 0.0f32;

        for sample in data {
            let ids = self.pipeline.tokenize(&sample.input);
            let (loss, predicted, probs) =
                self.pipeline.forward_only_with_probs(&ids, sample.label);
            total_loss += loss;
            y_true.push(sample.label);
            y_pred.push(predicted);
            all_probs.push(probs);
        }

        ClassifyEvalReport::from_predictions_with_probs(
            &y_pred,
            &y_true,
            &all_probs,
            total_loss,
            num_classes,
            label_names,
            start.elapsed().as_millis() as u64,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "classify_trainer_tests.rs"]
mod tests;
